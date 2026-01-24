"""Speed benchmark for bartz on GPU."""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from gc import collect
from os import putenv
from time import perf_counter
from typing import Any

from bartz import mcmcloop
from bartz.jaxext import split
from bartz.mcmcstep import State, init, make_p_nonterminal
from bartz.prepcovars import bin_predictors, quantilized_splits_from_matrix
from bartz.testing import gen_data
from equinox import Module, field
from jax import block_until_ready, debug, jit, random
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from jaxtyping import Array, Float32, Key, UInt

# allocate all gpu memory
putenv("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")

# Config
n_over_ntree = 8
fixed_ntree = None  # 200 or None
maxdepth = 6
nvec = [2**p for p in range(1, 30)]
n_over_p = 10
fixed_p = 100  # 100 or None
reps = 2
ndpost_per_rep = 15
cpu_max_memory = 16 * 2**30


class Data(Module):
    X: UInt[Array, "p n"]
    y: Float32[Array, "n"]
    max_split: UInt[Array, "p"]


@partial(jit, static_argnums=(1, 2))
def make_data(key: Key[Array, ""], n: int, p: int) -> Data:
    # generate data
    data = gen_data(
        key,
        n=n,
        p=p,
        k=1,
        q=2,
        sigma2_lin=1 / 3,
        sigma2_quad=1 / 3,
        sigma2_eps=1 / 3,
        lam=0.0,
    )

    # quantize predictors
    splits, max_split = quantilized_splits_from_matrix(data.x, 255)
    X = bin_predictors(data.x, splits)

    # squeeze away multivariate outcome
    y = data.y.squeeze(0)

    return Data(X=X, y=y, max_split=max_split)


def num2si(
    x: float, fmt=lambda x: f"{x:#.3g}".rstrip("."), si: bool = True, space: str = " "
):
    """Format a number using SI prefixes."""
    if x == 0:
        return fmt(x) + space
    exp = int(math.floor(math.log10(abs(x))))
    exp3 = exp - (exp % 3)
    x3 = x / (10**exp3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = space + "yzafpnμm kMGTPEZY"[(exp3 - (-24)) // 3]
    elif exp3 == 0:
        exp3_text = space
    else:
        exp3_text = f"e{exp3}"

    return f"{fmt(x3)}{exp3_text}"


def format_time(t: float):
    """Format a time as multiple of seconds."""
    return f"{num2si(t)}s"


class Benchmark(ABC):
    """Base class for benchmark harnesses."""

    @abstractmethod
    def setup(self, key: Key[Array, ""], data: Data, config: Any):
        """Set up the benchmark initial state."""
        ...

    @abstractmethod
    def run(self, key: Key[Array, ""]):
        """Run the thing being benchmarked, may update internal state."""
        ...


class BartzConfig(Module):
    """Configuration for `Bartz`."""

    n_save: int = field(static=True)
    num_trees: int = field(static=True)


class Bartz(Benchmark):
    """Benchmark harness for the bartz mcmc step."""

    def setup(self, key: Key[Array, ""], data: Data, config: BartzConfig):
        """Create the initial bart state and compile the mcmc loop."""
        print("initialize mcmc state...")
        self.state = init(
            X=data.X,
            y=data.y,
            offset=0.0,
            max_split=data.max_split,
            num_trees=config.num_trees,
            p_nonterminal=make_p_nonterminal(maxdepth, 0.95, 2),
            leaf_prior_cov_inv=jnp.float32(config.num_trees),
            error_cov_df=2.0,
            error_cov_scale=2.0,
            min_points_per_leaf=5,
        )

        print("compile mcmc loop...")

        @partial(jit, donate_argnums=(1,))
        def run_bart(key: Key[Array, ""], bart: State) -> State:
            def callback(**_):
                return debug.callback(lambda: print(".", end="", flush=True))

            bart, _, _ = mcmcloop.run_mcmc(key, bart, config.n_save, callback=callback)
            return bart

        self.run_bart = run_bart.lower(key, self.state).compile()

    def run(self, key: Key[Array, ""]):
        """Run a few iterations of the mcmc and update the state."""
        self.state = block_until_ready(self.run_bart(key, self.state))


def clock(f: Callable, *args: Any) -> float:
    """Time a function call."""
    start = perf_counter()
    f(*args)
    end = perf_counter()
    return end - start


# detect gpu/cpu
device_kind = jnp.empty(0).devices().pop().device_kind


def loop_body(key: Key[Array, ""], n: int, results: dict):
    # determine ntree and p for this n
    ntree = max(1, n // n_over_ntree) if fixed_ntree is None else fixed_ntree
    p = max(1, n // n_over_p) if fixed_p is None else fixed_p
    expected_memory_usage = n * (ntree + p)
    if device_kind == "cpu" and expected_memory_usage > cpu_max_memory:
        # on cpu, jax won't raise out of memory errors, and just hang forever
        return
    print(f"\nn = {n:_}, ntree = {ntree:_}, p = {p:_}")
    print(f"expected memory usage: {num2si(expected_memory_usage)}B")

    # split random seed
    keys = list(random.split(key, reps + 3))
    key = keys.pop()

    try:
        print("generate data...")
        data = make_data(keys.pop(), n, p)

        bench = Bartz()
        bench.setup(keys.pop(), data, BartzConfig(ndpost_per_rep, ntree))

        print("run...")
        times = []
        for i in range(reps):
            print(f"run {i + 1}/{reps} ", end="", flush=True)
            time = clock(bench.run, keys.pop())
            print(
                f" {ndpost_per_rep} iterations in {format_time(time)} ({format_time(time / ndpost_per_rep)} per iteration)"
            )
            times.append(time)
        per_iter = min(times) / ndpost_per_rep

    except JaxRuntimeError as exc:
        # suppress out-of-memory errors, without saving results
        if not exc.args[0].startswith(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate"
        ):
            raise

    else:
        # save results
        results.setdefault("n", []).append(n)
        results.setdefault("time_per_iter", []).append(per_iter)


# random seed
key = random.key(202404151128)

results = {}

for n in nvec:
    # split random key
    keys = split(key)
    key = keys.pop()

    # run benchmark unit
    loop_body(keys.pop(), n, results)

    # free memory
    collect()

# print machine-readable output
print(f"""
    {{
        'package': 'bartz',
        'device_kind': '{device_kind}',
        {"'n/ntree': " + str(n_over_ntree) if fixed_ntree is None else "'ntree': " + str(ntree)},
        {"'n/p': " + str(n_over_p) if fixed_p is None else "'p': " + str(p)},
        'maxdepth': {maxdepth},
        'results': {results},
    }},""")
