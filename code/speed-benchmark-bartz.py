"""Speed benchmark for bartz on GPU."""

import math
from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import partial
from gc import collect
from os import putenv
from time import perf_counter
from typing import Any, Literal

from bartz import mcmcloop
from bartz.jaxext import split
from bartz.mcmcstep import State, init, make_p_nonterminal
from bartz.prepcovars import bin_predictors, quantilized_splits_from_matrix
from bartz.testing import gen_data
from equinox import Module, field
from jax import (
    Device,
    block_until_ready,
    config,
    debug,
    default_device,
    device_put,
    devices,
    jit,
    random,
)
from jax import numpy as jnp
from jax.errors import JaxRuntimeError
from jaxtyping import Array, Float32, Key, UInt


class UnitConfig(Module):
    """Configuration for a single benchmark unit."""

    n: int
    ntree: int
    p: int
    maxdepth: int
    reps: int
    steps_per_rep: int
    cpu_max_memory: int
    device: Device


class Config(Module):
    """General configuration of the script."""

    fixed_ntree: int | None = 200
    fixed_p: int | None = 100
    n_over_ntree: int | None = None
    n_over_p: int | None = None
    maxdepth: int = 6
    # nvec: tuple[int, ...] = tuple(2**p for p in range(1, 30))
    nvec: tuple[int, ...] = tuple(2**p for p in range(1, 5))
    reps: int = 2
    steps_per_rep: int = 15
    cpu_max_memory: int = 16 * 2**30
    seed: int = 2026_01_24_16_54
    platform: Literal["cpu", "gpu"] = "cpu"

    def device(self) -> Device:
        """Get the jax device to use."""
        return devices(self.platform)[0]

    def unit_config(self, n: int) -> UnitConfig:
        """Return the specific config at sample size `n`."""
        return UnitConfig(
            n=n,
            ntree=max(1, n // self.n_over_ntree)
            if self.fixed_ntree is None
            else self.fixed_ntree,
            p=max(1, n // self.n_over_p) if self.fixed_p is None else self.fixed_p,
            maxdepth=self.maxdepth,
            reps=self.reps,
            steps_per_rep=self.steps_per_rep,
            cpu_max_memory=self.cpu_max_memory,
            device=self.device(),
        )


class Data(Module):
    X: UInt[Array, "p n"]
    y: Float32[Array, "n"]
    max_split: UInt[Array, "p"]


def make_data(key: Key[Array, ""], n: int, p: int) -> Data:
    """Generate data on cpu."""
    cpu = devices("cpu")[0]
    key = device_put(key, cpu)
    return _make_data(key, n, p)


@partial(jit, static_argnums=(1, 2))
def _make_data(key: Key[Array, ""], n: int, p: int) -> Data:
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
    def setup(self, key: Key[Array, ""], data: Data, cfg: UnitConfig) -> None:
        """Set up the benchmark initial state."""
        ...

    @abstractmethod
    def run(self, key: Key[Array, ""]):
        """Run the thing being benchmarked, may update internal state."""
        ...


class Bartz(Benchmark):
    """Benchmark harness for the bartz mcmc step."""

    def setup(self, key: Key[Array, ""], data: Data, cfg: UnitConfig):
        """Create the initial bart state and compile the mcmc loop."""
        print("initialize mcmc state...")
        cpu = devices("cpu")[0]
        with default_device(cpu):
            self.state = init(
                X=data.X,
                y=data.y,
                offset=0.0,
                max_split=data.max_split,
                num_trees=cfg.ntree,
                p_nonterminal=make_p_nonterminal(cfg.maxdepth, 0.95, 2),
                leaf_prior_cov_inv=jnp.float32(cfg.ntree),
                error_cov_df=2.0,
                error_cov_scale=2.0,
                min_points_per_leaf=5,
            )
        self.state = device_put(self.state, cfg.device)

        print("compile mcmc loop...")

        @partial(jit, donate_argnums=(1,))
        def run_bart(key: Key[Array, ""], bart: State) -> State:
            def callback(**_):
                return debug.callback(lambda: print(".", end="", flush=True))

            bart, _, _ = mcmcloop.run_mcmc(
                key, bart, cfg.steps_per_rep, callback=callback
            )
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


def loop_body(key: Key[Array, ""], cfg: UnitConfig, results: dict[str, list]):
    # determine ntree and p for this n
    expected_memory_usage = cfg.n * (cfg.ntree + cfg.p)
    if cfg.device.platform == "cpu" and expected_memory_usage > cfg.cpu_max_memory:
        # on cpu, jax won't raise out of memory errors, and just hang forever
        return
    print(f"\nn = {cfg.n:_}, ntree = {cfg.ntree:_}, p = {cfg.p:_}")
    print(f"expected memory usage: {num2si(expected_memory_usage)}B")

    # split random seed
    keys = list(random.split(key, cfg.reps + 3))
    key = keys.pop()

    try:
        print("generate data...")
        data = make_data(keys.pop(), cfg.n, cfg.p)

        # the harness prints its own messages
        bench = Bartz()
        bench.setup(keys.pop(), data, cfg)

        print("run...")
        times = []
        for i in range(cfg.reps):
            print(f"run {i + 1}/{cfg.reps} ", end="", flush=True)
            time = clock(bench.run, keys.pop())
            print(
                f" {cfg.steps_per_rep} iterations in {format_time(time)} ({format_time(time / cfg.steps_per_rep)} per iteration)"
            )
            times.append(time)
        per_iter = min(times) / cfg.steps_per_rep

    except JaxRuntimeError as exc:
        # suppress out-of-memory errors, without saving results
        if not exc.args[0].startswith(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate"
        ):
            raise

    else:
        # save results
        results.setdefault("n", []).append(cfg.n)
        results.setdefault("time_per_iter", []).append(per_iter)


def benchmarking_loop(config: Config) -> dict[str, list]:
    key = random.key(config.seed)

    results = {}
    for n in config.nvec:
        # split random key
        keys = split(key)
        key = keys.pop()

        # run benchmark unit
        loop_body(keys.pop(), config.unit_config(n), results)

        # free memory
        collect()

    return results


def save_results(cfg: Config, results: dict[str, list]):
    """Save results in machine-readable format."""
    print(f"""
{{
    'package': 'bartz',
    'device_kind': '{cfg.device().device_kind}',
    {"'n/ntree': " + str(cfg.n_over_ntree) if cfg.fixed_ntree is None else "'ntree': " + str(cfg.fixed_ntree)},
    {"'n/p': " + str(cfg.n_over_p) if cfg.fixed_p is None else "'p': " + str(cfg.fixed_p)},
    'maxdepth': {cfg.maxdepth},
    'results': {results},
}},""")


def setup_device(cfg: Config):
    """Configure the jax device."""
    match cfg.platform:
        case "cpu":
            # disable gpu altogether
            config.update("jax_platforms", "cpu")
        case "gpu":
            # allocate all gpu memory
            putenv("XLA_PYTHON_CLIENT_MEM_FRACTION", ".99")
            # force an error if gpu not found
            config.update("jax_platforms", "cuda,gpu")
        case _:
            raise ValueError(cfg.platform)


def main():
    """Entry point of the script."""
    cfg = Config()
    setup_device(cfg)
    results = benchmarking_loop(cfg)
    save_results(cfg, results)


if __name__ == "__main__":
    main()
