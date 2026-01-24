"""Speed benchmark for bartz on GPU."""

import os
from dataclasses import replace

from bartz.prepcovars import bin_predictors, quantilized_splits_from_matrix
from jax.errors import JaxRuntimeError

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".99"

import gc
import math
import time
from functools import partial

from bartz import mcmcloop, mcmcstep
from bartz.mcmcstep import make_p_nonterminal
from bartz.testing import gen_data
from jax import block_until_ready, debug, jit, random
from jax import numpy as jnp

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


@partial(jit, static_argnums=(1, 2, 3))
def init(key, p, n, ntree):
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
    data = replace(data, x=X)

    # initialize bart state
    return mcmcstep.init(
        X=data.x,
        y=data.y.squeeze(0),
        offset=0.0,
        max_split=max_split,
        num_trees=ntree,
        p_nonterminal=make_p_nonterminal(maxdepth, 0.95, 2),
        leaf_prior_cov_inv=jnp.float32(ntree),
        error_cov_df=2.0,
        error_cov_scale=2.0,
        min_points_per_leaf=5,
        target_platform="cpu",
    )


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.time = time.perf_counter() - self.start


def num2si(x, fmt=lambda x: f"{x:#.3g}".rstrip("."), si=True, space=" "):
    x = float(x)
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


def format_time(t):
    return f"{num2si(t)}s"


# random seed
key = random.key(202404151128)

# detect gpu/cpu
device_kind = jnp.empty(0).devices().pop().device_kind

results = {}

for n in nvec:
    # determine ntree and p for this n
    ntree = max(1, n // n_over_ntree) if fixed_ntree is None else fixed_ntree
    p = max(1, n // n_over_p) if fixed_p is None else fixed_p
    expected_memory_usage = n * (ntree + p)
    if device_kind == "cpu" and expected_memory_usage > cpu_max_memory:
        break
    print(f"\nn = {n:_}, ntree = {ntree:_}, p = {p:_}")
    print(f"expected memory usage: {expected_memory_usage * 1e-9:.1f} GB")

    # split random seed
    keys = list(random.split(key, reps + 3))
    key = keys.pop()

    try:
        print("initialize...")
        bart = init(keys.pop(), p, n, ntree)

        print("compile...")

        @partial(jit, donate_argnums=(1,))
        def run_bart(key, bart):
            def callback(**_):
                return debug.callback(lambda: print(".", end="", flush=True))

            bart, _, _ = mcmcloop.run_mcmc(key, bart, ndpost_per_rep, callback=callback)
            return bart

        run_bart = run_bart.lower(keys.pop(), bart).compile()

        print("run bart...")
        times = []
        for i in range(reps):
            print(f"run {i + 1}/{reps} ", end="", flush=True)
            with Timer() as timer:
                bart = block_until_ready(run_bart(keys.pop(), bart))
            print(
                f" {ndpost_per_rep} iterations in {format_time(timer.time)} ({format_time(timer.time / ndpost_per_rep)} per iteration)"
            )
            times.append(timer.time)
        per_iter = min(times) / ndpost_per_rep

    except JaxRuntimeError as exc:
        if exc.args[0].startswith(
            "RESOURCE_EXHAUSTED: Out of memory while trying to allocate"
        ):
            try:
                del bart
            except NameError:
                pass
            break
        else:
            raise

    # save results
    results.setdefault("n", []).append(n)
    results.setdefault("time_per_iter", []).append(per_iter)

    # free memory
    del bart
    gc.collect()

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
