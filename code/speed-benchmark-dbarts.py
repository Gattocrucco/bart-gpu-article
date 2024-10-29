import time
import gc
import functools

import jax
from jax import numpy as jnp
from jax import random
import rpy2

from rbartpackages import dbarts

# config
ndpost = 20
nvec = [2 ** p for p in range(1, 30)] # number of datapoints
n_over_ntree = 8
fixed_ntree = 200 # 200 or None
n_over_p = 10
fixed_p = 100 # 100 or None
max_memory = 16 * 2 ** 30 # bytes

# DGP definition
sigma = 0.1 # noise standard deviation
def f(x): # conditional mean
    T = 2
    return jnp.sum(jnp.cos(2 * jnp.pi / T * x), axis=-1) / jnp.sqrt(p)
def gen_X(key, n, p):
    return random.uniform(key, (n, p), float, -2, 2)
def gen_y(key, X):
    return f(X) + sigma * random.normal(key, X.shape[:1])

@functools.partial(jax.jit, static_argnums=(1, 2))
def gen_data(key, p, n):
    X = gen_X(key, p, n)
    y = gen_y(key, X)
    return X, y

# set up random seed
key = random.key(202403212335)

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.time = time.perf_counter() - self.start

results = {}

for n in nvec:

    # determine ntree and p for this n
    ntree = max(1, n // n_over_ntree) if fixed_ntree is None else fixed_ntree
    p = max(1, n // n_over_p) if fixed_p is None else fixed_p
    expected_memory_usage = 24 * n * (ntree + p)
    if expected_memory_usage > max_memory:
        break
    print(f'\n********* n = {n:_}, ntree = {ntree:_}, p = {p:_} *********\n')
    print(f'expected memory usage: {expected_memory_usage * 1e-9:.1f} GB')

    keys = list(random.split(key, 6))
    key = keys.pop()

    # generate data
    X, y = gen_data(keys.pop(), n, p)
    print(f'total data sdev: {y.std():.3g}')

    # seed for R
    seed = random.randint(keys.pop(), (), 0, jnp.uint32(2 ** 31)).item()
    sigest = 2 * sigma

    control = dbarts.dbartsControl(
        verbose=True,
        keepTrainingFits=False,
        keepTrees=False,
        n_cuts=255,
        n_trees=ntree,
        n_chains=1,
        n_threads=1,
        printEvery=1,
        rngSeed=seed,
    )
    with Timer() as timer_init:
        sampler = dbarts.dbarts(X, y, control=control, sigma=sigest)
    sampler.run(10, 1) # warm-up
    with Timer() as timer:
        sampler.run(0, ndpost)

    results.setdefault('n', []).append(n)
    results.setdefault('time_per_iter', []).append(timer.time / ndpost)
    results.setdefault('time_init', []).append(timer_init.time)

    # free memory
    del control, sampler, X, y
    gc.collect()
    rpy2.robjects.r('gc()')

# print machine-readable output
print(f"""
    {{
        'package': 'dbarts',
        'device_kind': 'cpu',
        {f"'n/ntree': {n_over_ntree}" if fixed_ntree is None else f"'ntree': {ntree}"},
        {f"'n/p': {n_over_p}" if fixed_p is None else f"'p': {p}"},
        'results': {results},
    }},""")
