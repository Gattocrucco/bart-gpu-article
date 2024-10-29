import os
import gc
import functools
import time

os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.00'

import jax
from jax import numpy as jnp
from jax import random
import xgboost

dev = jax.devices()[0]

# sizes config
nvec = [2 ** p for p in range(1, 30)] # number of datapoints
n_over_ntree = 8
fixed_ntree = None # 200 or None
n_over_p = 10
fixed_p = None # 100 or None
max_n_times_p = 2 ** 30 # memory
max_n_times_ntree = 2 ** 32 # time

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
def gen_data(key, n, p):
    X = gen_X(key, n, p)
    y = gen_y(key, X)
    return X, y

# set up random seed
cpu = jax.devices('cpu')[0]
with jax.default_device(cpu):
    key = jax.device_put(random.key(202403212335), cpu)

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
    if n * p > max_n_times_p or n * ntree > max_n_times_ntree:
        break
    print(f'\n********* n = {n:_}, ntree = {ntree:_}, p = {p:_} *********\n')

    keys = list(random.split(key, 6))
    key = keys.pop()

    # generate data
    X, y = gen_data(keys.pop(), n, p)
    assert X.devices().pop().platform == 'cpu'
    print(f'total data sdev: {y.std():.3g}')

    with jax.default_device(cpu):
        seed = random.randint(keys.pop(), (), 0, jnp.uint32(2 ** 31)).item()
    model = xgboost.XGBRegressor(
        n_estimators=ntree,
        n_jobs=1,
        random_state=seed,
        device=dev.platform,
        verbosity=2,
    )
    with Timer() as timer:
        model.fit(X, y, verbose=True)
    print(f'time = {timer.time:.3g} s')

    results.setdefault('n', []).append(n)
    results.setdefault('time_per_iter', []).append(timer.time)

    # free memory
    del X, y, model
    gc.collect()

# print machine-readable output
print(f"""
    {{
        'package': 'xgboost',
        'device_kind': {dev.device_kind!r},
        {f"'n/ntree': {n_over_ntree}" if fixed_ntree is None else f"'ntree': {ntree}"},
        {f"'n/p': {n_over_p}" if fixed_p is None else f"'p': {p}"},
        'results': {results},
    }},""")
