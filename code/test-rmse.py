import time
import functools
import math
import gc

import jax
import jaxlib
from jax import numpy as jnp
from jax import random
import polars as pl
import numpy as np
import rpy2

import bartz
from rbartpackages import BART, dbarts, bartMachine

# config
only_data = False # generate only the data, without fitting
n_over_ntree = 8
fixed_ntree = None # 200 or None
nvec = [2 ** p for p in range(2, 30)]
n_test = 1000
n_over_p = 10
fixed_p = 100 # 100 or None
max_n_times_ntree = 2 ** 24 # determined empirically on my laptop
max_n_times_p = 2 ** 27 # determined empirically on my laptop

@functools.partial(jax.jit, static_argnums=(1, 2))
def simulate_data(key, n, p, max_interactions):
    """ DGP. Uses data-based standardization, so you have to generate train &
    test at once. """

    # split random key
    keys = list(random.split(key, 4))
    
    # generate matrices
    X = random.uniform(keys.pop(), (p, n))
    beta = random.normal(keys.pop(), (p,))
    A = random.normal(keys.pop(), (p, p))
    error = random.normal(keys.pop(), (n,))
    
    # make A banded to limit the number of interactions
    num_nonzero = 1 + (max_interactions - 1) // 2
    num_nonzero = jnp.clip(num_nonzero, 0, p)
    interaction_pattern = jnp.arange(p) < num_nonzero
    multi_roll = jax.vmap(jnp.roll, in_axes=(None, 0))
    nonzero = multi_roll(interaction_pattern, jnp.arange(p))
    A *= nonzero

    # compute terms
    linear = beta @ X
    quadratic = jnp.einsum('ai,bi,ab->i', X, X, A)

    # equalize the terms
    linear /= jnp.std(linear)
    quadratic /= jnp.std(quadratic)

    # compute response
    y = linear + quadratic + error

    return X, y

def make_data(key, n_train, n_test, p):
    X, y = simulate_data(key, n_train + n_test, p, 5)
    X_train, y_train = X[:, :n_train], y[:n_train]
    X_test, y_test = X[:, n_train:], y[n_train:]
    return X_train, y_train, X_test, y_test

class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    def __exit__(self, *_):
        self.time = time.perf_counter() - self.start

def num2si(x, fmt=lambda x: f'{x:#.3g}'.rstrip('.'), si=True, space=' '):
    x = float(x)
    if x == 0:
        return fmt(x) + space
    exp = int(math.floor(math.log10(abs(x))))
    exp3 = exp - (exp % 3)
    x3 = x / (10 ** exp3)

    if si and exp3 >= -24 and exp3 <= 24 and exp3 != 0:
        exp3_text = space + 'yzafpnμm kMGTPEZY'[(exp3 - (-24)) // 3]
    elif exp3 == 0:
        exp3_text = space
    else:
        exp3_text = f'e{exp3}'

    return f'{fmt(x3)}{exp3_text}'

def format_time(t):
    return f'{num2si(t)}s'

# random seed
key = random.key(202409141324)
def make_r_seed(key):
    return int(random.randint(key, (), 0, jnp.uint32(2 ** 31), jnp.uint32))

results = {}

for n in nvec:

    # determine ntree and p for this n
    ntree = max(1, n // n_over_ntree) if fixed_ntree is None else fixed_ntree
    p = max(1, n // n_over_p) if fixed_p is None else fixed_p
    if n * ntree > max_n_times_ntree or n * p > max_n_times_p:
        break
    print(f'\nn = {n:_}, ntree = {ntree:_}, p = {p:_}')

    # split random seed
    keys = list(random.split(key, 6))
    key = keys.pop()

    print('generate data...')
    X_train, y_train, X_test, y_test = make_data(keys.pop(), n, n_test, p)
    total_var = jnp.concatenate([y_train, y_test]).var().item()

    if only_data:
        for name in 'bartz', 'BART', 'dbarts', 'bartMachine':
            result = results.setdefault(name, {})
            result.setdefault('n', []).append(n)
            result.setdefault('total_var', []).append(total_var)
        del X_train, y_train, X_test, y_test
        gc.collect()
        continue

    barts = {}

    print('run bartz...')
    kw_bartz = dict(
        x_test=X_test,
        sigest=1,
        usequants=False,
        numcut=100,
        nskip=1000,
        ndpost=1000,
        ntree=ntree,
        seed=keys.pop(),
    )
    with Timer() as timer:
        barts['bartz'] = bartz.BART.gbart(X_train, y_train, **kw_bartz)
    print(format_time(timer.time))

    print('run BART...')
    kw_BART = kw_bartz.copy()
    kw_BART.update(
        x_test=X_test.T,
        rm_const=False,
        mc_cores=1,
        seed=make_r_seed(keys.pop()),
    )
    with Timer() as timer:
        barts['BART'] = BART.mc_gbart(X_train.T, y_train, **kw_BART)
    print(format_time(timer.time))

    print('run dbarts...')
    kw_dbarts = kw_bartz.copy()
    kw_dbarts.update(
        x_test=X_test.T,
        seed=make_r_seed(keys.pop()),
    )
    with Timer() as timer:
        barts['dbarts'] = dbarts.bart(X_train.T, y_train, **kw_dbarts)
    print(format_time(timer.time))

    print('run bartMachine...')
    # I can't configure the splitting grid with bartMachine
    kw_bartMachine = kw_bartz.copy()
    kw_bartMachine.pop('x_test')
    kw_bartMachine.pop('usequants')
    kw_bartMachine.pop('numcut')
    kw_bartMachine.update(
        num_trees=kw_bartMachine.pop('ntree'),
        num_burn_in=kw_bartMachine.pop('nskip'),
        num_iterations_after_burn_in=kw_bartMachine.pop('ndpost'),
        run_in_sample=False,
        sig_sq_est=kw_bartMachine.pop('sigest'),
        mem_cache_for_speed=False, # set to False to use less memory
        seed=make_r_seed(keys.pop()),
    )
    with Timer() as timer:
        barts['bartMachine'] = bartMachine.bartMachine(
            pl.DataFrame(np.array(X_train.T)),
            pl.Series(np.array(y_train)),
            **kw_bartMachine,
        )
        barts['bartMachine'].yhat_test_mean = barts['bartMachine'].predict(pl.DataFrame(np.array(X_test.T)))
    print(format_time(timer.time))

    print('test...')
    rmses = {}
    for name, bart in barts.items():
        rmse = jnp.sqrt(jnp.mean(jnp.square(bart.yhat_test_mean - y_test)))
        rmse = float(rmse)
        print(f'{name} rmse={rmse:#.3g}')
        rmses[name] = rmse

    # save results
    for name, rmse in rmses.items():
        result = results.setdefault(name, {})
        result.setdefault('n', []).append(n)
        result.setdefault('total_var', []).append(total_var)
        result.setdefault('rmse', []).append(rmse)

    # free memory
    del X_train, y_train, X_test, y_test, barts, bart
    gc.collect()
    rpy2.robjects.r('gc()')

# print machine-readable output
for name, result in results.items():
    print(f"""\
    {{
        'package': '{name}',
        {f"'n/ntree': {n_over_ntree}" if fixed_ntree is None else f"'ntree': {ntree}"},
        {f"'n/p': {n_over_p}" if fixed_p is None else f"'p': {p}"},
        'results': {result},
    }},""")
