import time
import functools
import math
import gc

import jax
import jaxlib
from jax import numpy as jnp
from jax import random

import bartz

# Config
n_over_ntree = 8
fixed_ntree = None # 200 or None
maxdepth = 6
nvec = [2 ** p for p in range(1, 30)]
n_over_p = 10
fixed_p = 100 # 100 or None
reps = 2
ndpost_per_rep = 15
cpu_max_memory = 16 * 2 ** 30

# Totally fake DGP
def gen_data(p, n):
    X = jnp.arange(p * n, dtype=jnp.uint8).reshape(p, n)
    X = jax.vmap(jnp.roll)(X, jnp.arange(p))
    max_split = jnp.full(p, 255, jnp.uint8)
    y = jnp.cos(jnp.linspace(0, 2 * jnp.pi / 32 * n, n))
    return X, y, max_split

def make_p_nonterminal():
    depth = jnp.arange(maxdepth - 1)
    base = 0.95
    power = 2
    return base / (1 + depth).astype(float) ** power

@functools.partial(jax.jit, static_argnums=(1, 2, 3))
def init(key, p, n, ntree):
    X, y, max_split = gen_data(p, n)
    return bartz.mcmcstep.init(
        X=X,
        y=y,
        max_split=max_split,
        num_trees=ntree,
        p_nonterminal=make_p_nonterminal(),
        sigma2_alpha=1,
        sigma2_beta=1,
        min_points_per_leaf=5,
    )

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
key = random.key(202404151128)

# detect gpu/cpu
device_kind = jnp.empty(0).devices().pop().device_kind

results = {}

for n in nvec:

    # determine ntree and p for this n
    ntree = max(1, n // n_over_ntree) if fixed_ntree is None else fixed_ntree
    p = max(1, n // n_over_p) if fixed_p is None else fixed_p
    expected_memory_usage = n * (ntree + p)
    if device_kind == 'cpu' and expected_memory_usage > cpu_max_memory:
        break
    print(f'\nn = {n:_}, ntree = {ntree:_}, p = {p:_}')
    print(f'expected memory usage: {expected_memory_usage * 1e-9:.1f} GB')

    # split random seed
    keys = list(random.split(key, reps + 3))
    key = keys.pop()

    try:
        print('initialize...')
        bart = init(keys.pop(), p, n, ntree)

        print('compile...')
        callback = lambda **_: jax.debug.callback(lambda: print('.', end='', flush=True))
        def run_bart(bart, key):
            bart, _, _ = bartz.mcmcloop.run_mcmc(bart, 0, ndpost_per_rep, 1, callback, key)
            return bart
        run_bart = jax.jit(run_bart, donate_argnums=0).lower(bart, keys.pop()).compile()

        print('run bart...')
        times = []
        for i in range(reps):
            print(f'run {i + 1}/{reps} ', end='', flush=True)
            with Timer() as timer:
                bart = jax.block_until_ready(run_bart(bart, keys.pop()))
            print(f' {ndpost_per_rep} iterations in {format_time(timer.time)} ({format_time(timer.time / ndpost_per_rep)} per iteration)')
            times.append(timer.time)
        per_iter = min(times) / ndpost_per_rep

    except jaxlib.xla_extension.XlaRuntimeError as exc:
        if exc.args[0].startswith('RESOURCE_EXHAUSTED: Out of memory while trying to allocate'):
            try:
                del bart
            except NameError:
                pass
            break
        else:
            raise

    # save results
    results.setdefault('n', []).append(n)
    results.setdefault('time_per_iter', []).append(per_iter)

    # free memory
    del bart
    gc.collect()

# print machine-readable output
print(f"""
    {{
        'package': 'bartz',
        'device_kind': '{device_kind}',
        {f"'n/ntree': {n_over_ntree}" if fixed_ntree is None else f"'ntree': {ntree}"},
        {f"'n/p': {n_over_p}" if fixed_p is None else f"'p': {p}"},
        'maxdepth': {maxdepth},
        'results': {results},
    }},""")
