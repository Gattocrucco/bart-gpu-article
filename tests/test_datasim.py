"""Tests for :mod:`bart_gpu_article.datasim`."""

import pytest
from jax import numpy as jnp

from bart_gpu_article.datasim import _MAX_IO_NBYTES, make_data


def test_shapes_raw(keys):
    data = make_data(keys.pop(), n=1000, p=10, quantized_x=False)
    assert data.raw_X.shape == (10, 1000)
    assert data.raw_X.dtype == jnp.float32
    assert data.quantized_X is None
    assert data.y.shape == (1000,)
    assert data.max_split.shape == (10,)
    assert jnp.isfinite(data.prior_var)
    assert jnp.isfinite(data.pop_var)
    assert jnp.isfinite(data.eps_var)


def test_shapes_quantized(keys):
    data = make_data(keys.pop(), n=1000, p=10, quantized_x=True)
    assert data.raw_X is None
    assert data.quantized_X.shape == (10, 1000)
    assert data.quantized_X.dtype == jnp.uint8
    assert data.y.shape == (1000,)


def test_variance_ordering(keys):
    data = make_data(keys.pop(), n=100, p=5, quantized_x=False)
    assert data.eps_var > 0
    assert data.pop_var >= data.eps_var
    assert data.prior_var >= data.pop_var


def test_determinism(keys):
    key = keys.pop()
    a = make_data(key, n=500, p=8, quantized_x=False)
    b = make_data(key, n=500, p=8, quantized_x=False)
    assert jnp.array_equal(a.raw_X, b.raw_X)
    assert jnp.array_equal(a.y, b.y)


@pytest.mark.parametrize("quantized_x", [False, True])
def test_batching_exercised(keys, quantized_x):
    # pick n, p so batch_size < n, forcing multiple scan iterations
    p = 100
    bytes_per_sample = 4 + (1 if quantized_x else 4) * p
    batch_size = _MAX_IO_NBYTES // bytes_per_sample
    n = batch_size * 2 + batch_size // 3  # not a multiple of batch_size
    data = make_data(keys.pop(), n=n, p=p, quantized_x=quantized_x)
    x = data.quantized_X if quantized_x else data.raw_X
    assert x.shape == (p, n)
    assert data.y.shape == (n,)
