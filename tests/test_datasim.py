"""Tests for :mod:`bart_gpu_article.datasim`."""

from dataclasses import fields
from pathlib import Path

import pytest
from jax import numpy as jnp
from numpy.testing import assert_array_equal

from bart_gpu_article.datasim import _MAX_IO_NBYTES, load_data, make_data, save_data


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


def test_make_data_both_matches_separate(keys):
    key = keys.pop()
    a = make_data(key, n=300, p=6, quantized_x=False)
    b = make_data(key, n=300, p=6, quantized_x=True)
    c = make_data(key, n=300, p=6, quantized_x="both")
    assert_array_equal(c.raw_X, a.raw_X, strict=True)
    assert_array_equal(c.quantized_X, b.quantized_X, strict=True)
    assert_array_equal(c.y, a.y, strict=True)


def _assert_data_equal(actual, expected):
    for field in fields(expected):
        a = getattr(actual, field.name)
        e = getattr(expected, field.name)
        if e is None:
            assert a is None
        else:
            assert_array_equal(a, e, strict=True)


@pytest.mark.parametrize("quantized_x", [False, True, "both"])
@pytest.mark.parametrize("binary", [False, True])
def test_save_load_round_trip(keys, tmp_path: Path, quantized_x, binary):
    data = make_data(keys.pop(), n=150, p=4, quantized_x=quantized_x, binary=binary)
    assert (data.quantized_X is None) == (quantized_x is False)
    assert (data.raw_X is None) == (quantized_x is True)
    assert bool(data.binary) == binary
    save_data(data, tmp_path / "ds")
    loaded = load_data(tmp_path / "ds")
    assert (loaded.quantized_X is None) == (quantized_x is False)
    assert (loaded.raw_X is None) == (quantized_x is True)
    assert bool(loaded.binary) == binary
    _assert_data_equal(loaded, data)


def test_make_data_binary_y_values(keys):
    data = make_data(keys.pop(), n=500, p=6, quantized_x=False, binary=True)
    y = data.y
    assert y.dtype == jnp.float32
    assert jnp.array_equal(jnp.unique(y), jnp.array([0.0, 1.0], dtype=jnp.float32))
    assert bool(data.binary) is True


def test_save_overwrite(keys, tmp_path: Path):
    data = make_data(keys.pop(), n=50, p=3, quantized_x="both")
    target = tmp_path / "ds"
    save_data(data, target)
    with pytest.raises(Exception):
        save_data(data, target, overwrite=False)
    save_data(data, target, overwrite=True)
    loaded = load_data(target)
    _assert_data_equal(loaded, data)
