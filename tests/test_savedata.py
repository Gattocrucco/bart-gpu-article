"""Tests for :mod:`bart_gpu_article.scripts.savedata`."""

import pytest
from jax import numpy as jnp, random

from bart_gpu_article.datasim import load_data
from bart_gpu_article.scripts.savedata import main


def _int_seed(key) -> int:
    return random.randint(key, (), 0, jnp.uint32(2**31), jnp.uint32).item()


def test_savedata_cli(keys, tmp_path):
    seed = _int_seed(keys.pop())
    n, p = 100, 4

    main(["-n", str(n), "-p", str(p), "-s", str(seed), "-d", str(tmp_path)])

    target = tmp_path / f"savedata-{n}-{p}-{seed}"
    assert target.is_dir()

    data = load_data(target)
    assert data.raw_X is not None
    assert data.quantized_X is not None
    assert data.raw_X.shape == (p, n)
    assert data.quantized_X.shape == (p, n)
    assert data.raw_X.dtype == jnp.float32
    assert data.quantized_X.dtype == jnp.uint8
    assert data.y.shape == (n,)

    # Re-running without --overwrite fails; with --overwrite succeeds.
    with pytest.raises(Exception):
        main(["-n", str(n), "-p", str(p), "-s", str(seed), "-d", str(tmp_path)])
    main([
        "-n", str(n), "-p", str(p), "-s", str(seed),
        "-d", str(tmp_path), "--overwrite",
    ])
    assert target.is_dir()
