"""Simulated data generation helpers built on top of :mod:`bartz.testing`."""

from functools import partial
from typing import Any

from bartz.jaxext import autobatch
from bartz.testing import gen_data
from equinox import Module
from jax import jit, random
from jax import numpy as jnp
from jaxtyping import Array, Float, Float32, Key, UInt, UInt8


class Data(Module):
    """Simulated data."""

    raw_X: Float[Array, "p n"] | None
    quantized_X: UInt[Array, "p n"] | None
    y: Float32[Array, " n"]
    max_split: UInt[Array, " p"]
    prior_var: Float32[Array, ""]
    pop_var: Float32[Array, ""]
    eps_var: Float32[Array, ""]


@partial(jit, static_argnums=(1, 2, 3))
def make_data(key: Key[Array, ""], n: int, p: int, quantized_x: bool) -> Data:
    """Generate data."""

    @partial(
        autobatch,
        out_axes=Data(1, 1, 0, 0, 0, 0, 0),
        max_io_nbytes=2**26,
    )
    def make_data_batchable(keys: Key[Array, " n"]):
        return _make_data_batchable(keys[0], keys.size, p, quantized_x)

    unbatched_data = _make_data_unbatchable(key, p)
    batched_data = make_data_batchable(random.split(key, n))

    return Data(
        raw_X=batched_data.raw_X,
        quantized_X=batched_data.quantized_X,
        y=batched_data.y,
        max_split=unbatched_data.max_split,
        prior_var=unbatched_data.prior_var,
        pop_var=unbatched_data.pop_var,
        eps_var=unbatched_data.eps_var,
    )


def _gen_data_kwargs(n: int, p: int) -> dict[str, Any]:
    """Generate arguments for `gen_data`."""
    sigma2 = 1 / 3
    return dict(
        n=n,
        p=p,
        q=2 if p > 2 else 0,
        lam=1.0,
        sigma2_lin=sigma2,
        sigma2_quad=sigma2,
        sigma2_eps=sigma2,
    )


def _make_data_unbatchable(key: Key[Array, ""], p: int) -> Data:
    """Produce those parts of data that do not have 'n' amongst axis dims."""
    max_split = jnp.full((p,), 255, jnp.uint8)
    kwargs = _gen_data_kwargs(0, p)
    empty_data = gen_data(key, **kwargs)
    return Data(
        raw_X=None,
        quantized_X=None,
        y=None,
        max_split=max_split,
        prior_var=empty_data.params.sigma2_pri,
        pop_var=empty_data.params.sigma2_pop,
        eps_var=empty_data.params.sigma2_eps,
    )


def _make_data_batchable(
    key: Key[Array, ""], n: int, p: int, quantized_x: bool
) -> Data:
    """Internal implementation of make_data."""
    # generate data
    kwargs = _gen_data_kwargs(n, p)
    data = gen_data(key, **kwargs)

    # quantize predictors
    if quantized_x:
        qx = _quantize_uniform(data.x)

    return Data(
        raw_X=() if quantized_x else data.x,
        quantized_X=qx if quantized_x else (),
        y=data.y,
        max_split=(),
        prior_var=(),
        pop_var=(),
        eps_var=(),
    )


def _quantize_uniform(x: Float[Array, "p n"]) -> UInt8[Array, "p n"]:
    l = -(3**0.5)
    u = 3**0.5
    # see bartz.testing for the bounds
    qx = jnp.floor((x - l) / (u - l) * 256.0)
    return qx.astype(jnp.uint8)
