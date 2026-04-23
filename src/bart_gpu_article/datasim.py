"""Simulated data generation helpers built on top of :mod:`bartz.testing`."""

from functools import partial

from bartz.jaxext import split
from bartz.testing import gen_data_from_params, gen_params
from equinox import Module
from jax import jit, lax, random
from jax import numpy as jnp
from jaxtyping import Array, Float, Float32, Key, UInt, UInt8

_MAX_IO_NBYTES = 2**26


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
    keys = split(key)

    sigma2 = 1 / 3
    params = gen_params(
        keys.pop(),
        p=p,
        k=1,
        q=2 if p > 2 else 0,
        lam=1.0,
        sigma2_lin=sigma2,
        sigma2_quad=sigma2,
        sigma2_eps=sigma2,
    )

    x_dtype = jnp.uint8 if quantized_x else jnp.float32
    y_dtype = jnp.float32
    bytes_per_sample = jnp.dtype(y_dtype).itemsize + jnp.dtype(x_dtype).itemsize * p

    # round n up to a multiple of batch_size so scan has a single body path
    batch_size = max(1, min(n, _MAX_IO_NBYTES // bytes_per_sample))
    num_batches = -(-n // batch_size)
    total = num_batches * batch_size

    def body(_, key):
        dgp = gen_data_from_params(key, params, n=batch_size)
        x = dgp.x
        if quantized_x:
            x = _quantize_uniform(x)
        return None, (x, dgp.y.squeeze(0))

    _, (x_batches, y_batches) = lax.scan(body, None, keys.pop(num_batches))
    x = jnp.moveaxis(x_batches, 0, 1).reshape(p, total)[:, :n]
    y = y_batches.reshape(total)[:n]

    return Data(
        raw_X=None if quantized_x else x,
        quantized_X=x if quantized_x else None,
        y=y,
        max_split=jnp.full((p,), 255, jnp.uint8),
        prior_var=params.sigma2_pri,
        pop_var=params.sigma2_pop,
        eps_var=params.sigma2_eps,
    )


def _quantize_uniform(x: Float[Array, "p n"]) -> UInt8[Array, "p n"]:
    l = -(3**0.5)
    u = 3**0.5
    # see bartz.testing for the bounds
    qx = jnp.floor((x - l) / (u - l) * 256.0)
    return qx.astype(jnp.uint8)
