"""Simulated data generation helpers built on top of :mod:`bartz.testing`."""

from dataclasses import fields
from functools import partial
from os import PathLike
from typing import Literal

import jax
from bartz.jaxext import split
from bartz.testing import gen_data_from_params, gen_params
from equinox import Module
from jax import jit, lax, random
from jax import numpy as jnp
from jax.experimental.array_serialization import pytree_serialization
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
def make_data(
    key: Key[Array, ""], n: int, p: int, quantized_x: bool | Literal["both"]
) -> Data:
    """Generate data.

    `quantized_x` may be False (raw), True (quantized uint8), or "both"
    (populate both `raw_X` and `quantized_X` on the returned `Data`).
    """
    both = quantized_x == "both"
    want_raw = both or not quantized_x
    want_quant = both or quantized_x is True

    keys = split(key)

    sigma2 = 1 / 3
    params = gen_params(
        keys.pop(),
        p=p,
        k=None,
        q=2 if p > 2 else 0,
        sigma2_lin=sigma2,
        sigma2_quad=sigma2,
        sigma2_eps=sigma2,
    )

    # sizing for the scan: worst case is both forms materialised
    raw_bytes = jnp.dtype(jnp.float32).itemsize if want_raw else 0
    quant_bytes = jnp.dtype(jnp.uint8).itemsize if want_quant else 0
    bytes_per_sample = (
        jnp.dtype(jnp.float32).itemsize + (raw_bytes + quant_bytes) * p
    )

    # round n up to a multiple of batch_size so scan has a single body path
    batch_size = max(1, min(n, _MAX_IO_NBYTES // bytes_per_sample))
    num_batches = -(-n // batch_size)
    total = num_batches * batch_size

    def body(_, key):
        dgp = gen_data_from_params(key, params, n=batch_size)
        out: tuple = ()
        if want_raw:
            out += (dgp.x,)
        if want_quant:
            out += (_quantize_uniform(dgp.x),)
        out += (dgp.y,)
        return None, out

    _, batches = lax.scan(body, None, keys.pop(num_batches))
    *x_batches_list, y_batches = batches

    def _finalize_x(x_batches):
        return jnp.moveaxis(x_batches, 0, 1).reshape(p, total)[:, :n]

    raw_X = _finalize_x(x_batches_list[0]) if want_raw else None
    quant_X = _finalize_x(x_batches_list[-1]) if want_quant else None
    y = y_batches.reshape(total)[:n]

    return Data(
        raw_X=raw_X,
        quantized_X=quant_X,
        y=y,
        max_split=jnp.full((p,), 255, jnp.uint8),
        prior_var=params.sigma2_pri,
        pop_var=params.sigma2_pop,
        eps_var=params.sigma2_eps,
    )


def save_data(
    data: Data, path: str | PathLike[str], *, overwrite: bool = False
) -> None:
    """Save a `Data` pytree to `path` (a directory).

    The payload is written as a dict of fields so that
    :mod:`jax.experimental.array_serialization`'s pytree serializer, which only
    knows about built-in container types, does not need `Data` to be registered
    as a custom pytree node.
    """
    payload = {f.name: getattr(data, f.name) for f in fields(data)}
    pytree_serialization.save(payload, path, overwrite=overwrite)


def load_data(
    path: str | PathLike[str], *, device: jax.Device | None = None
) -> Data:
    """Load a `Data` pytree previously written by :func:`save_data`."""
    if device is None:
        device = jax.devices()[0]
    sharding = jax.sharding.SingleDeviceSharding(device)
    payload = pytree_serialization.load(path, sharding)
    return Data(**payload)


def _quantize_uniform(x: Float[Array, "p n"]) -> UInt8[Array, "p n"]:
    l = -(3**0.5)
    u = 3**0.5
    # see bartz.testing for the bounds
    qx = jnp.floor((x - l) / (u - l) * 256.0)
    return qx.astype(jnp.uint8)
