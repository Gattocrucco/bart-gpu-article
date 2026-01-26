# bartz/src/bartz/testing/_dgp.py
#
# Copyright (c) 2026, The Bartz Contributors
#
# This file is part of bartz.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Define `gen_data` that generates simulated data for testing."""

from dataclasses import replace

from bartz.jaxext import split
from equinox import Module, error_if
from jax import numpy as jnp
from jax import random
from jaxtyping import Array, Bool, Float, Integer, Key


def generate_x(key: Key[Array, ""], n: int, p: int) -> Float[Array, "p n"]:
    """Generate predictors with mean 0 and variance 1.

    x_rj ~iid U(-√3, √3)
    """
    return random.uniform(key, (p, n), minval=-jnp.sqrt(3.0), maxval=jnp.sqrt(3.0))


def generate_beta(
    key: Key[Array, ""], p: int, sigma2_lin: Float[Array, ""]
) -> Float[Array, " p"]:
    """Generate linear coefficients."""
    sigma2_beta = sigma2_lin / p
    return random.normal(key, (p,)) * jnp.sqrt(sigma2_beta)


def compute_linear_mean(
    beta: Float[Array, " p"], x: Float[Array, "p n"]
) -> Float[Array, " n"]:
    """mulin_j = beta_r x_rj."""
    return beta @ x


def interaction_pattern(p: int, q: Integer[Array, ""] | int) -> Bool[Array, "p p"]:
    """Create a symmetric interaction pattern for q interactions per variable.

    Parameters
    ----------
    p
        Number of predictors
    q
        Number of interactions per predictor (must be even)

    Returns
    -------
    Symmetric binary pattern of shape (p, p) where each row/col sums to q+1
    """
    q = error_if(q, q % 2 != 0, "q must be even")
    q = error_if(q, q >= p, "q must be less than p")

    i, j = jnp.ogrid[:p, :p]
    dist = jnp.minimum(jnp.abs(i - j), p - jnp.abs(i - j))
    return dist <= (q // 2)


def generate_A(
    key: Key[Array, ""],
    p: int,
    q: Integer[Array, ""],
    sigma2_quad: Float[Array, ""],
    kurt_x: float,
) -> Float[Array, "p p"]:
    """Generate quadratic coefficients."""
    pattern: Bool[Array, "p p"] = interaction_pattern(p, q)
    A: Float[Array, "p p"] = random.normal(key, (p, p))
    A = jnp.where(pattern, A, 0.0)
    sigma2_A = sigma2_quad / (p * (kurt_x - 1 + q))
    return A * jnp.sqrt(sigma2_A)


def compute_muquad(
    A: Float[Array, "p p"], x: Float[Array, "p n"]
) -> Float[Array, " n"]:
    """Compute quadratic mean.

    muquad_j = A_rs x_rj x_sj
    """
    return jnp.einsum("rs,rj,sj->j", A, x, x)


def generate_outcome(
    key: Key[Array, ""], mu: Float[Array, " n"], sigma2_eps: Float[Array, ""]
) -> Float[Array, " n"]:
    """Generate noisy outcome."""
    eps: Float[Array, " n"] = random.normal(key, mu.shape)
    return mu + eps * jnp.sqrt(sigma2_eps)


class DGP(Module):
    """Quadratic univariate DGP.

    Parameters
    ----------
    x
        Predictors of shape (p, n), variance 1
    y
        Noisy outcomes of shape (n,)
    beta
        Linear coefficients of shape (p,)
    mulin
        Linear part of latent mean of shape (n,)
    A
        Quadratic coefficients of shape (p, p)
    muquad
        Quadratic part of latent mean of shape (n,)
    mu
        True latent mean of shape (n,)
    q
        Number of interactions per predictor
    sigma2_lin
        Prior and expected population variance of mulin
    sigma2_quad
        Expected population variance of muquad
    sigma2_eps
        Variance of the error
    """

    # Main outputs
    x: Float[Array, "p n"]
    y: Float[Array, " n"]

    # Intermediate results
    beta: Float[Array, " p"]
    mulin: Float[Array, " n"]
    A: Float[Array, "p p"]
    muquad: Float[Array, " n"]
    mu: Float[Array, " n"]

    # Params
    q: Integer[Array, ""]
    sigma2_lin: Float[Array, ""]
    sigma2_quad: Float[Array, ""]
    sigma2_eps: Float[Array, ""]

    kurt_x: float = 9 / 5  # kurtosis of uniform distribution

    @property
    def sigma2_pri(self) -> Float[Array, ""]:
        """Prior variance of y."""
        return self.sigma2_pop + self.sigma2_mean

    @property
    def sigma2_pop(self) -> Float[Array, ""]:
        """Expected population variance of y."""
        return self.sigma2_lin + self.sigma2_quad + self.sigma2_eps

    @property
    def sigma2_mean(self) -> Float[Array, ""]:
        """Variance of the mean function."""
        return self.sigma2_quad / (self.kurt_x - 1 + self.q)

    def split(self, n_train: int | None = None) -> tuple["DGP", "DGP"]:
        """Split the data into training and test sets."""
        if n_train is None:
            n_train = self.x.shape[1] // 2
        assert 0 < n_train < self.x.shape[1], "n_train must be in (0, n)"
        train = replace(
            self,
            x=self.x[:, :n_train],
            y=self.y[:n_train],
            mulin=self.mulin[:n_train],
            muquad=self.muquad[:n_train],
            mu=self.mu[:n_train],
        )
        test = replace(
            self,
            x=self.x[:, n_train:],
            y=self.y[n_train:],
            mulin=self.mulin[n_train:],
            muquad=self.muquad[n_train:],
            mu=self.mu[n_train:],
        )
        return train, test


def gen_data(
    key: Key[Array, ""],
    *,
    n: int,
    p: int,
    q: Integer[Array, ""] | int,
    sigma2_lin: Float[Array, ""] | float,
    sigma2_quad: Float[Array, ""] | float,
    sigma2_eps: Float[Array, ""] | float,
) -> DGP:
    """Generate data from a quadratic univariate DGP.

    Parameters
    ----------
    key
        JAX random key
    n
        Number of observations
    p
        Number of predictors
    q
        Number of interactions per predictor (must be even and < p)
    sigma2_lin
        Prior and expected population variance of the linear term
    sigma2_quad
        Expected population variance of the quadratic term
    sigma2_eps
        Variance of the error term

    Returns
    -------
    An object with all generated data and parameters.
    """
    # check q
    q = jnp.asarray(q)
    q = error_if(q, q % 2 != 0, "q must be even")
    q = error_if(q, q >= p, "q must be less than p")

    keys = split(key, 4)

    sigma2_lin = jnp.asarray(sigma2_lin)
    sigma2_quad = jnp.asarray(sigma2_quad)
    sigma2_eps = jnp.asarray(sigma2_eps)

    x = generate_x(keys.pop(), n, p)
    beta = generate_beta(keys.pop(), p, sigma2_lin)
    mulin = compute_linear_mean(beta, x)
    A = generate_A(keys.pop(), p, q, sigma2_quad, DGP.kurt_x)
    muquad = compute_muquad(A, x)
    mu = mulin + muquad
    y = generate_outcome(keys.pop(), mu, sigma2_eps)

    return DGP(
        x=x,
        y=y,
        beta=beta,
        mulin=mulin,
        A=A,
        muquad=muquad,
        mu=mu,
        q=q,
        sigma2_lin=sigma2_lin,
        sigma2_quad=sigma2_quad,
        sigma2_eps=sigma2_eps,
    )
