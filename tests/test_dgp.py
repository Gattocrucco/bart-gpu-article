# bartz/tests/test_dgp.py
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

"""Tests `gen_data`."""

from collections.abc import Mapping
from functools import partial
from types import MappingProxyType

import pytest
from bartz.jaxext import split
from jax import jit, vmap
from jax import numpy as jnp
from jaxtyping import Array, Bool, Key
from numpy.testing import assert_array_equal, assert_array_less
from scipy.stats import norm

from bart_gpu_article.dgp import (
    DGP,
    gen_data,
    interaction_pattern,
)

# Test parameters
ALPHA = 5e-7  # probability of false positive (aaaaapprox)
SIGMA_THRESHOLD = norm.isf(ALPHA / 2)  # threshold for z tests
KWARGS: Mapping = MappingProxyType(
    dict(n=100, p=20, q=4, sigma2_eps=0.1, sigma2_lin=0.4, sigma2_quad=0.5)
)
REPS: int = 10_000  # number of datasets


@jit
@partial(vmap, in_axes=(0,))
def generate_dgps(key: Key[Array, "REPS"]) -> DGP:
    """Generate one dataset per random key."""
    return gen_data(key, **KWARGS)


@pytest.fixture
def dgps(keys: split) -> DGP:
    """Generate DGP instances using vmap and jit."""
    return generate_dgps(keys.pop(REPS))


def test_shapes_and_dtypes(keys: split):
    """Test that all DGP attributes have correct shapes and dtypes."""
    dgp = gen_data(keys.pop(), **KWARGS)
    n, p = KWARGS["n"], KWARGS["p"]

    # Test shapes
    assert dgp.x.shape == (p, n)
    assert dgp.y.shape == (n,)
    assert dgp.beta.shape == (p,)
    assert dgp.mulin.shape == (n,)
    assert dgp.A.shape == (p, p)
    assert dgp.muquad.shape == (n,)
    assert dgp.mu.shape == (n,)
    assert dgp.q.shape == ()
    assert dgp.sigma2_lin.shape == ()
    assert dgp.sigma2_quad.shape == ()
    assert dgp.sigma2_eps.shape == ()

    # Test dtypes
    assert jnp.issubdtype(dgp.x.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.y.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.beta.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mulin.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.A.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.muquad.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.mu.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.q.dtype, jnp.integer)
    assert jnp.issubdtype(dgp.sigma2_lin.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.sigma2_quad.dtype, jnp.floating)
    assert jnp.issubdtype(dgp.sigma2_eps.dtype, jnp.floating)


class TestGenerateX:
    """Test the _generate_x method."""

    def test_x_mean(self, dgps: DGP):
        """Test that x has mean close to 0."""
        x_samples = dgps.x  # Shape: (REPS, P, N)
        n_reps = x_samples.shape[0]

        # Compute mean and std of mean for each element
        means = jnp.mean(x_samples, axis=0)  # Shape: (P, N)
        stds_of_mean = jnp.std(x_samples, axis=0) / jnp.sqrt(n_reps)  # Shape: (P, N)

        # All means should be within SIGMA_THRESHOLD standard deviations of 0
        z_scores = jnp.abs(means / stds_of_mean)
        assert_array_less(z_scores, SIGMA_THRESHOLD)

    def test_x_variance(self, dgps: DGP):
        """Test that x has variance close to 1."""
        x_samples = dgps.x  # Shape: (REPS, P, N)
        n_reps = x_samples.shape[0]

        # Compute variance for each element
        var = jnp.var(x_samples, axis=0)  # Shape: (P, N)
        expected_var = 1.0

        # Standard deviation of sample variance for each element
        std_of_var = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))

        # All variances should be within SIGMA_THRESHOLD standard deviations of 1
        z_scores = jnp.abs((var - expected_var) / std_of_var)
        assert_array_less(z_scores, SIGMA_THRESHOLD)


class TestGenerateBeta:
    """Test the _generate_beta method."""

    def test_beta_mean(self, dgps: DGP):
        """Test that beta has mean close to 0."""
        beta_samples = dgps.beta  # Shape: (REPS, P)
        n_reps = beta_samples.shape[0]

        means = jnp.mean(beta_samples, axis=0)  # Shape: (P,)
        stds_of_mean = jnp.std(beta_samples, axis=0) / jnp.sqrt(n_reps)  # Shape: (P,)

        z_scores = jnp.abs(means / stds_of_mean)
        assert_array_less(z_scores, SIGMA_THRESHOLD)


@pytest.mark.parametrize(
    "which",
    [
        "mulin",
        "muquad",
        "mu",
        "y",
    ],
)
def test_outcome_prior_variance(dgps: DGP, which: str):
    """Test that latent mean and outcome have the expected elementwise variance."""
    samples = getattr(dgps, which)  # Shape: (REPS, N)
    n_reps = samples.shape[0]

    var = jnp.var(samples, axis=0)  # Shape: (N,)

    if which == "mulin":
        expected_var = dgps.sigma2_lin
    elif which == "muquad":
        expected_var = dgps.sigma2_quad + dgps.sigma2_mean
    elif which == "mu":
        expected_var = dgps.sigma2_pri - dgps.sigma2_eps
    elif which == "y":
        expected_var = dgps.sigma2_pri
    else:
        raise KeyError(which)

    expected_var = expected_var[0].item()
    std_of_var = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))

    z_scores = jnp.abs((var - expected_var) / std_of_var)
    assert_array_less(z_scores, SIGMA_THRESHOLD)


@pytest.mark.parametrize(
    "which",
    [
        "mulin",
        "muquad",
        "mu",
        "y",
    ],
)
def test_outcome_pop_variance(dgps: DGP, which: str):
    """Test that latent mean and outcome have the expected elementwise variance."""
    samples = getattr(dgps, which)  # Shape: (REPS, N)
    n_reps = samples.shape[0]

    var = jnp.var(samples, axis=-1, ddof=1)  # Shape: (REPS,)
    var = jnp.mean(var, axis=0)  # Shape: ()

    if which == "mulin":
        expected_var = dgps.sigma2_lin
    elif which == "muquad":
        expected_var = dgps.sigma2_quad
    elif which == "mu":
        expected_var = dgps.sigma2_pop - dgps.sigma2_eps
    elif which == "y":
        expected_var = dgps.sigma2_pop
    else:
        raise KeyError(which)

    expected_var = expected_var[0].item()
    std_of_var = jnp.sqrt(2 * expected_var**2 / (n_reps - 1))

    z_scores = jnp.abs((var - expected_var) / std_of_var)
    assert_array_less(z_scores, SIGMA_THRESHOLD)


def test_variance_relationships(dgps: DGP):
    """Check some simple inequalities on variances."""
    assert jnp.all(dgps.sigma2_pri >= 0)
    assert jnp.all(dgps.sigma2_pop >= 0)
    assert jnp.all(dgps.sigma2_mean >= 0)
    assert jnp.all(dgps.sigma2_pri >= dgps.sigma2_pop)
    assert jnp.all(dgps.sigma2_pop >= dgps.sigma2_eps)


class TestInteractionPattern:
    """Test the interaction_pattern function."""

    @pytest.fixture
    def pattern(self) -> Bool[Array, "p p"]:
        """Return the predictor interaction pattern."""
        return interaction_pattern(p=10, q=4)

    def test_symmetry(self, pattern: Bool[Array, "p p"]):
        """Test that interaction pattern is symmetric."""
        assert_array_equal(pattern, pattern.T)

    def test_diagonal(self, pattern: Bool[Array, "p p"]):
        """Test that diagonal is True."""
        assert_array_equal(jnp.diag(pattern), True)

    def test_row_sums(self, pattern: Bool[Array, "p p"]):
        """Test that each row sums to q+1."""
        row_sums = jnp.sum(pattern, axis=1)
        assert_array_equal(row_sums, 4 + 1)
