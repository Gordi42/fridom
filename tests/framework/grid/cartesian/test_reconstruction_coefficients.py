"""Tests for the polynomial reconstruction coefficients."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr


# ================================================================
#  Helpers
# ================================================================
def point_values(degree, stencil_size):
    # point values of x^degree at the stencil positions x_{i+1/2}
    i = jnp.arange(stencil_size)
    return (i + 0.5) ** degree


def cell_averages(degree, stencil_size):
    # cell averages of x^degree over the cells [i, i+1]
    i = jnp.arange(stencil_size)
    return ((i + 1) ** (degree + 1) - i ** (degree + 1)) / (degree + 1)


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("stencil_size", [1, 2, 3])
def test_pointwise_coefficients_reconstruct_polynomials(stencil_size):
    # coefficients of a stencil of size n reconstruct polynomials of
    # degree < n exactly at the grid points x_k = k
    coeffs = fr.grid.cartesian.compute_polynomial_coefficients_pointwise(
        stencil_size=stencil_size)
    assert coeffs.shape == (stencil_size + 1, stencil_size)

    for degree in range(stencil_size):
        values = point_values(degree, stencil_size)
        for k in range(stencil_size + 1):
            reconstructed = jnp.sum(coeffs[k] * values)
            assert reconstructed == pytest.approx(float(k) ** degree)


@pytest.mark.parametrize("stencil_size", [1, 2, 3])
def test_cell_average_coefficients_reconstruct_polynomials(stencil_size):
    # coefficients of a stencil of size n reconstruct the point values
    # of polynomials of degree < n from their cell averages
    cell_average = (
        fr.grid.cartesian.compute_polynomial_coefficients_cell_average)
    coeffs = cell_average(stencil_size=stencil_size)
    assert coeffs.shape == (stencil_size + 1, stencil_size)

    for degree in range(stencil_size):
        averages = cell_averages(degree, stencil_size)
        for k in range(stencil_size + 1):
            reconstructed = jnp.sum(coeffs[k] * averages)
            assert reconstructed == pytest.approx(float(k) ** degree)


@pytest.mark.parametrize("method", ["pointwise", "cell_average"])
def test_dispatch(method):
    dispatched = fr.grid.cartesian.compute_polynomial_coefficients(
        stencil_size=2, method=method)
    if method == "pointwise":
        direct = fr.grid.cartesian.compute_polynomial_coefficients_pointwise(
            stencil_size=2)
    else:
        cell_average = (
            fr.grid.cartesian.compute_polynomial_coefficients_cell_average)
        direct = cell_average(stencil_size=2)
    assert jnp.array_equal(dispatched, direct)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="Invalid method bogus"):
        fr.grid.cartesian.compute_polynomial_coefficients(
            stencil_size=2, method="bogus")
