"""Tests for the centered polynomial interpolation module."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr

# interior slice away from the periodically wrapped halo cells
INTERIOR = (slice(6, -6), slice(6, -6))


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(16, 16), domain_size=(16.0, 16.0))
    mset = fr.ModelSettingsBase(grid=grid)
    mset.halo = 3
    return mset.setup()


# ================================================================
#  Helpers
# ================================================================
def cell_average(x, degree):
    # cell average of x^degree over [x - 1/2, x + 1/2]
    return (((x + 0.5) ** (degree + 1) - (x - 0.5) ** (degree + 1))
            / (degree + 1))


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("order", [2, -1, 0])
def test_invalid_order_raises(order):
    with pytest.raises(ValueError, match="Only odd orders"):
        fr.grid.cartesian.PolynomialInterpolation(order=order)


@pytest.mark.parametrize("order", [1, 3])
def test_polynomial_exactness_cell_average(mset, order):
    # a centered interpolation of order n reconstructs the face point
    # values of polynomials of degree <= n from their cell averages
    module = fr.grid.cartesian.PolynomialInterpolation(order=order)
    module.setup(mset=mset)
    x, _y = mset.grid.x_mesh

    f = fr.ScalarField(mset, name="f")
    f.arr = cell_average(x, order)
    res = module.interpolate(f, f.position.shift(0))

    expected = (x + 0.5) ** order
    assert jnp.abs(res.arr - expected)[INTERIOR].max() < 1e-10


def test_polynomial_exactness_pointwise(mset):
    # with the pointwise method, the input values are point values
    module = fr.grid.cartesian.PolynomialInterpolation(
        order=1, method="pointwise")
    module.setup(mset=mset)
    x, _y = mset.grid.x_mesh

    f = fr.ScalarField(mset, name="f")
    f.arr = 2.0 * x + 1.0
    res = module.interpolate(f, f.position.shift(0))

    expected = 2.0 * (x + 0.5) + 1.0
    assert jnp.abs(res.arr - expected)[INTERIOR].max() < 1e-10


def test_axes_without_extent_are_skipped(mset):
    # fields with no extent along the interpolation axis are not
    # interpolated along that axis
    module = fr.grid.cartesian.PolynomialInterpolation(order=1)
    module.setup(mset=mset)

    f = fr.ScalarField(mset, name="f", topo=(False, True))
    res = module.interpolate(f, f.position.shift(0))

    assert res.position == f.position.shift(0)
    assert res.arr.shape == f.arr.shape
