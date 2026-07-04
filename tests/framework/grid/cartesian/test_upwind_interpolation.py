"""Tests for the upwind interpolation module."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr

# interior slice away from the periodically wrapped halo cells
INTERIOR = slice(6, -6)


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(16,), domain_size=(16.0,))
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


def interpolate(module, mset, values, bias_sign):
    f = fr.ScalarField(mset, name="f")
    f.arr = values
    destination = f.position.shift(0)
    bias = bias_sign * jnp.ones_like(values)
    return module.interpolate(f, bias, destination)


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("order", [3, -2])
def test_invalid_order_raises(order):
    with pytest.raises(ValueError, match="Only even orders"):
        fr.grid.cartesian.UpwindInterpolation(order=order)


@pytest.mark.parametrize("order", [2, 4])
@pytest.mark.parametrize("bias_sign", [1.0, -1.0])
def test_polynomial_exactness_cell_average(mset, order, bias_sign):
    # an upwind interpolation of order n reconstructs the face point
    # values of polynomials of degree <= n from their cell averages
    module = fr.grid.cartesian.UpwindInterpolation(order=order)
    module.setup(mset=mset)
    x = mset.grid.x_mesh[0]

    values = cell_average(x, order)
    res = interpolate(module, mset, values, bias_sign)

    expected = (x + 0.5) ** order
    error = jnp.abs(res.arr - expected)[INTERIOR].max()
    assert error < 1e-10


@pytest.mark.parametrize("bias_sign", [1.0, -1.0])
def test_polynomial_exactness_pointwise(mset, bias_sign):
    # with the pointwise method, the input values are point values
    module = fr.grid.cartesian.UpwindInterpolation(
        order=2, method="pointwise")
    module.setup(mset=mset)
    x = mset.grid.x_mesh[0]

    res = interpolate(module, mset, x**2, bias_sign)

    expected = (x + 0.5) ** 2
    error = jnp.abs(res.arr - expected)[INTERIOR].max()
    assert error < 1e-10


def test_upwind_orientation(mset):
    # order 0 is a pure upwind scheme: positive bias takes the cell on
    # the left of the face, negative bias the cell on the right
    module = fr.grid.cartesian.UpwindInterpolation(order=0)
    module.setup(mset=mset)
    x = mset.grid.x_mesh[0]
    values = jnp.arange(x.size, dtype=fr.utils.dtype_real())

    left = interpolate(module, mset, values, bias_sign=1.0)
    right = interpolate(module, mset, values, bias_sign=-1.0)

    assert jnp.array_equal(left.arr[INTERIOR], values[INTERIOR])
    assert jnp.array_equal(right.arr[INTERIOR], values[INTERIOR] + 1)
