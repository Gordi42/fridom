"""Tests for the WENO interpolation module."""

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
def interpolate(module, mset, values, bias_sign):
    f = fr.ScalarField(mset, name="f")
    f.arr = values
    destination = f.position.shift(0)
    bias = bias_sign * jnp.ones_like(values)
    return module.interpolate(f, bias, destination)


# ================================================================
#  Tests
# ================================================================
def test_even_order_raises():
    with pytest.raises(ValueError, match="not odd"):
        fr.grid.cartesian.InterWENO(order=2)


@pytest.mark.parametrize("order", [1, 7])
def test_unsupported_order_raises(order):
    # no smoothness-indicator coefficients exist for these orders
    with pytest.raises(ValueError, match="not supported"):
        fr.grid.cartesian.InterWENO(order=order)


@pytest.mark.parametrize("order", [3, 5])
@pytest.mark.parametrize("bias_sign", [1.0, -1.0])
def test_linear_field_is_exact(mset, order, bias_sign):
    # all candidate stencils reconstruct a linear field exactly, so the
    # weighted combination is exact regardless of the weights
    module = fr.grid.cartesian.InterWENO(order=order)
    module.setup(mset=mset)
    x = mset.grid.x_mesh[0]

    res = interpolate(module, mset, 2.0 * x + 1.0, bias_sign)

    expected = 2.0 * (x + 0.5) + 1.0
    error = jnp.abs(res.arr - expected)[INTERIOR].max()
    assert error < 1e-12


@pytest.mark.parametrize("bias_sign", [1.0, -1.0])
def test_pointwise_method(mset, bias_sign):
    module = fr.grid.cartesian.InterWENO(order=3, method="pointwise")
    module.setup(mset=mset)
    x = mset.grid.x_mesh[0]

    res = interpolate(module, mset, 2.0 * x + 1.0, bias_sign)

    expected = 2.0 * (x + 0.5) + 1.0
    error = jnp.abs(res.arr - expected)[INTERIOR].max()
    assert error < 1e-12


@pytest.mark.parametrize("order", [3, 5])
def test_essentially_non_oscillatory(mset, order):
    # interpolating a step function must not produce over- or
    # undershoots (a centered polynomial interpolation overshoots by
    # about 8 percent)
    module = fr.grid.cartesian.InterWENO(order=order)
    module.setup(mset=mset)
    x = mset.grid.x_mesh[0]
    step = jnp.where(x > 8.0, 1.0, 0.0)

    res = interpolate(module, mset, step, bias_sign=1.0)

    assert res.arr[INTERIOR].min() > -1e-12
    assert res.arr[INTERIOR].max() < 1 + 1e-12
