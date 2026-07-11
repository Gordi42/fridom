"""Tests for the linear interpolation module."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr

# interior slice away from the periodically wrapped halo cells
INTERIOR = slice(2, -2)


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(16,), domain_size=(16.0,))
    mset = fr.ModelSettingsBase(grid=grid)
    return mset.setup()


# ================================================================
#  Tests
# ================================================================
def test_linear_field_is_exact(mset):
    module = fr.grid.cartesian.LinearInterpolation()
    module.setup(mset=mset)
    x = mset.grid.x_mesh[0]

    f = fr.ScalarField(mset, name="f")
    f.arr = 2.0 * x + 1.0
    res = module.interpolate(f, f.position.shift(0))

    expected = 2.0 * (x + 0.5) + 1.0
    assert jnp.abs(res.arr - expected)[INTERIOR].max() < 1e-12
    assert res.position == f.position.shift(0)
