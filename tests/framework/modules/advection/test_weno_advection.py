"""Tests for the WENO advection scheme."""

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
#  Tests
# ================================================================
@pytest.mark.parametrize("u0", [1.5, -1.5])
def test_linear_advection_is_exact(mset, u0):
    # WENO reconstructs linear fields exactly, so the advection of
    # q = 2x + 1 by a constant velocity gives dq = -2 u0
    advection = fr.modules.advection.WENO(order=3)
    advection.setup(mset=mset)
    x = mset.grid.x_mesh[0]

    q = fr.ScalarField(mset, name="q")
    q.arr = 2.0 * x + 1.0
    u = fr.ScalarField(mset, name="u", position=q.position.shift(0))
    u.arr = u0 * jnp.ones_like(u.arr)
    velocity = fr.VectorField(mset, field_list=[u])

    res = advection.advection(velocity, q)

    expected = -u0 * 2.0 * jnp.ones_like(x)
    assert jnp.abs(res.arr - expected)[INTERIOR].max() < 1e-12


def test_default_biased_interpolation():
    advection = fr.modules.advection.WENO(order=3)
    assert isinstance(advection.biased_inter, fr.grid.cartesian.InterWENO)
    assert advection.biased_inter.order == 3


def test_custom_biased_interpolation():
    biased = fr.grid.cartesian.UpwindInterpolation(order=2)
    advection = fr.modules.advection.WENO(order=3, biased_inter=biased)
    assert advection.biased_inter is biased
