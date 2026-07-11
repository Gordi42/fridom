"""Tests for the upwind advection scheme."""

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


@pytest.fixture
def advection(mset):
    advection = fr.modules.advection.UpwindAdvection(order=3)
    return advection.setup(mset=mset)


# ================================================================
#  Helpers
# ================================================================
def make_velocity(mset, quantity, u0):
    u = fr.ScalarField(mset, name="u", position=quantity.position.shift(0))
    u.arr = u0 * jnp.ones_like(u.arr)
    return fr.VectorField(mset, field_list=[u])


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("u0", [1.5, -1.5])
def test_polynomial_advection_is_exact(mset, advection, u0):
    # for a constant velocity and exactly reconstructed face values,
    # the tendency equals the cell average of -u0 dq/dx:
    # q = avg(x^2) -> dq = -u0 * 2 x
    x = mset.grid.x_mesh[0]
    q = fr.ScalarField(mset, name="q")
    q.arr = x**2 + 1.0 / 12.0
    velocity = make_velocity(mset, q, u0)

    res = advection.advection(velocity, q)

    expected = -u0 * 2.0 * x
    assert jnp.abs(res.arr - expected)[INTERIOR].max() < 1e-12


def test_uniform_tracer_has_zero_tendency(mset, advection):
    q = fr.ScalarField(mset, name="q")
    q.arr = 3.3 * jnp.ones_like(q.arr)
    velocity = make_velocity(mset, q, u0=1.5)

    res = advection.advection(velocity, q)

    assert jnp.abs(res.arr)[INTERIOR].max() == 0


def test_default_interpolation_modules():
    advection = fr.modules.advection.UpwindAdvection(order=3)
    assert isinstance(advection.interp_module,
                      fr.grid.cartesian.PolynomialInterpolation)
    assert isinstance(advection.biased_inter,
                      fr.grid.cartesian.UpwindInterpolation)


def test_custom_interpolation_modules():
    symmetric = fr.grid.cartesian.PolynomialInterpolation(order=3)
    biased = fr.grid.cartesian.UpwindInterpolation(order=4)
    advection = fr.modules.advection.UpwindAdvection(
        order=3, symmetric_inter=symmetric, biased_inter=biased)

    assert advection.interp_module is symmetric
    assert advection.biased_inter is biased

    # the required halo is the maximum of the interpolation modules
    assert advection.required_halo == max(symmetric.required_halo,
                                          biased.required_halo)
