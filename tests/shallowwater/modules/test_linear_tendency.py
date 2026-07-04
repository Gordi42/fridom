"""Tests for the linear tendency module of the shallow water model."""

import jax.numpy as jnp
import pytest

import fridom.shallowwater as sw

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(params=[
    pytest.param("cartesian", id="cartesian"),
    pytest.param("spectral", id="spectral"),
])
def mset(request):
    if request.param == "cartesian":
        grid = sw.grid.cartesian.Grid(
            shape=(31, 31), domain_size=(2*PI, 2*PI))
    else:
        grid = sw.grid.spectral.Grid(
            shape=(32, 32), domain_size=(2*PI, 2*PI))
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-7), order=3)
    mset = sw.ModelSettings(
        grid, f0=1, csqr=1, rossby_number=0.1, time_stepper=time_stepper)
    mset = mset.setup()
    mset.tendencies.advection.disable()
    return mset


# ================================================================
#  Helper functions
# ================================================================
def to_model_space(mset, z):
    """Transform the state to the space the model expects."""
    if isinstance(mset.grid, sw.grid.spectral.Grid):
        return z.fft()
    return z


def to_physical_space(z):
    """Transform the state back to physical space."""
    return z.ifft() if z.is_spectral else z


# ================================================================
#  Tests
# ================================================================
def test_linear_energy_conservation(mset):
    z = sw.initial_conditions.SingleWave(mset, k=(2, 3), s=1)
    e_ini = z.etot.integrate().arr.item()

    model = sw.Model(mset)
    model.z = to_model_space(mset, z)
    model.run(runlen=1.0)

    z_final = to_physical_space(model.z)
    e_final = z_final.etot.integrate().arr.item()

    # the wave propagates (the state changes) ...
    assert (z_final - z).norm_l2() > 1e-3
    # ... but the linear model conserves the total energy
    assert abs(1 - e_final / e_ini) < 1e-6


def test_geostrophic_state_has_no_tendency(mset):
    z = sw.initial_conditions.SingleWave(mset, k=(2, 3), s=0)

    model = sw.Model(mset)
    model.z = to_model_space(mset, z)
    model.run(runlen=1.0)

    z_final = to_physical_space(model.z)

    # a geostrophically balanced state is stationary in the linear model
    assert (z_final - z).norm_l2() < 1e-10
