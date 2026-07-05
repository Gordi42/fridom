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
@pytest.fixture
def mset():
    grid = sw.grid.cartesian.Grid(
        shape=(31, 31), domain_size=(2*PI, 2*PI))
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-7), order=3)
    mset = sw.ModelSettings(
        grid, f0=1, csqr=1, rossby_number=0.1, time_stepper=time_stepper)
    mset = mset.setup()
    mset.tendencies.advection.disable()
    return mset


# ================================================================
#  Tests
# ================================================================
def test_linear_energy_conservation(mset):
    z = sw.initial_conditions.SingleWave(mset, k=(2, 3), s=1)
    e_ini = z.etot.integrate().arr.item()

    model = sw.Model(mset)
    model.z = z
    model.run(runlen=1.0)

    z_final = model.z
    e_final = z_final.etot.integrate().arr.item()

    # the wave propagates (the state changes) ...
    assert (z_final - z).norm_l2() > 1e-3
    # ... but the linear model conserves the total energy
    assert abs(1 - e_final / e_ini) < 1e-6


def test_geostrophic_state_has_no_tendency(mset):
    z = sw.initial_conditions.SingleWave(mset, k=(2, 3), s=0)

    model = sw.Model(mset)
    model.z = z
    model.run(runlen=1.0)

    # a geostrophically balanced state is stationary in the linear model
    assert (model.z - z).norm_l2() < 1e-10
