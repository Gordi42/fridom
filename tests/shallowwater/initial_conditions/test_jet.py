"""Tests for the unstable jet initial condition."""

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
    grid = sw.grid.cartesian.Grid(shape=(31, 31), domain_size=(2*PI, 2*PI))
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-5), order=3)
    return sw.ModelSettings(
        grid, f0=1, csqr=1, rossby_number=0.3,
        time_stepper=time_stepper).setup()


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("geo_proj", [
    pytest.param(True, id="geo_proj"),
    pytest.param(False, id="no_proj"),
])
def test_jet_amplitude(mset, geo_proj):
    waveamp = 0.1
    z = sw.initial_conditions.Jet(mset, waveamp=waveamp, geo_proj=geo_proj)

    # the jet is normalized to a unit velocity amplitude, the
    # perturbation adds at most `waveamp` on top of it
    speed = (z.u.arr**2 + z.v.arr**2)**0.5

    assert speed.max() == pytest.approx(1.0, abs=2*waveamp)


def test_jet_is_geostrophic(mset):
    z = sw.initial_conditions.Jet(mset, geo_proj=True)

    # both the jet and the perturbation are geostrophic, so the
    # projection onto the geostrophic subspace does not change the state
    proj_geo = sw.projection.GeostrophicSpectral(mset)
    z_proj = proj_geo(z)

    assert (z_proj - z).norm_l2() / z.norm_l2() < 1e-10
