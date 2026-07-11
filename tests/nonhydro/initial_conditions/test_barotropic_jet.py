"""Tests for the barotropic jet initial condition."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = nh.grid.cartesian.Grid(shape=(16, 16, 16), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(
        grid, f0=1, stratification_n2=1, dsqr=0.2**2)
    mset.time_stepper.dt = np.timedelta64(10, "ms")
    return mset.setup()


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("geo_proj", [
    pytest.param(True, id="geo_proj"),
    pytest.param(False, id="no_proj"),
])
def test_two_opposing_jets(mset, geo_proj):
    z = nh.initial_conditions.BarotropicJet(mset, geo_proj=geo_proj)

    y = z.u.get_mesh()[1]
    ly = mset.grid.domain_size[1]

    # eastward jet in the north, westward jet in the south
    assert z.u.arr[jnp.abs(y - 0.75*ly) < 0.2].mean() > 0
    assert z.u.arr[jnp.abs(y - 0.25*ly) < 0.2].mean() < 0


def test_perturbation_amplitude(mset):
    waveamp = 0.1
    z = nh.initial_conditions.BarotropicJet(
        mset, waveamp=waveamp, geo_proj=False)

    # the meridional velocity perturbation has the given amplitude
    assert jnp.abs(z.v.arr).max() == pytest.approx(waveamp, rel=0.1)


def test_projected_jet_is_geostrophic(mset):
    z = nh.initial_conditions.BarotropicJet(mset, geo_proj=True)

    proj_geo = nh.projection.GeostrophicSpectral(mset)
    z_proj = proj_geo(z)

    assert (z_proj - z).norm_l2() / z.norm_l2() < 1e-10
