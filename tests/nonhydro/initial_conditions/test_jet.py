"""Tests for the 3D jet initial condition."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro as nh


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = nh.grid.cartesian.Grid(shape=(16, 16, 8), domain_size=(4, 4, 1))
    mset = nh.ModelSettings(
        grid, f0=1, stratification_n2=1.0, dsqr=0.2**2, rossby_number=0.1)
    mset.time_stepper.dt = np.timedelta64(100, "ms")
    return mset.setup()


# ================================================================
#  Tests
# ================================================================
def test_jet_without_projection(mset):
    pert_strength = 0.05
    z = nh.initial_conditions.Jet(
        mset, pert_strength=pert_strength, geo_proj=False)

    # the jets dominate the zonal velocity
    assert jnp.abs(z.u.arr).max() > pert_strength
    # the perturbation is normalized to the given strength
    assert jnp.abs(z.v.arr).max() == pytest.approx(pert_strength)
    # without projection, the buoyancy perturbation comes from the
    # geostrophic single wave only
    assert jnp.abs(z.b.arr).max() == pytest.approx(0.0)


def test_jet_with_projection(mset):
    z = nh.initial_conditions.Jet(mset, geo_proj=True)

    # the projection is idempotent: projecting again does not change
    # the state
    proj_geo = nh.projection.GeostrophicSpectral(mset)
    z_proj = proj_geo(z)

    assert (z_proj - z).norm_l2() < 1e-10
    # the geostrophic jet is in thermal wind balance (nonzero buoyancy)
    assert jnp.abs(z.b.arr).max() > 0
