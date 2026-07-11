"""Tests for the spectral projections."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework as fr
import fridom.nonhydro as nh
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
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-6), order=3)
    return sw.ModelSettings(
        grid, f0=1, csqr=1, rossby_number=0.1,
        time_stepper=time_stepper).setup()


@pytest.fixture
def z(mset):
    z_geo = sw.initial_conditions.RandomGeostrophicSpectra(mset, seed=1)
    z_wave = sw.initial_conditions.SingleWave(mset, k=(2, 3), s=1)
    return z_geo + 0.5 * z_wave


@pytest.fixture
def mset_nh():
    grid = nh.grid.cartesian.Grid(shape=(8, 8, 8), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=1, stratification_n2=4, dsqr=1)
    mset.time_stepper.dt = np.timedelta64(100, "ms")
    return mset.setup()


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("use_discrete", [
    pytest.param(True, id="discrete"),
    pytest.param(False, id="analytical"),
])
def test_projections_are_idempotent(mset, z, use_discrete):
    for proj_type in (fr.projection.GeostrophicSpectral,
                      fr.projection.WaveSpectral):
        proj = proj_type(mset, use_discrete=use_discrete)
        z_proj = proj(z)

        assert (proj(z_proj) - z_proj).norm_l2() / z.norm_l2() < 1e-10


def test_projections_are_orthogonal(mset, z):
    proj_geo = fr.projection.GeostrophicSpectral(mset)
    proj_wave = fr.projection.WaveSpectral(mset)

    assert proj_geo(proj_wave(z)).norm_l2() / z.norm_l2() < 1e-10
    assert proj_wave(proj_geo(z)).norm_l2() / z.norm_l2() < 1e-10


def test_projections_are_complete(mset, z):
    proj_geo = fr.projection.GeostrophicSpectral(mset)
    proj_wave = fr.projection.WaveSpectral(mset)

    # in the shallow water model, the geostrophic and the wave
    # subspaces span the full state space
    z_sum = proj_geo(z) + proj_wave(z)

    assert (z_sum - z).norm_l2() / z.norm_l2() < 1e-10


def test_divergence_projection(mset_nh):
    proj_div = fr.projection.DivergenceSpectral(mset_nh)

    # geostrophic and wave modes have no divergent component
    z_wave = nh.initial_conditions.SingleWave(mset_nh, k=(1, 2, 3), s=1)
    z_geo = nh.initial_conditions.SingleWave(mset_nh, k=(1, 2, 3), s=0)

    assert proj_div(z_wave).norm_l2() < 1e-10
    assert proj_div(z_geo).norm_l2() < 1e-10

    # a divergent state has a nonzero, idempotent projection
    z_div = nh.State(mset_nh)
    x, _y, _z = z_div.u.get_mesh()
    z_div.u.arr = jnp.sin(x)
    z_div.sync()
    z_proj = proj_div(z_div)

    assert z_proj.norm_l2() > 1
    assert (proj_div(z_proj) - z_proj).norm_l2() < 1e-10
