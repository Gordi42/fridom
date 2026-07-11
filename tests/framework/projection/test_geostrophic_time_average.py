"""Tests for the geostrophic time-average projection."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr
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


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("equidistant_chunks", [
    pytest.param(True, id="equidistant"),
    pytest.param(False, id="constant_periods"),
])
@pytest.mark.parametrize("backward_forward", [
    pytest.param(False, id="forward"),
    pytest.param(True, id="backward_forward"),
])
def test_time_average_removes_waves(
        mset, z, backward_forward, equidistant_chunks):
    proj_geo = sw.projection.GeostrophicSpectral(mset)
    proj_wave = sw.projection.WaveSpectral(mset)
    wave_energy_before = proj_wave(z).norm_l2()

    gta = fr.projection.GeostrophicTimeAverage(
        mset,
        max_period=2*PI,
        n_ave=2,
        equidistant_chunks=equidistant_chunks,
        backward_forward=backward_forward)
    z_ave = gta(z)

    # the time-averaging strongly damps the wave component ...
    wave_energy_after = proj_wave(z_ave).norm_l2()
    assert wave_energy_after < wave_energy_before / 50

    # ... while the geostrophic component is preserved
    z_geo = proj_geo(z)
    assert (proj_geo(z_ave) - z_geo).norm_l2() / z_geo.norm_l2() < 1e-10
