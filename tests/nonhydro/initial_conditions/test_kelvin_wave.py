"""Tests for the Kelvin wave initial condition."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro as nh


# ================================================================
#  Fixtures
# ================================================================
def make_mset(periodic_bounds):
    grid = nh.grid.cartesian.Grid(
        shape=(16, 16, 16), domain_size=(1, 1, 1),
        periodic_bounds=periodic_bounds)
    mset = nh.ModelSettings(grid, f0=1, stratification_n2=1)
    mset.time_stepper.dt = np.timedelta64(10, "ms")
    return mset.setup()


@pytest.fixture
def mset_ns():
    """Model settings with walls in the y-direction (north/south)."""
    return make_mset((True, False, True))


@pytest.fixture
def mset_ew():
    """Model settings with walls in the x-direction (east/west)."""
    return make_mset((False, True, True))


# ================================================================
#  Helper functions
# ================================================================
def check_wave(z, x_normal, l_normal, flip):
    """Check the decay away from the boundary."""
    # the wave decays away from the boundary
    if flip:
        x_normal = l_normal - x_normal
    amplitude = jnp.abs(z.w.arr)
    near_wall = amplitude[x_normal < 0.2].max()
    far_away = amplitude[x_normal > 0.8].max()

    assert near_wall > 100 * far_away

    # the frequency and the wavenumbers are stored
    assert z.om > 0
    assert z.k_parallel == pytest.approx(2 * jnp.pi)
    assert z.kz == pytest.approx(4 * jnp.pi)


@pytest.mark.parametrize("side", ["N", "S"])
def test_kelvin_wave_north_south(mset_ns, side):
    z = nh.initial_conditions.KelvinWave(mset_ns, side, kz=2, k_parallel=1)

    # the wall-normal velocity vanishes
    assert jnp.abs(z.v.arr).max() == 0
    y = z.w.get_mesh()[1]
    check_wave(z, y, mset_ns.grid.domain_size[1], flip=(side == "N"))


@pytest.mark.parametrize("side", ["E", "W"])
def test_kelvin_wave_east_west(mset_ew, side):
    z = nh.initial_conditions.KelvinWave(mset_ew, side, kz=2, k_parallel=1)

    # the wall-normal velocity vanishes
    assert jnp.abs(z.u.arr).max() == 0
    x = z.w.get_mesh()[0]
    check_wave(z, x, mset_ew.grid.domain_size[0], flip=(side == "E"))


def test_phase_shift(mset_ns):
    z_0 = nh.initial_conditions.KelvinWave(
        mset_ns, "S", kz=2, k_parallel=1, phase=0)
    z_pi = nh.initial_conditions.KelvinWave(
        mset_ns, "S", kz=2, k_parallel=1, phase=jnp.pi)

    # a phase shift of pi flips the sign of the state
    assert (z_0 + z_pi).norm_l2() < 1e-10


def test_invalid_side_raises(mset_ns):
    with pytest.raises(ValueError, match="Invalid side 'X'"):
        nh.initial_conditions.KelvinWave(mset_ns, "X", kz=2, k_parallel=1)
