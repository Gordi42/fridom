"""Tests for the wave package initial condition."""

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
def test_wave_package_is_localized(mset):
    z = nh.initial_conditions.WavePackage(
        mset,
        mask_pos=(PI, None, None),
        mask_width=(0.8, None, None),
        k=(2, 3, 4),
        s=1)

    # the wave package has a frequency and a period
    assert z.omega.real > 0
    assert z.period == pytest.approx(2*PI / z.omega.real)

    # the wave amplitude is localized around the mask position
    x = z.w.get_mesh()[0]
    amplitude = jnp.abs(z.w.arr)
    near_mask = amplitude[jnp.abs(x - PI) < 0.8].max()
    far_away = amplitude[jnp.abs(x - PI) > 2.5].max()

    assert near_mask > 100 * far_away


def test_unmasked_directions_are_not_masked(mset):
    z = nh.initial_conditions.WavePackage(
        mset,
        mask_pos=(PI, None, None),
        mask_width=(0.8, None, None),
        k=(2, 3, 4),
        s=1)

    # the wave is not masked in the y-direction: the amplitude close
    # to and far away from the center are comparable
    y = z.w.get_mesh()[1]
    amplitude = jnp.abs(z.w.arr)
    near_center = amplitude[jnp.abs(y - PI) < 0.8].max()
    far_away = amplitude[jnp.abs(y - PI) > 2.5].max()

    assert near_center < 10 * far_away


def test_geostrophic_package_has_no_frequency(mset):
    z = nh.initial_conditions.WavePackage(
        mset,
        mask_pos=(PI, None, None),
        mask_width=(0.8, None, None),
        k=(2, 3, 4),
        s=0)

    # the geostrophic mode has no wave frequency
    assert not hasattr(z, "omega")
    assert not hasattr(z, "period")
