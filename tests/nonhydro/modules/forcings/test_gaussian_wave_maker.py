"""Tests for the gaussian wave maker of the nonhydrostatic model."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr
import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
FREQUENCY = 0.25
AMPLITUDE = 2.0


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = nh.grid.cartesian.Grid(shape=(8, 8, 8), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=0.0, stratification_n2=0.0)
    return mset.setup()


@pytest.fixture
def wave_maker(mset):
    wave_maker = nh.modules.forcings.GaussianWaveMaker(
        position=(PI, PI, None), width=(0.5, 0.5, None),
        frequency=FREQUENCY, amplitude=AMPLITUDE)
    return wave_maker.setup(mset=mset)


@pytest.fixture
def expected_mask(mset):
    # axes with position or width None are constant
    x, y, _z = mset.grid.x_mesh
    return (AMPLITUDE * jnp.exp(-(x - PI)**2 / 0.5**2)
                      * jnp.exp(-(y - PI)**2 / 0.5**2))


# ================================================================
#  Tests
# ================================================================
def test_mask(wave_maker, expected_mask):
    assert jnp.array_equal(wave_maker.mask, expected_mask)


def test_update_forces_u(mset, wave_maker, expected_mask):
    mz = fr.ModelState(mset)
    # sin(2 pi f t) = 1 at t = 1 / (4 f)
    mz.clock.time = 1.0 / (4 * FREQUENCY)

    mz = wave_maker.update(mz=mz)

    assert jnp.array_equal(mz.dz.u.arr, expected_mask)
    assert jnp.abs(mz.dz.v.arr).max() == 0
    assert jnp.abs(mz.dz.w.arr).max() == 0
    assert jnp.abs(mz.dz.b.arr).max() == 0


def test_no_forcing_at_time_zero(mset, wave_maker):
    mz = fr.ModelState(mset)
    mz = wave_maker.update(mz=mz)
    assert jnp.abs(mz.dz.u.arr).max() == 0


def test_custom_variable(mset):
    wave_maker = nh.modules.forcings.GaussianWaveMaker(
        position=(PI, PI, PI), width=(0.5, 0.5, 0.5),
        frequency=FREQUENCY, amplitude=1.0, variable="b")
    wave_maker.setup(mset=mset)

    mz = fr.ModelState(mset)
    mz.clock.time = 1.0 / (4 * FREQUENCY)
    mz = wave_maker.update(mz=mz)

    assert jnp.abs(mz.dz.b.arr).max() > 0
    assert jnp.abs(mz.dz.u.arr).max() == 0


def test_info(wave_maker):
    info = wave_maker.info
    assert info["position"] == (PI, PI, None)
    assert info["width"] == (0.5, 0.5, None)
    assert info["frequency"] == FREQUENCY
    assert info["amplitude"] == AMPLITUDE
