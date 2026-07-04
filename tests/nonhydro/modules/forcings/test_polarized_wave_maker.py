"""Tests for the polarized wave maker of the nonhydrostatic model."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr
import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
AMPLITUDE = 0.5


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = nh.grid.cartesian.Grid(shape=(16, 16, 16), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=1.0, stratification_n2=4.0)
    return mset.setup()


@pytest.fixture
def wave_maker(mset):
    wave_maker = nh.modules.forcings.PolarizedWaveMaker(
        position=(PI, PI, PI), width=(1.0, 1.0, 1.0),
        k=(2, 0, 1), amplitude=AMPLITUDE)
    return wave_maker.setup(mset=mset)


# ================================================================
#  Tests
# ================================================================
def test_frequency_from_dispersion_relation(mset, wave_maker):
    # the frequency of the wave maker is the (real part of the)
    # frequency of the corresponding wave package
    wave = nh.initial_conditions.WavePackage(
        mset, mask_pos=(PI, PI, PI), mask_width=(1.0, 1.0, 1.0), k=(2, 0, 1))
    assert wave_maker.frequency == wave.omega.real


def test_update_adds_polarized_source(mset, wave_maker):
    mz = fr.ModelState(mset)
    # sin(omega t) = 1 at t = pi / (2 omega)
    mz.clock.time = 0.5 * PI / wave_maker.frequency

    mz = wave_maker.update(mz=mz)

    for name in ("u", "v", "w", "b"):
        source = wave_maker.source.fields[name].arr
        assert jnp.abs(source).max() > 0
        assert jnp.array_equal(mz.dz.fields[name].arr, source)


def test_no_forcing_at_time_zero(mset, wave_maker):
    mz = fr.ModelState(mset)
    mz = wave_maker.update(mz=mz)
    for df in mz.dz:
        assert jnp.abs(df.arr).max() == 0


def test_info(wave_maker):
    info = wave_maker.info
    assert info["position"] == (PI, PI, PI)
    assert info["width"] == (1.0, 1.0, 1.0)
    assert info["frequency"] == wave_maker.frequency
    assert info["amplitude"] == AMPLITUDE
