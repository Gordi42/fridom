"""Tests for the nonlinear normal mode decomposition."""

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
def z_geo(mset):
    return sw.initial_conditions.RandomGeostrophicSpectra(mset, seed=1)


def wave_production(mset, z):
    """Measure the wave energy produced by a model run started from z."""
    proj_wave = sw.projection.WaveSpectral(mset)
    model = sw.Model(mset)
    model.z = z
    model.run(runlen=2.0)
    return proj_wave(model.z - z).norm_l2()


# ================================================================
#  Tests
# ================================================================
def test_order_zero_equals_geostrophic_projection(mset, z_geo):
    nnmd = fr.projection.NNMD(mset, order=0)
    proj_geo = sw.projection.GeostrophicSpectral(mset)

    assert (nnmd(z_geo) - proj_geo(z_geo)).norm_l2() < 1e-10


def test_base_point_is_preserved(mset, z_geo):
    nnmd = fr.projection.NNMD(mset, order=2, use_model=False)
    proj_geo = sw.projection.GeostrophicSpectral(mset)

    z_bal = nnmd(z_geo)

    z_base = proj_geo(z_geo)
    diff = (proj_geo(z_bal) - z_base).norm_l2()
    assert diff / z_base.norm_l2() < 1e-10


def test_balancing_improves_with_order(mset, z_geo):
    nnmd0 = fr.projection.NNMD(mset, order=0, use_model=False)
    nnmd1 = fr.projection.NNMD(mset, order=1, use_model=False)

    production0 = wave_production(mset, nnmd0(z_geo))
    production1 = wave_production(mset, nnmd1(z_geo))

    # the first-order balanced state produces much less waves than
    # the naive geostrophic projection (order zero)
    assert production1 < production0 / 3


def test_derivative_with_model(mset, z_geo):
    nnmd = fr.projection.NNMD(mset, order=3, use_model=True)
    proj_geo = sw.projection.GeostrophicSpectral(mset)

    z_bal = nnmd(z_geo)

    # the balanced state is finite and preserves the base point
    assert not z_bal.has_nan()
    z_base = proj_geo(z_geo)
    diff = (proj_geo(z_bal) - z_base).norm_l2()
    assert diff / z_base.norm_l2() < 1e-10


def test_create_state_spectral(mset, z_geo):
    nnmd = fr.projection.NNMD(mset, order=1, use_model=False)
    _ = nnmd(z_geo)

    z_spectral = nnmd.create_state(order=1, spectral=True)
    z_physical = nnmd.create_state(order=1, spectral=False)

    assert z_spectral.is_spectral
    assert not z_physical.is_spectral
    assert (z_spectral.ifft() - z_physical).norm_l2() < 1e-10


def test_bilinear_form_is_symmetric(mset, z_geo):
    nnmd = fr.projection.NNMD(mset, order=1, use_model=False)
    proj_geo = sw.projection.GeostrophicSpectral(mset)
    proj_wave = sw.projection.WaveSpectral(mset)

    z_wave = sw.initial_conditions.SingleWave(mset, k=(2, 3), s=1)
    z1 = proj_geo(z_geo).fft()
    z2 = proj_wave(z_wave).fft()

    s12 = nnmd.bilinear_form(z1, z2)
    s21 = nnmd.bilinear_form(z2, z1)

    assert (s12 - s21).ifft().norm_l2() < 1e-10

    # for identical arguments, S(z, z) equals the advection term N(z)
    s11 = nnmd.bilinear_form(z1, z1)
    advection = nnmd._advect_state(z1.ifft()).fft()

    assert (s11 - advection).ifft().norm_l2() < 1e-10


def test_getitem_invalid_keys(mset, z_geo):
    nnmd = fr.projection.NNMD(mset, order=1, use_model=False)
    _ = nnmd(z_geo)

    with pytest.raises(TypeError, match="Key must be a tuple"):
        _ = nnmd[0]
    with pytest.raises(ValueError, match="Mode must be 0, 1, or 2"):
        _ = nnmd[3, 0, 0]
    with pytest.raises(ValueError, match="Order must be 0 for mode 0"):
        _ = nnmd[0, 1, 0]
