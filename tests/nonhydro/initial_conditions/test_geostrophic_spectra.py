"""Tests for the random geostrophic spectra initial condition."""

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
def test_energy_spectrum_shape():
    k = jnp.linspace(0.01, 20, 1000)
    k0 = 6
    spectrum = nh.initial_conditions.geostrophic_energy_spectrum(
        k, 0*k, 0*k, k0=k0)

    # the spectrum is positive and peaks close to k0
    assert (spectrum >= 0).all()
    assert k[jnp.argmax(spectrum)] == pytest.approx(k0, rel=0.1)

    # the vertical spectrum decays with the vertical wavenumber
    kz = jnp.linspace(0, 10, 100)
    spectrum = nh.initial_conditions.geostrophic_energy_spectrum(
        0*kz + k0, 0*kz, kz)
    assert (jnp.diff(spectrum) < 0).all()


def test_normalization(mset):
    z = nh.initial_conditions.RandomGeostrophicSpectra(mset, seed=7)

    # the maximum zonal velocity is normalized to one
    assert z.u.arr.max() == pytest.approx(1.0)


def test_seed_reproducibility(mset):
    ic = nh.initial_conditions
    z1 = ic.RandomGeostrophicSpectra(mset, seed=7)
    z2 = ic.RandomGeostrophicSpectra(mset, seed=7)
    z3 = ic.RandomGeostrophicSpectra(mset, seed=8)

    # same seed => same state, different seed => different state
    assert (z1 - z2).norm_l2() == 0
    assert (z1 - z3).norm_l2() > 1e-3


def test_state_is_geostrophic(mset):
    z = nh.initial_conditions.RandomGeostrophicSpectra(mset, seed=7)

    proj_geo = nh.projection.GeostrophicSpectral(mset)
    z_proj = proj_geo(z)

    assert (z_proj - z).norm_l2() / z.norm_l2() < 1e-10
