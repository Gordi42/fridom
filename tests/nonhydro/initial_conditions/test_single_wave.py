"""Tests for the single wave initial condition."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
F0 = 1e-4
N2 = (50 * F0) ** 2
SHAPE = (16, 16, 16)
PI = jnp.pi


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = nh.grid.cartesian.Grid(shape=SHAPE, domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=F0, stratification_n2=N2)
    mset.time_stepper.dt = np.timedelta64(5, "s")
    mset.tendencies.advection.disable()
    return mset.setup()


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("use_discrete", [
    pytest.param(True, id="discrete"),
    pytest.param(False, id="analytical"),
])
@pytest.mark.parametrize("s", [
    pytest.param(0, id="geostrophic"),
    pytest.param(1, id="wave_plus"),
    pytest.param(-1, id="wave_minus"),
])
def test_monochromatic(mset, s, use_discrete):
    """The wave contains exactly the requested wavenumber."""
    k = (2, 3, 4)
    z = nh.initial_conditions.SingleWave(
        mset, k=k, s=s, use_discrete=use_discrete)

    # the state is normalized to a unit L2 norm
    assert z.norm_l2() == pytest.approx(1.0)

    kx_mesh, ky_mesh, kz_mesh = mset.grid.k_mesh
    meshes = (kx_mesh, ky_mesh, kz_mesh)
    for field in z.fields.values():
        spectral_mag = jnp.abs(field.fft().arr)
        if spectral_mag.max() == 0:
            continue
        nonzero = spectral_mag > 1e-10 * spectral_mag.max()
        for k_mesh, k_i in zip(meshes, k, strict=True):
            # only the wavenumbers +-k_i are present
            assert jnp.all(jnp.abs(k_mesh[nonzero]) == k_i)


def test_phase_shift(mset):
    """A phase shift of pi flips the sign of the state."""
    z_0 = nh.initial_conditions.SingleWave(mset, k=(2, 3, 4), phase=0)
    z_pi = nh.initial_conditions.SingleWave(mset, k=(2, 3, 4), phase=PI)

    assert (z_0 + z_pi).norm_l2() < 1e-10


def test_wave_returns_after_one_period(mset):
    """A single wave reproduces itself after one wave period."""
    z = nh.initial_conditions.SingleWave(mset, k=(2, 3, 4), s=1)

    assert z.period > 0

    model = nh.Model(mset)
    model.z = z
    model.run(runlen=np.timedelta64(int(z.period), "s"))

    assert (model.z - z).norm_l2() < 1e-2


def test_geostrophic_mode_is_stationary(mset):
    """The geostrophic mode does not evolve in the linear model."""
    z = nh.initial_conditions.SingleWave(mset, k=(2, 3, 4), s=0)

    model = nh.Model(mset)
    model.z = z
    model.run(runlen=np.timedelta64(1, "h"))

    assert (model.z - z).norm_l2() < 1e-10
