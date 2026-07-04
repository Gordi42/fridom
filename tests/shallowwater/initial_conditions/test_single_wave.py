"""Tests for the single wave initial condition."""

import jax.numpy as jnp
import pytest

import fridom.shallowwater as sw

# ================================================================
#  Constants
# ================================================================
RESOLUTION = 5  # N = 2**RESOLUTION - 1 = 31
PI = jnp.pi


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = sw.grid.cartesian.Grid(
        shape=(2**RESOLUTION - 1, 2**RESOLUTION - 1),
        domain_size=(2*PI, 2*PI),
    )
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-7), order=3)
    mset = sw.ModelSettings(
        grid, f0=1, csqr=1, rossby_number=0.3, time_stepper=time_stepper)
    mset = mset.setup()
    mset.tendencies.advection.disable()
    return mset


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
    k = (2, 3)
    z = sw.initial_conditions.SingleWave(
        mset, k=k, s=s, use_discrete=use_discrete)

    # the state is normalized to a unit L2 norm
    assert z.norm_l2() == pytest.approx(1.0)

    for field in z.fields.values():
        spectral_mag = jnp.abs(field.fft().arr)
        if spectral_mag.max() == 0:
            continue
        nonzero = spectral_mag > 1e-10 * spectral_mag.max()
        for k_mesh, k_i in zip(mset.grid.k_mesh, k, strict=True):
            # only the wavenumbers +-k_i are present
            assert jnp.all(jnp.abs(k_mesh[nonzero]) == k_i)


def test_phase_shift(mset):
    """A phase shift of pi flips the sign of the state."""
    z_0 = sw.initial_conditions.SingleWave(mset, k=(2, 3), phase=0)
    z_pi = sw.initial_conditions.SingleWave(mset, k=(2, 3), phase=PI)

    assert (z_0 + z_pi).norm_l2() < 1e-10


def test_wave_returns_after_one_period(mset):
    """A single wave reproduces itself after one wave period."""
    z = sw.initial_conditions.SingleWave(mset, k=(2, 3), s=1)

    assert z.period > 0

    model = sw.Model(mset)
    model.z = z
    model.run(runlen=z.period)

    assert (model.z - z).norm_l2() < 5e-2


def test_geostrophic_mode_is_stationary(mset):
    """The geostrophic mode does not evolve in the linear model."""
    z = sw.initial_conditions.SingleWave(mset, k=(2, 3), s=0)

    model = sw.Model(mset)
    model.z = z
    model.run(runlen=1.0)

    assert (model.z - z).norm_l2() < 1e-10
