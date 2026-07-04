"""Tests for the spectral advection scheme of the nonhydrostatic model."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
ROSSBY_NUMBER = 0.1


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = nh.grid.spectral.Grid(shape=(16, 16, 16), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(
        grid, f0=1, stratification_n2=1, rossby_number=ROSSBY_NUMBER)
    mset.time_stepper.dt = np.timedelta64(10, "ms")
    mset.tendencies.pressure_solver = \
        nh.modules.pressure_solvers.SpectralPressureSolver()
    return mset.setup()


@pytest.fixture
def z_ini(mset):
    return nh.initial_conditions.RandomGeostrophicSpectra(mset, seed=1)


# ================================================================
#  Tests
# ================================================================
def test_spectral_advection_is_selected(mset):
    advection = mset.tendencies.advection

    assert isinstance(advection, nh.modules.advection.SpectralAdvection)
    assert advection.scaling == ROSSBY_NUMBER


def test_energy_conservation(mset, z_ini):
    e_ini = z_ini.etot.integrate().value

    model = nh.Model(mset)
    model.z = z_ini
    model.run(runlen=np.timedelta64(1, "s"))

    z_final = model.z.ifft()
    e_final = z_final.etot.integrate().value

    assert not jnp.isnan(z_final.u.arr).any()
    assert abs(1 - e_final / e_ini) < 1e-2


def test_advection_changes_the_solution(mset, z_ini):
    model = nh.Model(mset)
    model.z = z_ini
    model.run(runlen=np.timedelta64(1, "s"))
    z_nonlinear = model.z.ifft()

    mset.tendencies.advection.disable()
    model = nh.Model(mset)
    model.z = z_ini
    model.run(runlen=np.timedelta64(1, "s"))
    z_linear = model.z.ifft()

    assert (z_nonlinear - z_linear).norm_l2() > 1e-3


def test_tracer_advection_respects_no_adv_flag(mset, z_ini):
    advection = mset.tendencies.advection
    z = z_ini.fft()

    # by default, the buoyancy is an advected tracer
    dz = advection.advect_state(z, z * 0)
    assert jnp.abs(dz.b.arr).max() > 0

    # with the NO_ADV flag, the buoyancy is skipped
    z.b.flags = {"NO_ADV": True}
    dz = advection.advect_state(z, z * 0)
    assert jnp.abs(dz.b.arr).max() == 0
