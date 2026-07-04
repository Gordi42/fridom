"""Tests for the spectral advection scheme of the shallow water model."""

from collections import OrderedDict

import jax.numpy as jnp
import pytest

import fridom.shallowwater as sw

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
    grid = sw.grid.spectral.Grid(shape=(32, 32), domain_size=(2*PI, 2*PI))
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-7), order=3)
    mset = sw.ModelSettings(
        grid, f0=1, csqr=1, rossby_number=ROSSBY_NUMBER,
        time_stepper=time_stepper)
    mset.tendencies.advection = sw.modules.advection.SpectralAdvection()
    return mset.setup()


@pytest.fixture
def z_ini(mset):
    return sw.initial_conditions.RandomGeostrophicSpectra(mset, seed=1)


# ================================================================
#  Tests
# ================================================================
def test_scaling_is_set_on_setup(mset):
    assert mset.tendencies.advection.scaling == ROSSBY_NUMBER


def test_energy_conservation(mset, z_ini):
    e_ini = z_ini.etot.integrate().arr.item()

    model = sw.Model(mset)
    model.z = z_ini.fft()
    model.run(runlen=1.0)

    z_final = model.z.ifft()
    e_final = z_final.etot.integrate().arr.item()

    assert not jnp.isnan(z_final.u.arr).any()
    assert abs(1 - e_final / e_ini) < 1e-5


def test_advection_changes_the_solution(mset, z_ini):
    model = sw.Model(mset)
    model.z = z_ini.fft()
    model.run(runlen=1.0)
    z_nonlinear = model.z.ifft()

    mset.tendencies.advection.disable()
    model = sw.Model(mset)
    model.z = z_ini.fft()
    model.run(runlen=1.0)
    z_linear = model.z.ifft()

    assert (z_nonlinear - z_linear).norm_l2() > 1e-3


def test_tracer_advection(mset, z_ini):
    # add two tracers, one of them with disabled advection
    x, _y = mset.grid.x_mesh
    fields = OrderedDict(z_ini.fields)
    for name, no_adv in (("c1", False), ("c2", True)):
        tracer = sw.ScalarField(mset, name=name)
        tracer.arr = jnp.sin(x)
        tracer.flags = {"NO_ADV": no_adv}
        fields[name] = tracer
    z = sw.State(mset, field_list=fields).fft()

    advection = mset.tendencies.advection
    dz = advection.advect_state(z, z * 0)

    # momentum and the advected tracer receive a tendency
    assert jnp.abs(dz.u.arr).max() > 0
    assert jnp.abs(dz["c1"].arr).max() > 0
    # the tracer with the NO_ADV flag is skipped
    assert jnp.abs(dz["c2"].arr).max() == 0
