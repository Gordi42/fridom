"""Tests for the Sadourny advection scheme of the shallow water model."""

import jax.numpy as jnp
import numpy as np

import fridom.framework as fr
import fridom.shallowwater as sw

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
ROSSBY_NUMBER = 0.1

# interior slice away from the periodically wrapped halo cells
INTERIOR = (slice(3, -3),) * 2


# ================================================================
#  Helpers
# ================================================================
def make_mset(custom_fields=()):
    grid = sw.grid.cartesian.Grid(shape=(31, 31), domain_size=(2*PI, 2*PI))
    mset = sw.ModelSettings(grid, f0=1.0, csqr=1.0,
                            rossby_number=ROSSBY_NUMBER)
    mset.time_stepper.dt = np.timedelta64(10, "ms")
    for name in custom_fields:
        mset.custom_state_fields.append(fr.FieldMetadata(name=name))
    return mset.setup()


# ================================================================
#  Tests
# ================================================================
def test_sadourny_is_default_advection():
    mset = make_mset()
    advection = mset.tendencies.advection
    assert isinstance(advection, sw.modules.advection.SadournyAdvection)
    assert advection.scaling == ROSSBY_NUMBER


def test_energy_conservation():
    # the sadourny scheme is designed to conserve the total energy
    mset = make_mset()
    z = sw.initial_conditions.Jet(mset)
    initial_energy = z.etot.integrate().value

    model = sw.Model(mset)
    model.z = z
    model.run(runlen=np.timedelta64(1, "s"))

    final_energy = model.z.etot.integrate().value
    assert not jnp.isnan(model.z.u.arr).any()
    assert abs(1 - final_energy / initial_energy) < 1e-6


def test_uniform_tracer_has_zero_tendency():
    # for a uniform tracer, the advection terms -div(vC) and C div(v)
    # cancel exactly
    mset = make_mset(custom_fields=["c"])
    advection = mset.tendencies.advection

    z = sw.State(mset)
    x, y = mset.grid.x_mesh
    z.u.arr = jnp.sin(y)
    z.fields["c"].arr = 3.3 * jnp.ones_like(x)
    z.sync()

    dz = advection.advect_state(z, sw.State(mset))
    assert jnp.abs(dz.fields["c"].arr[INTERIOR]).max() == 0


def test_tracer_respects_no_adv_flag():
    mset = make_mset(custom_fields=["c"])
    advection = mset.tendencies.advection

    x, y = mset.grid.x_mesh
    z = sw.State(mset)
    z.u.arr = jnp.sin(y)
    z.fields["c"].arr = jnp.sin(x)
    z.fields["c"].flags = {"NO_ADV": True}
    z.sync()

    dz = advection.advect_state(z, sw.State(mset))
    assert jnp.abs(dz.fields["c"].arr).max() == 0


def test_tracer_is_advected():
    mset = make_mset(custom_fields=["c"])
    advection = mset.tendencies.advection

    x, _y = mset.grid.x_mesh
    z = sw.State(mset)
    z.u.arr = jnp.ones_like(x)
    z.fields["c"].arr = jnp.sin(2 * x)
    z.sync()

    dz = advection.advect_state(z, sw.State(mset))

    # dc = -scale * u dc/dx (up to discretization errors)
    expected = -ROSSBY_NUMBER * 2 * jnp.cos(2 * x)
    error = jnp.abs(dz.fields["c"].arr - expected)[INTERIOR].max()
    assert error < 1e-2


def test_background_advection_with_disabled_nonlinear():
    mset = make_mset()
    advection = mset.tendencies.advection

    background = sw.State(mset)
    background.u.arr = jnp.ones_like(background.u.arr)
    background.sync()
    advection.background = background
    advection.disable_nonlinear = True

    x, _y = mset.grid.x_mesh
    z = sw.State(mset)
    z.p.arr = jnp.sin(2 * x)
    z.u.arr = jnp.sin(2 * x)
    # the v field must vary along x to be advected by the background
    z.v.arr = jnp.sin(2 * x)
    z.sync()

    dz = advection.advect_state(z, sw.State(mset))

    # the pressure is advected by the background flow:
    # dp = -scale * u_b dp/dx
    expected = -ROSSBY_NUMBER * 2 * jnp.cos(2 * x)
    error = jnp.abs(dz.p.arr - expected)[INTERIOR].max()
    assert error < 1e-2

    # the momentum is advected by the background flow
    assert jnp.abs(dz.u.arr[INTERIOR]).max() > 0
    assert jnp.abs(dz.v.arr[INTERIOR]).max() > 0


def test_disabled_nonlinear_without_background():
    mset = make_mset()
    advection = mset.tendencies.advection
    advection.disable_nonlinear = True

    z = sw.State(mset)
    z.u.arr = jnp.ones_like(z.u.arr)
    z.sync()

    dz = sw.State(mset)
    result = advection.advect_state(z, dz)

    assert result is dz
    for field in result:
        assert jnp.abs(field.arr).max() == 0
