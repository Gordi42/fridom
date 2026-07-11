"""Tests for the Smagorinsky-Lilly closure of the nonhydrostatic model."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr
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
    mset = nh.ModelSettings(grid, f0=0.0, stratification_n2=0.0)
    return mset.setup()


@pytest.fixture
def closure(mset):
    return nh.modules.closures.SmagorinskyLilly().setup(mset=mset)


@pytest.fixture
def sheared_state(mset):
    x, y, z_mesh = mset.grid.x_mesh
    z = nh.State(mset)
    z.u.arr = jnp.sin(y) * jnp.sin(z_mesh)
    z.v.arr = jnp.sin(x)
    z.w.arr = jnp.sin(y)
    z.b.arr = jnp.sin(x) * jnp.sin(y)
    return z.sync()


def apply_closure(closure, z):
    mz = fr.ModelState(z.mset)
    mz.z = z
    return closure.update(mz)


# ================================================================
#  Tests
# ================================================================
def test_default_buoyancy_multiplier():
    closure = nh.modules.closures.SmagorinskyLilly(
        turbulent_prandtl_number=2.0)
    assert closure.buoyancy_multiplier == 0.5

    closure = nh.modules.closures.SmagorinskyLilly(
        turbulent_prandtl_number=2.0, buoyancy_multiplier=3.0)
    assert closure.buoyancy_multiplier == 3.0


def test_filter_width_is_cell_size(mset, closure):
    assert closure.filter_width == pytest.approx(
        float(mset.grid.cell_volume) ** (1 / 3))


def test_dissipates_energy(closure, sheared_state):
    mz = apply_closure(closure, sheared_state)

    for name in ("u", "v", "w", "b"):
        f = sheared_state.fields[name]
        df = mz.dz.fields[name]
        tendency = (f * df).integrate().value
        assert tendency < 0
        assert not jnp.isnan(df.arr).any()


def test_stratification_damps_viscosity(sheared_state):
    # strong stable stratification: gamma -> 0, only the background
    # viscosity remains
    grid = nh.grid.cartesian.Grid(shape=(16, 16, 16), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=0.0, stratification_n2=1e6).setup()
    closure = nh.modules.closures.SmagorinskyLilly().setup(mset=mset)

    z = nh.State(mset)
    for name in ("u", "v", "w", "b"):
        z.fields[name].arr = sheared_state.fields[name].arr
    z.sync()

    mz = apply_closure(closure, z)

    # the friction is at the level of the background viscosity, which
    # is orders of magnitude smaller than the unstratified smagorinsky
    # friction of the same state
    max_du = jnp.abs(mz.dz.u.arr).max()
    assert max_du < 10 * closure.background_viscosity


def test_background_diffusivity_limit(mset, closure):
    # for zero velocity the smagorinsky viscosity vanishes and the
    # buoyancy diffuses with the background diffusivity:
    # db = kappa_bg * laplacian(b) = -kappa_bg * b for b = sin(x)
    x, _y, _z = mset.grid.x_mesh
    z = nh.State(mset)
    z.b.arr = jnp.sin(x)
    z.sync()

    mz = apply_closure(closure, z)

    expected = -closure.background_diffusivity * z.b
    error = (mz.dz.b - expected).norm_l2() / expected.norm_l2()
    assert error < 5e-2


def test_mixing_respects_field_flag(closure, sheared_state):
    sheared_state.b.flags = {"ENABLE_MIXING": False}

    mz = apply_closure(closure, sheared_state)

    assert jnp.abs(mz.dz.b.arr).max() == 0
    assert jnp.abs(mz.dz.u.arr).max() > 0


def test_disabled_module_is_skipped(closure, sheared_state):
    closure.disable()

    mz = fr.ModelState(sheared_state.mset)
    mz.z = sheared_state
    mz = closure.update(mz=mz)

    for df in mz.dz:
        assert jnp.abs(df.arr).max() == 0
