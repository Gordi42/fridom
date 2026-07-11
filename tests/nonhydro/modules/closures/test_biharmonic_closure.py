"""Tests for the biharmonic closure of the nonhydrostatic model."""

import jax.numpy as jnp
import pytest

import fridom.framework as fr
import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
WAVENUMBER = 2.0


# ================================================================
#  Fixtures
# ================================================================
def make_closure(mode):
    # the closure must be added to the tendencies before the setup so
    # that the grid is created with the required halo of two
    grid = nh.grid.cartesian.Grid(shape=(16, 16, 16), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=0.0, stratification_n2=0.0)
    closure = nh.modules.closures.BiharmonicClosure(kh=1.0, kv=1.0, mode=mode)
    mset.tendencies.add_module(closure)
    mset.setup()
    return mset, closure


@pytest.fixture(params=["free-slip", "no-slip"])
def mset_and_closure(request):
    return make_closure(request.param)


def apply_closure(closure, z):
    mz = fr.ModelState(z.mset)
    mz.z = z
    return closure.update(mz)


def single_mode_state(mset, axis):
    z = nh.State(mset)
    z.u.arr = jnp.sin(WAVENUMBER * mset.grid.x_mesh[axis])
    return z.sync()


def discrete_damping_rate(mset, axis):
    # eigenvalue of the discrete second derivative for sin(k x)
    dx = mset.grid.dx[axis]
    k_eff2 = (2 - 2 * jnp.cos(WAVENUMBER * dx)) / dx**2
    return k_eff2**2


# ================================================================
#  Tests
# ================================================================
def test_invalid_mode_raises():
    with pytest.raises(ValueError, match="Invalid mode 'bogus'"):
        nh.modules.closures.BiharmonicClosure(kh=1.0, kv=1.0, mode="bogus")


@pytest.mark.parametrize(("axis", "direction"), [
    pytest.param(0, "horizontal", id="horizontal"),
    pytest.param(2, "vertical", id="vertical"),
])
def test_single_mode_damping(mset_and_closure, axis, direction):
    # a single fourier mode is a discrete eigenfunction of the closure:
    # du = - coeff * k_eff^4 u
    mset, closure = mset_and_closure
    z = single_mode_state(mset, axis)
    mz = apply_closure(closure, z)

    if direction == "horizontal":
        coeff = closure._hor_diff_coeff
    else:
        coeff = closure._ver_diff_coeff
    expected = -coeff * discrete_damping_rate(mset, axis) * z.u

    error = (mz.dz.u - expected).norm_l2() / expected.norm_l2()
    assert error < 1e-10

    # the closure dissipates energy
    assert (z.u * mz.dz.u).integrate().value < 0


def test_modes_agree_without_land():
    # on a fully periodic domain without land, the free-slip and no-slip
    # modes are identical
    tendencies = []
    for mode in ("free-slip", "no-slip"):
        mset, closure = make_closure(mode)
        z = single_mode_state(mset, axis=0)
        tendencies.append(apply_closure(closure, z).dz.u.arr)

    assert jnp.abs(tendencies[0] - tendencies[1]).max() == 0


def test_coefficient_setters_rescale_tendency(mset_and_closure):
    mset, closure = mset_and_closure
    z_hor = single_mode_state(mset, axis=0)
    z_ver = single_mode_state(mset, axis=2)

    du_hor = apply_closure(closure, z_hor).dz.u
    du_ver = apply_closure(closure, z_ver).dz.u

    closure.kh = 2.0
    closure.kv = 3.0
    assert closure.kh == 2.0
    assert closure.kv == 3.0

    du_hor2 = apply_closure(closure, z_hor).dz.u
    du_ver2 = apply_closure(closure, z_ver).dz.u

    assert (du_hor2 - 2.0 * du_hor).norm_l2() / du_hor.norm_l2() < 1e-10
    assert (du_ver2 - 3.0 * du_ver).norm_l2() / du_ver.norm_l2() < 1e-10
