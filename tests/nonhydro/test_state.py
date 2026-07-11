"""Tests for the state vector of the nonhydrostatic model."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework as fr
import fridom.nonhydro as nh

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
F0 = 1.0
N2 = 4.0
ROSSBY_NUMBER = 0.5

# interior slice away from the periodically wrapped halo cells
INTERIOR = (slice(3, -3),) * 3


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = nh.grid.cartesian.Grid(shape=(16, 16, 16), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=F0, stratification_n2=N2,
                            rossby_number=ROSSBY_NUMBER)
    mset.time_stepper.dt = np.timedelta64(1, "s")
    return mset.setup()


@pytest.fixture
def solid_body_rotation(mset):
    # u = -y, v = x is linear, so the finite differences are exact in
    # the interior: rel_vort_z = dv/dx - du/dy = 2
    _x, y, _z = mset.grid.x_mesh
    x, _y, _z = mset.grid.x_mesh
    z = nh.State(mset)
    z.u.arr = -y
    z.v.arr = x
    return z.sync()


# ================================================================
#  Tests
# ================================================================
def test_velocity_and_tracers(mset):
    z = nh.State(mset)
    assert [f.name for f in z.velocity] == ["u", "v", "w"]
    assert [f.name for f in z.tracers] == ["b"]


def test_field_setters(mset):
    z = nh.State(mset)
    z.u = z.u + 1.0
    z.v = z.v + 2.0
    z.w = z.w + 3.0
    z.b = z.b + 4.0
    assert z.fields["u"].arr.max() == 1.0
    assert z.fields["v"].arr.max() == 2.0
    assert z.fields["w"].arr.max() == 3.0
    assert z.fields["b"].arr.max() == 4.0


def test_epot_without_stratification():
    grid = nh.grid.cartesian.Grid(shape=(16, 16, 16), domain_size=(2*PI,)*3)
    mset = nh.ModelSettings(grid, f0=F0, stratification_n2=0.0).setup()

    z = nh.State(mset)
    z.b += 1.0

    # for zero background stratification: epot = b * z
    z_mesh = mset.grid.x_mesh[2]
    assert jnp.allclose(z.epot.arr, z_mesh)


def test_curl_of_gradient_vanishes(mset):
    # on the staggered grid, the discrete curl of a discrete gradient
    # is exactly zero
    x, y, z_mesh = mset.grid.x_mesh
    phi = fr.ScalarField(mset, name="phi")
    phi.arr = jnp.sin(x) * jnp.sin(y) * jnp.sin(z_mesh)
    phi.sync()

    z = nh.State(mset)
    z.u = z.u + phi.diff(axis=0)
    z.v = z.v + phi.diff(axis=1)
    z.w = z.w + phi.diff(axis=2)
    z.sync()

    for vort in (z.rel_vort_x, z.rel_vort_y, z.rel_vort_z):
        assert jnp.abs(vort.arr[INTERIOR]).max() < 1e-12


def test_solid_body_rotation_vorticity(solid_body_rotation):
    z = solid_body_rotation

    assert jnp.allclose(z.rel_vort_z.arr[INTERIOR], 2.0)
    assert jnp.abs(z.rel_vort_x.arr[INTERIOR]).max() < 1e-12
    assert jnp.abs(z.rel_vort_y.arr[INTERIOR]).max() < 1e-12

    # the rel_vort property collects all three components
    vort = z.rel_vort
    assert [f.name for f in vort] == ["vort_x", "vort_y", "vort_z"]


def test_local_rossby_number(solid_body_rotation):
    ro_local = solid_body_rotation.local_rossby_number
    expected = ROSSBY_NUMBER * 2.0 / F0
    assert jnp.allclose(ro_local.arr[INTERIOR], expected)


def test_pot_vort_at_rest(mset):
    # for a state at rest, the potential vorticity is f0 * N²
    z = nh.State(mset)
    assert jnp.allclose(z.pot_vort.arr, F0 * N2)


def test_pot_vort_spectral_raises(mset):
    z = nh.State(mset).fft()
    with pytest.raises(NotImplementedError, match="spectral"):
        _ = z.pot_vort


def test_linear_pot_vort(mset):
    # for a state with only buoyancy b = sin(z):
    # Q = Ro * f0 / N² * cos(z) (up to discretization errors)
    z_mesh = mset.grid.x_mesh[2]
    z = nh.State(mset)
    z.b.arr = jnp.sin(z_mesh)
    z.sync()

    pot_vort = z.linear_pot_vort
    expected = ROSSBY_NUMBER * F0 / N2 * jnp.cos(z_mesh)
    error = jnp.abs(pot_vort.arr - expected)[INTERIOR].max()
    assert error < 1e-2


def test_cfl(mset):
    z = nh.State(mset)
    z.u += 2.0
    z.v += 1.0
    z.w += 0.5

    dx = mset.grid.dx[0]
    cfl = z.cfl
    assert cfl.name == "cfl"
    assert jnp.allclose(cfl.arr, 2.0 * 1.0 / dx)


def test_diagnostic_state(mset):
    z_diag = fr.ModelState(mset).z_diag

    assert z_diag.p.name == "p"
    assert z_diag.div.name == "div"

    z_diag.p = z_diag.p + 1.0
    z_diag.div = z_diag.div + 2.0
    assert z_diag.fields["p"].arr.max() == 1.0
    assert z_diag.fields["div"].arr.max() == 2.0
