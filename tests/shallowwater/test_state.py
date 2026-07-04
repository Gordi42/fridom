"""Tests for the state vector of the shallow water model."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.shallowwater as sw

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi
F0 = 1.0
CSQR = 2.0
ROSSBY_NUMBER = 0.5

# interior slice away from the periodically wrapped halo cells
INTERIOR = (slice(3, -3),) * 2


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = sw.grid.cartesian.Grid(shape=(31, 31), domain_size=(2*PI, 2*PI))
    mset = sw.ModelSettings(grid, f0=F0, csqr=CSQR,
                            rossby_number=ROSSBY_NUMBER)
    mset.time_stepper.dt = np.timedelta64(1, "s")
    return mset.setup()


@pytest.fixture
def solid_body_rotation(mset):
    # u = -y, v = x is linear, so the finite differences are exact in
    # the interior: rel_vort = dv/dx - du/dy = 2
    x, y = mset.grid.x_mesh
    z = sw.State(mset)
    z.u.arr = -y
    z.v.arr = x
    return z.sync()


# ================================================================
#  Tests
# ================================================================
def test_velocity_and_tracers(mset):
    z = sw.State(mset)
    assert [f.name for f in z.velocity] == ["u", "v"]
    assert [f.name for f in z.tracers] == []


def test_spectral_ekin(mset):
    x, _y = mset.grid.x_mesh
    z = sw.State(mset)
    z.u.arr = jnp.sin(2 * x)
    z.sync()

    ekin = z.fft().spectral_ekin

    # the spectral kinetic energy density is real and non-negative
    # (up to rounding errors of the unnormalized fourier amplitudes)
    scale = jnp.abs(ekin.arr).max()
    assert jnp.abs(ekin.arr.imag).max() < 1e-12 * scale
    assert ekin.arr.real.min() > -1e-12 * scale

    # a single fourier mode has exactly two non-zero entries
    assert jnp.sum(jnp.abs(ekin.arr) > 1e-6) == 2

    # the physical state raises an error
    with pytest.raises(sw.exceptions.FieldSpaceError):
        _ = z.spectral_ekin


def test_pot_vort_at_rest(mset):
    # for a state at rest, the potential vorticity is f0 / c²
    z = sw.State(mset)
    assert jnp.allclose(z.pot_vort.arr, F0 / CSQR)


def test_pot_vort_solid_body_rotation(solid_body_rotation):
    pot_vort = solid_body_rotation.pot_vort
    expected = (2.0 + F0) / CSQR
    assert jnp.allclose(pot_vort.arr[INTERIOR], expected)


def test_pot_vort_spectral_raises(mset):
    z = sw.State(mset).fft()
    with pytest.raises(NotImplementedError, match="spectral"):
        _ = z.pot_vort


def test_local_rossby_number(solid_body_rotation):
    ro_local = solid_body_rotation.local_rossby_number
    expected = ROSSBY_NUMBER * 2.0 / F0
    assert jnp.allclose(ro_local.arr[INTERIOR], expected)


def test_cfl(mset):
    z = sw.State(mset)
    z.u += 2.0
    z.v += 1.0

    dx = mset.grid.dx[0]
    cfl = z.cfl
    assert cfl.name == "cfl"
    assert jnp.allclose(cfl.arr, 2.0 * 1.0 / dx)
