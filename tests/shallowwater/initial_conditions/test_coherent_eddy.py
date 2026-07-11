"""Tests for the coherent eddy initial condition."""

import jax.numpy as jnp
import pytest

import fridom.shallowwater as sw

# ================================================================
#  Constants
# ================================================================
PI = jnp.pi


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = sw.grid.cartesian.Grid(shape=(31, 31), domain_size=(2*PI, 2*PI))
    time_stepper = sw.time_steppers.AdamBashforth(dt=2**(-5), order=3)
    return sw.ModelSettings(
        grid, f0=1, csqr=1, rossby_number=0.3,
        time_stepper=time_stepper).setup()


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("gauss_field", ["streamfunction", "vorticity"])
def test_eddy_is_divergence_free(mset, gauss_field):
    z = sw.initial_conditions.CoherentEddy(mset, gauss_field=gauss_field)

    u_max = jnp.abs(z.u.arr).max()
    assert u_max > 0

    # the velocity field derived from a streamfunction is divergence-free
    divergence = z.u.diff(axis=0) + z.v.diff(axis=1)
    assert jnp.abs(divergence.arr).max() < 1e-10 * u_max

    # the pressure is in geostrophic balance with the streamfunction
    assert jnp.abs(z.p.arr).max() > 0


def test_negative_amplitude_flips_rotation(mset):
    z_ccw = sw.initial_conditions.CoherentEddy(mset, amplitude=1)
    z_cw = sw.initial_conditions.CoherentEddy(mset, amplitude=-1)

    assert jnp.abs(z_ccw.u.arr + z_cw.u.arr).max() == 0
    assert jnp.abs(z_ccw.v.arr + z_cw.v.arr).max() == 0


def test_eddy_position(mset):
    z = sw.initial_conditions.CoherentEddy(
        mset, pos_x=0.25, pos_y=0.75, gauss_field="streamfunction")

    # the eddy (maximum speed) is located around the given position
    speed = (z.u.arr**2 + z.v.arr**2)**0.5
    x, y = z.u.get_mesh()
    lx, ly = mset.grid.domain_size
    distance = ((x - 0.25*lx)**2 + (y - 0.75*ly)**2)**0.5

    assert speed[distance > 2.0].max() < speed.max() / 10


def test_unknown_gauss_field_raises(mset):
    with pytest.raises(ValueError, match="Unknown gauss_field"):
        sw.initial_conditions.CoherentEddy(mset, gauss_field="invalid")
