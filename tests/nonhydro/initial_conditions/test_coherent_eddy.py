"""Tests for the coherent eddy initial condition."""

import jax.numpy as jnp
import numpy as np
import pytest

import fridom.nonhydro as nh


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = nh.grid.cartesian.Grid(shape=(32, 32, 4), domain_size=(3, 3, 1))
    mset = nh.ModelSettings(grid, f0=1, beta=0.2)
    mset.time_stepper.dt = np.timedelta64(4, "ms")
    return mset.setup()


# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("gauss_field", ["streamfunction", "vorticity"])
def test_eddy_is_horizontally_divergence_free(mset, gauss_field):
    z = nh.initial_conditions.CoherentEddy(mset, gauss_field=gauss_field)

    u_max = jnp.abs(z.u.arr).max()
    assert u_max > 0

    # the eddy is purely horizontal
    assert jnp.abs(z.w.arr).max() == 0
    # the velocity field derived from a streamfunction is divergence-free
    divergence = z.u.diff(axis=0) + z.v.diff(axis=1)
    assert jnp.abs(divergence.arr).max() < 1e-10 * u_max


def test_negative_amplitude_flips_rotation(mset):
    z_ccw = nh.initial_conditions.CoherentEddy(mset, amplitude=1)
    z_cw = nh.initial_conditions.CoherentEddy(mset, amplitude=-1)

    assert jnp.abs(z_ccw.u.arr + z_cw.u.arr).max() == 0
    assert jnp.abs(z_ccw.v.arr + z_cw.v.arr).max() == 0


def test_eddy_position(mset):
    z = nh.initial_conditions.CoherentEddy(mset, pos_x=0.25, pos_y=0.75)

    # the eddy (maximum speed) is located around the given position
    speed = (z.u.arr**2 + z.v.arr**2)**0.5
    x, y, _z = z.u.get_mesh()
    lx, ly, _lz = mset.grid.domain_size
    distance = ((x - 0.25*lx)**2 + (y - 0.75*ly)**2)**0.5

    assert speed[distance > 1.0].max() < speed.max() / 10


def test_unknown_gauss_field_raises(mset):
    with pytest.raises(ValueError, match="Unknown gauss_field"):
        nh.initial_conditions.CoherentEddy(mset, gauss_field="invalid")
