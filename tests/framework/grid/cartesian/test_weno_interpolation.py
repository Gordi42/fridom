"""Tests for the WENO interpolation module."""

import pytest

import fridom.framework as fr

# ================================================================
#  Fixtures
# ================================================================
CENTER = fr.grid.AxisPosition.CENTER
FACE = fr.grid.AxisPosition.FACE

# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("periods",[
    (True, False),
    (False, True),
    (False, False),
])
def test_nonperiodic_grid(periods):
    grid = fr.grid.cartesian.Grid(N=(4, 4), L=(1.0, 1.0), periodic_bounds=periods)
    mset = fr.ModelSettingsBase(grid).setup()
    weno = fr.grid.cartesian.InterWENO(order=3)
    msg = "WENO only works on periodic grids"
    with pytest.raises(ValueError, match=msg):
        weno.setup(mset)

@pytest.mark.parametrize("order", [2, 4])
def test_even_order(order):
    msg = "Please use an odd order for WENO."
    with pytest.raises(ValueError, match=msg):
        fr.grid.cartesian.InterWENO(order=order)

@pytest.mark.parametrize("order", [7, 9])
def test_order_too_high(order):
    msg = f"Order {order} is too high."
    with pytest.raises(ValueError, match=msg):
        fr.grid.cartesian.InterWENO(order=order)

# ----------------------------------------------------------------
#  Tests for the reconstruction axis
# ----------------------------------------------------------------

@pytest.mark.parametrize(*("pos, dest, expected_axis", [
    ((CENTER, CENTER), (FACE, CENTER), 0),
    ((CENTER, CENTER), (CENTER, FACE), 1),
    ((FACE, CENTER), (CENTER, CENTER), 0),
    ((FACE, CENTER), (FACE, FACE), 1),
    ((CENTER, FACE), (FACE, FACE), 0),
    ((CENTER, FACE), (CENTER, CENTER), 1),
    ((FACE, FACE), (CENTER, FACE), 0),
    ((FACE, FACE), (FACE, CENTER), 1),
]))
def test_reconstuction_axis(pos, dest, expected_axis):
    grid = fr.grid.cartesian.Grid(N=(4, 4), L=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid).setup()
    weno = fr.grid.cartesian.InterWENO()
    weno.setup(mset)

    f = fr.ScalarField(mset)
    f.position = fr.grid.Position(pos)

    axis = weno._get_reconstruction_axis(f, fr.grid.Position(dest))

    assert axis == expected_axis

@pytest.mark.parametrize(*("pos, dest, msg", [
    ((CENTER, CENTER), (CENTER, CENTER), "same as the field position"),
    ((FACE, CENTER), (FACE, CENTER), "same as the field position"),
    ((CENTER, FACE), (CENTER, FACE), "same as the field position"),
    ((FACE, FACE), (FACE, FACE), "same as the field position"),
    ((CENTER, CENTER), (FACE, FACE), "one dimension at a time"),
    ((FACE, CENTER), (CENTER, FACE), "one dimension at a time"),
    ((CENTER, FACE), (FACE, CENTER), "one dimension at a time"),
    ((FACE, FACE), (CENTER, CENTER), "one dimension at a time"),
]))
def test_reconstruction_wrong_position(pos, dest, msg):
    grid = fr.grid.cartesian.Grid(N=(4, 4), L=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid).setup()
    weno = fr.grid.cartesian.InterWENO()
    weno.setup(mset)

    f = fr.ScalarField(mset)
    f.position = fr.grid.Position(pos)

    with pytest.raises(ValueError, match=msg):
        weno.reconstruct(f, fr.grid.Position(dest))

def test_stencil_slices():
    grid = fr.grid.cartesian.Grid(N=(4, 4), L=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid).setup()
    weno = fr.grid.cartesian.InterWENO(order=5)
    weno.setup(mset)

    expected_slices = [
        [(slice(None, -2), slice(None)),
         (slice(1, -1), slice(None)),
         (slice(2, None), slice(None))],
        [(slice(None), slice(None, -2)),
         (slice(None), slice(1, -1)),
         (slice(None), slice(2, None))],
    ]

    assert weno.stencil_slices == expected_slices