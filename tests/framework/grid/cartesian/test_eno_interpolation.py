"""Tests for the ENO interpolation module."""

import pytest

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(params=["pointwise", "cell_average"])
def method(request):
    return request.param

# ================================================================
#  Tests
# ================================================================
@pytest.mark.parametrize("periods",[
    (True, True),
    (True, False),
    (False, True),
    (False, False),
])
def test_nonperiodic_grid(periods):
    grid = fr.grid.cartesian.Grid(N=(4, 4), L=(1.0, 1.0), periodic_bounds=periods)
    mset = fr.ModelSettingsBase(grid).setup()
    eno = fr.grid.cartesian.InterENO(order=1)
    if not all(periods):
        msg = "ENO interpolation only works on periodic grids"
        with pytest.raises(ValueError, match=msg):
            eno.setup(mset)
    else:
        eno.setup(mset)

@pytest.mark.parametrize("order", [1, 2, 3])
def test_cell_average_reconstruction_coefficients(order):
    """
    Check polynomial reconstruction coefficients for cell average.

    Test that the polynomial reconstruction coefficients are consistent with
    the coefficients in the table of Shu (1998).
    """
    expected_coeffs = {
        1: fr.config.ncp.array([
            [3/2, -1/2],
            [1/2, 1/2],
            [-1/2, 3/2],
        ]),
        2: fr.config.ncp.array([
            [11/6, -7/6, 1/3],
            [1/3, 5/6, -1/6],
            [-1/6, 5/6, 1/3],
            [1/3, -7/6, 11/6],
        ]),
        3: fr.config.ncp.array([
            [25/12, -23/12, 13/12, -1/4],
            [1/4, 13/12, -5/12, 1/12],
            [-1/12, 7/12, 7/12, -1/12],
            [1/12, -5/12, 13/12, 1/4],
            [-1/4, 13/12, -23/12, 25/12],
        ]),
    }[order]

    grid = fr.grid.GridBase(1)
    mset = fr.ModelSettingsBase(grid).setup()
    eno = fr.grid.cartesian.InterENO(order=order)
    eno.setup(mset)
    coeffs = eno._coeffs
    # check that the coefficients have the expected shape
    assert coeffs.shape == expected_coeffs.shape
    # check that the coefficients are equal to the expected coefficients
    assert fr.config.ncp.allclose(coeffs, expected_coeffs)

@pytest.mark.parametrize("order", [2, 3, 4, 5])
@pytest.mark.parametrize(("position", "grid_points"), [
    (fr.grid.AxisPosition.CENTER, 11),
    (fr.grid.AxisPosition.FACE, 10),
])
def test_1d_eno(order, method, position, grid_points):
    grid = fr.grid.cartesian.Grid(N=(grid_points,), L=(2.3,))
    mset = fr.ModelSettingsBase(grid, halo=order).setup()
    eno = fr.grid.cartesian.InterENO(order=order, method=method)
    eno.setup(mset)

    # check that the halo is set correctly
    assert grid.halo == order

    # create the field
    f = fr.ScalarField(mset)
    f.position = fr.grid.Position((position,))
    x, = f.get_mesh()
    lx, = grid.L
    f.arr = (x < lx/2) * x**2 + (x >= lx/2) * (lx - x)**2

    f_eno = eno.interpolate(f, f.position.shift(axis=0))
    # check that the new position is correct
    assert f_eno.position == f.position.shift(axis=0)
    # create the expected field
    x, = f_eno.get_mesh()
    expected = (x < lx/2) * x**2 + (x >= lx/2) * (lx - x)**2
    # check that the values are correct
    # for the cell averaged method, we expect the values to have a small offset
    # that is constant in space
    if method == "cell_average":
        offset = f_eno.arr[0] - expected[0]
        expected += offset

    assert fr.config.ncp.allclose(f_eno.arr, expected)

@pytest.mark.parametrize("order", [2, 3, 4, 5])
def test_2d_random(method, order):
    """Test the ENO interpolation in 2D with random data."""
    grid = fr.grid.cartesian.Grid(N=(12, 11), L=(2.0, 1.0))
    mset = fr.ModelSettingsBase(grid, halo=order).setup()
    eno = fr.grid.cartesian.InterENO(order=order, method=method)
    eno.setup(mset)

    all_positions = [
        grid.cell_center,
        grid.cell_center.shift(axis=0),
        grid.cell_center.shift(axis=1),
        grid.cell_center.shift(axis=0).shift(axis=1),
    ]

    for start_pos in all_positions:
        for end_pos in all_positions:
            # create the field
            f = fr.ScalarField(mset)
            f.position = start_pos
            f.set_random()

            # interpolate to the face position
            f_eno = eno.interpolate(f, end_pos)
            # check that the new position is correct
            assert f_eno.position == end_pos

            # check that the new field has the correct shape
            assert f_eno.arr.shape == f.arr.shape
