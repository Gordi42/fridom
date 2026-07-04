"""Tests for the grid base class."""

import pytest

import fridom.framework as fr
import fridom.shallowwater as sw


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def grid():
    grid = sw.grid.cartesian.Grid(shape=(15, 15), domain_size=(1.0, 1.0))
    mset = sw.ModelSettings(grid, f0=1.0, csqr=1.0)
    mset.setup()
    return grid


# ================================================================
#  Tests
# ================================================================
def test_base_get_mesh(grid):
    # the base class implementation only supports the cell center
    # (concrete grids override it)
    physical = fr.grid.GridBase.get_mesh(grid)
    assert len(physical) == 2

    spectral = fr.grid.GridBase.get_mesh(grid, spectral=True)
    assert len(spectral) == 2

    with pytest.raises(NotImplementedError):
        fr.grid.GridBase.get_mesh(
            grid, position=grid.cell_center.shift(0))


def test_omega_properties_are_cached(grid):
    analytical = grid.omega_analytical
    assert analytical.shape == (15, 15)
    assert grid.omega_analytical is analytical

    space_discrete = grid.omega_space_discrete
    assert space_discrete.shape == (15, 15)
    assert grid.omega_space_discrete is space_discrete

    time_discrete = grid.omega_time_discrete
    assert time_discrete.shape == (15, 15)
    assert grid.omega_time_discrete is time_discrete


def test_module_setters(grid):
    diff_module = fr.grid.cartesian.FiniteDifferences()
    grid.diff_module = diff_module
    assert grid.diff_module is diff_module

    with pytest.raises(TypeError, match="differential operator"):
        grid.diff_module = 42

    interp_module = fr.grid.cartesian.LinearInterpolation()
    grid.interp_module = interp_module
    assert grid.interp_module is interp_module

    with pytest.raises(TypeError, match="interpolation operator"):
        grid.interp_module = 42


def test_water_mask_setter(grid):
    mask = fr.grid.WaterMask()
    grid.water_mask = mask
    assert grid.water_mask is mask


def test_simple_properties(grid):
    assert grid.shape == (15, 15)
    assert grid.domain_size == (1.0, 1.0)
    assert grid.total_grid_points == 225
    assert len(grid.k_mesh) == 2
    assert len(grid.k_global) == 2
    assert isinstance(grid.mpi_available, bool)
