"""Tests for the upwind flux function."""

import pytest

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(params=[1, 2, 3])
def n_dims(request):
    return request.param

@pytest.fixture
def shape(n_dims):
    return (3, 10, 4)[:n_dims]

@pytest.fixture
def grid(shape):
    return fr.grid.cartesian.Grid(N=shape, L=(1, 2, 3)[:len(shape)])

@pytest.fixture
def mset(grid):
    mset = fr.ModelSettingsBase(grid)
    mset.halo = 1
    mset.setup()
    return mset

# ================================================================
#  Tests
# ================================================================

@pytest.mark.parametrize("position",
                         [fr.grid.AxisPosition.CENTER, fr.grid.AxisPosition.FACE])
def test_upwind_random(mset, n_dims, position):
    # create scalar fields for velocity and flux
    velocity = fr.ScalarField(mset)
    flux = fr.ScalarField(mset)
    flux.name = "flux"

    # fill with random values
    velocity.set_random(seed=12345)
    flux.set_random(seed=12345)

    # set the position of the flux
    flux.position = fr.grid.Position(tuple([position] * n_dims))

    # create the upwind flux function and setup
    upwind = fr.modules.flux_functions.Upwind()
    upwind.setup(mset)

    # test the upwind flux function in every axis
    for axis in range(n_dims):
        # compute the upwind flux
        upwind_flux = upwind.compute(flux, velocity, axis)
        # check if the shape is correct
        assert upwind_flux.arr.shape == flux.arr.shape
        # check that the metadata is correct
        assert upwind_flux.mdata == flux.mdata

def test_1d_upwind_face():
    # create a 1D grid
    grid = fr.grid.cartesian.Grid(N=(4,), L=(1,))
    mset = fr.ModelSettingsBase(grid)
    mset.halo = 1
    mset.setup()

    # create scalar fields for velocity and flux
    velocity = fr.ScalarField(mset)
    flux = fr.ScalarField(mset)
    velocity.position = fr.grid.Position((fr.grid.AxisPosition.FACE,))
    flux.position = fr.grid.Position((fr.grid.AxisPosition.FACE,))

    velocity.arr = grid.pad(fr.config.ncp.array([-1.0, 0.0, 1.0, 1.0]))
    flux.arr = grid.pad(fr.config.ncp.array([1.0, 2.0, 3.0, 4.0]))

    # create the upwind flux function and setup
    upwind = fr.modules.flux_functions.Upwind()
    upwind.setup(mset)
    # compute the upwind flux
    upwind_flux = upwind.compute(flux, velocity, 0)
    # check the result
    expected = fr.config.ncp.array([1.0, 2.0, 2.0, 3.0])
    assert fr.config.ncp.allclose(grid.unpad(upwind_flux.arr), expected)

def test_1d_upwind_center():
    # create a 1D grid
    grid = fr.grid.cartesian.Grid(N=(4,), L=(1,))
    mset = fr.ModelSettingsBase(grid)
    mset.halo = 1
    mset.setup()

    # create scalar fields for velocity and flux
    velocity = fr.ScalarField(mset)
    flux = fr.ScalarField(mset)
    velocity.position = fr.grid.Position((fr.grid.AxisPosition.CENTER,))
    flux.position = fr.grid.Position((fr.grid.AxisPosition.CENTER,))

    velocity.arr = grid.pad(fr.config.ncp.array([-1.0, 0.0, 1.0, 1.0]))
    flux.arr = grid.pad(fr.config.ncp.array([1.0, 2.0, 3.0, 4.0]))

    # create the upwind flux function and setup
    upwind = fr.modules.flux_functions.Upwind()
    upwind.setup(mset)
    # compute the upwind flux
    upwind_flux = upwind.compute(flux, velocity, 0)
    # check the result
    expected = fr.config.ncp.array([2.0, 2.0, 3.0, 4.0])
    assert fr.config.ncp.allclose(grid.unpad(upwind_flux.arr), expected)
