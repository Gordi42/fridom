"""Test for the cartesian grid class of the hydrostatic model."""
import pytest

import fridom.hydrostatic as hs


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(params=[
    pytest.param((3, 3, 3), id="3x3x3"),
    pytest.param((4, 4, 1), id="4x4x1"),
])
def shape(request):
    return request.param

@pytest.fixture(params=[
    pytest.param((3.0, 3.0, 3.3), id="3x3x3"),
])
def length(request):
    return request.param

@pytest.fixture(params=[
    pytest.param((True, True, True), id="Periodic"),
    pytest.param((False, False, False), id="Non-periodic"),
    pytest.param((True, False, True), id="Mixed"),
])
def periodic(request):
    return request.param

@pytest.fixture(params=[
    pytest.param((3, 3, 1), id="uneven"),
    pytest.param((4, 4, 2), id="even"),
])
def grid(request):
    grid = hs.grid.cartesian.Grid(shape = request.param,
                                  length = [1.5, 2.0, 3.0],
                                  periodic = [True, True, True])
    mset = hs.ModelSettings(grid)
    grid.setup(mset)
    return grid

@pytest.fixture(params=[
    "discrete",
    "continuous",
])
def spectral_method(request):
    return request.param

@pytest.fixture(params=[
    "0",
    "1",
    "-1",
    "d",
])
def mode(request):
    return request.param

# ================================================================
#  Tests
# ================================================================

def test_init(shape, length, periodic):
    grid = hs.grid.cartesian.Grid(shape, length, periodic)
    assert shape == grid.N
    assert length == grid.L
    assert periodic == grid.periodic_bounds

def test_setup(shape, length, periodic):
    grid = hs.grid.cartesian.Grid(shape, length, periodic)
    mset = hs.ModelSettings(grid)
    # check that the grid is not setup initially
    assert grid.X is None
    # setup the grid
    grid.setup(mset)
    # check if the grid is set up correctly
    assert grid.X is not None

# TODO(Silvano): Test one example with a model run
def test_model_run(): ...

# ----------------------------------------------------------------
#  Test the spectral analysis methods
# ----------------------------------------------------------------

def test_omega(grid, spectral_method):
    with pytest.raises(NotImplementedError):
        grid.omega(kvec=(1.0, 1.0, 1.0), method=spectral_method)

def test_vec_q(grid, spectral_method, mode):
    with pytest.raises(NotImplementedError):
        grid.vec_q(mode=mode, method=spectral_method)

def test_vec_p(grid, spectral_method, mode):
    with pytest.raises(NotImplementedError):
        grid.vec_p(mode=mode, method=spectral_method)
