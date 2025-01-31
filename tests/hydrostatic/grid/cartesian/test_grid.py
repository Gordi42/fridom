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

# ================================================================
#  Tests
# ================================================================

def test_init(shape, length, periodic):
    grid = hs.grid.cartesian.Grid(shape, length, periodic)
    assert shape == grid.N
    assert length == grid.L
    assert periodic == grid.periodic_bounds

# TODO(Silvano): Test once we have the model settings
def test_setup(): ...

# TODO(Silvano): Test one example with a model run
def test_model_run(): ...

# ----------------------------------------------------------------
#  Test the spectral analysis methods
# ----------------------------------------------------------------

# TODO(Silvano): Test once we have the model settings
def test_omega(): ...

# TODO(Silvano): Test once we have the model settings
def test_vec_q(): ...

# TODO(Silvano): Test once we have the model settings
def test_vec_p(): ...
