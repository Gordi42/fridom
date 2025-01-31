"""Test the Modelsettings of the hydrostatic model."""
import pytest

import fridom.hydrostatic as hs


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(params=[
    pytest.param(
        hs.grid.cartesian.Grid(
            shape=(4, 3, 2),
            length=(1.3, 1.0, 0.3),
        ), id="Cartesian"),
])
def grid(request):
    return request.param


# ================================================================
#  Tests
# ================================================================
def test_init(grid):
    mset = hs.ModelSettings(grid)
    assert mset.grid == grid
    assert mset.model_name == "3D - Hydrostatic model"
    assert mset.coriolis_parameter == 0.0
    assert mset.background_stratification == 0.0
    assert mset.rossby_number == 1.0


def test_setup(grid):
    mset = hs.ModelSettings(grid)
    mset.setup()
    assert mset.is_setup
    assert isinstance(mset.coriolis_parameter, hs.ScalarField)
    assert isinstance(mset.background_stratification, hs.ScalarField)
    # lets set the coriolis parameter to None and check if it will be setup
    # if we call the setup method again
    mset._coriolis_parameter = -1
    mset.setup()
    assert mset.coriolis_parameter == -1
    mset.setup(setup_mode="forced")
    assert isinstance(mset.coriolis_parameter, hs.ScalarField)

# TODO(Silvano): Test the remaining methods
# def test_state_constructor(grid):
#     mset = hs.ModelSettings(grid).setup()
#     state = mset.state_constructor()
#     assert isinstance(state, hs.State)
#     assert state.is_spectral == grid.spectral_grid

# def test_diagnostic_constructor(grid):
#     mset = hs.ModelSettings(grid).setup()
#     state = mset.diagnostic_state_constructor()
#     assert isinstance(state, hs.DiagnosticState)
#     assert state.is_spectral == grid.spectral_grid

def test_format_coriolis_parameter(): ...

def test_format_background_stratification(): ...

def test_repr(): ...

def test_coriolis_parameter(): ...

def test_background_stratification(): ...

def test_rossby_number(): ...
