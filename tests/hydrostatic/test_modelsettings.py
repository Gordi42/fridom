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

# TODO(Silvano): Test the remaining methods

def test_setup(): ...

def test_state_constructor(): ...

def test_diagnostic_constructor(): ...

def test_format_coriolis_parameter(): ...

def test_format_background_stratification(): ...

def test_repr(): ...

def test_coriolis_parameter(): ...

def test_background_stratification(): ...

def test_rossby_number(): ...
