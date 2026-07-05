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

def test_state_constructor(grid):
    mset = hs.ModelSettings(grid).setup()
    state = mset.state_constructor()
    assert isinstance(state, hs.State)
    assert not state.is_spectral

def test_diagnostic_constructor(grid):
    mset = hs.ModelSettings(grid).setup()
    state = mset.diagnostic_state_constructor()
    assert isinstance(state, hs.DiagnosticState)

# ----------------------------------------------------------------
#  Test formatting methods
# ----------------------------------------------------------------

def test_format_coriolis_parameter(grid):
    mset = hs.ModelSettings(grid)
    assert mset._format_coriolis_parameter() == "0 1/s"
    mset.setup()
    assert mset._format_coriolis_parameter() == "Variable"
    mset.coriolis_parameter = hs.ScalarField(
        mset, name="f", topo=(False, False, False))
    mset.coriolis_parameter += 1
    assert mset._format_coriolis_parameter() == "1.0 1/s"

def test_format_background_stratification(grid):
    mset = hs.ModelSettings(grid)
    assert mset._format_background_stratification() == "0 1/s^2"
    mset.setup()
    assert mset._format_background_stratification() == "Variable"
    mset.background_stratification = hs.ScalarField(mset,
                                                    name="N2",
                                                    topo=(False, False, False))
    mset.background_stratification += 1
    assert mset._format_background_stratification() == "1.0 1/s^2"

@pytest.mark.parametrize("entry", [
    "Coriolis parameter",
    "Background stratification",
    "Rossby number",
])
def test_repr(grid, entry):
    mset = hs.ModelSettings(grid)
    assert entry in repr(mset)

# ----------------------------------------------------------------
#  Test the properties
# ----------------------------------------------------------------
@pytest.mark.parametrize("name", ["coriolis_parameter",
                                  "background_stratification"])
def test_scalar_field_properties(grid, name):
    mset = hs.ModelSettings(grid)
    # first the field should be a float or int
    field = getattr(mset, name)
    assert isinstance(field, (int, float))
    # when setting it to a float value, it should be exactly that
    my_value = 3.14
    setattr(mset, name, my_value)
    field = getattr(mset, name)
    assert field == my_value
    # after setup it should be a ScalarField
    mset.setup()
    field = getattr(mset, name)
    assert isinstance(field, hs.ScalarField)
    # when setting it to a float value, it should be a ScalarField with
    # that value
    setattr(mset, name, my_value)
    field = getattr(mset, name)
    assert isinstance(field, hs.ScalarField)
    assert (field.arr == my_value).all()
    # when setting it to a ScalarField, it should be that
    my_field = hs.ScalarField(mset, name="test", topo=(False, False, False))
    setattr(mset, name, my_field)
    field = getattr(mset, name)
    assert field is my_field

# TODO(Silvano): Test once the advection term is implemented
def test_rossby_number(): ...
