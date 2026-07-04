"""Test the state vector of the hydrostatic model."""

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

@pytest.fixture
def mset(grid):
    return hs.ModelSettings(grid).setup()

# ================================================================
#  Tests
# ================================================================

def test_init(mset):
    state = hs.State(mset)
    assert state.mset == mset
    assert state.is_spectral == mset.grid.spectral_grid
    # check if the vector has 3 fields
    expected_fields = ["u", "v", "b"]
    assert state.vector_dim == len(expected_fields)

def test_with_custom_fields(mset):
    mset.custom_state_fields.append(
        hs.FieldMetadata(name="custom_field"),
    )
    state = hs.State(mset)
    # check if the vector has 4 fields
    expected_fields = ["u", "v", "b", "custom_field"]
    assert state.vector_dim == len(expected_fields)
    custom_field = state["custom_field"]
    assert isinstance(custom_field, hs.ScalarField)
    assert custom_field.name == "custom_field"

# ----------------------------------------------------------------
#  State Variables
# ----------------------------------------------------------------

@pytest.mark.parametrize("var_name", ["u", "v", "b"])
def test_state_variables(mset, var_name):
    state = hs.State(mset)
    # test getting
    field = getattr(state, var_name)
    assert field.name == var_name
    assert isinstance(field, hs.ScalarField)
    # test setting
    new_field = hs.ScalarField(mset, name=var_name)
    setattr(state, var_name, new_field)
    # check if the field was set
    assert getattr(state, var_name) is new_field

def test_velocity_property(mset):
    state = hs.State(mset)
    velocity = state.velocity
    assert isinstance(velocity, hs.VectorField)
    velocity_vector_dim = 2
    assert velocity.vector_dim == velocity_vector_dim
    # test if the fields are (u, v)
    u_vel, v_vel = velocity
    assert u_vel is state.u
    assert v_vel is state.v

@pytest.mark.parametrize("custom_fields", [
    pytest.param([], id="No custom fields"),
    pytest.param([hs.FieldMetadata(name="custom_field")],
                 id="With one custom fields"),
    pytest.param([hs.FieldMetadata(name="custom_field1"),
                  hs.FieldMetadata(name="custom_field2")],
                 id="With two custom fields"),
])
def test_tracers_property(mset, custom_fields):
    for field in custom_fields:
        mset.custom_state_fields.append(field)
    state = hs.State(mset)
    tracers = state.tracers
    assert isinstance(tracers, hs.VectorField)
    expected_dim = 1 + len(custom_fields)
    assert tracers.vector_dim == expected_dim
    b, *other_fields = tracers
    assert b is state.b
    for field in other_fields:
        assert field == state[field.name]
