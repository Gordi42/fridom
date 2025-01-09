"""Tests for the field_metadata module."""
from copy import copy

import pytest

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture(params=[1, 2, 3])
def n_dims(request):
    return request.param

@pytest.fixture
def mset(n_dims):
    grid = fr.grid.cartesian.Grid(N=(3,) * n_dims, L=(1,) * n_dims)
    return fr.ModelSettingsBase(grid=grid).setup()

# ================================================================
#  Tests
# ================================================================

def test_init():
    mdata = fr.FieldMetadata()
    assert mdata.name == "unnamed"
    assert mdata.long_name == "Unnamed"
    assert mdata.units == "n/a"
    assert mdata.nc_attrs is None
    assert mdata.is_spectral is False
    assert mdata._topo is None
    assert mdata.position is None
    assert mdata._bc_types is None
    assert mdata._flags == {"NO_ADV": False,
                               "ENABLE_MIXING": False,
                               "ENABLE_FRICTION": False}

def test_set_default(mset, n_dims):
    mdata = fr.FieldMetadata()
    mdata.set_default(mset)

    assert mdata.nc_attrs == {}
    assert mdata.position == mset.grid.cell_center
    assert mdata.topo == tuple([True] * n_dims)
    assert len(mdata.bc_types) == n_dims

@pytest.mark.parametrize("copy_method", [lambda x: x.copy(), copy])
def test_copy_methods(copy_method, mset):
    mdata = fr.FieldMetadata(name="test")
    mdata.set_default(mset)
    mdata_copy = copy_method(mdata)

    assert mdata_copy.name == "test"
    assert mdata_copy is not mdata

def test_serialize(mset):
    mdata = fr.FieldMetadata(name="test")
    mdata.set_default(mset)

    serializable = mdata.to_serializable()
    assert isinstance(serializable, dict)
    # all values should be default python types
    for value in serializable.values():
        assert isinstance(value, (str, int, tuple, list))

    mdata_new = fr.FieldMetadata.from_serializable(serializable)
    assert mdata_new.name == mdata.name
    assert mdata_new.long_name == mdata.long_name
    assert mdata_new.units == mdata.units
    assert mdata_new.nc_attrs == mdata.nc_attrs
    assert mdata_new.is_spectral == mdata.is_spectral
    assert mdata_new.topo == mdata.topo
    assert mdata_new.position == mdata.position
    assert mdata_new.bc_types == mdata.bc_types
    assert mdata_new.flags == mdata.flags
    assert mdata_new is not mdata

def test_bc_type_property(mset, n_dims):
    mdata = fr.FieldMetadata()
    mdata.set_default(mset)

    # with correct number of dimensions
    mdata.bc_types = [fr.grid.BCType.NEUMANN] * n_dims
    # with incorrect number of dimensions
    msg = "Number of BCType"
    with pytest.raises(ValueError, match=msg):
        mdata.bc_types = [fr.grid.BCType.NEUMANN] * (n_dims + 1)

@pytest.mark.parametrize(*(
    "flag_name, flag_value, error, error_msg",
    [("NO_ADV", 1, TypeError, "Flag NO_ADV must be a boolean"),
     ["NO_ADV", True, None, ""],
     ["INVALID_FLAG", False, KeyError, "Flag INVALID_FLAG not available"],
     ["ENABLE_MIXING", False, None, ""],
    ],
))
def test_flags_property(flag_name, flag_value, error, error_msg):
    mdata = fr.FieldMetadata()
    if not error_msg:
        mdata.flags = {flag_name: flag_value}
        assert mdata.flags[flag_name] is flag_value
        return

    with pytest.raises(error, match=error_msg):
        mdata.flags = {flag_name: flag_value}
