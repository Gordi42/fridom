"""Tests for the scalar field class."""
from copy import deepcopy

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

# default grid is 2D with shape (3, 10)
@pytest.fixture
def grid():
    return fr.grid.cartesian.Grid(N=(3, 10), L=(1, 2))

# for some tests we test different grid shapes
@pytest.fixture
def grid_all(shape):
    return fr.grid.cartesian.Grid(N=shape, L=(1, 2, 3)[:len(shape)])

# default model settings
@pytest.fixture
def mset(grid):
    mset = fr.ModelSettingsBase(grid)
    mset.halo = 1
    mset.setup()
    return mset

# model settings for different grid shapes
@pytest.fixture
def mset_all(grid_all):
    mset = fr.ModelSettingsBase(grid_all)
    mset.halo = 1
    mset.setup()
    return mset

@pytest.fixture(params=[True, False])
def is_spectral(request):
    return request.param

@pytest.fixture(params=[(True, True), (True, False), (False, True)])
def topo(request):
    return request.param

# ================================================================
#  Test helpers
# ================================================================

def not_implemented_for_non_full_domain_fields(operation) -> None:
    msg = "Operation not available for non full domain fields"
    with pytest.raises(NotImplementedError, match=msg):
        operation()

# ================================================================
#  Tests
# ================================================================

def test_init(mset_all, is_spectral, n_dims):
    field = fr.ScalarField(mset_all, is_spectral=is_spectral)
    # check if the field is a scalar field
    assert isinstance(field, fr.ScalarField)
    # check if the field has the correct grid
    assert field.is_spectral == is_spectral
    # check if the underlying data is a numpy array with the correct dimensions
    assert isinstance(field.arr, fr.config.ncp.ndarray)
    assert len(field.arr.shape) == n_dims
    # test if arr has the correct dtype
    c = fr.config
    expected_dtype = c.dtype_comp if is_spectral else c.dtype_real
    assert field.arr.dtype == expected_dtype

@pytest.mark.parametrize(*(
    "kwargs",
    [
        {"name": "test"},
        {"is_spectral": True},
        {"is_spectral": False},
        {"long_name": "test"},
        {"nc_attrs": {"var1": 1, "var2": 2}},
        {"topo": (True, False)},
        {"position": fr.grid.Position(
            (fr.grid.AxisPosition.CENTER, fr.grid.AxisPosition.FACE))},
        {"bc_types": (
            fr.grid.BCType.DIRICHLET, fr.grid.BCType.NEUMANN)},
        {"units": "test"},
        {"name": "test", "is_spectral": False},
    ],
))
def test_kwargs(mset, kwargs):
    field = fr.ScalarField(mset, **kwargs)
    # test if the kwargs are set correctly
    for key, value in kwargs.items():
        assert getattr(field, key) == value

def test_topo_shape(mset, topo):
    field = fr.ScalarField(mset, topo=topo)
    # check if the field has the correct topo
    assert field.mdata.topo == topo
    # every dimension with topo=False should have a size of 1
    for i, t in enumerate(topo):
        if not t:
            assert field.arr.shape[i] == 1

def test_wrong_topo(mset):
    msg = "Topology must have extend in at least one direction"
    with pytest.raises(ValueError, match=msg):
        fr.ScalarField(mset, topo=(False, False))

# ----------------------------------------------------------------
#  Test properties
# ----------------------------------------------------------------

# I don't know how to set the fixture values for the expected values
# so I skip these attributes for now by setting the expected values to None
@pytest.mark.parametrize(*(
    "attr, expected_type, expected_value",
    [
        ("mset", fr.ModelSettingsBase, None),  # mset is set in the fixture
        ("grid", fr.grid.cartesian.Grid, None),  # grid is set in the fixture
        ("is_spectral", bool, False),
        ("arr", fr.config.ncp.ndarray, None),
        ("mdata", fr.FieldMetadata, None),  # too lazy to set the expected value
        ("name", str, "unnamed"),
        ("long_name", str, "Unnamed"),
        ("units", str, "n/a"),
        ("nc_attrs", dict, None),
        ("topo", tuple, (True, True)),
        ("position", fr.grid.Position, fr.grid.Position(
            (fr.grid.AxisPosition.CENTER, fr.grid.AxisPosition.CENTER))),
        ("bc_types", tuple, (fr.grid.BCType.NEUMANN, fr.grid.BCType.NEUMANN)),
        ("flags", dict, {"NO_ADV": False,
                         "ENABLE_MIXING": False,
                         "ENABLE_FRICTION": False}),
    ],
))
def test_get_attr(mset, attr, expected_type, expected_value):
    field = fr.ScalarField(mset)
    # check if the field has the attribute
    assert hasattr(field, attr)
    # check if the attribute is of the correct type
    attr_value = getattr(field, attr)
    assert isinstance(attr_value, expected_type)
    # check if the attribute has the correct value
    if expected_value is not None:
        assert attr_value == expected_value

@pytest.mark.parametrize(*(
    "attr, value",
    [
        ("mset", "readonly"),
        ("grid", "readonly"),
        ("is_spectral", "readonly"),
        ("arr", fr.config.ncp.array([1, 2, 3])),
        ("mdata", fr.FieldMetadata()),
        ("name", "new_name"),
        ("long_name", "New Name"),
        ("units", "new units"),
        ("nc_attrs", {"var1": 1, "var2": 2}),
        ("topo", "readonly"),
        ("position", fr.grid.Position(
            (fr.grid.AxisPosition.FACE, fr.grid.AxisPosition.CENTER))),
        ("bc_types", (fr.grid.BCType.DIRICHLET, fr.grid.BCType.NEUMANN)),
    ],
))
def test_set_attr(mset, attr, value):
    field = fr.ScalarField(mset)
    # check that readonly attributes cannot be set
    if isinstance(value, str) and value == "readonly":
        with pytest.raises(AttributeError):
            setattr(field, attr, value)
        return
    # set the attribute
    setattr(field, attr, value)
    if isinstance(value, fr.config.ncp.ndarray):
        assert (field.arr == value).all()
    else:
        assert getattr(field, attr) == value

# ----------------------------------------------------------------
#  Test general methods
# ----------------------------------------------------------------

def test_fft_ifft(mset_all):
    # create the field
    field = fr.ScalarField(mset_all)
    # set the field to a random value
    field.arr = field.grid.create_random_array(seed=12345)
    # it should be impossible to perform an ifft on a physical field
    msg = "Field is not in spectral space, cannot perform ifft"
    with pytest.raises(ValueError, match=msg):
        field.ifft()
    field_fft = field.fft()
    # check if the field is in spectral space
    assert field_fft.is_spectral
    # it should be impossible to perform an fft on a spectral field
    msg = "Field is in spectral space, cannot perform fft"
    with pytest.raises(ValueError, match=msg):
        field_fft.fft()
    # compute the inverse fft
    field_ifft = field_fft.ifft()
    assert not field_ifft.is_spectral
    assert fr.config.ncp.allclose(field.arr, field_ifft.arr)

def test_fft_ifft_topo(mset, topo, is_spectral):
    field = fr.ScalarField(mset, topo=topo, is_spectral=is_spectral)
    if all(topo):
        # fft should work on full domain fields
        field.ifft() if is_spectral else field.fft()
        return
    op = field.ifft if is_spectral else field.fft
    not_implemented_for_non_full_domain_fields(op)

def test_sync(mset, topo, is_spectral):
    ncp = fr.config.ncp
    field = fr.ScalarField(mset, topo=topo, is_spectral=is_spectral)
    field.arr = field.grid.create_random_array(seed=12345)
    # if the field is spectral, sync should do nothing (also no error)
    if is_spectral:
        field.sync()
        return
    # if the field is not fully extended, sync should raise an error
    if not all(topo):
        not_implemented_for_non_full_domain_fields(field.sync)
        return
    # check if the sync method does not raise an error
    synced_field = field.sync()
    # check if the field is synced in place
    assert field is synced_field
    # let's differentiate the field which should make ghost points unsynced
    diff_field = field.diff(axis=0)
    # check if the field is not synced anymore
    field_copy = deepcopy(diff_field)
    synced_field = field_copy.sync()
    # check if the inner points are the same
    assert ncp.allclose(diff_field.unpad(), synced_field.unpad())
    # but the ghost points should be different
    assert not ncp.allclose(diff_field.arr, synced_field.arr)
    # we did not rigourously check if the ghost points are correct since this
    # is tested in the grid class

def test_apply_watermask(mset, topo, is_spectral):
    # TODO: should test a custom watermask array
    field = fr.ScalarField(mset, topo=topo, is_spectral=is_spectral)
    field.arr = field.grid.create_random_array(seed=12345)
    # if the field is not fully extended, apply_watermask should raise an error
    if not all(topo):
        not_implemented_for_non_full_domain_fields(field.apply_water_mask)
        return
    # if the field is spectral, apply_watermask should raise an error
    if is_spectral:
        msg = "Cannot apply watermask to spectral field"
        with pytest.raises(ValueError, match=msg):
            field.apply_water_mask()
        return
    # check if the apply_watermask method does not raise an error
    masked_field = field.apply_water_mask()
    assert masked_field is field  # should be in place

def test_has_nan(mset, topo, is_spectral):
    field = fr.ScalarField(mset, topo=topo, is_spectral=is_spectral)
    # field should not have any nan values initially
    assert not field.has_nan()
    # set some nan values
    field.arr = fr.utils.modify_array(field.arr, (0, 0), fr.config.ncp.nan)
    assert field.has_nan()

def test_copy(): ...

def test_unpad(): ...

def test_get_mesh(): ...

def test_interpolate(): ...

# ----------------------------------------------------------------
#  Test differential operators
# ----------------------------------------------------------------

def test_diff(): ...

def test_grad(): ...

def test_laplacian(): ...

def test_div(): ...

# ----------------------------------------------------------------
#  Test xarray interface
# ----------------------------------------------------------------

def test_xr(): ...

def test_xrs(): ...

def test_from_xr(): ...

def test_netcdf_save_load(): ...

# ----------------------------------------------------------------
#  Test slicing methods
# ----------------------------------------------------------------

def test_getitem(): ...

def test_setitem(): ...

# ----------------------------------------------------------------
#  Pickling
# ----------------------------------------------------------------

def test_pickle(): ...

# ----------------------------------------------------------------
#  Test arithmetic operations
# ----------------------------------------------------------------

def test_apply_operator(): ...

def test_abs(): ...

def test_sum(): ...

def test_max(): ...

def test_min(): ...

def test_integrate(): ...

def test_norm_l2(): ...

def test_dot(): ...

def test_conj(): ...
