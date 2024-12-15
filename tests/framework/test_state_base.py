"""Tests for the StateBase class."""
import tempfile
from copy import copy, deepcopy

import numpy as np
import pytest
import xarray as xr

import fridom.framework as fr

# ================================================================
#  Basic fixtures
# ================================================================

@pytest.fixture
def directory():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture(params=[2, 3], ids=["2D", "3D"])
def n_dims(request):
    return request.param

@pytest.fixture(params=[True, False], ids=["Spectral", "Physical"])
def is_spectral(request):
    return request.param

@pytest.fixture
def dtype_in(is_spectral):
    return fr.config.dtype_comp if is_spectral else fr.config.dtype_real

@pytest.fixture
def dtype_out(is_spectral):
    return fr.config.dtype_real if is_spectral else fr.config.dtype_comp

@pytest.fixture(params=[3, 5], ids=["n=3", "n=5"])
def n_fields(request):
    return request.param

@pytest.fixture
def mset(n_dims):
    grid = fr.grid.cartesian.Grid(N=tuple([32]*n_dims), L=tuple([1.0]*n_dims))
    mset = fr.ModelSettingsBase(grid)
    mset.halo = 1
    mset.setup()
    return mset

@pytest.fixture
def position(n_dims):
    return fr.grid.Position(
        tuple([fr.grid.AxisPosition.CENTER]*n_dims))

@pytest.fixture
def field_list(mset, is_spectral, n_fields, position):
    field_list = [fr.FieldVariable(
        mset, name=f"v{i}", is_spectral=is_spectral, position=position)
                  for i in range(n_fields)]
    for field in field_list:
        field.arr = mset.grid.create_random_array(spectral=is_spectral)
    return field_list

@pytest.fixture
def state(mset, field_list, is_spectral):
    return fr.StateBase(mset, field_list, is_spectral=is_spectral)

@pytest.fixture
def mset_1d():
    grid = fr.grid.cartesian.Grid(N=(3,), L=(1.0,))
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset

@pytest.fixture
def position_1d():
    return fr.grid.Position((fr.grid.AxisPosition.CENTER,))

@pytest.fixture
def zeros_p(mset_1d, position_1d):
    return fr.FieldVariable(
        mset_1d, is_spectral=False, name="zeros_p", position=position_1d)

@pytest.fixture
def zeros_s(mset_1d, position_1d):
    return fr.FieldVariable(
        mset_1d, is_spectral=True, name="zeros_s", position=position_1d)

@pytest.fixture
def ones_p(mset_1d, position_1d):
    return fr.FieldVariable(
        mset_1d, is_spectral=False, name="ones_p", position=position_1d) + 1.0

@pytest.fixture
def ones_s(mset_1d, position_1d):
    return fr.FieldVariable(
        mset_1d, is_spectral=True, name="ones_s", position=position_1d) + 1.0

@pytest.fixture
def imag_s(mset_1d, position_1d):
    return fr.FieldVariable(
        mset_1d, is_spectral=True, name="imag_s", position=position_1d) + 1.0j

@pytest.fixture
def state_01(mset_1d, position_1d):
    v1 = fr.FieldVariable(
        mset_1d, is_spectral=False, name="v1", position=position_1d)
    v2 = fr.FieldVariable(
        mset_1d, is_spectral=False, name="v2", position=position_1d) + 1.0
    return fr.StateBase(mset_1d, [v1, v2], is_spectral=False)

@pytest.fixture
def state_11(mset_1d, position_1d):
    v1 = fr.FieldVariable(
        mset_1d, is_spectral=False, name="v1", position=position_1d) + 1.0
    v2 = fr.FieldVariable(
        mset_1d, is_spectral=False, name="v2", position=position_1d) + 1.0
    return fr.StateBase(mset_1d, [v1, v2], is_spectral=False)

@pytest.fixture
def state_2d(is_spectral):
    grid = fr.grid.cartesian.Grid(N=(16, 16), L=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid)
    mset.halo = 1
    mset.setup()
    v1 = fr.FieldVariable(mset, is_spectral=is_spectral, name="v1")
    v2 = fr.FieldVariable(mset, is_spectral=is_spectral, name="v2")
    v1.arr = mset.grid.create_random_array(seed=12345)
    v2.arr = mset.grid.create_random_array(seed=54321)
    return fr.StateBase(mset, [v1, v2])

def fields_are_equal(f1, f2):
    return fr.config.ncp.allclose(f1.arr, f2.arr)

def state_fields_are_equal(state1, state2):
    for key in state1.fields:
        f1 = state1.fields[key]
        f2 = state2.fields[key]
        if not fields_are_equal(f1, f2):
            return False
    return True

# ================================================================
#  Tests
# ================================================================

def test_init(mset, field_list):
    state = fr.StateBase(mset, field_list)
    assert state.mset is mset
    assert state.grid is mset.grid
    field_dict = {f.name: f for f in field_list}
    assert state.fields == field_dict

# ----------------------------------------------------------------
#  Basic operators
# ----------------------------------------------------------------

def test_fft(state, dtype_in, dtype_out, is_spectral):
    if is_spectral:
        msg = "State is in spectral space, cannot perform fft"
        with pytest.raises(ValueError, match=msg):
            state.fft()
        return

    state_fft = state.fft()
    assert state_fft.is_spectral
    assert state_fft.field_list[0].arr.dtype == dtype_out
    state_fft_ifft = state_fft.ifft()
    assert not state_fft_ifft.is_spectral
    assert state_fft_ifft.field_list[0].arr.dtype == dtype_in
    # Check that the data is the same
    assert state_fields_are_equal(state, state_fft_ifft)

def test_ifft(state, is_spectral):
    if not is_spectral:
        msg = "State is not in spectral space, cannot perform ifft"
        with pytest.raises(ValueError, match=msg):
            state.ifft()
        return
    state_ifft = state.ifft()
    assert not state_ifft.is_spectral

def test_sync(mset, field_list):
    if field_list[0].is_spectral:
        # skip test for spectral fields
        return
    # make sure we have a halo of at least 1
    assert mset.halo >= 1
    # modify the field list so that the ghost points are not synced
    for field in field_list:
        field.arr = field.diff(axis=0).arr
    field_list2 = deepcopy(field_list)
    for field in field_list2:
        field.arr = field.sync().arr

    # synced and non-synced fields should be different
    fields1 = {f.name: f for f in field_list}
    fields2 = {f.name: f for f in field_list2}
    for key, field in fields1.items():
        assert not fields_are_equal(field, fields2[key])

    # create a state and sync it
    state = fr.StateBase(mset, field_list)
    state = state.sync()
    # synced and non-synced fields should be the same
    for key in state.fields:
        f1 = state.fields[key]
        f2 = fields2[key]
        assert fields_are_equal(f1, f2)

def test_project(mset, field_list, is_spectral):
    if is_spectral:
        return
    state = fr.StateBase(mset, field_list)
    vec_q = state * 2.0
    vec_p = state * (-1.75)

    # check that the result is a StateBase object
    ref = state.project(vec_p, vec_q)
    assert isinstance(ref, fr.StateBase)

    # check with spectral projection vectors
    res = state.project(vec_p.fft(), vec_q)
    assert state_fields_are_equal(ref, res)

    res = state.project(vec_p, vec_q.fft())
    assert state_fields_are_equal(ref, res)

    res = state.project(vec_p.fft(), vec_q.fft())
    assert state_fields_are_equal(ref, res)

    # check with spectral state
    res = state.fft().project(vec_p, vec_q).ifft()
    assert state_fields_are_equal(ref, res)

def test_dot(ones_p, ones_s, state_01, state_11, position_1d):
    mset_1d = state_01.mset
    state = state_01
    state2 = state_11
    dot = state.dot(state2)
    assert isinstance(dot, fr.FieldVariable)
    assert fields_are_equal(dot, ones_p)

    # test complex
    v1 = fr.FieldVariable(
        mset_1d, is_spectral=True, name="v1", position=position_1d) + 1
    v2 = fr.FieldVariable(
        mset_1d, is_spectral=True, name="v2", position=position_1d) + 1 - 1j
    state = fr.StateBase(mset_1d, [v1, v2], is_spectral=True)
    v1 = fr.FieldVariable(
        mset_1d, is_spectral=True, name="v1", position=position_1d) + 1j
    v2 = fr.FieldVariable(
        mset_1d, is_spectral=True, name="v2", position=position_1d) + 1
    state2 = fr.StateBase(mset_1d, [v1, v2], is_spectral=True)
    dot = state.dot(state2)
    assert fields_are_equal(dot, ones_s-2j)

def test_norm_l2(state_01, state_11):
    state = state_01
    state2 = state_11
    # yields state = [(0, 1), (0, 1), (0, 1)]
    # with l2 norm = sqrt((1+1+1) * 1/3) = 1
    #                               ^^^
    #                               dV
    norm = state.norm_l2()
    assert fr.config.ncp.allclose(norm, 1.0)

    # yields state = [(1, 1), (1, 1), (1, 1)]
    # with l2 norm = sqrt((2+2+2) * 1/3) = sqrt(2)
    norm = state2.norm_l2()
    assert fr.config.ncp.allclose(norm, 2**0.5)

def test_norm_of_diff(state_01, state_11):
    # test norm of difference between two identical states
    # should be 0
    norm = state_01.norm_of_diff(state_01)
    assert norm == 0

    # test norm of difference between two different states
    # the l2 norm of state - state2 is:
    # sqrt((1+1+1) * 1/3) = 1
    assert (state_01 - state_11).norm_l2() == 1

    # hence, the norm of the difference should be
    # 2 * |z - z'|_2 / (|z|_2 + |z'|_2)
    # 2 *    1       / ( 1    + 2**(1/2)) due to the above
    norm = state_01.norm_of_diff(state_11)
    assert norm == 2 / (1 + 2**0.5)

def test_has_nan(mset, field_list):
    # initially the array should not have nan
    state = fr.StateBase(mset, field_list)
    assert not state.has_nan()
    # set the array to have nan
    n_dims = mset.grid.n_dims
    arr = field_list[0].arr
    position = tuple(0 for _ in range(n_dims))
    arr = fr.utils.modify_array(arr, position, np.nan)
    # check if the array has nan
    assert fr.config.ncp.any(fr.config.ncp.isnan(arr))
    # set the array back and create a state
    field_list[0].arr = arr
    state = fr.StateBase(mset, field_list)
    # check if the state has nan
    assert state.has_nan()

def test_repr(state_2d):
    # check that the repr does not raise an error
    res = repr(state_2d)
    # check that the repr is a string
    assert isinstance(res, str)
    # check that the repr is correct
    assert res == "State with fields:\n  v1: Unnamed  [n/a]\n  v2: Unnamed  [n/a]\n"

# ----------------------------------------------------------------
#  xarray conversion
# ----------------------------------------------------------------

def test_xr(state_2d):
    n_grid_points = 16
    # convert to xarray dataset
    ds = state_2d.xr
    # check that the dataset is an xarray dataset
    assert isinstance(ds, xr.Dataset)
    # check that the dataset has the correct dimensions
    dim_names = list(ds.dims)
    expected_dim_names = {"kx", "ky"} if state_2d.is_spectral else {"x", "y"}
    assert set(dim_names) == expected_dim_names
    for dim_name in dim_names:
        assert ds[dim_name].size == n_grid_points
    # check that the dataset contains the correct variables
    expected_var_names = {"v1", "v2"}
    expected_vars = expected_var_names | expected_dim_names
    var_names = set(ds.variables)
    assert var_names == expected_vars
    # check that the variables have the correct shape
    for var_name in expected_var_names:
        assert ds[var_name].shape == (n_grid_points, n_grid_points)

def test_xrs(state_2d):
    n_grid_points = 16
    # convert to sliceable
    sl = state_2d.xrs
    # check that the dataset is an xarray dataset
    assert isinstance(sl, fr.utils.SliceableAttribute)
    ds = sl[(0, slice(None, None, 2))]
    # check that the dataset is an xarray dataset
    assert isinstance(ds, xr.Dataset)
    # check that the dataset has the correct dimensions
    dim_names = list(ds.dims)
    expected_dim_names = {"ky"} if state_2d.is_spectral else {"y"}
    assert set(dim_names) == expected_dim_names
    for dim_name in dim_names:
        assert ds[dim_name].size == n_grid_points // 2
    # check that the dataset contains the correct variables
    expected_var_names = {"v1", "v2"}
    expected_vars = expected_var_names | expected_dim_names
    var_names = set(ds.variables)
    assert var_names == expected_vars
    # check that the variables have the correct shape
    for var_name in expected_var_names:
        assert ds[var_name].shape == (n_grid_points // 2, )

# ----------------------------------------------------------------
#  Field variable access
# ----------------------------------------------------------------

def test_get_field(state_2d):
    # get a field
    field = state_2d["v1"]
    assert isinstance(field, fr.FieldVariable)
    assert field.name == "v1"
    # get a field that does not exist
    with pytest.raises(KeyError):
        state_2d["v3"]

def test_set_field(state_2d):
    # set a field
    field = fr.FieldVariable(state_2d.mset, name="v2")
    state_2d["v2"] = field
    assert state_2d["v2"] is field
    # set a field that does not exist
    with pytest.raises(KeyError):
        state_2d["v3"] = field

# ----------------------------------------------------------------
#  Creating copies
# ----------------------------------------------------------------

def test_copy(state):
    state_copy = copy(state)
    # model settings should be the same
    assert state_copy.mset is state.mset
    # the fields should be different
    for key in state.fields:
        assert state.fields[key] is not state_copy.fields[key]
        assert fields_are_equal(state.fields[key], state_copy.fields[key])

# ----------------------------------------------------------------
#  NetCDF I/O
# ----------------------------------------------------------------

def test_save_load(directory, state):
    is_spectral = state.is_spectral
    conf = fr.config
    dtype = conf.dtype_comp if is_spectral else conf.dtype_real
    # check that the dtype is correct
    assert state.field_list[0].arr.dtype == dtype
    mset = state.mset
    # save the state
    state.to_netcdf(directory + "/state.nc")
    # load the state
    state2 = fr.StateBase.from_netcdf(mset, directory + "/state.nc")
    # check that the two states are the same
    assert state_fields_are_equal(state, state2)

# ----------------------------------------------------------------
#  Operator overloading
# ----------------------------------------------------------------

def test_add(state_01, state_11, zeros_p, ones_p):
    state = state_01
    state2 = state_11

    # add two states
    state3 = state + state2
    assert fields_are_equal(state.field_list[0], zeros_p)
    assert fields_are_equal(state.field_list[1], ones_p)
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], ones_p)
    assert fields_are_equal(state3.field_list[1], ones_p + 1)

    # add state and scalar
    state3 = state + 1
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], ones_p)
    assert fields_are_equal(state3.field_list[1], ones_p + 1)

    # add state and array
    state3 = state + ones_p
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], ones_p)
    assert fields_are_equal(state3.field_list[1], ones_p + 1)

    # add number and state
    state3 = 1 + state
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], ones_p)
    assert fields_are_equal(state3.field_list[1], ones_p + 1)

def test_sub(ones_p, state_01, state_11):
    state = state_01
    state2 = state_11

    # subtract two states
    state3 = state - state2
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], ones_p * -1)
    assert fr.config.ncp.allclose(state3.field_list[1].arr, 0)

    # subtract state and scalar
    state3 = state - 1
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], ones_p * -1)
    assert fr.config.ncp.allclose(state3.field_list[1].arr, 0)

    # subtract state and array
    state3 = state - ones_p
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], ones_p * -1)
    assert fr.config.ncp.allclose(state3.field_list[1].arr, 0)

    # subtract number and state
    state3 = 1 - state
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], ones_p)
    assert fr.config.ncp.allclose(state3.field_list[1].arr, 0)

def test_mul(zeros_p, ones_p, state_01, state_11):
    state = state_01
    state2 = state_11

    # multiply two states
    state3 = state * state2
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], zeros_p)
    assert fields_are_equal(state3.field_list[1], ones_p)

    # multiply state and scalar
    state3 = state * 2
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], zeros_p)
    assert fr.config.ncp.allclose(state3.field_list[1].arr, 2)

    # multiply state and array
    state3 = state * ones_p
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], zeros_p)
    assert fields_are_equal(state3.field_list[1], ones_p)

    # multiply number and state
    state3 = 2 * state
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], zeros_p)
    assert fr.config.ncp.allclose(state3.field_list[1].arr, 2)

def test_truediv(zeros_p, ones_p, state_01, state_11):
    state = state_01
    state2 = state_11

    # divide two states
    state3 = state / state2
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], zeros_p)
    assert fields_are_equal(state3.field_list[1], ones_p)

    # divide state and scalar
    state3 = state / 2
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], zeros_p)
    assert fr.config.ncp.allclose(state3.field_list[1].arr, 0.5)

    # divide state and array
    state3 = state / ones_p
    assert isinstance(state3, fr.StateBase)
    assert fields_are_equal(state3.field_list[0], zeros_p)
    assert fr.config.ncp.allclose(state3.field_list[1].arr, 1)

    # divide number and state
    state3 = 2 / state2
    assert isinstance(state3, fr.StateBase)
    assert fr.config.ncp.allclose(state3.field_list[0].arr, 2)
    assert fr.config.ncp.allclose(state3.field_list[1].arr, 2)
