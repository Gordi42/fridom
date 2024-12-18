"""Test for the vector field class."""
import tempfile
from copy import copy, deepcopy
from pathlib import Path
from collections import OrderedDict

import dill
import pytest
import xarray as xr

import fridom.framework as fr

# ================================================================
#  Fixtures
# ================================================================

@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def grid():
    return fr.grid.cartesian.Grid(N=(3, 10), L=(1, 2))

@pytest.fixture
def mset(grid):
    mset = fr.ModelSettingsBase(grid)
    mset.halo = 1
    mset.setup()
    return mset

@pytest.fixture(params=[
    pytest.param(True, id="spectral"),
    pytest.param(False, id="physical"),
])
def is_spectral(request):
    return request.param

@pytest.fixture(params=[None, [0], [1], [0, 1]])
def axes(request):
    return request.param

@pytest.fixture(params=[
    pytest.param(lambda x, y: x.dot(y), id="dot"),
    pytest.param(lambda x, y: x @ y, id="matmul")])
def dot_op(request):
    return request.param

@pytest.fixture(params=[1, 2, 3])
def vector_dim(request):
    return request.param

@pytest.fixture(params=[
    pytest.param((True, True), id="full extend"),
    pytest.param((True, False), id="x extend"),
    pytest.param((False, True), id="y extend"),
    pytest.param((False, False), id="no extend"),
])
def topo(request):
    return request.param

@pytest.fixture
def vector(mset):
    return fr.VectorField(mset, vector_dim=2).set_random()

# ================================================================
#  Test helpers
# ================================================================

def not_implemented_for_non_full_domain_fields(operation) -> None:
    msg = "Operation not available for non full domain fields"
    with pytest.raises(NotImplementedError, match=msg):
        operation()

def not_implemented_for_spectral_fields(operation) -> None:
    msg = "Operation not available for spectral fields"
    with pytest.raises(NotImplementedError, match=msg):
        operation()

# ================================================================
#  Tests
# ================================================================

def test_init(mset, is_spectral, vector_dim):
    vec = fr.VectorField(mset, is_spectral=is_spectral, vector_dim=vector_dim)
    # check that the instances are correct
    assert isinstance(vec, fr.VectorField)
    for f in vec:
        assert isinstance(f, fr.ScalarField)

    # check that the number of fields is correct
    assert len(vec.fields) == vector_dim

    # check that the name of the fields are correct
    for i, f in enumerate(vec):
        assert f.name == f"f{i}"

    # test the is_spectral attribute
    assert vec.is_spectral == is_spectral
    for f in vec:
        assert f.is_spectral == is_spectral

@pytest.mark.parametrize(*("field_names, double_names", [
    (["a", "b", "c"], False),
    (["a", "b", "b"], True),
]))
def test_init_from_field_list(mset, is_spectral, field_names, double_names):
    # try with list
    fields = [fr.ScalarField(mset, is_spectral=is_spectral, name=name)
              for name in field_names]
    if double_names:
        msg = "Duplicated field names"
        with pytest.raises(ValueError, match=msg):
            fr.VectorField(mset, field_list=fields)
    else:
        vec = fr.VectorField(mset, field_list=fields)
        for f, name in zip(vec, field_names):
            assert f.name == name

def test_init_from_field_dict(mset, is_spectral):
    field_names = ["a", "b", "c"]
    fields = OrderedDict((name, fr.ScalarField(mset, is_spectral=is_spectral, name=name))
                         for name in field_names)
    vec = fr.VectorField(mset, field_list=fields)
    for f, name in zip(vec, field_names):
        assert f.name == name

    fields = {name: fr.ScalarField(mset, is_spectral=is_spectral, name=name)
              for name in field_names}
    with pytest.raises(TypeError, match="Invalid field list type"):
        fr.VectorField(mset, field_list=fields)

def test_init_topo(mset, is_spectral, topo):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    for f in vec:
        assert f.topo == topo
        if not topo[0]:
            assert f.arr.shape[0] == 1
        if not topo[1]:
            assert f.arr.shape[1] == 1

@pytest.mark.parametrize("kwargs",
                         [{"topo": (True, False)}, {"is_spectral": True}])
def test_init_list_and_kwargs(mset, kwargs):
    fields = [fr.ScalarField(mset, name="a"), fr.ScalarField(mset, name="b")]
    msg = "Keyword arguments not allowed when passing a list or dict"
    with pytest.raises(TypeError, match=msg):
        fr.VectorField(mset, field_list=fields, **kwargs)

@pytest.mark.parametrize("n_dims", [1, 2, 3])
def test_init_different_dims(n_dims):
    grid = fr.grid.cartesian.Grid(N=(3,) * n_dims, L=(1,) * n_dims)
    mset = fr.ModelSettingsBase(grid).setup()
    vec = fr.VectorField(mset, vector_dim=2)
    assert isinstance(vec, fr.VectorField)

@pytest.mark.parametrize(*("kwargs, allowed", [
    ({"name": "a"}, False),
    ({"long_name": "b"}, False),
    ({"units": "c"}, False),
    ({"nc_attrs": {"a": 1}}, True),
    ({"topo": (True, False)}, True),
    ({"position": fr.grid.Position((
        fr.grid.AxisPosition.CENTER, fr.grid.AxisPosition.FACE))}, True),
    ({"bc_types": (fr.grid.BCType.NEUMANN, fr.grid.BCType.DIRICHLET)}, True),
]))
def test_init_kwargs(mset, kwargs, allowed):
    if allowed:
        vec = fr.VectorField(mset, **kwargs, vector_dim=2)
        for f in vec:
            for key, value in kwargs.items():
                assert getattr(f, key) == value
    else:
        msg = "Invalid keyword argument"
        with pytest.raises(TypeError, match=msg):
            fr.VectorField(mset, **kwargs, vector_dim=2)

# ----------------------------------------------------------------
#  Test properties
# ----------------------------------------------------------------

def test_get_attr(): ...

def test_set_attr(): ...

def test_repr(): ...

# ----------------------------------------------------------------
#  Test general methods
# ----------------------------------------------------------------

def test_fft_ifft(mset):
    vec = fr.VectorField(mset, vector_dim=2).set_random()
    # compute the fft
    vec_hat = vec.fft()
    # check that the vector is spectral and the values have changed
    assert vec_hat.is_spectral
    for f, f_hat in zip(vec, vec_hat):
        assert f.name == f_hat.name
        assert f.arr.shape != f_hat.arr.shape
    # compute the inverse fft
    vec_inv = vec_hat.ifft()
    # check that the vector is physical and the values are the same
    assert not vec_inv.is_spectral
    for f, f_inv in zip(vec, vec_inv):
        assert f.name == f_inv.name
        assert fr.config.ncp.allclose(f.arr, f_inv.arr)

def test_fft_ifft_topo(mset, topo, is_spectral):
    vec = fr.VectorField(mset, topo=topo, is_spectral=is_spectral, vector_dim=2)
    if all(topo):
        # fft should work on full domain fields
        vec.ifft() if is_spectral else vec.fft()
        return
    op = vec.ifft if is_spectral else vec.fft
    not_implemented_for_non_full_domain_fields(op)

def test_sync(vector):
    # let's differentiate the vector so that the ghost points are not synced
    vector = vector.diff(axis=0)
    # create a copy of the vector and sync the copy
    vec_sync = copy(vector).sync()
    # check that the fields are different
    for f, f_sync in zip(vector, vec_sync):
        assert not fr.config.ncp.allclose(f.arr, f_sync.arr)
    # sync the original vector
    vector.sync()
    # check that the fields are the same
    for f, f_sync in zip(vector, vec_sync):
        assert fr.config.ncp.allclose(f.arr, f_sync.arr)

def test_apply_watermask(mset, topo, is_spectral):
    # TODO(Silvano): should test a custom watermask array
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    # if the field is not fully extended, apply_watermask should raise an error
    if not all(topo):
        not_implemented_for_non_full_domain_fields(vec.apply_water_mask)
        return
    # if the field is spectral, apply_watermask should raise an error
    if is_spectral:
        not_implemented_for_spectral_fields(vec.apply_water_mask)
        return
    # check if the apply_watermask method does not raise an error
    masked_field = vec.apply_water_mask()
    assert masked_field is vec  # should be in place

def test_has_nan(vector):
    # field should not have any nan values initially
    assert not vector.has_nan()
    # set some nan values
    f1, f2 = vector
    f2.arr = fr.utils.modify_array(f2.arr, (0, 0), fr.config.ncp.nan)
    assert f2.has_nan()
    assert vector.has_nan()

def test_copy(vector):
    vec_copy = copy(vector)
    # check that the copied field is not the same as the original field
    assert vec_copy is not vector
    # check that the model settings is the same
    assert vec_copy.mset is vector.mset
    # check that the fields are not the same
    for f, f_copy in zip(vector, vec_copy):
        assert f is not f_copy
        assert f.arr.shape == f_copy.arr.shape
        assert fr.config.ncp.allclose(f.arr, f_copy.arr)

def test_set_random(mset, topo, is_spectral):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if not all(topo):
        not_implemented_for_non_full_domain_fields(vec.set_random)
        return
    # check that the field is all zeros initially
    for f in vec:
        assert fr.config.ncp.allclose(f.arr, 0)
    vec.set_random(seed=12345)
    # check if the field is not all zeros
    for f in vec:
        assert not fr.config.ncp.allclose(f.arr, 0)
    # check that the individual fields differ
    f1, f2 = vec
    assert not fr.config.ncp.allclose(f1.arr, f2.arr)
    # check that the field is reproducible
    vec2 = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    vec2.set_random(seed=12345)
    for f1, f2 in zip(vec, vec2):
        assert fr.config.ncp.allclose(f1.arr, f2.arr)

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
#  Pickling with dill
# ----------------------------------------------------------------

def test_dill(): ...

# ----------------------------------------------------------------
#  Test shrink / extend methods
# ----------------------------------------------------------------

def test_extend(mset, is_spectral):
    topo = (False, True)
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    for new_topo in [(False, False), (True, False)]:
        msg = "Cannot shrink the field in any direction"
        with pytest.raises(ValueError, match=msg):
            vec.extend(new_topo)

def test_sum(): ...

def test_max(): ...

def test_min(): ...

def test_integrate(): ...

# ----------------------------------------------------------------
#  Test arithmetic operations
# ----------------------------------------------------------------

def test_apply_operator_with_scalar_field(): ...

def test_apply_operator_with_vector_field(): ...

def test_apply_operator_with_tensor_field(): ...

def test_apply_operator_with_scalar(): ...

def test_dot_with_scalar_field(): ...

def test_dot_with_vector_field(): ...

def test_dot_with_tensor_field(): ...

def test_abs(): ...

def test_conj(): ...

# ================================================================
#  JAX JIT tests
# ================================================================

def test_jit(): ...
