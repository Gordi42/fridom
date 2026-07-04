"""Test for the vector field class."""
import tempfile
from collections import OrderedDict
from copy import copy
from pathlib import Path

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
    return fr.grid.cartesian.Grid(shape=(3, 10), domain_size=(1, 2))

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

def not_implemented_axes(operation) -> None:
    msg = "Operation not available for specific axes"
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
        for f, name in zip(vec, field_names, strict=False):
            assert f.name == name

def test_init_from_field_dict(mset, is_spectral):
    field_names = ["a", "b", "c"]
    fields = OrderedDict((name, fr.ScalarField(mset, is_spectral=is_spectral, name=name))
                         for name in field_names)
    vec = fr.VectorField(mset, field_list=fields)
    for f, name in zip(vec, field_names, strict=False):
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
    grid = fr.grid.cartesian.Grid(shape=(3,) * n_dims, domain_size=(1,) * n_dims)
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

def test_is_constant(mset, topo, is_spectral):
    vec = fr.VectorField(mset, topo=topo, is_spectral=is_spectral, vector_dim=2)
    expected = not any(topo)
    assert vec.is_constant == expected

# ----------------------------------------------------------------
#  Test general methods
# ----------------------------------------------------------------

def test_fft_ifft(mset):
    vec = fr.VectorField(mset, vector_dim=2).set_random()
    # compute the fft
    vec_hat = vec.fft()
    # check that the vector is spectral and the values have changed
    assert vec_hat.is_spectral
    for f, f_hat in zip(vec, vec_hat, strict=False):
        assert f.name == f_hat.name
        assert f.arr.shape != f_hat.arr.shape
    # compute the inverse fft
    vec_inv = vec_hat.ifft()
    # check that the vector is physical and the values are the same
    assert not vec_inv.is_spectral
    for f, f_inv in zip(vec, vec_inv, strict=False):
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
    for f, f_sync in zip(vector, vec_sync, strict=False):
        assert not fr.config.ncp.allclose(f.arr, f_sync.arr)
    # sync the original vector
    vector.sync()
    # check that the fields are the same
    for f, f_sync in zip(vector, vec_sync, strict=False):
        assert fr.config.ncp.allclose(f.arr, f_sync.arr)

def test_apply_watermask(mset, topo, is_spectral):
    # TODO(Silvano): should test a custom watermask array
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    # if the field is not fully extended, apply_watermask should raise an error
    if not all(topo):
        with pytest.raises(fr.exceptions.PartialDomainError):
            vec.apply_water_mask()
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
    _f1, f2 = vector
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
    for f, f_copy in zip(vector, vec_copy, strict=False):
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
    for f1, f2 in zip(vec, vec2, strict=False):
        assert fr.config.ncp.allclose(f1.arr, f2.arr)

# ----------------------------------------------------------------
#  Test differential operators
# ----------------------------------------------------------------

def test_diff(mset, topo, is_spectral):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if not all(topo):
        not_implemented_for_non_full_domain_fields(
            lambda: vec.diff(axis=0))
        return
    # if the field is spectral, diff should raise an error
    if is_spectral:
        not_implemented_for_spectral_fields(
            lambda: vec.diff(axis=0))
        return
    diff_vec = vec.diff(axis=0)
    for f, f_diff in zip(vec, diff_vec, strict=False):
        assert fr.config.ncp.allclose(f_diff.arr, f.diff(axis=0).arr)

def test_grad(mset, topo, is_spectral):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    msg = "grad not implemented yet"
    with pytest.raises(NotImplementedError, match=msg):
        vec.grad()

def test_laplacian(mset, topo, is_spectral):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if not all(topo):
        not_implemented_for_non_full_domain_fields(vec.laplacian)
        return
    # if the field is spectral, diff should raise an error
    if is_spectral:
        not_implemented_for_spectral_fields(vec.laplacian)
        return
    lap_vec = vec.laplacian()
    for f, f_lap in zip(vec, lap_vec, strict=False):
        assert fr.config.ncp.allclose(f_lap.arr, f.laplacian().arr)

def test_div(mset, topo, is_spectral):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    msg = "div not implemented yet"
    with pytest.raises(NotImplementedError, match=msg):
        vec.div()

@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_cumulative_integral(mset, topo, is_spectral, direction):
    pos = mset.grid.cell_center
    if direction == "backward":
        pos = pos.shift(axis=1)
    vec = fr.VectorField(mset,
                         is_spectral=is_spectral,
                         topo=topo,
                         vector_dim=2,
                         position=pos)
    # we need to shift the field position for the backward direction
    # if the field is spectral, diff should raise an error
    if is_spectral:
        with pytest.raises(fr.exceptions.FieldSpaceError):
            vec.cumulative_integral(axis=1, direction=direction)
        return
    if not all(topo):
        with pytest.raises(fr.exceptions.PartialDomainError):
            vec.cumulative_integral(axis=1, direction=direction)
        return

    cumvec = vec.cumulative_integral(axis=1, direction=direction)
    for cv, f in zip(cumvec, vec, strict=False):
        cum_f = f.cumulative_integral(axis=1, direction=direction)
        # check if the fields are the same
        assert fr.config.ncp.allclose(cv.arr, cum_f.arr)


# ----------------------------------------------------------------
#  Test xarray interface
# ----------------------------------------------------------------

def test_xr(mset, is_spectral, topo):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if not all(topo):
        not_implemented_for_non_full_domain_fields(lambda: vec.xr)
        return
    ds = vec.xr
    assert isinstance(ds, xr.Dataset)

@pytest.mark.parametrize(*(
    "key, expected_shape, dim_names",
    [
        (slice(None), (10, 3), slice(None)),
        (-1, (10,), 1),
        ((slice(None), -1), (3,), 0),
        ((slice(None), slice(1, 5)), (4, 3), slice(None)),
    ],
))
def test_xrs(mset, is_spectral, key, expected_shape, dim_names):
    vec = fr.VectorField(mset, is_spectral=is_spectral, vector_dim=2)
    ds = vec.xrs[key]
    assert isinstance(ds, xr.Dataset)
    # check that the variables are in the dataset and have the correct shape
    var_names = [f.name for f in vec]
    for var_name in var_names:
        assert var_name in ds.variables
        assert ds[var_name].shape == expected_shape
    # check if the dimenstion names are correct
    exp_names = (["kx", "ky"] if is_spectral else ["x", "y"])[dim_names]
    exp_names = {exp_names} if isinstance(exp_names, str) else set(exp_names)
    assert set(ds.dims) == exp_names

@pytest.mark.parametrize(*(
    "key, possible",
    [
        (slice(None), True),
        (-1, False),
        (slice(None, 3), False),
    ],
))
def test_from_xr(mset, topo, is_spectral, key, possible):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if not all(topo):
        not_implemented_for_non_full_domain_fields(lambda: vec.xrs[key])
        return
    ds = vec.xrs[key]
    if not possible:
        msg = "Cannot convert sliced dataarray to ScalarField"
        with pytest.raises(ValueError, match=msg):
            fr.VectorField.from_xarray(mset, ds)
        return
    new_vec = fr.VectorField.from_xarray(mset, ds)
    # new vector should not be the same as the original vector
    assert new_vec is not vec
    # check that the fields are the same
    for f, f_new in zip(vec, new_vec, strict=False):
        assert fr.config.ncp.allclose(f.arr, f_new.arr)

def test_netcdf_save_load(mset, is_spectral, tmp_dir):
    vec = fr.VectorField(mset, is_spectral=is_spectral, vector_dim=2).set_random()
    # save the field to a netcdf file
    vec.to_netcdf(tmp_dir + "/vec.nc")
    # load the field from the netcdf file
    new_vec = fr.VectorField.from_netcdf(mset, tmp_dir + "/vec.nc")
    # check that the fields are the same
    for f, f_new in zip(vec, new_vec, strict=False):
        assert fr.config.ncp.allclose(f.arr, f_new.arr)

# ----------------------------------------------------------------
#  Test slicing methods
# ----------------------------------------------------------------

def test_getitem_int(mset, is_spectral):
    vec = fr.VectorField(mset, vector_dim=3, is_spectral=is_spectral).set_random()
    # test with index
    f = vec[0]
    assert isinstance(f, fr.ScalarField)
    assert f.name == "f0"

def test_getitem_str(mset, is_spectral):
    vec = fr.VectorField(mset, vector_dim=3, is_spectral=is_spectral).set_random()
    # test with string
    f = vec["f1"]
    assert isinstance(f, fr.ScalarField)
    assert f.name == "f1"

def test_getitem_slice(mset, is_spectral):
    vec = fr.VectorField(mset, vector_dim=3, is_spectral=is_spectral).set_random()
    # test with slice
    vec_slice = vec[:2]
    assert isinstance(vec_slice, fr.VectorField)
    assert vec_slice.vector_dim == 2
    assert vec_slice[0].name == "f0"
    assert vec_slice[1].name == "f1"
    # test with another slice
    vec_slice = vec[1:]
    assert isinstance(vec_slice, fr.VectorField)
    assert vec_slice.vector_dim == 2
    assert vec_slice[0].name == "f1"
    assert vec_slice[1].name == "f2"

def test_setitem_int(mset, is_spectral):
    field = fr.ScalarField(mset, is_spectral=is_spectral, name="f0")
    vec2 = fr.VectorField(mset, vector_dim=2, is_spectral=is_spectral).set_random()
    # it should be possible to set the field with the same name
    vec2[0] = field
    assert vec2[0] is field
    # it should not be possible to set the field with a different name
    msg = "Field name mismatch"
    with pytest.raises(ValueError, match=msg):
        vec2[1] = field

def test_setitem_str(mset, is_spectral):
    field = fr.ScalarField(mset, is_spectral=is_spectral, name="f0")
    vec2 = fr.VectorField(mset, vector_dim=2, is_spectral=is_spectral).set_random()
    # it should be possible to set the field via the name
    vec2["f0"] = field
    assert vec2[0] is field
    # it should not be possible to set the field with a different name
    msg = "Field name mismatch"
    with pytest.raises(ValueError, match=msg):
        vec2["f1"] = field

def test_setitem_slice(mset, is_spectral):
    # test with slice access
    vec2 = fr.VectorField(mset, vector_dim=2, is_spectral=is_spectral).set_random()
    vec3 = fr.VectorField(mset, vector_dim=3, is_spectral=is_spectral).set_random()
    # it should be possible to set the field with the correct names
    vec3[:2] = vec2
    assert vec3[0] is vec2[0]
    assert vec3[1] is vec2[1]
    # it should not be possible to set the field with different names
    msg = "Field name mismatch"
    with pytest.raises(ValueError, match=msg):
        vec3[1:] = vec2

# ----------------------------------------------------------------
#  Pickling with dill
# ----------------------------------------------------------------

def test_dill(mset, is_spectral, topo, tmp_dir):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    path = Path(tmp_dir + "/vec.pkl")
    # check that the file does not exist
    assert not path.exists()
    # check if the field can be pickled with dill
    with path.open("wb") as f:
        dill.dump(vec, f)
    # check if the file exists
    assert path.exists()
    # load the field
    with path.open("rb") as f:
        new_vec = dill.load(f)  # noqa: S301
    # check that the fields are the same
    for f, f_new in zip(vec, new_vec, strict=False):
        assert fr.config.ncp.allclose(f.arr, f_new.arr)

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

def test_sum(mset, is_spectral, topo, axes):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if not all(topo):
        not_implemented_for_non_full_domain_fields(lambda: vec.sum(axes))
        return
    if axes is not None:
        not_implemented_axes(lambda: vec.sum(axes))
        return
    result = vec.sum()
    assert isinstance(result, fr.VectorField)
    for f in result:
        assert f.topo == (False, False)
        assert f.arr.shape == (1, 1)

def test_max(mset, is_spectral, topo, axes):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if not all(topo):
        not_implemented_for_non_full_domain_fields(lambda: vec.max(axes))
        return
    if axes is not None:
        not_implemented_axes(lambda: vec.max(axes))
        return
    result = vec.max()
    assert isinstance(result, fr.VectorField)
    for f in result:
        assert f.topo == (False, False)
        assert f.arr.shape == (1, 1)

def test_min(mset, is_spectral, topo, axes):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if not all(topo):
        not_implemented_for_non_full_domain_fields(lambda: vec.min(axes))
        return
    if axes is not None:
        not_implemented_axes(lambda: vec.min(axes))
        return
    result = vec.min()
    assert isinstance(result, fr.VectorField)
    for f in result:
        assert f.topo == (False, False)
        assert f.arr.shape == (1, 1)

def test_integrate(mset, is_spectral, topo, axes):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if is_spectral:
        with pytest.raises(fr.exceptions.FieldSpaceError):
            vec.integrate(axes)
        return
    if not all(topo):
        with pytest.raises(fr.exceptions.PartialDomainError):
            vec.integrate(axes)
        return
    vec_int = vec.integrate(axes)
    axes = axes or (0, 1)
    for f in vec_int:
        for axis in axes:
            assert not f.topo[axis]
            assert f.arr.shape[axis] == 1

def test_mean(mset, is_spectral, topo, axes):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if is_spectral:
        with pytest.raises(fr.exceptions.FieldSpaceError):
            vec.mean(axes)
        return
    if not all(topo):
        with pytest.raises(fr.exceptions.PartialDomainError):
            vec.mean(axes)
        return
    vec_mean = vec.mean(axes)
    axes = axes or (0, 1)
    for f in vec_mean:
        for axis in axes:
            assert not f.topo[axis]
            assert f.arr.shape[axis] == 1

# ----------------------------------------------------------------
#  Test arithmetic operations
# ----------------------------------------------------------------

@pytest.mark.parametrize(*(
    "op",
    [
        pytest.param(lambda x, y: x + y, id="add"),
        pytest.param(lambda x, y: y + x, id="radd"),
        pytest.param(lambda x, y: x - y, id="sub"),
        pytest.param(lambda x, y: y - x, id="rsub"),
        pytest.param(lambda x, y: x * y, id="mul"),
        pytest.param(lambda x, y: y * x, id="rmul"),
        pytest.param(lambda x, y: x / y, id="div"),
        pytest.param(lambda x, y: y / x, id="rdiv"),
        pytest.param(lambda x, y: x ** y, id="pow"),
        pytest.param(lambda x, y: y ** x, id="rpow"),
    ],
))
def test_apply_operator_with_scalar_field(mset, topo, is_spectral, op):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    scalar = fr.ScalarField(mset, is_spectral=is_spectral)
    # TODO(Silvano): set random also for non full domain fields
    if all(topo):
        vec.set_random()
        scalar.set_random()
    # we need to make sure that the arrays are never zero, as otherwise
    # division or power operations may fail. Just add 20 to all values
    vec += 20
    scalar += 20
    # test if the operation works
    new_vec = op(vec, scalar)
    # check if the result is a vector field
    assert isinstance(new_vec, fr.VectorField)
    # check if the fields are correct
    for f, f_new in zip(vec, new_vec, strict=False):
        assert fr.config.ncp.allclose(op(f.arr, scalar.arr), f_new.arr)

@pytest.mark.parametrize(*(
    "op",
    [
        pytest.param(lambda x, y: x + y, id="add"),
        pytest.param(lambda x, y: x - y, id="sub"),
        pytest.param(lambda x, y: x * y, id="mul"),
        pytest.param(lambda x, y: x / y, id="div"),
        pytest.param(lambda x, y: x ** y, id="pow"),
    ],
))
def test_apply_operator_with_vector_field(mset, topo, is_spectral, op):
    vec1 = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    vec2 = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    # TODO(Silvano): set random also for non full domain fields
    if all(topo):
        vec1.set_random(seed=12345)
        vec2.set_random(seed=54321)
    vec1 += 20
    vec2 += 20
    # test if the operation works
    new_vec = op(vec1, vec2)
    # check if the result is a vector field
    assert isinstance(new_vec, fr.VectorField)
    # check if the fields are correct
    for f1, f2, f_new in zip(vec1, vec2, new_vec, strict=False):
        assert fr.config.ncp.allclose(op(f1.arr, f2.arr), f_new.arr)

def test_apply_operator_with_invalid_vector_field(mset):
    vec2 = fr.VectorField(mset, vector_dim=2)
    vec3 = fr.VectorField(mset, vector_dim=3)

    msg = "Vector dimensions do not match"
    with pytest.raises(ValueError, match=msg):
        vec2 + vec3

@pytest.mark.parametrize(*(
    "op",
    [
        pytest.param(lambda x, y: x + y, id="add"),
        pytest.param(lambda x, y: y + x, id="radd"),
        pytest.param(lambda x, y: x - y, id="sub"),
        pytest.param(lambda x, y: y - x, id="rsub"),
        pytest.param(lambda x, y: x * y, id="mul"),
        pytest.param(lambda x, y: y * x, id="rmul"),
        pytest.param(lambda x, y: x / y, id="div"),
        pytest.param(lambda x, y: y / x, id="rdiv"),
        pytest.param(lambda x, y: x ** y, id="pow"),
        pytest.param(lambda x, y: y ** x, id="rpow"),
    ],
))
def test_apply_operator_with_scalar(mset, topo, is_spectral, op):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    vec += 20
    scalar = 2
    # test if the operation works
    new_vec = op(vec, scalar)
    # check if the result is a vector field
    assert isinstance(new_vec, fr.VectorField)
    # check if the fields are correct
    for f, f_new in zip(vec, new_vec, strict=False):
        assert fr.config.ncp.allclose(op(f.arr, scalar), f_new.arr)

@pytest.mark.parametrize(*(
    "other",
    [
        pytest.param("test", id="str"),
        pytest.param([1, "test"], id="list"),
        pytest.param({"key": "value"}, id="dict"),
        pytest.param({3, "test"}, id="set"),
        pytest.param((1, ), id="tuple"),
    ],
))
def test_apply_operator_with_wrong_type(mset, other):
    field = fr.VectorField(mset, vector_dim=2)
    with pytest.raises(TypeError):
        field + other

def test_dot_with_scalar_field(mset, topo, is_spectral, dot_op):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    # if the spectral flag is different, the dot product should raise an error
    scalar = fr.ScalarField(mset, is_spectral=not is_spectral)
    msg = "Cannot take dot product of spectral and real fields"
    with pytest.raises(ValueError, match=msg):
        dot_op(vec, scalar)
    scalar = fr.ScalarField(mset, is_spectral=is_spectral)
    if all(topo):
        vec.set_random(seed=32145)
        scalar.set_random(seed=54321)
    # test if the operation works
    result = dot_op(vec, scalar)
    # check if the result is a vector field
    assert isinstance(result, fr.VectorField)
    # check if the fields are correct
    for f, f_new in zip(vec, result, strict=False):
        assert fr.config.ncp.allclose(f.arr * scalar.arr.conj(), f_new.arr)

def test_dot_with_vector_field(mset, topo, is_spectral, dot_op):
    vec1 = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    vec2 = fr.VectorField(mset, is_spectral=not is_spectral, topo=topo, vector_dim=2)
    msg = "Cannot take dot product of spectral and real fields"
    with pytest.raises(ValueError, match=msg):
        dot_op(vec1, vec2)
    vec2 = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if all(topo):
        vec1.set_random(seed=12345)
        vec2.set_random(seed=54321)
    # test if the operation works
    result = dot_op(vec1, vec2)
    # check if the result is a scalar field
    assert isinstance(result, fr.ScalarField)
    # check if the fields are correct
    expected = sum(f1 * f2.conj() for f1, f2 in zip(vec1, vec2, strict=False))
    assert fr.config.ncp.allclose(expected.arr, result.arr)

def test_dot_with_invalid_vector_field(mset, dot_op):
    vec2 = fr.VectorField(mset, vector_dim=2)
    vec3 = fr.VectorField(mset, vector_dim=3)

    msg = "Vector dimensions do not match"
    with pytest.raises(ValueError, match=msg):
        dot_op(vec2, vec3)

def test_dot_with_tensor_field(): ...

def test_abs(mset, topo, is_spectral):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if all(topo):
        vec.set_random()
    vec_abs = abs(vec)
    for f, f_abs in zip(vec, vec_abs, strict=False):
        assert fr.config.ncp.allclose(abs(f.arr), f_abs.arr)

def test_conj(mset, topo, is_spectral):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if all(topo):
        vec.set_random()
    vec_conj = vec.conj()
    for f, f_conj in zip(vec, vec_conj, strict=False):
        assert fr.config.ncp.allclose(f.arr.conj(), f_conj.arr)

def test_neg(mset, topo, is_spectral):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if all(topo):
        vec.set_random()
    vec_neg = -vec
    for f, f_neg in zip(vec, vec_neg, strict=False):
        assert fr.config.ncp.allclose(-f.arr, f_neg.arr)

def test_norm_l2(mset, topo, is_spectral):
    vec = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if is_spectral:
        with pytest.raises(fr.exceptions.FieldSpaceError):
            vec.norm_l2()
        return
    if not all(topo):
        with pytest.raises(fr.exceptions.PartialDomainError):
            vec.norm_l2()
        return
    norm = vec.norm_l2()
    assert isinstance(norm, float)

def test_norm_of_diff(mset, topo, is_spectral):
    vec1 = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2) + 1
    vec2 = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    if is_spectral:
        with pytest.raises(fr.exceptions.FieldSpaceError):
            vec1.norm_of_diff(vec2)
        return
    if not all(topo):
        with pytest.raises(fr.exceptions.PartialDomainError):
            vec1.norm_of_diff(vec2)
        return
    norm_of_diff = vec1.norm_of_diff(vec2)
    assert isinstance(norm_of_diff, float)

def test_norm_of_diff_invalid(mset, topo, is_spectral):
    vec1 = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=2)
    vec2 = fr.VectorField(mset, is_spectral=is_spectral, topo=topo, vector_dim=3)
    msg = "Vector dimensions do not match"
    with pytest.raises(ValueError, match=msg):
        vec1.norm_of_diff(vec2)

# ================================================================
#  JAX JIT tests
# ================================================================

@pytest.mark.parametrize("op", [
    pytest.param(lambda x: x * 2, id="mul"),
    pytest.param(lambda x: x.fft(), id="fft"),
    pytest.param(lambda x: x.sync(), id="sync"),
    pytest.param(lambda x: x.apply_water_mask(), id="apply_water_mask"),
    pytest.param(lambda x: x.set_random(seed=12345), id="set_random"),
])
def test_jit(mset, op):
    vec = fr.VectorField(mset, vector_dim=2).set_random(seed=12345)
    # check that the field can be jitted
    @fr.utils.jaxjit
    def func(f) -> fr.ScalarField:
        return op(f)
    new_vec = func(vec)
    assert isinstance(new_vec, fr.VectorField)
    for f_exp, f_new in zip(op(vec), new_vec, strict=False):
        assert fr.config.ncp.allclose(f_exp.arr, f_new.arr)
    if not fr.config.backend_is_jax:
        return
    # check if a gradient can be computed
    import jax
    grad_func = jax.grad(lambda f: func(f).sum()[0].arr.item().real)
    grad_func(vec)
