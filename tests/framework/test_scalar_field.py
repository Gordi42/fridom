"""Tests for the scalar field class."""
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

@pytest.fixture(params=[(True, True), (True, False), (False, True), (False, False)])
def topo(request):
    return request.param

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

# ----------------------------------------------------------------

def test_set_attr(): ...

# ----------------------------------------------------------------
#  Test general methods
# ----------------------------------------------------------------

def test_fft(): ...

def test_ifft(): ...

def test_sync(): ...

def test_apply_watermask(): ...

def test_has_nan(): ...

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
