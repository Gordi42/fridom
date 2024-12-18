"""Test for the vector field class."""
import tempfile
from copy import copy, deepcopy
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

# ================================================================
#  Test helpers
# ================================================================

# ================================================================
#  Tests
# ================================================================

def test_init(): ...

def test_kwargs(): ...

# ----------------------------------------------------------------
#  Test properties
# ----------------------------------------------------------------

def test_get_attr(): ...

def test_set_attr(): ...

def test_repr(): ...

# ----------------------------------------------------------------
#  Test general methods
# ----------------------------------------------------------------

def test_fft_ifft(): ...

def test_sync(): ...

def test_apply_watermask(): ...

def test_has_nan(): ...

def test_copy(): ...

def test_set_random(): ...

def test_extend(): ...

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
#  Test shrinking methods
# ----------------------------------------------------------------

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
