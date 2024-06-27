from mpi4py import MPI
import pytest
from copy import deepcopy
import numpy as np
import fridom.framework as fr
from fridom.framework import config

# --------------------------------------------------------------
#  Create fixtures for the tests
# --------------------------------------------------------------
is_parallel = MPI.COMM_WORLD.Get_size() > 1

# skip n_dims=1 if parallel
@pytest.fixture(
        params=[pytest.param(1, id="1D", marks=pytest.mark.skipif(
                        is_parallel, reason="Skip n_dims=1 if parallel")), 
                pytest.param(2, id="2D"),
                pytest.param(3, id="3D")])
def n_dims(request):
    return request.param

@pytest.fixture()
def mset(backend, n_dims):
    grid = fr.grid.CartesianGrid([64]*n_dims, [1.0]*n_dims, shared_axes=[0])
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset

@pytest.fixture()
def shape_phy(mset):
    return mset.grid.subdomain_phy.shape

@pytest.fixture()
def shape_spe(mset):
    return mset.grid.subdomain_spe.shape

@pytest.fixture(params=[True, False], ids=["Spectral", "Physical"])
def spectral(request):
    return request.param

@pytest.fixture()
def shape(spectral, shape_phy, shape_spe):
    return shape_spe if spectral else shape_phy

@pytest.fixture()
def dtype(spectral):
    return np.complex128 if spectral else np.float64

# --------------------------------------------------------------
#  Testing
# --------------------------------------------------------------
def test_zeros(mset, spectral, dtype, n_dims, shape):
    """Test the FieldVariable() constructor with no input array."""
    fv = fr.FieldVariable(mset, is_spectral=spectral)
    assert fv.mset == mset
    assert fv.grid == mset.grid
    assert len(fv.shape) == n_dims
    assert fv.is_spectral == spectral
    assert fv.dtype == dtype
    ncp = config.ncp
    arr = ncp.zeros(shape, dtype=dtype)
    assert ncp.allclose(fv[:], arr)

def test_constructor_with_input(mset, spectral, dtype, shape):
    ncp = config.ncp
    arr = ncp.ones(shape, dtype=dtype)
    fv = fr.FieldVariable(mset, is_spectral=spectral, arr=arr)
    assert ncp.allclose(fv[:], arr)

def teste_copy(mset, spectral):
    ncp = config.ncp
    fv = fr.FieldVariable(mset, is_spectral=spectral)
    copy = deepcopy(fv)
    # Test that the copy is not the same object
    assert fv is not copy
    # Test that the copy has the same data
    assert ncp.allclose(fv[:], copy[:])
    # Change the copy and test that the original is not changed
    copy[:] = 1.0
    assert not ncp.allclose(fv[:], copy[:])

@pytest.fixture()
def random_fields_real(mset, shape_phy):
    ncp = config.ncp
    field = fr.FieldVariable(mset, is_spectral=False, name="Test")
    field.arr = ncp.random.rand(*shape_phy)
    field.sync()
    return field

def test_fft(random_fields_real):
    field_hat = random_fields_real.fft()
    # Check that the field is now spectral
    assert field_hat.is_spectral
    # Check that the type of the field is complex
    assert field_hat.dtype == config.dtype_comp

def test_fft_ifft(random_fields_real):
    u = random_fields_real
    v = u.fft()
    w = v.fft()

    # Test that the fft is the inverse of itself
    ncp = config.ncp
    assert ncp.allclose(u.arr, w.arr)

def test_sqrt(random_fields_real):
    field = random_fields_real
    field_sqrt = field.sqrt()
    # Check that the square of the field is the original field
    ncp = config.ncp
    assert ncp.allclose(field.arr, field_sqrt.arr**2)

def test_norm_l2(random_fields_real):
    field = random_fields_real
    norm = field.norm_l2()
    ncp = config.ncp
    assert norm == ncp.linalg.norm(field.arr)

@pytest.fixture(params=[True, False], ids=["Periodic", "Non-periodic"])
def periodic(request):
    return request.param

@pytest.fixture()
def mset_131(backend, periodic):
    grid = fr.grid.CartesianGrid(N=[1, 3, 1], L=[1, 1, 1], 
                                 periodic_bounds=[periodic]*3)
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset

@pytest.fixture()
def field_123(mset_131):
    field = fr.FieldVariable(mset_131, is_spectral=False, name="Test")
    ncp = config.ncp
    field.arr[0,:,0] = ncp.array([1,2,3])
    return field

@pytest.mark.mpi_skip
def test_pad_raw(field_123):
    ncp = config.ncp
    pad = field_123.pad_raw(pad_width=((0,0),(0,0),(0,0)))
    assert ncp.allclose(pad, field_123.arr)
    pad = field_123.pad_raw(pad_width=((0,0),(1,1),(0,0)))
    aim = [3,1,2,3,1]
    assert ncp.allclose(pad[0,:,0], aim)

@pytest.fixture(params=[1, -1], ids=["Forward", "Backward"])
def direction(request):
    return request.param

@pytest.mark.mpi_skip
def test_average(field_123, direction, periodic):
    if direction == 1 and periodic:
        aim = [1.5, 2.5, 2]
    elif direction == 1 and not periodic:
        aim = [1.5, 2.5, 1.5]
    elif direction == -1 and periodic:
        aim = [2, 1.5, 2.5]
    elif direction == -1 and not periodic:
        aim = [0.5, 1.5, 2.5]
    averaged = field_123.ave(axis=1, shift=direction)
    ncp = config.ncp
    assert ncp.allclose(averaged[0,:,0], aim)

@pytest.mark.mpi_skip
def test_diff_forward(field_123, periodic):
    if periodic:
        aim = [1*3, 1*3, -2*3]
    else:
        aim = [1*3, 1*3, -3*3]

    diff = field_123.diff_forward(axis=1)
    ncp = config.ncp
    assert ncp.allclose(diff.arr[0,:,0], aim)

@pytest.mark.mpi_skip
def test_diff_backward(field_123, periodic):
    if periodic:
        aim = [-2*3, 1*3, 1*3]
    else:
        aim = [1*3, 1*3, 1*3]

    diff = field_123.diff_backward(axis=1)
    ncp = config.ncp
    assert ncp.allclose(diff.arr[0,:,0], aim)

def test_setitem(random_fields_real, n_dims):
    field = random_fields_real
    field.arr *= 0
    ncp = config.ncp
    assert ncp.allclose(field[:], 0)

    # test a single value
    ind = [0]*n_dims
    field[ind] = 1.0
    assert ncp.allclose(field.arr[ind], 1.0)

    # test a slice
    ind = [slice(None)]
    if n_dims > 1:
        ind = [0]*(n_dims-1) + ind
    ind = tuple(ind)
    field[ind] = 2.0
    assert ncp.allclose(field.arr[ind], 2.0)

@pytest.fixture()
def mset_3(backend, n_dims):
    grid = fr.grid.CartesianGrid(N=[3]*n_dims, L=[1]*n_dims)
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset

@pytest.fixture()
def zeros(mset_3):
    field = fr.FieldVariable(mset_3, is_spectral=False, name="Test")
    return field

@pytest.fixture()
def ones(mset_3):
    ncp = config.ncp
    field = fr.FieldVariable(mset_3, is_spectral=False, name="Test",
                             arr=ncp.ones(mset_3.grid.N))
    return field

@pytest.mark.mpi_skip
def test_add(zeros, ones):
    ncp = config.ncp
    # test sum with scalar
    sum = ones + 1.0
    assert ncp.allclose(sum.arr, 2.0)
    # test sum with FieldVariable
    sum = zeros + ones
    assert ncp.allclose(sum.arr, 1.0)
    # test sum with array
    sum = zeros + ncp.ones(zeros.shape)
    assert ncp.allclose(sum.arr, 1.0)

@pytest.mark.mpi_skip
def test_radd(zeros, ones):
    ncp = config.ncp
    # test sum with scalar
    sum = 1.0 + ones
    assert ncp.allclose(sum.arr, 2.0)

@pytest.mark.mpi_skip
def test_sub(zeros, ones):
    ncp = config.ncp
    # test difference with scalar
    diff = ones - 1.0
    assert ncp.allclose(diff.arr, 0.0)
    # test difference with FieldVariable
    diff = zeros - ones
    assert ncp.allclose(diff.arr, -1.0)
    # test difference with array
    diff = zeros - ncp.ones(ones.shape)
    assert ncp.allclose(diff.arr, -1.0)

@pytest.mark.mpi_skip
def test_rsub(zeros, ones):
    ncp = config.ncp
    # test difference with scalar
    diff = 1.0 - ones
    assert ncp.allclose(diff.arr, 0.0)

@pytest.mark.mpi_skip
def test_mul(zeros, ones):
    ncp = config.ncp
    # test product with scalar
    prod = ones * 2.0
    assert ncp.allclose(prod.arr, 2.0)
    # test product with FieldVariable
    prod = (ones * 2.0) * (ones * 2.0)
    assert ncp.allclose(prod.arr, 4.0)
    # test product with array
    prod = ones * (ncp.ones(ones.shape)*3)
    assert ncp.allclose(prod.arr, 3.0)

@pytest.mark.mpi_skip
def test_rmul(zeros, ones):
    ncp = config.ncp
    # test product with scalar
    prod = 2.0 * ones
    assert ncp.allclose(prod.arr, 2.0)

@pytest.mark.mpi_skip
def test_truediv(zeros, ones):
    ncp = config.ncp
    # test division with scalar
    div = ones / 2.0
    assert ncp.allclose(div.arr, 0.5)
    # test division with FieldVariable
    div = (ones / 2.0) / (ones / 4.0)
    assert ncp.allclose(div.arr, 2.0)
    # test division with array
    div = ones / (ncp.ones(ones.shape)*3)
    assert ncp.allclose(div.arr, 1/3)

@pytest.mark.mpi_skip
def test_rtruediv(zeros, ones):
    ncp = config.ncp
    # test division with scalar
    div = 2.0 / ones
    assert ncp.allclose(div.arr, 2.0)

@pytest.mark.mpi_skip
def test_pow(zeros, ones):
    ncp = config.ncp
    # test power with scalar
    power = (ones*2)**2
    assert ncp.allclose(power.arr, 4.0)
    # test power with FieldVariable
    power = (ones*2) ** (ones*2)
    assert ncp.allclose(power.arr, 4.0)
    # test power with array
    power = (ones*2) ** (ncp.ones(ones.shape)*3)
    assert ncp.allclose(power.arr, 8.0)