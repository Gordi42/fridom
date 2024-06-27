import pytest
import numpy as np
from fridom.framework import ModelSettingsBase, GridBaseOld, FieldVariable
from fridom.framework.boundary_conditions \
    import Periodic, Dirichlet, Neumann, BoundaryConditions

# --------------------------------------------------------------
#  Create fixtures for the tests
# --------------------------------------------------------------

@pytest.fixture(params=[1, 2, 3], ids=['1D', '2D', '3D'])
def n_dims(request):
    return request.param

@pytest.fixture(params=[True, False], ids=["Spectral", "Physical"])
def spectral(request):
    return request.param

@pytest.fixture()
def dtype(spectral):
    return np.complex128 if spectral else np.float64

@pytest.fixture()
def mset(enable_gpu, n_dims):
    return ModelSettingsBase(n_dims, gpu=enable_gpu)

@pytest.fixture()
def grid(mset):
    return GridBaseOld(mset)


def test_zeros(mset, grid, spectral, dtype, n_dims):
    """Test the FieldVariable() constructor with no input array."""
    fv = FieldVariable(grid, is_spectral=spectral)
    assert fv.grid == grid
    assert fv.mset == mset
    assert len(fv.shape) == n_dims
    assert fv.is_spectral == spectral
    assert fv.dtype == dtype
    cp = grid.cp
    arr = cp.zeros(mset.N, dtype=dtype)
    assert cp.allclose(fv[:], arr)


def test_constructor_with_input(mset, grid, spectral, dtype):
    cp = grid.cp
    arr = cp.ones(mset.N, dtype=dtype)
    fv = FieldVariable(grid, is_spectral=spectral, arr=arr)
    assert cp.allclose(fv[:], arr)

def test_copy(grid, spectral):
    cp = grid.cp
    fv = FieldVariable(grid, is_spectral=spectral)
    copy = fv.copy()
    # Test that the copy is not the same object
    assert fv is not copy
    # Test that the copy has the same grid and mset
    assert fv.mset is copy.mset
    assert fv.grid is copy.grid
    # Test that the copy has the same data
    assert cp.allclose(fv[:], copy[:])
    # Change the copy and test that the original is not changed
    copy[:] = 1.0
    assert not cp.allclose(fv[:], copy[:])

@pytest.fixture(params=[Periodic, Dirichlet, Neumann], 
                ids=["Periodic", "Dirichlet", "Neumann"])
def boundary_condition(request):
    return request.param

@pytest.fixture()
def periodic(boundary_condition):
    return boundary_condition == Periodic

@pytest.fixture()
def random_fields_real(enable_gpu, n_dims, boundary_condition, periodic):
    m = ModelSettingsBase(n_dims, gpu=enable_gpu, 
                          periodic_bounds=[periodic]*n_dims)
    # create array of random integers between 1 and 10
    m.N = [np.random.randint(1,10) for i in range(n_dims)]
    bounds = [boundary_condition(m, i, 1) for i in range(n_dims)]
    bc = BoundaryConditions(bounds)
    g = GridBaseOld(m)
    cp = g.cp
    field = FieldVariable(
        g, is_spectral=False, name="Test", bc=bc)
    field.arr = cp.random.rand(*m.N)
    return field

def test_fft(random_fields_real):
    field_hat = random_fields_real.fft()
    # Check that the field is now spectral
    assert field_hat.is_spectral
    # Check that the type of the field is complex
    assert field_hat.dtype == np.complex128

def test_fft_ifft(random_fields_real):
    field = random_fields_real
    field_hat = field.fft()
    field_hat_hat = field_hat.fft()

    # Test that the fft is the inverse of itself
    cp = field.grid.cp
    assert cp.allclose(field.arr, field_hat_hat.arr)

def test_sqrt(random_fields_real):
    field = random_fields_real
    field_sqrt = field.sqrt()
    # Check that the square of the field is the original field
    cp = field.grid.cp
    assert cp.allclose(field.arr, field_sqrt.arr**2)

def test_norm_l2(random_fields_real):
    field = random_fields_real
    norm = field.norm_l2()
    cp = field.grid.cp
    assert norm == cp.linalg.norm(field.arr)

def test_pad_raw(enable_gpu):
    m = ModelSettingsBase(n_dims=3, gpu=enable_gpu)
    m.N = [1, 3, 1]
    g = GridBaseOld(m)
    cp = g.cp
    field = FieldVariable(g, is_spectral=False, name="Test")
    field.arr[0,:,0] = cp.array([1,2,3])

    pad = field.pad_raw(pad_width=((0,0),(0,0),(0,0)))
    assert cp.allclose(pad, field.arr)

    pad = field.pad_raw(pad_width=((0,0),(1,1),(0,0)))
    aim = [3,1,2,3,1]
    assert cp.allclose(pad[0,:,0], aim)

@pytest.fixture(params=[True, False], ids=["Periodic", "Non-Periodic"])
def periodic(request):
    return request.param

@pytest.fixture(params=[1, -1], ids=["Forward", "Backward"])
def direction(request):
    return request.param

@pytest.fixture()
def field(enable_gpu, periodic):
    m = ModelSettingsBase(n_dims=3, gpu=enable_gpu)
    m.N = [1, 3, 1]
    m.periodic_bounds = [periodic]*3
    g = GridBaseOld(m)
    cp = g.cp
    field = FieldVariable(g, is_spectral=False, name="Test")
    field.arr[0,:,0] = cp.array([1,2,3])
    return field

def test_average(field, direction, periodic):
    if direction == 1 and periodic:
        aim = [1.5, 2.5, 2]
    elif direction == 1 and not periodic:
        aim = [1.5, 2.5, 1.5]
    elif direction == -1 and periodic:
        aim = [2, 1.5, 2.5]
    elif direction == -1 and not periodic:
        aim = [0.5, 1.5, 2.5]
    averaged = field.ave(axis=1, shift=direction)
    cp = field.grid.cp
    assert cp.allclose(averaged[0,:,0], aim)

def test_diff_forward(field, periodic):
    if periodic:
        aim = [1*3, 1*3, -2*3]
    else:
        aim = [1*3, 1*3, -3*3]

    diff = field.diff_forward(axis=1)
    cp = field.grid.cp
    assert cp.allclose(diff.arr[0,:,0], aim)

def test_diff_backward(field, periodic):
    if periodic:
        aim = [-2*3, 1*3, 1*3]
    else:
        aim = [1*3, 1*3, 1*3]

    diff = field.diff_backward(axis=1)
    cp = field.grid.cp
    assert cp.allclose(diff.arr[0,:,0], aim)

def test_setitem(random_fields_real, n_dims):
    field = random_fields_real
    field.arr *= 0
    cp = field.grid.cp
    assert cp.allclose(field[:], 0)

    # test a single value
    ind = [0]*n_dims
    field[ind] = 1.0
    assert cp.allclose(field.arr[ind], 1.0)

    # test a slice
    ind = [slice(None)]
    if n_dims > 1:
        ind = [0]*(n_dims-1) + ind
    ind = tuple(ind)
    field[ind] = 2.0
    assert cp.allclose(field.arr[ind], 2.0)

def test_getitem():
    # TODO
    pass

def test_getattr():
    # TODO
    pass

@pytest.fixture()
def zeros(enable_gpu, n_dims):
    m = ModelSettingsBase(n_dims, gpu=enable_gpu)
    m.N = [3] * n_dims
    g = GridBaseOld(m)
    cp = g.cp
    field = FieldVariable(g, is_spectral=False, name="Test")
    return field

@pytest.fixture()
def ones(enable_gpu, n_dims):
    m = ModelSettingsBase(n_dims, gpu=enable_gpu)
    m.N = [3] * n_dims
    g = GridBaseOld(m)
    cp = g.cp
    field = FieldVariable(g, is_spectral=False, name="Test")
    field.arr = cp.ones(m.N)
    return field

def test_add(zeros, ones):
    cp = zeros.grid.cp
    # test sum with scalar
    sum = ones + 1.0
    assert cp.allclose(sum.arr, 2.0)
    # test sum with FieldVariable
    sum = zeros + ones
    assert cp.allclose(sum.arr, 1.0)
    # test sum with array
    sum = zeros + cp.ones(zeros.shape)
    assert cp.allclose(sum.arr, 1.0)

def test_radd(zeros, ones):
    cp = zeros.grid.cp
    # test sum with scalar
    sum = 1.0 + ones
    assert cp.allclose(sum.arr, 2.0)

def test_sub(zeros, ones):
    cp = zeros.grid.cp
    # test difference with scalar
    diff = ones - 1.0
    assert cp.allclose(diff.arr, 0.0)
    # test difference with FieldVariable
    diff = zeros - ones
    assert cp.allclose(diff.arr, -1.0)
    # test difference with array
    diff = zeros - cp.ones(ones.shape)
    assert cp.allclose(diff.arr, -1.0)

def test_rsub(zeros, ones):
    cp = zeros.grid.cp
    # test difference with scalar
    diff = 1.0 - ones
    assert cp.allclose(diff.arr, 0.0)

def test_mul(zeros, ones):
    cp = zeros.grid.cp
    # test product with scalar
    prod = ones * 2.0
    assert cp.allclose(prod.arr, 2.0)
    # test product with FieldVariable
    prod = (ones * 2.0) * (ones * 2.0)
    assert cp.allclose(prod.arr, 4.0)
    # test product with array
    prod = ones * (cp.ones(ones.shape)*3)
    assert cp.allclose(prod.arr, 3.0)

def test_rmul(zeros, ones):
    cp = zeros.grid.cp
    # test product with scalar
    prod = 2.0 * ones
    assert cp.allclose(prod.arr, 2.0)

def test_truediv(zeros, ones):
    cp = zeros.grid.cp
    # test division with scalar
    div = ones / 2.0
    assert cp.allclose(div.arr, 0.5)
    # test division with FieldVariable
    div = (ones / 2.0) / (ones / 4.0)
    assert cp.allclose(div.arr, 2.0)
    # test division with array
    div = ones / (cp.ones(ones.shape)*3)
    assert cp.allclose(div.arr, 1/3)

def test_rtruediv(zeros, ones):
    cp = zeros.grid.cp
    # test division with scalar
    div = 2.0 / ones
    assert cp.allclose(div.arr, 2.0)

def test_pow(zeros, ones):
    cp = zeros.grid.cp
    # test power with scalar
    power = (ones*2)**2
    assert cp.allclose(power.arr, 4.0)
    # test power with FieldVariable
    power = (ones*2) ** (ones*2)
    assert cp.allclose(power.arr, 4.0)
    # test power with array
    power = (ones*2) ** (cp.ones(ones.shape)*3)
    assert cp.allclose(power.arr, 8.0)