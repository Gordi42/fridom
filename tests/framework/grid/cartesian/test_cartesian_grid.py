import pytest
from mpi4py import MPI
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
                pytest.param(2, id="2D")])
def n_dims(request):
    return request.param

@pytest.fixture()
def L(n_dims):
    match n_dims:
        case 1:
            return [1.0]
        case 2:
            return [1.0, 2.0]

@pytest.fixture()
def N(n_dims):
    match n_dims:
        case 1:
            return [64]
        case 2:
            return [32, 128]
        
@pytest.fixture()
def dx(L, N):
    return [li/ni for li, ni in zip(L, N)]

@pytest.fixture()
def grid(backend, L, N):
    grid = fr.grid.CartesianGrid(N, L, shared_axes=[0])
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset.grid

@pytest.fixture()
def local_shape_phy(grid):
    return grid.get_subdomain().shape

@pytest.fixture()
def local_shape_spe(grid):
    return grid.get_subdomain(spectral=True).shape

# --------------------------------------------------------------
#  Testing
# --------------------------------------------------------------

def test_backend(grid):
    x = grid.X[0]
    assert isinstance(x, config.ncp.ndarray)

def test_x(grid, n_dims, N, L, dx):
    x = grid.x_global
    assert len(x) == n_dims
    for i in range(n_dims):
        assert len(x[i]) == N[i]
        assert x[i][1] - x[i][0] == dx[i]
        assert x[i][-1] == L[i] - dx[i]/2.0

def test_X(grid, n_dims, local_shape_phy):
    X = grid.X
    assert len(X) == n_dims
    for i in range(n_dims):
        assert X[i].shape == local_shape_phy

def test_k(grid, n_dims, N, L):
    k = grid.k_global
    ncp = config.ncp
    assert len(k) == n_dims
    for i in range(n_dims):
        assert k[i][0] == 0
        k_max = np.pi * N[i] / L[i]
        assert max(ncp.abs(k[i])) == k_max

def test_K(grid, n_dims, local_shape_spe):
    K = grid.K
    assert len(K) == n_dims
    for i in range(n_dims):
        assert K[i].shape == local_shape_spe