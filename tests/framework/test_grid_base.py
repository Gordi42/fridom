import pytest
import numpy as np
from fridom.framework import ModelSettingsBase, GridBase

# --------------------------------------------------------------
#  Create fixtures for the tests
# --------------------------------------------------------------

@pytest.fixture(params=[1, 2], ids=['1D', '2D'])
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
def dg(L, N):
    return [li/ni for li, ni in zip(L, N)]

@pytest.fixture()
def mset(n_dims, L, N, enable_gpu):
    settings = ModelSettingsBase(n_dims, gpu=enable_gpu)
    settings.L = L
    settings.N = N
    return settings

@pytest.fixture()
def grid(mset):
    return GridBase(mset)

# --------------------------------------------------------------
#  Testing
# --------------------------------------------------------------

def test_backend(enable_gpu, grid):
    if enable_gpu:
        import cupy as cp
        assert grid.cp == cp
    else:
        assert grid.cp == np

def test_x(mset, grid):
    x = grid.x
    assert len(x) == mset.n_dims
    for i in range(mset.n_dims):
        assert len(x[i]) == mset.N[i]
        assert x[i][1] == mset.dg[i]

def test_X(mset, grid):
    X = grid.X
    assert len(X) == mset.n_dims
    for i in range(mset.n_dims):
        assert X[i].shape == tuple(mset.N)

def test_k(mset, grid):
    k = grid.k
    cp = grid.cp
    assert len(k) == mset.n_dims
    for i in range(mset.n_dims):
        assert k[i][0] == 0
        k_max = np.pi * mset.N[i] / mset.L[i]
        assert max(cp.abs(k[i])) == k_max

def test_K(mset, grid):
    K = grid.K
    assert len(K) == mset.n_dims
    for i in range(mset.n_dims):
        assert K[i].shape == tuple(mset.N)