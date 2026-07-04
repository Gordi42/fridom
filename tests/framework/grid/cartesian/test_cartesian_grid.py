import jax.numpy as jnp
import numpy as np
import pytest

import fridom.framework as fr

# --------------------------------------------------------------
#  Create fixtures for the tests
# --------------------------------------------------------------

# skip n_dims=1 if parallel
@pytest.fixture(
        params=[pytest.param(1, id="1D"),
                pytest.param(2, id="2D")])
def n_dims(request):
    return request.param

@pytest.fixture
def L(n_dims):
    match n_dims:
        case 1:
            return (1.0, )
        case 2:
            return (1.0, 2.0)

@pytest.fixture
def N(n_dims):
    match n_dims:
        case 1:
            return (64, )
        case 2:
            return (32, 128)

@pytest.fixture
def dx(L, N):
    return [li/ni for li, ni in zip(L, N, strict=False)]

@pytest.fixture
def grid(L, N):
    grid = fr.grid.cartesian.Grid(N, L)
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset.grid

# --------------------------------------------------------------
#  Testing
# --------------------------------------------------------------

def test_array_type(grid):
    x = grid.x_mesh[0]
    assert isinstance(x, jnp.ndarray)

def test_x(grid, n_dims, N, L, dx):
    x = grid.x_global
    assert len(x) == n_dims
    for i in range(n_dims):
        assert len(x[i]) == N[i]
        assert x[i][1] - x[i][0] == dx[i]
        assert x[i][-1] == L[i] - dx[i]/2.0

def test_X(grid, n_dims):
    X = grid.x_mesh
    assert len(X) == n_dims

def test_k(grid, n_dims, N, L):
    k = grid.k_global
    assert len(k) == n_dims
    for i in range(n_dims):
        assert k[i][0] == 0
        k_max = np.pi * N[i] / L[i]
        assert max(jnp.abs(k[i])) == k_max

def test_K(grid, n_dims):
    K = grid.k_mesh
    assert len(K) == n_dims
