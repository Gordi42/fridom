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

# --------------------------------------------------------------
#  Construction errors
# --------------------------------------------------------------

def test_mismatched_shape_and_domain_size_raises():
    with pytest.raises(ValueError, match="same number of dimensions"):
        fr.grid.cartesian.Grid(shape=(16, 16), domain_size=(1.0,))

def test_mismatched_periodic_bounds_raises():
    with pytest.raises(ValueError, match="periodic_bounds"):
        fr.grid.cartesian.Grid(shape=(16, 16), domain_size=(1.0, 1.0),
                               periodic_bounds=(True,))

# --------------------------------------------------------------
#  Setup variants
# --------------------------------------------------------------

def test_setup_with_new_halo():
    grid = fr.grid.cartesian.Grid(shape=(8, 8), domain_size=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    assert grid.halo == 1

    # a new setup with a different halo reconstructs the decomposition
    grid.setup(mset, req_halo=3)
    assert grid.halo == 3

def test_setup_without_fourier_transform():
    grid = fr.grid.cartesian.Grid(shape=(8, 8), domain_size=(1.0, 1.0))
    grid.fourier_transform_available = False
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    assert grid.k_mesh is None
    assert grid.k_global is None

def test_extend_topo():
    grid = fr.grid.cartesian.Grid(shape=(8, 8), domain_size=(1.0, 1.0))
    assert grid._extend_topo((False, True), (0,)) == (True, True)

# --------------------------------------------------------------
#  Reductions
# --------------------------------------------------------------

@pytest.fixture
def wave_field():
    grid = fr.grid.cartesian.Grid(shape=(16, 16), domain_size=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    f = fr.ScalarField(mset, name="f")
    x, _y = grid.x_mesh
    f.arr = jnp.sin(2 * jnp.pi * 2 * x)
    return f.sync()

def test_max_and_min(wave_field):
    grid = wave_field.mset.grid
    maximum = grid.max(wave_field)
    minimum = grid.min(wave_field)
    assert float(maximum.arr.squeeze()) == pytest.approx(
        float(wave_field.arr.max()))
    assert float(minimum.arr.squeeze()) == pytest.approx(
        float(wave_field.arr.min()))

# --------------------------------------------------------------
#  Property setters
# --------------------------------------------------------------

def test_domain_size_and_shape_setters():
    grid = fr.grid.cartesian.Grid(shape=(16, 16), domain_size=(1.0, 1.0))

    grid.domain_size = (2.0, 2.0)
    assert grid.dx == (0.125, 0.125)

    grid.shape = (8, 8)
    assert grid.dx == (0.25, 0.25)
    assert grid.total_grid_points == 64

def test_k_local():
    grid = fr.grid.cartesian.Grid(shape=(8, 8), domain_size=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    # the local k-vectors are not set on a single process
    assert grid.k_local is None or len(grid.k_local) == 2
