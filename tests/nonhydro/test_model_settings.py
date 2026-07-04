import jax.numpy as jnp
import pytest

import fridom.nonhydro as nh


@pytest.fixture(params=[(16, 16, 16), (16, 13, 9)],
                ids=["16x16x16", "16x13x9"])
def N(request):
    return request.param

@pytest.fixture(params=[(1.0, 1.0, 1.0), (1.0, 2.0, 3.0)],
                ids=["1x1x1", "1x2x3"])
def L(request):
    return request.param

@pytest.fixture
def grid_ini(N, L):
    return nh.grid.cartesian.Grid(N, L)

@pytest.fixture(params=[1e-4, 2], ids=["f=1e-4", "f=2"])
def f(request):
    return request.param

@pytest.fixture(params=[1e-4, 2],
                ids=["stratification_n2=1e-4", "stratification_n2=2"])
def N2(request):
    return request.param

def test_model_settings(grid_ini, f, N2):
    mset = nh.ModelSettings(grid_ini, f0=f, stratification_n2=N2)
    mset.setup()
    assert mset.grid == grid_ini
    assert jnp.allclose(mset.f_coriolis.arr, f)
    assert mset.stratification_n2 == N2
