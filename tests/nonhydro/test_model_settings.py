import pytest
import fridom.nonhydro as nh

@pytest.fixture(params=[[64, 64, 64], [64, 53, 33]], ids=['64x64x64', '64x53x33'])
def N(request):
    return request.param

@pytest.fixture(params=[[1.0, 1.0, 1.0], [1.0, 2.0, 3.0]], ids=['1x1x1', '1x2x3'])
def L(request):
    return request.param

@pytest.fixture()
def grid_ini(backend, N, L):
    return nh.grid.CartesianGrid(N, L)

@pytest.fixture(params=[1e-4, 2], ids=["f=1e-4", "f=2"])
def f(request):
    return request.param

@pytest.fixture(params=[1e-4, 2], ids=["N2=1e-4", "N2=2"])
def N2(request):
    return request.param

def test_model_settings(grid_ini, f, N2):
    mset = nh.ModelSettings(grid_ini, f_coriolis=f, N2=N2)
    assert mset.grid == grid_ini
    assert mset.f_coriolis == f
    assert mset.N2 == N2