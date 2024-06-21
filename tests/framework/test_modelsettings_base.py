import pytest
import numpy as np
from fridom.framework import ModelSettingsBase

@pytest.fixture(scope='class', params=[1,2,3], ids=['1D', '2D', '3D'])
def n_dims(request):
    return request.param


def test_constructor(self, enable_gpu, n_dims):
    mset = ModelSettingsBase(n_dims, gpu=enable_gpu)
    assert mset.n_dims == n_dims
    assert mset.dtype == np.float64
    assert mset.ctype == np.complex128
    assert mset.model_name == "Unnamed model"
    assert mset.gpu == enable_gpu

def test_set_L(self, enable_gpu, n_dims):
    mset = ModelSettingsBase(n_dims, gpu=enable_gpu)
    L = [128.0]*n_dims
    mset.L = L
    assert mset.L == L
    N = mset.N
    # make sure that the grid spacing is updated
    dg = [li / ni for li, ni in zip(L, N)]
    assert mset.dg == dg

def test_set_N(self, enable_gpu, n_dims):
    mset = ModelSettingsBase(n_dims, gpu=enable_gpu)
    N = [16]*n_dims
    mset.N = N
    assert mset.N == N
    L = mset.L
    dg = [li / ni for li, ni in zip(L, N)]
    assert mset.dg == dg
    assert mset.total_grid_points == np.prod(N)