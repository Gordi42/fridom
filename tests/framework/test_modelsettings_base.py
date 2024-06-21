import pytest
import numpy as np
from fridom.framework import ModelSettingsBase

@pytest.fixture(scope='class', params=[1,2,3], ids=['1D', '2D', '3D'])
def n_dims(request):
    return request.param


def test_constructor(enable_gpu, n_dims):
    mset = ModelSettingsBase(n_dims, gpu=enable_gpu)
    assert mset.n_dims == n_dims
    assert mset.dtype == np.float64
    assert mset.ctype == np.complex128
    assert mset.model_name == "Unnamed model"
    assert mset.gpu == enable_gpu

def test_set_L(enable_gpu, n_dims):
    mset = ModelSettingsBase(n_dims, gpu=enable_gpu)
    L = [128.0]*n_dims
    mset.L = L
    assert mset.L == L
    N = mset.N
    # make sure that the grid spacing is updated
    dg = [li / ni for li, ni in zip(L, N)]
    assert mset.dg == dg

def test_set_N(enable_gpu, n_dims):
    mset = ModelSettingsBase(n_dims, gpu=enable_gpu)
    N = [16]*n_dims
    mset.N = N
    assert mset.N == N
    L = mset.L
    dg = [li / ni for li, ni in zip(L, N)]
    assert mset.dg == dg
    assert mset.total_grid_points == np.prod(N)

def test_doc_example(enable_gpu):
    class MyModelSettings(ModelSettingsBase):
        def __init__(self, n_dims:int, dtype=np.float64, ctype=np.complex128,
                     **kwargs) -> None:
            super().__init__(n_dims, dtype=dtype, ctype=ctype)
            self.model_name = "MyModel"
            self.L = [2*np.pi, 2*np.pi]
            self.N = [63, 63]
            self.my_parameter = 1.0
            self.set_attributes(**kwargs)
        def __str__(self) -> str:
            res = super().__str__()
            res += "  My parameter: {}\n".format(self.my_parameter)
            return res

    mset = MyModelSettings(2, gpu=enable_gpu, L=[4*np.pi, 4*np.pi], 
                           my_parameter=2.0)
    assert mset.model_name == "MyModel"
    assert mset.L == [4*np.pi, 4*np.pi]
    assert mset.N == [63, 63]
    assert mset.my_parameter == 2.0
    assert mset.gpu == enable_gpu
    assert "My parameter: 2" in str(mset)