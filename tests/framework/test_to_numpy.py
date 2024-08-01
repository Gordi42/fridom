import fridom.framework as fr
from fridom.framework.utils import to_numpy
import numpy as np
import pytest

@pytest.mark.parametrize("variable", [1, 1.0, "string", 'c'])
def test_standard_variable_to_numpy(backend, variable):
    np.variable = to_numpy(variable)
    assert np.variable == variable


def test_cupy_array_to_numpy(backend):
    x = fr.utils.random_array((10,))
    if backend != fr.config.Backend.NUMPY:
        assert not isinstance(x, np.ndarray)
    y = to_numpy(x)
    assert isinstance(y, np.ndarray)

def test_cupy_list_to_numpy(backend):
    x = [fr.utils.random_array((10,)) for _ in range(3)]
    if backend != fr.config.Backend.NUMPY:
        assert not isinstance(x[0], np.ndarray)
    y = to_numpy(x)
    for yi in y:
        assert isinstance(yi, np.ndarray)

def test_cupy_tuple_to_numpy(backend):
    x = tuple([fr.utils.random_array((10,)) for _ in range(3)])
    if backend != fr.config.Backend.NUMPY:
        assert not isinstance(x[0], np.ndarray)
    y = to_numpy(x)
    for yi in y:
        assert isinstance(yi, np.ndarray)

def test_recursion(backend):
    x = fr.utils.random_array((10,))
    y = [x, x]
    z = to_numpy(y)
    assert isinstance(z[0], np.ndarray)
    assert y[0] is y[1]
    assert z[0] is z[1]

def test_mset_to_numpy(backend):
    grid = fr.grid.cartesian.Grid(N=(32, 32, 8), L=(1, 1, 1))
    mset = fr.ModelSettingsBase(grid)
    mset.setup()

    mset_cpu = to_numpy(mset)
    grid_cpu = mset_cpu.grid

    assert isinstance(grid_cpu.X[0], np.ndarray)
