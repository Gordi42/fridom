import numpy as np
import pytest

import fridom.framework as fr
from fridom.framework.utils import to_numpy
from fridom.framework.utils.numpy_utils import _handle_iterable, to_seconds


@pytest.mark.parametrize("variable", [1, 1.0, "string", "c"])
def test_standard_variable_to_numpy(variable):
    np.variable = to_numpy(variable)
    assert np.variable == variable


def test_array_to_numpy():
    x = fr.utils.random_array((10,))
    assert not isinstance(x, np.ndarray)
    y = to_numpy(x)
    assert isinstance(y, np.ndarray)

def test_list_to_numpy():
    x = [fr.utils.random_array((10,)) for _ in range(3)]
    assert not isinstance(x[0], np.ndarray)
    y = to_numpy(x)
    for yi in y:
        assert isinstance(yi, np.ndarray)

def test_tuple_to_numpy():
    x = tuple([fr.utils.random_array((10,)) for _ in range(3)])
    assert not isinstance(x[0], np.ndarray)
    y = to_numpy(x)
    for yi in y:
        assert isinstance(yi, np.ndarray)

def test_recursion():
    x = fr.utils.random_array((10,))
    y = [x, x]
    z = to_numpy(y)
    assert isinstance(z[0], np.ndarray)
    assert y[0] is y[1]
    assert z[0] is z[1]

def test_mset_to_numpy():
    grid = fr.grid.cartesian.Grid(shape=(32, 32, 8), domain_size=(1, 1, 1))
    mset = fr.ModelSettingsBase(grid)
    mset.setup()

    mset_cpu = to_numpy(mset)
    grid_cpu = mset_cpu.grid

    assert isinstance(grid_cpu.x_mesh[0], np.ndarray)

def test_dict_to_numpy():
    x = {"a": fr.utils.random_array((10,)), "b": 1.0}
    y = to_numpy(x)
    assert isinstance(y["a"], np.ndarray)
    assert y["b"] == 1.0

def test_set_to_numpy():
    x = {1, 2, 3}
    y = to_numpy(x)
    assert y == x

def test_non_iterable_raises():
    with pytest.raises(TypeError, match="is not iterable"):
        _handle_iterable(42, memo={})

def test_custom_to_numpy_method():
    class MyObject:
        def __to_numpy__(self, memo):
            return "converted"

    assert to_numpy(MyObject()) == "converted"

def test_cpu_attribute_is_cached():
    class MyObject:
        def __init__(self):
            self._cpu = None
            self.arr = fr.utils.random_array((10,))

    obj = MyObject()

    # the first conversion caches the result in the _cpu attribute
    first = to_numpy(obj)
    assert isinstance(first.arr, np.ndarray)
    assert obj._cpu is first

    # the second conversion returns the cached object
    second = to_numpy(obj)
    assert second is first

@pytest.mark.parametrize(("time", "expected"), [
    pytest.param(2.5, 2.5, id="float"),
    pytest.param(np.timedelta64(2, "h"), 7200.0, id="hours"),
    pytest.param(np.timedelta64(90, "s"), 90.0, id="seconds"),
    pytest.param(np.timedelta64(500, "ms"), 0.5, id="milliseconds"),
    pytest.param(np.timedelta64(1, "W"), 7 * 24 * 3600.0, id="weeks"),
    pytest.param(np.datetime64("1970-01-01T01:00:00"), 3600.0,
                 id="datetime"),
])
def test_to_seconds(time, expected):
    assert to_seconds(time) == pytest.approx(expected)
