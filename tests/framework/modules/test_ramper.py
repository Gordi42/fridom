"""Test of the ramper module."""
from unittest.mock import MagicMock

import pytest

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(N=(4, ), L=(1, ))
    return fr.ModelSettingsBase(grid).setup()

@pytest.fixture(params=[
    "exponential",
    "power_3",
    "cosine",
    "linear",
    lambda x: x**2,
    ])
def ramp_func(request):
    return request.param

# ================================================================
#  Tests
# ================================================================
def test_ramp_function(ramp_func):
    ramper = fr.modules.Ramper(0, 1, ramp_function=ramp_func)
    # check if the value at t = 0 is 0
    assert ramper.ramp_function(0) == 0
    # check if the value at t = 1 is 1
    assert ramper.ramp_function(1) == 1

@pytest.mark.parametrize(*("start_time, ramp_period", [
    pytest.param(0, 1),
    pytest.param(1, 4),
    pytest.param(0.5, 0.8),
]))
def test_update(mset, ramp_func, start_time, ramp_period):
    def update_parameters(_mz, ramped_value) -> None:
        mset.my_parameter = ramped_value

    # Wrap the update_parameters method to check if it is called
    update_parameters = MagicMock(wraps=update_parameters)
    ramper = fr.modules.Ramper(start_time,
                               ramp_period,
                               update_parameters,
                               ramp_func)

    # Set the clock time to the start time
    delta_time = 1e-3
    mz = fr.ModelState(mset)
    mz.clock.time = start_time - delta_time  # just before the start time

    # Check if the parameter is not updated
    ramper.update(mz)
    assert not update_parameters.called

    # Check calling update with the start time
    mz.clock.time = start_time
    ramper.update(mz)
    assert update_parameters.called
    expected_parameter = 0
    assert mset.my_parameter == expected_parameter

    # Check calling update with the start_time + ramp_period
    mz.clock.time = start_time + ramp_period
    ramper.update(mz)
    expected_call_count = 2
    assert update_parameters.call_count == expected_call_count
    expected_parameter = 1
    assert mset.my_parameter == expected_parameter

    # Check calling update with a time after the ramp period
    mz.clock.time = start_time + ramp_period + delta_time
    ramper.update(mz)
    assert update_parameters.call_count == expected_call_count
    assert mset.my_parameter == expected_parameter

def test_custom_ramp_function():
    my_custom_ramp_func = MagicMock(wraps=lambda x: x)
    my_custom_ramp_func.__name__ = "test_func"
    ramper = fr.modules.Ramper(0, 1, ramp_function=my_custom_ramp_func)

    # call the ramp_runction method and check if the custom function is called
    ramper.ramp_function(0)
    assert my_custom_ramp_func.called
    assert ramper._ramp_name == "test_func"

def test_to_string(ramp_func):
    ramper = fr.modules.Ramper(3, 5, ramp_function=ramp_func)
    ramp_str = str(ramper)
    assert "Ramper" in ramp_str
    assert "start_time: 3" in ramp_str
    assert "ramp_period: 5" in ramp_str
    assert "ramp_function" in ramp_str
