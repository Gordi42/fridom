"""Test the nan_checker module."""
import numpy as np
import pytest

import fridom.framework as fr


@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(128, 64), domain_size=(1, 1))
    mset = fr.ModelSettingsBase(grid=grid)
    mset.setup()
    def _state_constructor() -> fr.VectorField:
        var1 = fr.ScalarField(
            mset, name="var1", long_name="Variable 1", units="unit1")
        var2 = fr.ScalarField(
            mset, name="var2", long_name="Variable 2", units="unit2")
        return fr.VectorField(mset, field_list=[var1, var2])
    mset.state_constructor = _state_constructor
    return mset

def test_nan_checker_no_nan(mset):
    nan_checker = fr.modules.NaNChecker()
    # set the clock trigger to trigger every step
    nan_checker.clock_trigger = fr.ClockTrigger()
    model_state = fr.ModelState(mset)
    model_state = nan_checker.update(model_state)
    assert not model_state.panicked

def test_nan_checker_nan(mset):
    nan_checker = fr.modules.NaNChecker()
    # set the clock trigger to trigger every step
    nan_checker.clock_trigger = fr.ClockTrigger()
    model_state = fr.ModelState(mset)
    var1 = model_state.z["var1"]
    var1.arr = fr.utils.modify_array(var1.arr, (0, 0), np.nan)
    model_state = nan_checker.update(model_state)
    assert model_state.panicked

@pytest.mark.parametrize(*(
    "number_of_steps, step_size, expected_panic",
    [(9, 10, False), (10, 10, True)],
))
def test_nan_checker_clock_trigger(number_of_steps, step_size, expected_panic, mset):
    clock_trigger = fr.ClockTrigger(step_size=step_size)
    nan_checker = fr.modules.NaNChecker(clock_trigger=clock_trigger)
    model_state = fr.ModelState(mset)
    var1 = model_state.z["var1"]
    var1.arr = fr.utils.modify_array(var1.arr, (0, 0), np.nan)
    # clock_trigger always triggers at the first step so we call it manually
    nan_checker.clock_trigger.check(model_state.clock)
    for _ in range(number_of_steps):
        model_state.clock.tick(1.0)
        model_state = nan_checker.update(model_state)
    assert model_state.panicked == expected_panic

@pytest.mark.parametrize("reset", [True, False])
def test_reset(reset, mset):
    nan_checker = fr.modules.NaNChecker(fr.ClockTrigger(step_size=10))
    model_state = fr.ModelState(mset)
    var1 = model_state.z["var1"]
    var1.arr = fr.utils.modify_array(var1.arr, (0, 0), np.nan)
    for _ in range(3):
        nan_checker.clock_trigger.check(model_state.clock)
        for _ in range(8):
            model_state.clock.tick(1.0)
            model_state = nan_checker.update(model_state)
        if reset:
            nan_checker.reset()
    # if the nan_checker is reset, the clock_trigger starts over and over again
    # and will never reach the trigger of 10 steps
    assert model_state.panicked != reset
