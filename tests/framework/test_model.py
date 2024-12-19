"""Test the main model class."""
from __future__ import annotations

import tempfile

import numpy as np
import pytest

import fridom.framework as fr


@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(N=(16, ), L=(1, ))
    mset = fr.ModelSettingsBase(grid)
    mset.setup()
    return mset

@pytest.mark.parametrize("module_name", [
    "timer",
    "nan_checker",
    "progress_bar",
    "restart_module",
    "time_stepper",
    "tendencies",
    "diagnostics",
])
def test_model_modules(mset, module_name):
    """Test if the model modules are immutable and point to the settings."""
    model = fr.Model(mset)
    assert hasattr(model, module_name)

    model_obj = getattr(model, module_name)
    mset_obj = getattr(mset, module_name)
    assert model_obj is mset_obj


@pytest.mark.parametrize(
        "start_step",
        [0, 1235],
)
def test_run_steps(mset, start_step):
    """Test the run method."""
    number_of_steps = 10
    counter_diag = fr.modules.Counter()
    counter_tend = fr.modules.Counter()
    mset.diagnostics.add_module(counter_diag)
    mset.tendencies.add_module(counter_tend)
    mset.setup()

    model = fr.Model(mset)
    model.run(steps=number_of_steps, start_step=start_step)

    # check that the counters are correct
    assert counter_diag.counter == number_of_steps
    assert counter_tend.counter == number_of_steps

    # check that the clock is correct
    clock = model.model_state.clock
    assert clock.it == start_step + number_of_steps
    assert np.isclose(clock._passed_time, number_of_steps * mset.time_stepper.dt)


@pytest.mark.parametrize(*(
    "start_time, runlen, expected_time",
    [
        (0, 10.0, 10.0),
        (0, np.timedelta64(10, "s"), 10.0),
        (np.datetime64("2021-01-01T00:00:00"),
         10.0,
         np.datetime64("2021-01-01T00:00:10")),
    ],
))
def test_correct_clock_runlen(mset, start_time, runlen, expected_time):
    model = fr.Model(mset)
    model.run(start_time=start_time, runlen=runlen)
    clock = model.model_state.clock

    runlen_in_seconds = fr.utils.to_seconds(runlen)
    assert np.isclose(clock._passed_time, runlen_in_seconds)
    assert clock.get_total_time() == expected_time

@pytest.mark.parametrize(*(
    "start_time, steps, expected_time",
    [
        (0, 101, 10.1),
        (np.datetime64("2021-01-01T00:00:00"),
         101,
         np.datetime64("2021-01-01T00:00:10")),
    ],
))
def test_correct_clock_steps(mset, start_time, steps, expected_time):
    mset.time_stepper.dt = 0.1
    mset.setup()

    model = fr.Model(mset)
    model.run(start_time=start_time, steps=steps)
    clock = model.model_state.clock

    assert clock.it == steps
    if isinstance(expected_time, np.datetime64):
        assert clock.get_total_time() == expected_time
    else:
        assert np.isclose(clock.get_total_time(), expected_time)

@pytest.mark.parametrize(*(
    "start_time, end_time, expected_time",
    [
        (0, 10.0, 10.0),
        (10.0, 20.0, 20.0),
        (np.datetime64("2021-01-01T00:00:00"),
         np.datetime64("2021-01-01T00:00:10"),
         np.datetime64("2021-01-01T00:00:10")),
    ],
))
def test_correct_clock_end_time(mset, start_time, end_time, expected_time):
    mset.setup()

    model = fr.Model(mset)
    model.run(start_time=start_time, end_time=end_time)
    clock = model.model_state.clock

    if isinstance(expected_time, np.datetime64):
        assert clock.get_total_time() == expected_time
    else:
        assert np.isclose(clock.get_total_time(), expected_time)

@pytest.mark.parametrize(*(
    "run_args",
    [
        {"steps": 0},
        {"runlen": 0.0},
        {"start_time": np.datetime64("2021-01-01T00:00:00"),
         "end_time": np.datetime64("2021-01-01T00:00:00")},
    ],
))
def test_satified_end_condition(mset, run_args):
    """Test if the end condition is already satisfied."""
    counter_diag = fr.modules.Counter()
    counter_tend = fr.modules.Counter()
    mset.diagnostics.add_module(counter_diag)
    mset.tendencies.add_module(counter_tend)
    mset.setup()

    model = fr.Model(mset)

    model.run(**run_args)

    # check that the counters are correct
    assert counter_diag.counter == 0
    assert counter_tend.counter == 0

def test_manual_stepping(mset):
    number_of_steps = 10

    counter_tend = fr.modules.Counter()
    mset.tendencies.add_module(counter_tend)
    mset.setup()

    model = fr.Model(mset)
    model.start()
    for _ in range(number_of_steps):
        model.step()
    model.stop()

    # check that the counters are correct
    assert counter_tend.counter == number_of_steps

@pytest.fixture
def directory():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

def test_save_load(mset, directory):
    """Test the save and load methods."""
    number_of_steps = 10

    counter_diag = fr.modules.Counter()
    counter_tend = fr.modules.Counter()
    mset.diagnostics.add_module(counter_diag)
    mset.tendencies.add_module(counter_tend)
    mset.setup()

    model = fr.Model(mset)

    model.run(steps=number_of_steps)
    # check that the counters are reset
    assert model.model_state.clock.it == number_of_steps
    assert counter_diag.counter == number_of_steps
    assert counter_tend.counter == number_of_steps
    model.save(directory + "/model_state")

    model.reset()
    # check that the counters are reset
    assert model.model_state.clock.it == 0
    assert counter_diag.counter == 0
    assert counter_tend.counter == 0

    # load the model state
    model = fr.Model(mset)
    model.load(directory + "/model_state")

    counter_diag = model.diagnostics.get("Counter")[0]
    counter_tend = model.tendencies.get("Counter")[0]

    # check that the counters are reset
    assert model.model_state.clock.it == number_of_steps
    assert counter_diag.counter == number_of_steps
    assert counter_tend.counter == number_of_steps

    model.run(steps=number_of_steps)

    # check that the counters are reset
    assert counter_diag.counter == number_of_steps * 2
    assert counter_tend.counter == number_of_steps * 2
