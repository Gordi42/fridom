"""test_clock_trigger.py - Test the ClockTrigger class."""

import numpy as np
import pytest

import fridom.framework as fr


# ----------------------------------------------------------------
#  Fixtures
# ----------------------------------------------------------------
@pytest.fixture(params=[None, 10, np.datetime64("2023-01-01")])
def start_date(request):
    return request.param

@pytest.fixture(params=[None, 10])
def start_step(request):
    return request.param

@pytest.fixture(params=[None, 10, np.datetime64("2023-01-02")])
def stop_date(request):
    return request.param

@pytest.fixture(params=[None, 10])
def stop_step(request):
    return request.param

@pytest.fixture(params=[None, 10, np.timedelta64(1, "h")])
def time_interval(request):
    return request.param

@pytest.fixture(params=[None, 10])
def step_number(request):
    return request.param

# ----------------------------------------------------------------
#  Test initialization of the ClockTrigger class
# ----------------------------------------------------------------
def test_initialization_start_arg(start_date, start_step):
    """Test the initialization of the ClockTrigger class with start arguments."""
    # if both arguments are provided, raise an error
    if start_date is not None and start_step is not None:
        with pytest.raises(fr.exceptions.TooManyArgumentsError):
            fr.ClockTrigger(start_date=start_date, start_step=start_step)
        return
    # otherwise, create the ClockTrigger object
    clock_trigger = fr.ClockTrigger(start_date=start_date, start_step=start_step)
    assert isinstance(clock_trigger, fr.ClockTrigger)

def test_initialization_stop_arg(stop_date, stop_step):
    """Test the initialization of the ClockTrigger class with stop arguments."""
    # if both arguments are provided, raise an error
    if stop_date is not None and stop_step is not None:
        with pytest.raises(fr.exceptions.TooManyArgumentsError):
            fr.ClockTrigger(stop_date=stop_date, stop_step=stop_step)
        return
    # otherwise, create the ClockTrigger object
    clock_trigger = fr.ClockTrigger(stop_date=stop_date, stop_step=stop_step)
    assert isinstance(clock_trigger, fr.ClockTrigger)

def test_initialization_time_interval(time_interval, step_number):
    """Test the initialization of the ClockTrigger class with time interval arguments."""
    # if both arguments are provided, raise an error
    if time_interval is not None and step_number is not None:
        with pytest.raises(fr.exceptions.TooManyArgumentsError):
            fr.ClockTrigger(time_interval=time_interval, step_number=step_number)
        return
    # otherwise, create the ClockTrigger object
    clock_trigger = fr.ClockTrigger(time_interval=time_interval, step_number=step_number)
    assert isinstance(clock_trigger, fr.ClockTrigger)

# ----------------------------------------------------------------
#  Count the number of advanced steps
# ----------------------------------------------------------------
@pytest.mark.parametrize(*(
    "time_interval, step_number, expected_triggers",
    [
        pytest.param(None, None, 10, id="No time interval"),
        pytest.param(1, None, 3, id="Time interval"),
        pytest.param(np.timedelta64(1, "s"), None, 3, id="Time interval in seconds"),
        pytest.param(None, 2, 5, id="No time interval, step number"),
    ],
))
def test_count_triggers_no_start_no_stop(time_interval, step_number, expected_triggers):
    """Count the number of advanced steps."""
    clock_trigger = fr.ClockTrigger(
        time_interval=time_interval, step_number=step_number)
    clock = fr.Clock()
    n_ticks = 10
    tick_amount = 0.3
    trigger_count = 0
    for _ in range(n_ticks):
        trigger_count += clock_trigger.check(clock)
        clock.tick(tick_amount)

    assert trigger_count == expected_triggers

@pytest.mark.parametrize(*(
    "time_interval, step_number, expected_triggers",
    [
        pytest.param(None, None, 6, id="No time interval"),
        pytest.param(1, None, 2, id="Time interval"),
        pytest.param(np.timedelta64(1, "s"), None, 2, id="Time interval in seconds"),
        pytest.param(None, 2, 3, id="No time interval, step number"),
    ],
))
def test_count_triggers_with_start(time_interval, step_number, expected_triggers):
    """Count the number of advanced steps."""
    clock_trigger = fr.ClockTrigger(
        start_date=1,
        time_interval=time_interval,
        step_number=step_number)
    clock = fr.Clock()
    n_ticks = 10
    tick_amount = 0.3
    trigger_count = 0
    for _ in range(n_ticks):
        trigger_count += clock_trigger.check(clock)
        clock.tick(tick_amount)

    assert trigger_count == expected_triggers

@pytest.mark.parametrize(*(
    "time_interval, step_number, expected_triggers",
    [
        pytest.param(None, None, 7, id="No time interval"),
        pytest.param(1, None, 2, id="Time interval"),
        pytest.param(np.timedelta64(1, "s"), None, 2, id="Time interval in seconds"),
        pytest.param(None, 2, 4, id="No time interval, step number"),
    ],
))
def test_count_triggers_with_stop(time_interval, step_number, expected_triggers):
    """Count the number of advanced steps."""
    clock_trigger = fr.ClockTrigger(
        stop_date=2,
        time_interval=time_interval,
        step_number=step_number)
    clock = fr.Clock()
    n_ticks = 10
    tick_amount = 0.3
    trigger_count = 0
    for _ in range(n_ticks):
        trigger_count += clock_trigger.check(clock)
        clock.tick(tick_amount)

    assert trigger_count == expected_triggers
