"""test_clock.py - Test the Clock class."""

import numpy as np
import pytest

import fridom.framework as fr


# ----------------------------------------------------------------
#  Test the initialization of the Clock class
# ----------------------------------------------------------------
@pytest.mark.parametrize(*(
    "start_date, start_time, raises",
    [
        pytest.param(None, None, False, id="No start date or time"),
        pytest.param(np.datetime64("2023-01-01"), None, False,
                     id="Start date only"),
        pytest.param(None, 0.0, False, id="Start time only"),
        pytest.param(
            np.datetime64("2023-01-01"), 0.0, True,
            id="Both start date and time",
        ),
    ],
))
def test_clock_initialization(start_date, start_time, raises):
    """Test the initialization of the Clock class."""
    if raises:
        with pytest.raises(fr.exceptions.TooManyArgumentsError):
            fr.Clock(start_date=start_date, start_time=start_time)
        return
    clock = fr.Clock(start_date=start_date, start_time=start_time)
    assert isinstance(clock, fr.Clock)

# ----------------------------------------------------------------
#  Test the tick method of the Clock class
# ----------------------------------------------------------------
@pytest.mark.parametrize(*(
    "time_steps, expected",
    [
        pytest.param([10], 10, id="Single time step"),
        pytest.param([10, 5], 15, id="Multiple time steps"),
        pytest.param([10, -5], 5, id="Negative time step"),
        pytest.param([-10], -10, id="Negative initial time step"),
    ],
))
@pytest.mark.parametrize(*(
    "start_date",
    [
        pytest.param(None, id="No start date"),
        pytest.param(np.datetime64("2023-01-01"), id="Start date provided"),
    ],
))
def test_tick(start_date, time_steps, expected):
    """Test the tick method of the Clock class."""
    clock = fr.Clock(start_date=start_date)
    _ = [clock.tick(step) for step in time_steps]
    assert clock.passed_time == expected

# ----------------------------------------------------------------
#  Test the get_total_time method of the Clock class
# ----------------------------------------------------------------
@pytest.mark.parametrize(*(
    "start_date, start_time, passed_time, expected",
    [
        pytest.param(
            np.datetime64("2023-01-01"),
            None,
            3600,
            np.datetime64("2023-01-01T01:00:00"),
            id="Start date",
        ),
        pytest.param(None, 1500.0, 3600, 5100.0, id="Start time"),
    ],
))
def test_get_total_time(start_date, start_time, passed_time, expected):
    """Test the get_total_time method of the Clock class."""
    clock = fr.Clock(start_date=start_date, start_time=start_time)
    total_time = clock.get_total_time(passed_time)
    assert total_time == expected

# ----------------------------------------------------------------
#  Test the time property
# ----------------------------------------------------------------
@pytest.mark.parametrize(*(
    "start_date, start_time, delta_time, expected",
    [
        pytest.param(None, 0.0, 3600, 3600.0, id="Start time"),
        pytest.param(
            np.datetime64("1970-01-02"),
            None,
            3600,
            86400 + 3600,  # 1970-01-01 is the reference date
            id="Start date",
        ),
    ],
))
def test_time(start_date, start_time, delta_time, expected):
    """Test the time property of the Clock class."""
    clock = fr.Clock(start_date=start_date, start_time=start_time)
    clock.tick(delta_time)
    assert clock.time == expected

# ----------------------------------------------------------------
#  Test the reset method
# ----------------------------------------------------------------
@pytest.mark.parametrize(*(
    "start_time",
    [
        pytest.param(None, id="No start time"),
        pytest.param(100.0, id="Start time"),
    ],
))
def test_reset(start_time):
    """Test the reset method of the Clock class."""
    clock = fr.Clock(start_time=start_time)
    clock.tick(10)
    assert clock.passed_time == 10
    clock.reset()
    assert clock.passed_time == 0
    if start_time is not None:
        assert clock.time == start_time
