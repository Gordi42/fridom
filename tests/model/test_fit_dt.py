"""Tests for fridom.model.fit_dt."""
import math

import numpy as np
import pytest

import fridom as fr
from fridom.model.model import _lands_on_a_step


# ================================================================
#  Fitting the window
# ================================================================
@pytest.mark.parametrize(("runlen", "max_dt"), [
    pytest.param(12.0, 0.0013810679320049755, id="cfl-like"),
    pytest.param(12.0, 0.12, id="exact-multiple"),
    pytest.param(6.0 * 3600.0, 4.7, id="long-window"),
    pytest.param(1.0, 0.9, id="barely-two-steps"),
    pytest.param(117.4, 0.031, id="non-round-window"),
])
def test_fitted_dt_divides_the_window(runlen, max_dt):
    """The result is a whole number of steps at or below the bound."""
    dt = fr.model.fit_dt(runlen, max_dt)
    assert dt <= max_dt * (1.0 + 1e-9)
    assert _lands_on_a_step(runlen / dt)
    # n is minimal: one step fewer would break the bound
    n = round(runlen / dt)
    assert n == 1 or runlen / (n - 1) > max_dt


def test_exact_multiple_returns_the_bound():
    dt = fr.model.fit_dt(12.0, 0.12)
    assert dt == pytest.approx(0.12, rel=1e-12)
    assert round(12.0 / dt) == 100


def test_warning_example_case():
    """The docstring case: 8688.93 steps rounds up to 8689."""
    dt = fr.model.fit_dt(12.0, 0.0013810679320049755)
    assert round(12.0 / dt) == 8689


def test_noise_tolerant_snap():
    """Float noise in the quotient does not force an extra step."""
    max_dt = 1.0 / 3.0
    dt = fr.model.fit_dt(1.0, max_dt)
    assert round(1.0 / dt) == 3
    assert dt == pytest.approx(max_dt, rel=1e-9)


def test_bound_larger_than_window_gives_one_step():
    assert fr.model.fit_dt(5.0, 10.0) == 5.0


def test_timedelta64_durations():
    dt = fr.model.fit_dt(
        np.timedelta64(6, "h"), np.timedelta64(90, "s"))
    assert dt == 90.0


# ================================================================
#  parts= (uniform subwindows)
# ================================================================
def test_parts_rounds_up_to_a_multiple():
    dt = fr.model.fit_dt(10.0, 0.3, parts=8)
    n = round(10.0 / dt)
    assert n == 40  # ceil(10/0.3) = 34, rounded up to 5 * 8
    # each of the 8 subwindows is a whole number of steps
    assert _lands_on_a_step((10.0 / 8) / dt)


def test_parts_already_a_multiple_is_untouched():
    dt = fr.model.fit_dt(12.0, 1.0, parts=6)
    assert dt == pytest.approx(1.0, rel=1e-12)


def test_parts_dominates_a_loose_bound():
    dt = fr.model.fit_dt(12.0, 100.0, parts=480)
    assert dt == pytest.approx(12.0 / 480, rel=1e-12)


@pytest.mark.parametrize("parts", [3, 7, 480])
def test_parts_subwindows_are_whole_steps(parts):
    runlen, max_dt = 117.4, 0.031
    dt = fr.model.fit_dt(runlen, max_dt, parts=parts)
    assert dt <= max_dt * (1.0 + 1e-9)
    k = (runlen / parts) / dt
    assert _lands_on_a_step(k)
    assert round(runlen / dt) == parts * round(k)


# ================================================================
#  Errors
# ================================================================
@pytest.mark.parametrize("runlen", [0.0, -1.0])
def test_non_positive_runlen_raises(runlen):
    with pytest.raises(ValueError, match="runlen= must be a positive"):
        fr.model.fit_dt(runlen, 0.1)


@pytest.mark.parametrize("max_dt", [0.0, -0.5])
def test_non_positive_max_dt_raises(max_dt):
    with pytest.raises(ValueError, match="max_dt= must be a positive"):
        fr.model.fit_dt(1.0, max_dt)


@pytest.mark.parametrize(("bad", "name"), [
    pytest.param("12", "runlen", id="string"),
    pytest.param(True, "runlen", id="bool"),
    pytest.param(None, "max_dt", id="none"),
])
def test_non_duration_spelling_raises(bad, name):
    args = {"runlen": 1.0, "max_dt": 0.1, name: bad}
    with pytest.raises(
            TypeError,
            match=f"{name}= takes float seconds or np.timedelta64"):
        fr.model.fit_dt(args["runlen"], args["max_dt"])


@pytest.mark.parametrize("parts", [0, -3])
def test_non_positive_parts_raises(parts):
    with pytest.raises(ValueError, match="parts= must be a positive"):
        fr.model.fit_dt(1.0, 0.1, parts=parts)


@pytest.mark.parametrize("parts", [2.5, True])
def test_non_int_parts_raises(parts):
    with pytest.raises(TypeError, match="parts= must be a positive"):
        fr.model.fit_dt(1.0, 0.1, parts=parts)


# ================================================================
#  The run planner agrees
# ================================================================
def test_quotient_is_whole_for_the_planner():
    """runlen/dt passes the planner's own whole-steps gate."""
    for runlen, max_dt in [(12.0, 0.00138), (55.0, 0.72), (8.0, 3.0)]:
        dt = fr.model.fit_dt(runlen, max_dt)
        quotient = runlen / dt
        assert _lands_on_a_step(quotient)
        assert math.ceil(quotient - 1e-9) == round(quotient)
