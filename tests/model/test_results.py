"""Tests for the run-loop result types (model/results.py).

Covers the RunStatus vocabulary, the frozen result dataclasses, and
the PanicError/RunTargetError raising surfaces (attribution
attributes, default-message content, exception bases).
"""
import dataclasses

import pytest

from fridom.model.results import (
    AdvanceResult,
    PanicError,
    RunResult,
    RunStatus,
    RunTargetError,
    _fmt_duration,
)


# ================================================================
#  RunStatus
# ================================================================
def test_run_status_vocabulary():
    assert {status.name for status in RunStatus} == {
        "COMPLETED", "NAN_ABORT", "WALLTIME", "INTERRUPTED"}
    assert RunStatus.COMPLETED.value == "completed"
    assert RunStatus.NAN_ABORT.value == "nan_abort"
    assert RunStatus.WALLTIME.value == "walltime"
    assert RunStatus.INTERRUPTED.value == "interrupted"


# ================================================================
#  AdvanceResult / RunResult (frozen host values)
# ================================================================
def test_advance_result_construction():
    result = AdvanceResult(steps_done=10, panicked=False,
                           panic_it=None, wall_seconds=0.5)
    assert result.steps_done == 10
    assert result.panicked is False
    assert result.panic_it is None
    assert result.wall_seconds == 0.5


def test_advance_result_is_frozen():
    result = AdvanceResult(steps_done=1, panicked=True, panic_it=7,
                           wall_seconds=0.1)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.steps_done = 2


def test_run_result_construction():
    result = RunResult(
        status=RunStatus.COMPLETED, steps_done=100, final_it=100,
        final_time=6000.0, compile_seconds=1.5, run_seconds=2.0,
        steps_per_second=50.0)
    assert result.status is RunStatus.COMPLETED
    assert result.final_it == 100
    assert result.steps_per_second == 50.0


def test_run_result_is_frozen():
    result = RunResult(
        status=RunStatus.WALLTIME, steps_done=1, final_it=1,
        final_time=60.0, compile_seconds=0.0, run_seconds=0.1,
        steps_per_second=10.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.status = RunStatus.COMPLETED


# ================================================================
#  Duration formatting / repr
# ================================================================
@pytest.mark.parametrize(("seconds", "expected"), [
    pytest.param(0.2843997, "284 ms", id="milliseconds"),
    pytest.param(8.42e-5, "0.0842 ms", id="sub-millisecond"),
    pytest.param(0.0, "0 ms", id="zero"),
    pytest.param(47.5233250, "47.52 s", id="seconds"),
    pytest.param(120.0000000002, "0:02:00", id="minutes"),
    pytest.param(5025.0, "1:23:45", id="hours"),
    pytest.param(2 * 86400 + 3 * 3600, "2 d 3:00:00", id="days"),
    pytest.param(-47.5233250, "-47.52 s", id="negative"),
    pytest.param(float("nan"), "nan s", id="nan"),
    pytest.param(float("inf"), "inf s", id="inf"),
])
def test_fmt_duration(seconds, expected):
    assert _fmt_duration(seconds) == expected


def test_run_result_repr_is_a_compact_one_liner():
    result = RunResult(
        status=RunStatus.COMPLETED, steps_done=11520, final_it=11520,
        final_time=120.00000000001964,
        compile_seconds=0.2843997199088335,
        run_seconds=47.523325093556195,
        steps_per_second=242.4072805789851)
    assert repr(result) == (
        "RunResult(completed, steps=11520, it=11520, time=0:02:00, "
        "compile=284 ms, run=47.52 s, 242.4 steps/s)")
    assert "\n" not in repr(result)
    assert "RunStatus." not in repr(result)


def test_run_result_repr_slow_rate_keeps_significance():
    result = RunResult(
        status=RunStatus.NAN_ABORT, steps_done=3, final_it=3,
        final_time=0.03, compile_seconds=1.0, run_seconds=6.0,
        steps_per_second=0.512345)
    text = repr(result)
    assert "nan_abort" in text
    assert "0.512 steps/s" in text


# ================================================================
#  PanicError
# ================================================================
def test_panic_error_carries_attribution():
    partial = AdvanceResult(steps_done=41, panicked=True,
                            panic_it=42, wall_seconds=1.0)
    error = PanicError(model_name="ocean", first_bad_it=42,
                       partial=partial)
    assert isinstance(error, RuntimeError)
    assert error.model_name == "ocean"
    assert error.first_bad_it == 42
    assert error.partial is partial


def test_panic_error_default_message_names_model_and_iteration():
    message = str(PanicError(model_name="ocean", first_bad_it=42))
    assert "'ocean'" in message
    assert "42" in message
    assert "reset" in message  # resume guidance


def test_panic_error_is_raisable_with_defaults():
    with pytest.raises(PanicError, match="panicked"):
        raise PanicError


def test_panic_error_explicit_message_wins():
    error = PanicError("custom text", model_name="atm")
    assert str(error) == "custom text"
    assert error.model_name == "atm"


# ================================================================
#  RunTargetError
# ================================================================
def test_run_target_error_is_a_value_error():
    assert issubclass(RunTargetError, ValueError)


def test_run_target_error_raising_surface():
    with pytest.raises(RunTargetError, match=r"\(end - t0\)"):
        raise RunTargetError(
            "run target unreachable: (end - t0) * dt <= 0")
    with pytest.raises(RunTargetError, match="exactly one"):
        raise RunTargetError(
            "give exactly one of steps/runlen/end_time")
