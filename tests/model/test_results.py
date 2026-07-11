"""Tests for the run-loop result types (framework2/model/results.py).

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
