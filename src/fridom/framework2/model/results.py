"""
Run outcomes.

Description
-----------
The typed run-loop returns (CS-3): ``RunStatus``, ``AdvanceResult``,
``RunResult``, the ``PanicError``/``RunTargetError`` exceptions, and
the reserved ``PendingAdvance`` slot. Owning class spec:
``notes/framework2/model/classes/model.md`` (section 5, "Run-loop
result types", and the error registry, section 7).

All host values — no pytrees, nothing traced. ``advance()`` raises
``PanicError``; ``run()`` catches and converts to
``RunResult(NAN_ABORT)`` unless ``raise_on_nan=True`` (notebooks
want the carry; context managers need exceptions — both contracts
hold). Scripts and SLURM drivers branch on ``RunResult.status`` —
``run()`` never exits the process.
"""
# Wave 4 B: RunStatus, AdvanceResult, RunResult, PanicError,
#    RunTargetError
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


# ================================================================
#  RunStatus
# ================================================================
class RunStatus(Enum):

    """
    Shared status vocabulary (``RunResult``, Session aggregation).

    Description
    -----------
    ``COMPLETED``: the planned steps ran. ``NAN_ABORT``: the S5
    panic flag read true at a chunk boundary. ``WALLTIME``: the
    predictive walltime check stopped the run (elapsed + predicted
    chunk + snapshot margin). ``INTERRUPTED``: first Ctrl-C — the
    in-flight chunk completed, the carry is consistent, zero steps
    lost.
    """

    COMPLETED = "completed"
    NAN_ABORT = "nan_abort"
    WALLTIME = "walltime"
    INTERRUPTED = "interrupted"


# ================================================================
#  AdvanceResult
# ================================================================
@dataclass(frozen=True)
class AdvanceResult:

    """
    ``advance()``'s minimal frozen return.

    Description
    -----------
    The IO-free primitive's typed contract (CS-3) — everything else
    (status, timing breakdown) stays ``RunResult``'s.

    Parameters
    ----------
    steps_done : int
        Steps completed.
    panicked : bool
        The panic flag as read at the last chunk boundary.
    panic_it : int | None
        First-failure iteration (the S5 record), or None.
    wall_seconds : float
        Wall time spent advancing.
    """

    steps_done: int
    panicked: bool
    panic_it: int | None
    wall_seconds: float


# ================================================================
#  RunResult
# ================================================================
@dataclass(frozen=True)
class RunResult:

    """
    ``run()``'s plain, aggregatable return.

    Description
    -----------
    Scripts and SLURM drivers branch on :attr:`status`; ``run()``
    never exits the process.

    Parameters
    ----------
    status : RunStatus
        The run outcome.
    steps_done : int
        Steps completed.
    final_it : int
        The carry clock's iteration counter at the end.
    final_time : float
        The carry clock's model time at the end (seconds).
    compile_seconds : float
        AOT ``lower().compile()`` accounting.
    run_seconds : float
        Wall time spent stepping (compile excluded).
    steps_per_second : float
        ``steps_done / run_seconds`` throughput.
    """

    status: RunStatus
    steps_done: int
    final_it: int
    final_time: float
    compile_seconds: float
    run_seconds: float
    steps_per_second: float


# ================================================================
#  PanicError
# ================================================================
class PanicError(RuntimeError):

    """
    Typed NaN abort raised at a panicked chunk boundary.

    Description
    -----------
    Raised by ``advance()`` at the chunk boundary where the panic
    flag reads true, and on entry with an already-panicked carry.
    Carries attribution for multi-model sessions (the model
    ``name=`` plus the S5 first-failure iteration); carries are left
    in memory for autopsy. Resume guidance: ``set_fields`` /
    ``set_state`` / ``load_snapshot`` / ``reset`` clear the flag
    (``update_parameters`` and ``set_aux`` deliberately do not).

    Parameters
    ----------
    message : str | None, optional
        Override of the default attributed message (default: None).
    model_name : str | None, optional
        The panicking model's ``name=`` (default: None).
    first_bad_it : int | None, optional
        First-failure iteration, the S5 record (default: None).
    partial : AdvanceResult | None, optional
        Steps completed before the abort (default: None).
    """

    def __init__(
        self,
        message: str | None = None,
        *,
        model_name: str | None = None,
        first_bad_it: int | None = None,
        partial: AdvanceResult | None = None,
    ) -> None:
        """Attach attribution and build the default message."""
        if message is None:
            who = (f"model {model_name!r}" if model_name is not None
                   else "the model")
            where = (f" (first non-finite values at iteration "
                     f"{first_bad_it})" if first_bad_it is not None
                     else "")
            message = (
                f"{who} panicked: non-finite values in the "
                f"state{where}. The carry is left in memory for "
                "autopsy; set_fields/set_state/load_snapshot/reset "
                "clear the panic flag")
        super().__init__(message)
        self.model_name = model_name
        self.first_bad_it = first_bad_it
        self.partial = partial


# ================================================================
#  RunTargetError
# ================================================================
class RunTargetError(ValueError):

    """
    Raised by ``run()`` planning on a bad run target.

    Description
    -----------
    The named error of the run-target rules (spec deviation: the
    design mandates a named error; the name is coined here, in
    ``results.py``). Two raise conditions share it:

    - the sign precondition ``(end - t0) * dt > 0`` is violated —
      the requested end time cannot be reached with the current
      signed dt (flip ``fr.params.TIME_STEP`` via
      ``update_parameters`` for backward legs);
    - inconsistent target combinations — ``run()`` takes exactly
      one of ``steps``/``runlen``/``end_time`` (none or several is
      ambiguous).
    """
