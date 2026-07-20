"""
Run outcomes.

Description
-----------
The typed run-loop returns (CS-3): ``RunStatus``, ``AdvanceResult``,
``RunResult``, the ``PanicError``/``RunTargetError`` exceptions, and
the reserved ``PendingAdvance`` slot. Owning class spec:
``design/specs/model/classes/model.md`` (section 5, "Run-loop
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

import math
from dataclasses import dataclass
from enum import Enum


# ================================================================
#  Duration formatting
# ================================================================
def _fmt_duration(seconds: float) -> str:
    """
    Format a duration in seconds for human eyes.

    Description
    -----------
    Picks the unit by magnitude: milliseconds below one second
    (``284 ms``), seconds with four significant digits below one
    minute (``47.52 s``), ``h:mm:ss`` below one day (``1:23:45``),
    and a day prefix above (``2 d 3:00:00``). Negative durations
    keep a leading sign; non-finite values fall back to a plain
    seconds rendering (``nan s``).

    Parameters
    ----------
    seconds : float
        The duration in seconds.

    Returns
    -------
    str
        The humanized duration.
    """
    if not math.isfinite(seconds):
        return f"{seconds:.4g} s"
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    if seconds < 1:
        return f"{sign}{seconds * 1e3:.3g} ms"
    if seconds < 60:  # noqa: PLR2004 — seconds per minute
        return f"{sign}{seconds:.4g} s"
    whole = round(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    days, hours = divmod(hours, 24)
    day = f"{days} d " if days else ""
    return f"{sign}{day}{hours}:{minutes:02d}:{secs:02d}"


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
    #  Introspection
    # ================================================================
    def __repr__(self) -> str:
        """
        Compact one-line summary.

        Description
        -----------
        Notebooks and the example gallery show a bare
        ``model.run(...)`` result, so the repr stays one readable
        line: the status value, the integer counters, humanized
        durations, and the throughput —
        ``RunResult(completed, steps=11520, it=11520, time=0:02:00,
        compile=284 ms, run=47.52 s, 242.4 steps/s)``.

        Returns
        -------
        str
            The one-line summary.
        """
        rate = self.steps_per_second
        rate_str = (f"{rate:.1f}" if rate >= 10  # noqa: PLR2004
                    else f"{rate:.3g}")  # one decimal reads best
        return (
            f"RunResult({self.status.value}, "
            f"steps={self.steps_done}, it={self.final_it}, "
            f"time={_fmt_duration(self.final_time)}, "
            f"compile={_fmt_duration(self.compile_seconds)}, "
            f"run={_fmt_duration(self.run_seconds)}, "
            f"{rate_str} steps/s)")


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
        The S5 detection iteration — the boundary of the chunk
        whose state was non-finite; the exact first-bad step for
        chunk length 1 or via ``replay_nan`` (default: None).
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
            where = (f" (non-finite values detected at iteration "
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
