"""
SLURM helpers (``fr.slurm``).

Description
-----------
The ``restart_module`` successor, reduced to plain functions:
``in_job``/``job_id`` read the environment, ``resubmit_current``
re-submits the running job via the old scontrol -> sbatch
autodetection, and ``resubmit()`` is the factory for the
``Snapshots(on_walltime=...)`` action (the ``fr.io.resubmit()``
acceptance-surface spelling — re-exported by the io namespace).
Subprocess and executable lookups sit behind the small mockable
seams ``_run``/``_which``. Owning class spec:
``design/specs/model/classes/io_ops.md``.
"""
# Wave 4 C: in_job, job_id, resubmit_current, resubmit
from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence


# ================================================================
#  Mockable seams
# ================================================================
def _run(args: Sequence[str]) -> str:
    """Run one subprocess and return its stdout (mockable seam).

    Parameters
    ----------
    args : sequence of str
        The argv; the executable is an absolute path resolved by
        ``_which``.

    Returns
    -------
    str
        The process's stdout.
    """
    completed = subprocess.run(  # noqa: S603 — fixed argv, resolved executable, no shell
        list(args), capture_output=True, text=True, check=True)
    return completed.stdout


def _which(name: str) -> str | None:
    """Resolve an executable on PATH (mockable seam)."""
    return shutil.which(name)


# ================================================================
#  Environment queries
# ================================================================
def in_job() -> bool:
    """Whether the process runs inside a SLURM allocation.

    Returns
    -------
    bool
        True iff ``SLURM_JOB_ID`` is set in the environment.
    """
    return "SLURM_JOB_ID" in os.environ


def job_id() -> str | None:
    """Return the current SLURM job id, if any.

    Returns
    -------
    str or None
        The value of ``SLURM_JOB_ID``, or ``None`` outside a job.
    """
    return os.environ.get("SLURM_JOB_ID")


# ================================================================
#  Resubmission
# ================================================================
def resubmit_current() -> None:
    """Re-submit the current job (scontrol -> sbatch autodetect).

    Description
    -----------
    Reads the running job's batch script from
    ``scontrol show job <id>`` (the ``Command=`` field) and hands
    it back to ``sbatch`` — the old restart-module behavior, kept
    as a plain function. Rank-0-only on multi-process launches
    (non-zero ``SLURM_PROCID`` returns without submitting); the
    cross-process walltime consensus is the parked 3.2/3.3
    residual.
    """
    jid = job_id()
    if jid is None:
        raise RuntimeError(
            "fr.slurm.resubmit_current() requires a SLURM "
            "allocation (SLURM_JOB_ID is not set)")
    if os.environ.get("SLURM_PROCID", "0") != "0":
        return  # rank-0-only submission guard
    scontrol = _which("scontrol")
    sbatch = _which("sbatch")
    if scontrol is None or sbatch is None:
        raise RuntimeError(
            "scontrol/sbatch not found on PATH — cannot resubmit "
            "the current job")
    info = _run([scontrol, "show", "job", jid])
    command = None
    for line in info.splitlines():
        stripped = line.strip()
        if stripped.startswith("Command="):
            command = stripped.split("=", 1)[1].strip()
    if not command:
        raise RuntimeError(
            f"no batch script found in 'scontrol show job {jid}' "
            "output — cannot autodetect the resubmission command")
    _run([sbatch, *shlex.split(command)])


def resubmit() -> Callable[[], None]:
    """Return the ``on_walltime`` action re-submitting this job.

    Description
    -----------
    The ``fr.io.resubmit()`` acceptance-surface spelling (sketch
    7.7): a factory whose result wraps
    ``fr.slurm.resubmit_current`` for
    ``fr.io.Snapshots(on_walltime=...)``.

    Returns
    -------
    callable
        A zero-argument action calling ``resubmit_current()``.
    """
    def _action() -> None:
        """Re-submit the current SLURM job."""
        resubmit_current()
    return _action
