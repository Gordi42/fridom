"""Fit a time step so a run window is a whole number of steps."""
from __future__ import annotations

import math

import numpy as np

# the relative snap tolerance for "the quotient is already a whole
# number of steps"; mirrors the run-target snap in
# fridom.model.model (_TARGET_EPS), so a dt this helper returns is
# always exact by the run planner's own measure
_EPS = 1e-9


def _seconds(value: object, *, name: str) -> float:
    """Convert a duration spelling to float seconds."""
    if isinstance(value, np.timedelta64):
        return float(value / np.timedelta64(1, "s"))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{name}= takes float seconds or np.timedelta64; got "
            f"{value!r}")
    return float(value)


# ================================================================
#  fit_dt (run-window time-step fitting)
# ================================================================
def fit_dt(
    runlen: float | np.timedelta64,
    max_dt: float | np.timedelta64,
    *,
    parts: int | None = None,
) -> float:
    r"""
    Fit the largest time step that divides a run window.

    Description
    -----------
    Returns :math:`dt = \mathrm{runlen}/n` with the smallest
    :math:`n` such that :math:`dt \le \mathrm{max\_dt}` — the
    largest stable time step for which
    ``run(runlen=runlen)`` is a whole number of steps, so the run
    lands exactly on the window and the rounding warning never
    fires. ``max_dt`` is the stability bound the step must respect
    (a CFL estimate, a relaxation timescale, ...).

    With ``parts`` given, :math:`n` is additionally rounded up to a
    multiple of ``parts``, so each of the ``parts`` equal
    subwindows (e.g. animation frames sampled with
    ``every(time_units=runlen / parts)``) is itself a whole number
    of steps and the samples come out exactly uniform.

    A quotient ``runlen / max_dt`` within relative ``1e-9`` of a
    whole number counts as exact (the same snap tolerance the run
    planner uses), so float noise in the operands never forces an
    extra step; in that case the returned dt may exceed ``max_dt``
    by that same relative sliver.

    .. code-block:: python

        dt = fr.model.fit_dt(runlen, dt_cfl, parts=frames)
        model.run(runlen=runlen, outputs=(writer,))  # lands exactly

    Parameters
    ----------
    runlen : float | np.timedelta64
        The run window to divide, in model seconds; must be
        positive.
    max_dt : float | np.timedelta64
        The upper bound on the time step, in model seconds; must be
        positive.
    parts : int | None, optional
        Also make each of the ``parts`` equal subwindows a whole
        number of steps; None fits the window alone (default:
        None).

    Returns
    -------
    float
        The fitted time step in model seconds:
        ``runlen / steps`` for a whole ``steps``.

    Raises
    ------
    TypeError
        On a duration that is neither a number nor a
        ``np.timedelta64``, or a ``parts`` that is not an int.
    ValueError
        On a non-positive ``runlen`` or ``max_dt``, or a ``parts``
        below one.
    """
    runlen_s = _seconds(runlen, name="runlen")
    max_dt_s = _seconds(max_dt, name="max_dt")
    if runlen_s <= 0.0:
        raise ValueError(
            f"runlen= must be a positive duration; got {runlen!r}")
    if max_dt_s <= 0.0:
        raise ValueError(
            f"max_dt= must be a positive duration; got {max_dt!r}")
    if parts is not None:
        if isinstance(parts, bool) or not isinstance(parts, int):
            raise TypeError(
                f"parts= must be a positive int; got {parts!r}")
        if parts < 1:
            raise ValueError(
                f"parts= must be a positive int; got {parts!r}")
    quotient = runlen_s / max_dt_s
    # noise-tolerant ceil: an exact-multiple bound does not force an
    # extra step (same discipline as the run-target snap)
    n_steps = max(1, math.ceil(
        quotient - _EPS * max(1.0, quotient)))
    if parts is not None:
        n_steps = parts * -(-n_steps // parts)
    return runlen_s / n_steps
