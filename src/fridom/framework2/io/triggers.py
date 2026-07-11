"""
Output triggers and the plan-time lowering.

Description
-----------
The declarative trigger algebra — the ``every``/``at`` factories
(surfaced as ``fr.every``/``fr.at``), ``|`` unions, and
``after=``/``until=`` windows — plus ``lower_trigger``, the public
plan-time lowering from a trigger to a sorted step-index set (CS-7:
importable by drivers, no model or clock access). Owning class spec:
``notes/framework2/model/classes/io_ops.md``.

Triggers are frozen host-side data: never callbacks, never traced,
never pytrees. The lowering works in step space,
``k = (t - t0) / dt``, snapped up with ``ceil(k)`` — sign-agnostic
by construction: for ``dt < 0`` the division maps decreasing model
times to increasing ``k`` and the same ceil snap applies; no branch
on the dt sign exists anywhere in this module. The concrete node
classes (``Every``, ``At``, ``Union``, ``Window``) are public here
but not re-exported; the factories are the user surface.
"""
# Wave 4 C: Trigger nodes, every/at factories, lower_trigger
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence


# ================================================================
#  Time spellings
# ================================================================
# model-time cadence kwargs, in seconds per unit
_CADENCE_SECONDS: Final[dict[str, float]] = {
    "seconds": 1.0,
    "minutes": 60.0,
    "hours": 3600.0,
    "days": 86400.0,
}

# walltime string spellings: "7.5h" (sketch 7.7)
_WALLTIME_UNITS: Final[dict[str, float]] = {
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}

_WALLTIME_PATTERN: Final = re.compile(
    r"^\s*(\d+(?:\.\d*)?|\.\d+)\s*([smhd])\s*$")


def _model_seconds(value: object, *, name: str) -> float:
    """Convert a model-time spelling to float seconds.

    Parameters
    ----------
    value : float or np.timedelta64
        The model time; float values are already seconds.
    name : str
        The kwarg name, for the error message.

    Returns
    -------
    float
        The value in seconds.
    """
    if isinstance(value, np.timedelta64):
        return float(value / np.timedelta64(1, "s"))
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name}= takes float seconds or np.timedelta64; "
            f"got {value!r}") from exc


def _wall_seconds(value: str | float | np.timedelta64) -> float:
    """Convert a walltime spelling to float wall seconds.

    Parameters
    ----------
    value : str or float or np.timedelta64
        Wall seconds, a ``"7.5h"``-style string (units s/m/h/d), or
        a timedelta.

    Returns
    -------
    float
        The walltime budget in wall seconds.
    """
    if isinstance(value, np.timedelta64):
        return float(value / np.timedelta64(1, "s"))
    if isinstance(value, str):
        match = _WALLTIME_PATTERN.match(value)
        if match is None:
            raise ValueError(
                f"cannot parse walltime {value!r}; use a "
                "number-and-unit spelling like '7.5h' (units: s, m, "
                "h, d), plain wall seconds, or np.timedelta64")
        return float(match.group(1)) * _WALLTIME_UNITS[match.group(2)]
    return float(value)


# ================================================================
#  Snapped step-space arithmetic
# ================================================================
# tolerance for "this time is an exact step multiple up to float
# noise" — without it (0.3 - 0.0) / 0.1 would snap UP a whole step.
_SNAP_RTOL: Final[float] = 1e-9


def _snap(k: float) -> float:
    """Round ``k`` to the nearest integer when within float noise."""
    nearest = round(k)
    if abs(k - nearest) <= _SNAP_RTOL * max(1.0, abs(k)):
        return float(nearest)
    return k


def _snap_ceil(k: float) -> int:
    """Snap up: the ceil of ``k``, noise-tolerant at exact hits."""
    return math.ceil(_snap(k))


def _snap_floor(k: float) -> int:
    """Snap down: the floor of ``k``, noise-tolerant at exact hits."""
    return math.floor(_snap(k))


# ================================================================
#  The trigger nodes (public, not re-exported)
# ================================================================
@dataclass(frozen=True)
class Trigger:

    """
    Base of the trigger algebra; immutable declarative data.

    Description
    -----------
    Frozen host-side data — never a callback, never traced, never a
    pytree. Compose with ``|`` (union); build instances through the
    ``every``/``at`` factories, never directly.
    """

    def __or__(self, other: Trigger) -> Trigger:
        """Union: the composed trigger fires when any operand fires."""
        if not isinstance(other, Trigger):
            return NotImplemented
        return Union(_flat(self) + _flat(other))

    @property
    def has_walltime(self) -> bool:
        """Whether any component is walltime-based.

        Walltime triggers are snapshot/action-only; data streams
        reject them at bind (nondeterministic output grids).
        """
        return False


@dataclass(frozen=True)
class Every(Trigger):

    """
    Periodic trigger node; exactly one cadence slot is set.

    Description
    -----------
    Built by ``every()``, which folds the ``minutes``/``hours``/
    ``days`` spellings into ``seconds`` and parses ``walltime``
    strings into wall seconds. Step and model-time cadences include
    step 0 (the ``execute_at_start`` successor); walltime cadences
    are never lowered to steps.

    Parameters
    ----------
    steps : int, optional
        Fire every ``steps`` steps (including step 0).
    seconds : float, optional
        Fire every ``seconds`` model seconds along the run
        direction (including step 0).
    walltime : float, optional
        Wall-clock budget in seconds; evaluated predictively at
        chunk boundaries by the ``WalltimeGuard``.
    """

    steps: int | None = None
    seconds: float | None = None
    walltime: float | None = None

    @property
    def has_walltime(self) -> bool:
        """Whether this node is the walltime cadence."""
        return self.walltime is not None


@dataclass(frozen=True)
class At(Trigger):

    """
    Fire at explicit model times; validated at run planning.

    Parameters
    ----------
    times : tuple of float
        Model times in seconds; times outside the run's interval
        error at planning, never silently drop.
    """

    times: tuple[float, ...]


@dataclass(frozen=True)
class Union(Trigger):

    """
    Union node: fires when any operand fires.

    Parameters
    ----------
    operands : tuple of Trigger
        The (flattened) union operands.
    """

    operands: tuple[Trigger, ...]

    @property
    def has_walltime(self) -> bool:
        """Whether any operand is walltime-based."""
        return any(op.has_walltime for op in self.operands)


@dataclass(frozen=True)
class Window(Trigger):

    """
    Window node: filters the inner trigger's firing set.

    Description
    -----------
    ``after``/``until`` are model times, interpreted along the run
    direction; the endpoints are mapped to step space by the same
    ``k = (t - t0) / dt`` rule as everything else, so windows are
    sign-agnostic too.

    Parameters
    ----------
    inner : Trigger
        The trigger whose firing set is filtered.
    after : float, optional
        Keep firings at or after this model time (run direction).
    until : float, optional
        Keep firings at or before this model time (run direction).
    """

    inner: Trigger
    after: float | None = None
    until: float | None = None

    @property
    def has_walltime(self) -> bool:
        """Whether the windowed trigger is walltime-based."""
        return self.inner.has_walltime


def _flat(trigger: Trigger) -> tuple[Trigger, ...]:
    """Flatten nested unions into their operands."""
    if isinstance(trigger, Union):
        return trigger.operands
    return (trigger,)


# ================================================================
#  The factories — the user surface (fr.every / fr.at)
# ================================================================
def every(
    *,
    steps: int | None = None,
    seconds: float | np.timedelta64 | None = None,
    minutes: float | np.timedelta64 | None = None,
    hours: float | np.timedelta64 | None = None,
    days: float | np.timedelta64 | None = None,
    walltime: str | float | np.timedelta64 | None = None,
    after: float | np.timedelta64 | None = None,
    until: float | np.timedelta64 | None = None,
) -> Trigger:
    """Build a periodic trigger; exactly one cadence kwarg.

    Description
    -----------
    Model-time and step cadences include step 0 (the
    ``execute_at_start`` successor). Exactly one cadence kwarg per
    call — composition is ``|``, never multiple cadences in one
    node. ``walltime=`` triggers are snapshot/action-only.

    Parameters
    ----------
    steps : int, optional
        Fire every ``steps`` steps.
    seconds, minutes, hours, days : float or np.timedelta64, optional
        Fire at this model-time cadence along the run direction.
    walltime : str or float or np.timedelta64, optional
        Wall-clock cadence: seconds, ``"7.5h"``-style string, or
        timedelta; evaluated by the ``WalltimeGuard``, never
        lowered to steps.
    after, until : float or np.timedelta64, optional
        Window: keep only firings inside ``[after, until]`` (model
        time, run direction).

    Returns
    -------
    Trigger
        The frozen trigger node (wrapped in a window if bounded).
    """
    cadences: dict[str, object] = {
        "steps": steps, "seconds": seconds, "minutes": minutes,
        "hours": hours, "days": days, "walltime": walltime}
    given = [name for name, value in cadences.items() if value is not None]
    if len(given) != 1:
        spelled = ", ".join(given) if given else "none"
        raise ValueError(
            "fr.every() takes exactly one cadence kwarg (steps=, "
            f"seconds=, minutes=, hours=, days= or walltime=); got "
            f"{spelled}. Compose cadences with |, e.g. "
            "fr.every(steps=10) | fr.every(hours=1).")
    if steps is not None:
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError(f"steps= must be an int; got {steps!r}")
        if steps < 1:
            raise ValueError(f"steps= must be >= 1; got {steps}")
        node: Trigger = Every(steps=steps)
    elif walltime is not None:
        wall = _wall_seconds(walltime)
        if wall <= 0.0:
            raise ValueError(
                f"walltime= must be positive; got {walltime!r}")
        node = Every(walltime=wall)
    else:
        name = given[0]
        value = cadences[name]
        period = _model_seconds(value, name=name)
        if not isinstance(value, np.timedelta64):
            # plain numbers are counted in the kwarg's unit; a
            # timedelta64 is already an absolute duration
            period *= _CADENCE_SECONDS[name]
        if period <= 0.0:
            raise ValueError(
                f"{name}= must be positive; got {cadences[name]!r}")
        node = Every(seconds=period)
    return _windowed(node, after=after, until=until)


def at(
    times: Sequence[float | np.timedelta64],
    *,
    after: float | np.timedelta64 | None = None,
    until: float | np.timedelta64 | None = None,
) -> Trigger:
    """Build a trigger firing at explicit model times.

    Description
    -----------
    Times are validated at run planning (``lower_trigger``): a time
    outside the run's interval errors there, never silently drops.
    The realized (snapped) time ``t0 + ceil(k) * dt`` is what lands
    on the output time axis.

    Parameters
    ----------
    times : sequence of float or np.timedelta64
        The model times, in seconds or as timedeltas.
    after, until : float or np.timedelta64, optional
        Window: keep only firings inside ``[after, until]`` (model
        time, run direction).

    Returns
    -------
    Trigger
        The frozen trigger node (wrapped in a window if bounded).
    """
    if isinstance(times, (int, float, np.timedelta64)):
        raise TypeError(
            f"fr.at() takes a sequence of model times; got the "
            f"scalar {times!r} — spell it fr.at([{times!r}])")
    fired = tuple(_model_seconds(t, name="times") for t in times)
    if not fired:
        raise ValueError("fr.at() needs at least one model time")
    return _windowed(At(fired), after=after, until=until)


def _windowed(
    node: Trigger,
    *,
    after: float | np.timedelta64 | None,
    until: float | np.timedelta64 | None,
) -> Trigger:
    """Wrap ``node`` in a Window when a bound is given."""
    if after is None and until is None:
        return node
    return Window(
        node,
        after=(None if after is None
               else _model_seconds(after, name="after")),
        until=(None if until is None
               else _model_seconds(until, name="until")))


# ================================================================
#  lower_trigger — the public plan-time lowering (CS-7)
# ================================================================
def lower_trigger(
    trigger: Trigger,
    *,
    t0: float,
    dt: float,
    n_steps: int,
) -> tuple[int, ...]:
    """Lower a trigger to a sorted step-index set in step space.

    Description
    -----------
    Step space is ``k = (t - t0) / dt``, snapped up with ``ceil(k)``
    — sign-agnostic by construction: for ``dt < 0`` the division
    maps decreasing model times to increasing ``k`` and the same
    ceil snap applies. ``every()`` cadences enumerate firing times
    in the run direction and snap up; step 0 is included. ``fr.at``
    times outside the run interval (``ceil(k)`` not in
    ``[0, n_steps]``) error at planning. Walltime components lower
    to the empty set (they are boundary-evaluated by the
    ``WalltimeGuard``). The realized time of a firing is
    ``t0 + k * dt``; logging it is the caller's (bind/plan) duty.

    Parameters
    ----------
    trigger : Trigger
        A trigger built by ``fr.every``/``fr.at`` (unions/windows
        included).
    t0 : float
        The run's start model time in seconds.
    dt : float
        The (signed, nonzero) time step in seconds.
    n_steps : int
        The number of steps in the run; firing indices lie in
        ``[0, n_steps]``.

    Returns
    -------
    tuple of int
        The sorted, deduplicated firing step indices.
    """
    if not isinstance(trigger, Trigger):
        raise TypeError(
            f"expected a Trigger built by fr.every/fr.at; got "
            f"{trigger!r}")
    dt = float(dt)
    if dt == 0.0:
        raise ValueError("dt must be nonzero")
    if isinstance(n_steps, bool) or not isinstance(n_steps, int):
        raise TypeError(f"n_steps must be an int; got {n_steps!r}")
    if n_steps < 0:
        raise ValueError(f"n_steps must be >= 0; got {n_steps}")
    ks = _lower(trigger, t0=float(t0), dt=dt, n_steps=n_steps)
    return tuple(sorted(ks))


def _lower(
    trigger: Trigger, *, t0: float, dt: float, n_steps: int,
) -> set[int]:
    """Dispatch the lowering over the node types."""
    if isinstance(trigger, Union):
        fired: set[int] = set()
        for operand in trigger.operands:
            fired |= _lower(operand, t0=t0, dt=dt, n_steps=n_steps)
        return fired
    if isinstance(trigger, Window):
        return _lower_window(trigger, t0=t0, dt=dt, n_steps=n_steps)
    if isinstance(trigger, Every):
        return _lower_every(trigger, dt=dt, n_steps=n_steps)
    if isinstance(trigger, At):
        return _lower_at(trigger, t0=t0, dt=dt, n_steps=n_steps)
    raise TypeError(
        f"cannot lower {type(trigger).__name__}; triggers are built "
        "by fr.every/fr.at")


def _lower_every(node: Every, *, dt: float, n_steps: int) -> set[int]:
    """Lower a periodic node to its firing step set."""
    if node.walltime is not None:
        # boundary-evaluated by the WalltimeGuard, never step-lowered
        return set()
    if node.steps is not None:
        return set(range(0, n_steps + 1, node.steps))
    # Model-time cadence: the m-th firing along the run direction is
    # t_m = t0 + m * T * (dt / |dt|), so its step-space coordinate
    # is (t_m - t0) / dt = m * T / |dt| — positive for BOTH dt
    # signs; the |dt| falls out of the sign-agnostic division, it is
    # not a code branch on the sign.
    period = node.seconds if node.seconds is not None else 0.0
    step_period = period / abs(dt)
    if step_period <= 1.0:
        # cadence at or below the step length: every step fires
        return set(range(n_steps + 1))
    fired: set[int] = set()
    m = 0
    while True:
        k = _snap_ceil(m * period / abs(dt))
        if k > n_steps:
            return fired
        fired.add(k)
        m += 1


def _lower_at(
    node: At, *, t0: float, dt: float, n_steps: int,
) -> set[int]:
    """Lower explicit times; out-of-interval errors at planning."""
    fired: set[int] = set()
    for time in node.times:
        k = _snap_ceil((time - t0) / dt)
        if k < 0 or k > n_steps:
            raise ValueError(
                f"fr.at time {time} lies outside the run: it lowers "
                f"to step {k}, but the run covers steps "
                f"0..{n_steps} (t0={t0}, dt={dt}). Out-of-interval "
                "trigger times error at planning, never silently "
                "drop.")
        fired.add(k)
    return fired


def _lower_window(
    node: Window, *, t0: float, dt: float, n_steps: int,
) -> set[int]:
    """Filter the inner firing set by the window's step interval.

    The lower endpoint snaps up (the same ceil rule as firings);
    the upper endpoint snaps down, so no kept firing's realized
    time lies past ``until``.
    """
    inner = _lower(node.inner, t0=t0, dt=dt, n_steps=n_steps)
    lo = (0 if node.after is None
          else _snap_ceil((node.after - t0) / dt))
    hi = (n_steps if node.until is None
          else _snap_floor((node.until - t0) / dt))
    return {k for k in inner if lo <= k <= hi}
