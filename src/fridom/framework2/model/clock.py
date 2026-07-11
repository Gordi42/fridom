"""
The traced clock (``fr.Clock``).

Description
-----------
``Clock``: the traced model clock — dynamic ``start``/``elapsed``/
``it`` leaves at the global width (float64/int64 under the default
x64-on run), a static host-only ``start_date`` calendar anchor, and
purely functional advancement (``tick``/``shifted``/``reset``).
Owning class spec:
``notes/framework2/model/classes/time_steppers.md`` ("Clock");
precision rules: ``notes/framework2/model/02_rules.md`` ("Clock
precision").

The *authoritative* clock is host-side float64 numpy always —
calendar, run targets, trigger times, and snapshot times live in
numpy, unaffected by the x64 flag. The traced leaves here take the
global width: under x64-off the model (wave 4.2) re-anchors the
traced ``elapsed`` from the host clock at every chunk boundary
through :meth:`Clock.reanchored` — this class only provides the
primitive.
"""
# Wave 4 B: Clock
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Final

import jax.numpy as jnp
import numpy as np

from fridom.framework.utils import dtype_real, jaxify

if TYPE_CHECKING:  # pragma: no cover
    import jax

# the fixed field set (also the pytree leaf order for the dynamic
# subset); Clock is final and frozen — every attribute is set
# exactly once
_CLOCK_FIELDS: Final[tuple[str, ...]] = (
    "start", "elapsed", "it", "start_date")

# nanoseconds per second (the ``date`` conversion resolution)
_NS_PER_S: Final[float] = 1e9


def _dtype_int() -> jnp.dtype:
    """Return the default integer dtype at the global width.

    int64 under the default x64-on run, int32 otherwise — the
    integer twin of ``fr.utils.dtype_real`` (an explicit
    ``jnp.int64`` request would warn and truncate under x64-off).
    """
    return jnp.result_type(int)


# ================================================================
#  Clock
# ================================================================
@partial(jaxify, dynamic=("start", "elapsed", "it"))
class Clock:

    """
    The traced model clock; host-side calendar anchor.

    Description
    -----------
    Concrete, final, and frozen: no mutating method exists — every
    advance builds a fresh ``Clock`` functionally (``tick``,
    ``shifted``, ``reset``, ``reanchored``). The dynamic leaves ride
    the carry; ``start_date`` is static, host-only treedef aux
    (changing it is a re-assembly-grade structure change — it never
    affects numerics and is not fingerprinted as a leaf).

    Signed-dt semantics: ``tick(-|dt|)`` is the backward-run
    primitive; ``it`` still increments (it counts steps, not
    direction). Nothing on the Clock is direction-aware.

    The calendar is strictly host-side: no ``datetime64`` ever
    enters a trace; :attr:`date` is the single conversion point.

    Attributes
    ----------
    start : jax.Array
        Run-start model time, float64 seconds under x64 (traced).
    elapsed : jax.Array
        Accumulated signed model time since start (traced);
        accumulates by repeated ``+= dt`` (parity; survives
        adaptive dt, unlike ``it * dt``).
    it : jax.Array
        Iteration counter, int64 under x64 (traced); increments on
        every tick, forward or backward.

    Parameters
    ----------
    start : float, optional
        Run-start model time in seconds (default: 0.0).
    start_date : np.datetime64 | None, optional
        Static, host-only calendar anchor (default: None).

    Raises
    ------
    TypeError
        If ``start`` is not a real number or ``start_date`` is not
        a ``np.datetime64`` (or None).
    """

    def __init__(
        self,
        start: float = 0.0,
        *,
        start_date: np.datetime64 | None = None,
    ) -> None:
        """Build a fresh clock: elapsed 0, it 0; see class doc."""
        if isinstance(start, bool) or not isinstance(
                start, int | float | np.floating | np.integer):
            raise TypeError(
                f"start is the run-start model time in seconds "
                f"(a real number), got {start!r}")
        if start_date is not None and not isinstance(
                start_date, np.datetime64):
            raise TypeError(
                f"start_date is a host-side np.datetime64 calendar "
                f"anchor, got {start_date!r}")
        real = dtype_real()
        self.start = jnp.asarray(float(start), dtype=real)
        self.elapsed = jnp.asarray(0.0, dtype=real)
        self.it = jnp.asarray(0, dtype=_dtype_int())
        self.start_date = start_date

    # ================================================================
    #  Frozen discipline (write-once over the fixed field set)
    # ================================================================
    def __setattr__(self, name: str, value: object) -> None:
        """Set a clock field exactly once (frozen thereafter)."""
        # write-once: __init__, _build, and pytree unflattening set
        # fresh attributes; everything else raises
        if name in _CLOCK_FIELDS and name not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"Clock is frozen: cannot set {name!r} (advance "
            "functionally via tick/shifted/reset/reanchored)")

    def __delattr__(self, name: str) -> None:
        """Raise: Clock is frozen."""
        raise AttributeError(
            f"Clock is frozen: cannot delete {name!r}")

    @classmethod
    def _build(
        cls,
        start: jax.Array,
        elapsed: jax.Array,
        it: jax.Array,
        start_date: np.datetime64 | None,
    ) -> Clock:
        """Assemble a clock from already-built leaves (internal)."""
        clock = object.__new__(cls)
        clock.start = start
        clock.elapsed = elapsed
        clock.it = it
        clock.start_date = start_date
        return clock

    # ================================================================
    #  Derived reads
    # ================================================================
    @property
    def time(self) -> jax.Array:
        """``start + elapsed`` — the traced time axis, in seconds."""
        return self.start + self.elapsed

    @property
    def date(self) -> np.datetime64:
        """
        Host-side calendar timestamp of the elapsed time.

        Description
        -----------
        ``start_date + timedelta64(elapsed)`` at nanosecond
        resolution — the single trace-to-calendar conversion point,
        consumed by progress reporting, writers, and time-target
        parsing only. Never call in a trace (it synchronizes).

        Returns
        -------
        np.datetime64
            The calendar timestamp.

        Raises
        ------
        ValueError
            If no ``start_date`` anchor was given.
        """
        if self.start_date is None:
            raise ValueError(
                "this clock has no calendar anchor; construct it "
                "with Clock(start_date=np.datetime64(...)) to read "
                "dates")
        nanoseconds = round(float(self.elapsed) * _NS_PER_S)
        return self.start_date + np.timedelta64(nanoseconds, "ns")

    # ================================================================
    #  Functional advancement
    # ================================================================
    def tick(self, dt: jax.Array) -> Clock:
        """
        Return the clock advanced by one step: the step primitive.

        Description
        -----------
        ``elapsed += dt`` (signed — a negative dt is the
        backward-run primitive) and ``it += 1`` (always: ``it``
        counts steps, not direction). The stepper owns the call
        (once per step, after the primary advance).

        Parameters
        ----------
        dt : jax.Array
            The signed step size in seconds.

        Returns
        -------
        Clock
            The advanced clock; ``self`` is unchanged.
        """
        return self._build(
            self.start, self.elapsed + dt, self.it + 1,
            self.start_date)

    def shifted(self, tau: jax.Array) -> Clock:
        """
        Return the stage clock: shifted time, same iteration.

        Description
        -----------
        ``elapsed += tau`` with ``it`` unchanged — the RK stage
        clock (``clock.shifted(c_i * dt)``), superseding the old
        deep-copied clock. ``tau`` carries dt's sign.

        Parameters
        ----------
        tau : jax.Array
            The signed stage shift in seconds.

        Returns
        -------
        Clock
            The shifted clock; ``self`` is unchanged.
        """
        return self._build(
            self.start, self.elapsed + tau, self.it,
            self.start_date)

    def reset(self) -> Clock:
        """
        Return a fresh clock: start preserved, elapsed/it zeroed.

        Description
        -----------
        What restarts a Ramp leg (``model.reset()`` swaps this in);
        ``start_date`` is preserved.

        Returns
        -------
        Clock
            The fresh clock; ``self`` is unchanged.
        """
        return self._build(
            self.start, jnp.asarray(0.0, dtype=dtype_real()),
            jnp.asarray(0, dtype=_dtype_int()), self.start_date)

    def reanchored(self, elapsed: float) -> Clock:
        """
        Return the clock with ``elapsed`` re-anchored from the host.

        Description
        -----------
        The x64-off re-anchor primitive (02_rules, Clock precision):
        in a float32 run the authoritative host-side float64 clock
        overwrites the traced ``elapsed`` at every chunk boundary,
        bounding accumulation drift by one chunk. Driving the
        cadence is the MODEL's job (wave 4.2); this class only
        provides the primitive. ``start``/``it`` are unchanged.

        Parameters
        ----------
        elapsed : float
            The authoritative host-side elapsed time in seconds.

        Returns
        -------
        Clock
            The re-anchored clock; ``self`` is unchanged.
        """
        return self._build(
            self.start,
            jnp.asarray(elapsed, dtype=dtype_real()),
            self.it, self.start_date)

    # ================================================================
    #  Introspection
    # ================================================================
    def __repr__(self) -> str:
        """Compact host-side summary (synchronizes the leaves)."""
        anchor = (f", start_date={self.start_date!r}"
                  if self.start_date is not None else "")
        return (f"Clock(start={self.start!r}, "
                f"elapsed={self.elapsed!r}, it={self.it!r}{anchor})")
