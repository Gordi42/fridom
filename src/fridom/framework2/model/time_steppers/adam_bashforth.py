"""
Adams-Bashforth steppers.

Description
-----------
``AdamBashforth`` (explicit AB, orders 1-4, at cutover parity) and
``ABState`` (its carry: tendency ring + warm-up counter). Owning
class spec: ``notes/framework2/model/classes/time_steppers.md``
("AdamBashforth"); design source
``notes/framework2/model/03_time_stepping.md`` sections 5.3/5.7;
eps rule: ``notes/framework2/model/02_rules.md`` ("eps is
order-2-only").

The step body is the bitwise-parity algorithm (the traced
transcription of the old ``update_coeff_AB`` scheme): weights are
premultiplied ``row * dt`` FIRST (never ``dt * (c * h)``),
accumulation runs in ascending j over the newest-first ring, the
tendency is evaluated at the pre-tick time, and the ring shift is
structural. Warm-up is a dense zero-padded (order x order)
coefficient table row-indexed by a carried saturating int32
counter — the first chunk compiles the same trace as every other
(no Python branching on the step count, no unrolled first-K steps).
"""
# Wave 4 B: AdamBashforth, ABState (eps is order-2-only)
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, ClassVar, Final

import jax.numpy as jnp
import numpy as np

from fridom.framework.utils import dtype_real, jaxify
from fridom.framework2.model.stages import StageKind
from fridom.framework2.model.terms import Treatment
from fridom.framework2.model.time_steppers.base import TimeStepper

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.grid.fields.vector_field import VectorField
    from fridom.framework2.model.clock import Clock
    from fridom.framework2.model.schedule import BoundSchedule

# the settled Adams-Bashforth coefficient rows (order 2 gets its
# eps'd variant at construction; eps is order-2-only, 02_rules)
_AB_ROWS: Final[dict[int, tuple[float, ...]]] = {
    1: (1.0,),
    2: (3 / 2, -1 / 2),
    3: (23 / 12, -4 / 3, 5 / 12),
    4: (55 / 24, -59 / 24, 37 / 24, -3 / 8),
}

# the parity default of the order-2 computational-mode damper
_DEFAULT_EPS: Final[float] = 0.01

_MIN_ORDER: Final[int] = 1
_MAX_ORDER: Final[int] = 4
_EPS_ORDER: Final[int] = 2

# ABState's fixed field set (write-once frozen discipline)
_AB_STATE_FIELDS: Final[tuple[str, ...]] = ("history", "warmup")


# ================================================================
#  ABState — the AdamBashforth carry
# ================================================================
@partial(jaxify, dynamic=("history", "warmup"))
class ABState:

    """
    AdamBashforth carry: tendency ring + warm-up counter.

    Description
    -----------
    A per-family frozen fully-dynamic pytree (the StepperState
    conventions, ``base.py``): it flattens with the carry and is
    donated with it. Restart restores every leaf bitwise — with the
    counter in the carry, a mid-warm-up crash restores bitwise and
    the first chunk compiles the same trace as every other.

    Parameters
    ----------
    history : tuple[VectorField, ...]
        The tendency ring, length ``order``, **newest first**;
        PROGNOSTIC-only summed explicit tendencies (unprojected),
        shifted structurally.
    warmup : jax.Array
        The saturating warm-up counter (int32, saturates at
        ``order - 1``); stepper-local, never ``clock.it``.
    """

    history: tuple[VectorField, ...]
    warmup: jax.Array

    def __init__(
        self,
        history: tuple[VectorField, ...],
        warmup: jax.Array,
    ) -> None:
        """Freeze the carry entry; see the class docstring."""
        self.history = tuple(history)
        self.warmup = jnp.asarray(warmup, dtype=jnp.int32)

    def __setattr__(self, name: str, value: object) -> None:
        """Set a carry field exactly once (frozen thereafter)."""
        # write-once: __init__ and pytree unflattening set fresh
        # attributes; everything else raises
        if name in _AB_STATE_FIELDS and name not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"ABState is frozen: cannot set {name!r} (a fresh state "
            "is built per step by AdamBashforth.step)")

    def __delattr__(self, name: str) -> None:
        """Raise: ABState is frozen."""
        raise AttributeError(
            f"ABState is frozen: cannot delete {name!r}")

    def __repr__(self) -> str:
        """Compact host-side summary."""
        return (f"ABState({len(self.history)} history levels, "
                f"warmup={self.warmup!r})")


# ================================================================
#  AdamBashforth
# ================================================================
@partial(jaxify, dynamic=("dt",))
class AdamBashforth(TimeStepper):

    """
    Explicit AB1-4; quasi-AB2 eps damper at order 2 only.

    Description
    -----------
    Adams-Bashforth orders 1-4 at cutover parity; the nh preset
    default at cutover (``order=3``). A pytree whose only dynamic
    leaf is ``dt``; ``order``/``eps``/``table`` are static treedef
    aux, hashed by the restart fingerprint.

    Backward runs need nothing here: the table is dt-free, the
    premultiplied weights inherit dt's sign, and warm-up is
    sign-symmetric.

    Parameters
    ----------
    dt : float | np.timedelta64
        The signed step size (converted once; see ``TimeStepper``).
    order : int, optional
        The AB order, 1 to 4 (default: 3).
    eps : float | None, optional
        The quasi-AB2 computational-mode damper — legal at
        ``order=2`` only, where ``None`` resolves to 0.01 (parity
        with the old AB2); any other order rejects a non-None eps
        (default: None).

    Raises
    ------
    ValueError
        If ``order`` is outside 1..4, or ``eps`` is not None at an
        order other than 2.
    """

    supported_treatments: ClassVar[frozenset[Treatment]] = (
        frozenset({Treatment.EXPLICIT}))

    def __init__(
        self,
        dt: float | np.timedelta64,
        order: int = 3,
        eps: float | None = None,
    ) -> None:
        """Validate order/eps, build the warm-up table; see above."""
        super().__init__(dt)
        if (isinstance(order, bool) or not isinstance(order, int)
                or not _MIN_ORDER <= order <= _MAX_ORDER):
            raise ValueError(
                f"AdamBashforth supports orders {_MIN_ORDER} to "
                f"{_MAX_ORDER}, got order={order!r}")
        if order == _EPS_ORDER:
            eps = _DEFAULT_EPS if eps is None else float(eps)
        elif eps is not None:
            raise ValueError(
                f"eps is an order-2-only parameter (the quasi-AB2 "
                f"computational-mode damper); order={order} rejects "
                f"eps={eps!r}. Orders >= 3 warm up through textbook "
                "AB2 [3/2, -1/2]")
        self._order = order
        self._eps = eps
        self._table = _warmup_table(order, eps)

    # ================================================================
    #  Static read surface (fingerprinted treedef aux)
    # ================================================================
    @property
    def order(self) -> int:
        """The static AB order (1-4); fingerprinted."""
        return self._order

    @property
    def eps(self) -> float | None:
        """The static order-2 damper (None at every other order)."""
        return self._eps

    @property
    def table(self) -> tuple[tuple[float, ...], ...]:
        """
        The dense zero-padded warm-up coefficient table.

        Description
        -----------
        ``(order x order)``, row = warm-up level: row 0 is AB1,
        row 1 is AB2 — textbook ``[3/2, -1/2]`` at order >= 3, the
        eps'd row at order 2 — up to the full-order row. Zero-padded
        weights mask the zero-initialized ring rows (no NaN hazard).
        Static and constant-folded into the trace.

        Returns
        -------
        tuple[tuple[float, ...], ...]
            The coefficient rows.
        """
        return self._table

    def _statics(self) -> tuple:
        """Order and eps join the restart fingerprint."""
        return (("order", self._order), ("eps", self._eps))

    # ================================================================
    #  The scan-body protocol
    # ================================================================
    def init(self, tendency_template: VectorField) -> ABState:
        """
        Return the fresh carry: zeroed order-length ring, warmup 0.

        Parameters
        ----------
        tendency_template : VectorField
            The zero PROGNOSTIC tendency vector; ring entries take
            its exact structure.

        Returns
        -------
        ABState
            The fresh carry entry.
        """
        zeros = tendency_template.map(_zero_like)
        return ABState(
            history=(zeros,) * self._order,
            warmup=jnp.asarray(0, dtype=jnp.int32))

    def step(
        self,
        stepper_state: ABState,
        state: VectorField,
        stages: BoundSchedule,
        clock: Clock,
    ) -> tuple[ABState, VectorField, Clock]:
        """
        Advance one AB step (the normative parity body).

        Description
        -----------
        Single substage plus the final combination: prepare and
        evaluate the tendency at the **pre-tick** time, shift the
        newest-first ring structurally, premultiply the warm-up row
        by dt FIRST, accumulate in ascending j, apply the
        PROGNOSTIC-only increment, tick, then run the post-advance
        stage groups (S3' ADVANCE, S4 CONSTRAINT) at the ticked
        clock. S5/S6 are the chunk body's epilogue.

        The increment lands via ``state.add(**components)`` — the
        key-aligned ``State.add_prognostic`` sugar is a parked
        fields.md follow-up (open question 2); this is its exact
        semantics spelled through ``add``.

        Parameters
        ----------
        stepper_state : ABState
            The carry entry.
        state : VectorField
            The full assembled state vector.
        stages : BoundSchedule
            The per-step stage-group view (deviation D-1).
        clock : Clock
            The pre-step clock.

        Returns
        -------
        tuple[ABState, VectorField, Clock]
            The advanced carry entries.
        """
        # P0 + S1/S1' + S2 at the pre-tick time
        ctx = stages.context(clock, dt=self.dt, stage_dt=self.dt)
        state = stages.prepare(state, ctx)
        sums = stages.tendency(state, ctx)
        # structural ring shift, newest first (dataflow renaming)
        history = (sums.explicit, *stepper_state.history[:-1])
        # premultiply the warm-up row by dt FIRST (parity rule)
        table = jnp.asarray(self._table, dtype=dtype_real())
        weights = table[stepper_state.warmup] * self.dt
        # ascending-j accumulation over the newest-first ring
        increment = _weighted(history[0], weights[0])
        for j in range(1, self._order):
            increment = increment + _weighted(history[j], weights[j])
        state = state.add(**dict(increment.components))       # S3
        clock = clock.tick(self.dt)
        # post-advance groups at the ticked clock, sums attached
        ctx = stages.context(clock, dt=self.dt, stage_dt=self.dt,
                             sums=sums)
        if stages.schedule.kind_entries(StageKind.ADVANCE):
            state = stages.advance_stages(state, ctx)         # S3'
        state = stages.constrain(state, ctx)                  # S4
        warmup = jnp.minimum(stepper_state.warmup + 1,
                             self._order - 1)
        return ABState(history, warmup), state, clock

    # ================================================================
    #  Host-side analysis
    # ================================================================
    def time_discretization_effect(
        self,
        omega: np.ndarray,
        *,
        dt: float | None = None,
    ) -> np.ndarray:
        """
        Discrete AB dispersion of a frequency array (host/CPU).

        Description
        -----------
        For each continuous frequency the stability polynomial of
        the FULL-order coefficient row (including eps at order 2 —
        warm-up rows never enter the asymptotic analysis) is solved
        with ``np.roots``; the physical root — the one closest to
        the analytic ``exp(-i omega dt)`` — is mapped back to a
        discrete frequency ``i log(x) / dt``. The tendency
        convention is ``dU/dt = -i omega U``.

        Parameters
        ----------
        omega : np.ndarray
            Continuous frequencies (rad/s); any shape.
        dt : float | None, optional
            Override of the live ``dt`` leaf (default: None).

        Returns
        -------
        np.ndarray
            The complex discrete frequencies, same shape.
        """
        dt_value = float(self.dt) if dt is None else float(dt)
        row = np.asarray(self._table[-1], dtype=np.float64)
        omega = np.asarray(omega)
        result = np.empty(omega.shape, dtype=np.complex128)
        for index, value in np.ndenumerate(omega):
            # x^s - x^(s-1) + i*w*dt * sum_j c_j x^(s-1-j) = 0,
            # coefficients highest degree first
            z = 1j * value * dt_value
            poly = np.empty(self._order + 1, dtype=np.complex128)
            poly[0] = 1.0
            poly[1:] = z * row
            poly[1] -= 1.0
            roots = np.roots(poly)
            physical = np.exp(-1j * value * dt_value)
            x = roots[np.argmin(np.abs(roots - physical))]
            result[index] = 1j * np.log(x) / dt_value
        return result

    # ================================================================
    #  Introspection
    # ================================================================
    def __repr__(self) -> str:
        """Compact host-side summary."""
        damper = (f", eps={self._eps!r}"
                  if self._order == _EPS_ORDER else "")
        return (f"AdamBashforth(dt={self.dt!r}, "
                f"order={self._order}{damper})")


# ================================================================
#  Internals
# ================================================================
def _warmup_table(
    order: int,
    eps: float | None,
) -> tuple[tuple[float, ...], ...]:
    """
    Build the dense zero-padded (order x order) warm-up table.

    Description
    -----------
    Row r is the AB(r+1) coefficient row, zero-padded to ``order``;
    the final row is the full-order scheme. At order 2 the final row
    carries the eps damper; orders >= 3 warm up through textbook AB2
    (a deliberate startup-only delta vs the old code — 02_rules).

    Parameters
    ----------
    order : int
        The AB order (1-4).
    eps : float | None
        The order-2 damper (None at every other order).

    Returns
    -------
    tuple[tuple[float, ...], ...]
        The static coefficient table.
    """
    rows = []
    for level in range(1, order + 1):
        row = _AB_ROWS[level]
        if level == _EPS_ORDER and order == _EPS_ORDER:
            row = (row[0] + eps, row[1] - eps)
        rows.append(row + (0.0,) * (order - level))
    return tuple(rows)


def _zero_like(field: ScalarField) -> ScalarField:
    """Zero twin of one field (stored array; halos trivially valid)."""
    return type(field)(
        field.grid, field.function_space,
        jnp.zeros_like(field._data),  # noqa: SLF001 — plumbing-constructor seam
        field.metadata)


def _weighted(
    vector: VectorField,
    weight: jax.Array,
) -> VectorField:
    """
    Scale one ring entry: ``weight * vector``, componentwise.

    Description
    -----------
    The traced-scalar scaling ``weights[j] * history[j]`` of the
    normative step body, spelled on the true-shape data (the field
    dunders accept Python scalars only). The multiplication order is
    ``weight * data`` — part of the bitwise-parity op sequence.

    Parameters
    ----------
    vector : VectorField
        One newest-first ring entry.
    weight : jax.Array
        The dt-premultiplied scalar weight.

    Returns
    -------
    VectorField
        The scaled entry (metadata preserved by ``map``).
    """
    return vector.map(
        lambda field: field.with_data(weight * field.data))
