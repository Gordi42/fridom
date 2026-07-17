"""
IMEX multistep steppers.

Description
-----------
Wave 5 B: ``IMEXMultistep`` (the one generic IMEX-by-treatment
multistep driver), ``IMEXState`` (its carry: explicit-F ring, SBDF
state ring, warm-up counter), and the ``CNAB2`` / ``SBDF2`` thin
factories (SBDF3 / IMEX-RK are designed-for). Owning class spec:
``design/specs/model/classes/time_steppers.md`` ("IMEXMultistep");
design source ``design/specs/model/03_time_stepping.md`` sections
5.4/5.6/5.7.

One driver; the schemes are static coefficient-level sets. Warm-up
switches WHOLE ``(explicit_weights, state_weights, apply_weight,
gamma)`` tuples (gamma changes across levels: FB-Euler gamma=1 ->
CNAB2 gamma=1/2, SBDF1 gamma=1 -> SBDF2 gamma=2/3), which is exactly
why ``dt_gamma = gamma * dt`` is necessarily traced (the implicit
``solve`` receives it as a positional, never from ctx). The F-ring
stores the SUMMED explicit contribution (unprojected); the implicit
side buffers nothing (the CN forward apply is recomputed fresh each
step through ``op.apply`` — buffering ``L @ X`` across the projection
boundary is the staleness bug). Projection is project-the-state once
per step, post-advance (the only coherent IMEX arrangement).

Coefficient levels are stated solve-normalized (the ``solve`` protocol
is ``(1 - dt_gamma * L)^{-1}``, so the implicit mass is 1 and SBDF2 is
gamma=2/3 with state weights (4/3, -1/3) — deviation D-4). The
``apply_weight`` is the CNAB forward-apply coefficient carried in the
switched tuple (deviation D-3).
"""
# Wave 5 B: IMEXMultistep, IMEXState, CNAB2, SBDF2
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, ClassVar, Final, Literal

import jax.numpy as jnp

from fridom.framework.utils import dtype_real, jaxify
from fridom.model.schedule import TendencySums
from fridom.model.stages import StageKind
from fridom.model.terms import Treatment
from fridom.model.time_steppers.base import TimeStepper

if TYPE_CHECKING:  # pragma: no cover
    import jax
    import numpy as np

    from fridom.model.clock import Clock
    from fridom.model.context import StepContext
    from fridom.model.schedule import (
        BoundImplicitOperator,
        BoundSchedule,
    )
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField

# One warm-up level == (explicit_weights, state_weights, apply_weight,
# gamma), zero-padded to the scheme's history depth (D-3/D-4).
IMEXLevel = tuple[tuple[float, ...], tuple[float, ...], float, float]

# The solve-normalized coefficient levels per scheme; index == warm-up
# counter (level 0 the first step, level 1 the settled scheme).
_SCHEME_LEVELS: Final[dict[str, tuple[IMEXLevel, ...]]] = {
    "cnab2": (
        # FB Euler: explicit (1,0), state (1,), apply 0, gamma 1
        ((1.0, 0.0), (1.0,), 0.0, 1.0),
        # CNAB2: explicit (3/2,-1/2), state (1,), apply 1/2, gamma 1/2
        ((3 / 2, -1 / 2), (1.0,), 1 / 2, 1 / 2),
    ),
    "sbdf2": (
        # SBDF1: explicit (1,0), state (1,0), apply 0, gamma 1
        ((1.0, 0.0), (1.0, 0.0), 0.0, 1.0),
        # SBDF2: explicit (4/3,-2/3), state (4/3,-1/3), apply 0,
        # gamma 2/3
        ((4 / 3, -2 / 3), (4 / 3, -1 / 3), 0.0, 2 / 3),
    ),
}

_IMEX_STATE_FIELDS: Final[tuple[str, ...]] = (
    "f_history", "x_history", "warmup")


# ================================================================
#  IMEXState — the IMEXMultistep carry
# ================================================================
@partial(jaxify, dynamic=("f_history", "x_history", "warmup"))
class IMEXState:

    """
    IMEXMultistep carry: explicit-F ring, SBDF state ring, counter.

    Description
    -----------
    A per-family frozen fully-dynamic pytree (the StepperState
    conventions): it flattens with the carry and is donated with it.
    Rings are tuples of PROGNOSTIC-only vectors, newest first, shifted
    structurally, and hold PAST entries only — the newest explicit
    contribution ``f_n`` is computed fresh inside the step and never
    carried (the 2026-07-12 memory cut, matching ``ABState``).
    ``x_history`` exists only for SBDF (an empty tuple for CNAB2 — a
    treedef difference between schemes, consistent with
    scheme-as-static); it holds the PAST prognostic states beyond the
    current one (depth ``state_depth - 1``).

    Parameters
    ----------
    f_history : tuple[VectorField, ...]
        The summed PAST EXPLICIT contribution ring (newest first),
        length ``explicit_depth - 1``.
    x_history : tuple[VectorField, ...]
        The past-state ring (SBDF only; empty for CNAB2).
    warmup : jax.Array
        The saturating warm-up counter (int32); stepper-local.
    """

    f_history: tuple[VectorField, ...]
    x_history: tuple[VectorField, ...]
    warmup: jax.Array

    def __init__(
        self,
        f_history: tuple[VectorField, ...],
        x_history: tuple[VectorField, ...],
        warmup: jax.Array,
    ) -> None:
        """Freeze the carry entry; see the class docstring."""
        self.f_history = tuple(f_history)
        self.x_history = tuple(x_history)
        self.warmup = jnp.asarray(warmup, dtype=jnp.int32)

    def __setattr__(self, name: str, value: object) -> None:
        """Set a carry field exactly once (frozen thereafter)."""
        # write-once: __init__ and pytree unflattening set fresh
        # attributes; everything else raises
        if name in _IMEX_STATE_FIELDS and name not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"IMEXState is frozen: cannot set {name!r} (a fresh "
            "state is built per step by IMEXMultistep.step)")

    def __delattr__(self, name: str) -> None:
        """Raise: IMEXState is frozen."""
        raise AttributeError(
            f"IMEXState is frozen: cannot delete {name!r}")

    def __repr__(self) -> str:
        """Compact host-side summary."""
        return (f"IMEXState({len(self.f_history)} F, "
                f"{len(self.x_history)} X, warmup={self.warmup!r})")


# ================================================================
#  IMEXMultistep
# ================================================================
@partial(jaxify, dynamic=("dt",))
class IMEXMultistep(TimeStepper):

    """
    IMEX-by-treatment multistep: explicit combine + implicit solves.

    Description
    -----------
    One generic driver; the schemes (CNAB2, SBDF2) are static
    coefficient-level sets. A pytree whose only dynamic leaf is
    ``dt``; the scheme name and its level tuples are static treedef
    aux (fingerprinted). Empty-implicit compositions are legal — the
    solve set is empty and the scheme degrades to its explicit member
    (CNAB2 -> textbook AB2, SBDF2 -> the explicit two-step); only an
    IMPLICIT term under an explicit-only stepper is the assembly
    error (never a silent demotion).

    Parameters
    ----------
    dt : float | np.timedelta64
        The signed step size (converted once; see ``TimeStepper``).
    scheme : {"cnab2", "sbdf2"}, optional
        The coefficient-level set (default: "cnab2").

    Raises
    ------
    ValueError
        If ``scheme`` is not a supported name.
    """

    supported_treatments: ClassVar[frozenset[Treatment]] = (
        frozenset({Treatment.EXPLICIT, Treatment.IMPLICIT}))

    #: multistep outer driver: it runs a module-owned split ADVANCE
    #: (S3') subcycle with the per-treatment sums attached (03 5.4).
    supports_split_advance: ClassVar[bool] = True

    def __init__(
        self,
        dt: float | np.timedelta64,
        scheme: Literal["cnab2", "sbdf2"] = "cnab2",
    ) -> None:
        """Select the static coefficient-level tuple set."""
        super().__init__(dt)
        if scheme not in _SCHEME_LEVELS:
            raise ValueError(
                f"unknown IMEX scheme {scheme!r}; supported schemes: "
                f"{tuple(_SCHEME_LEVELS)}")
        self._scheme = scheme
        levels = _SCHEME_LEVELS[scheme]
        self._levels = levels
        self._explicit_table = tuple(level[0] for level in levels)
        self._state_table = tuple(level[1] for level in levels)
        self._apply_row = tuple(level[2] for level in levels)
        self._gamma_row = tuple(level[3] for level in levels)
        self._explicit_depth = len(levels[0][0])
        self._state_depth = len(levels[0][1])
        self._n_levels = len(levels)
        self._needs_apply = any(level[2] != 0.0 for level in levels)

    # ================================================================
    #  Static read surface (fingerprinted treedef aux)
    # ================================================================
    @property
    def scheme(self) -> str:
        """Static scheme name; fingerprinted."""
        return self._scheme

    @property
    def levels(self) -> tuple[IMEXLevel, ...]:
        """
        Static warm-up levels, index == counter value.

        Description
        -----------
        Each an ``(explicit_weights, state_weights, apply_weight,
        gamma)`` tuple, zero-padded to the scheme's history depth
        (deviations D-3/D-4).

        Returns
        -------
        tuple[IMEXLevel, ...]
            The coefficient levels.
        """
        return self._levels

    def _statics(self) -> tuple:
        """Scheme and levels join the restart fingerprint."""
        return (("scheme", self._scheme), ("levels", self._levels))

    # ================================================================
    #  The scan-body protocol
    # ================================================================
    def init(self, tendency_template: VectorField) -> IMEXState:
        """
        Return the fresh carry: zeroed rings, warmup 0.

        Parameters
        ----------
        tendency_template : VectorField
            The zero PROGNOSTIC tendency vector; ring entries take
            its exact structure.

        Returns
        -------
        IMEXState
            The fresh carry entry.
        """
        zero = _zero_vector(tendency_template)
        return IMEXState(
            f_history=(zero,) * (self._explicit_depth - 1),
            x_history=(zero,) * (self._state_depth - 1),
            warmup=jnp.asarray(0, dtype=jnp.int32))

    def step(
        self,
        stepper_state: IMEXState,
        state: VectorField,
        stages: BoundSchedule,
        clock: Clock,
    ) -> tuple[IMEXState, VectorField, Clock]:
        """
        Advance one IMEX step (the section-5.4 algorithm).

        Description
        -----------
        Pre-tick prepare/tendency (S1/S1'/S2); structural ring shifts
        (the carried rings hold PAST entries only — the newest-first
        F levels are ``(f_n, *f_history)``); a branch-free warm-up
        level gather; the AB-parity rhs combine (state weights on
        the current + past prognostic states, premultiplied explicit
        weights on the F levels, and the forward-apply term
        recomputed fresh through ``op.apply``);
        one ``op.solve`` per merged implicit operator (dt_gamma =
        gamma*dt, traced) with explicit-only fields taking the combine
        directly; the tick; then S3' ADVANCE and the once-per-step S4
        CONSTRAINT (project-the-state after the solves).

        Parameters
        ----------
        stepper_state : IMEXState
            The carry entry.
        state : VectorField
            The full assembled state vector.
        stages : BoundSchedule
            The per-step stage-group view (deviation D-1).
        clock : Clock
            The pre-step clock.

        Returns
        -------
        tuple[IMEXState, VectorField, Clock]
            The advanced carry entries.
        """
        dt = self.dt
        names = stages.schedule.prognostic
        # -- P0 + S1/S1' + S2 at the pre-tick time -------------------
        ctx = stages.context(clock, dt=dt, stage_dt=dt)
        state = stages.prepare(state, ctx)
        sums = stages.tendency(state, ctx)
        # -- newest-first F levels over the past-only carry ----------
        f_levels = (sums.explicit, *stepper_state.f_history)
        prognostic = _prognostic(state, names)
        if self._state_depth > 1:
            x_history = (prognostic,
                         *stepper_state.x_history[:-1])
        else:
            x_history = ()
        x_current = (prognostic, *stepper_state.x_history)
        # -- warm-up level gather (branch-free) ----------------------
        real = dtype_real()
        explicit_w = jnp.asarray(
            self._explicit_table, dtype=real)[stepper_state.warmup]
        state_w = jnp.asarray(
            self._state_table, dtype=real)[stepper_state.warmup]
        apply_w = jnp.asarray(
            self._apply_row, dtype=real)[stepper_state.warmup]
        gamma = jnp.asarray(
            self._gamma_row, dtype=real)[stepper_state.warmup]
        dt_gamma = gamma * dt
        # -- forward applies (fresh each step; CNAB only) ------------
        implicit_ops = stages.implicit
        applies = None
        if self._needs_apply and implicit_ops:
            applies = _forward_applies(
                implicit_ops, state, ctx, sums.explicit)
        # -- rhs combine (premultiplied, ascending j) ----------------
        rhs = _scaled(x_current[0], state_w[0])
        for j in range(1, self._state_depth):
            rhs = rhs + _scaled(x_current[j], state_w[j])
        for j in range(self._explicit_depth):
            rhs = rhs + _scaled(f_levels[j], explicit_w[j] * dt)
        if applies is not None:
            rhs = rhs + _scaled(applies, apply_w * dt)
        # -- solves (per merged operator); explicit-only take rhs ----
        updates: dict[str, ScalarField] = {
            name: rhs[name] for name in names}
        for operator in implicit_ops:
            op_rhs = {name: rhs[name] for name in operator.fields}
            updates.update(operator.solve(op_rhs, dt_gamma, ctx))
        state = state.replace(**updates)                      # S3
        clock = clock.tick(dt)
        # -- post-advance groups at the ticked clock -----------------
        post = TendencySums(explicit=sums.explicit, implicit=applies)
        ctx = stages.context(clock, dt=dt, stage_dt=dt, sums=post)
        if stages.schedule.kind_entries(StageKind.ADVANCE):
            state = stages.advance_stages(state, ctx)          # S3'
        state = stages.constrain(state, ctx)                  # S4
        warmup = jnp.minimum(stepper_state.warmup + 1,
                             self._n_levels - 1)
        # structural ring shift (dataflow renaming): carry only the
        # explicit_depth-1 newest F levels — the oldest is dead
        return IMEXState(f_levels[:-1], x_history, warmup), state, clock

    def time_discretization_effect(
        self,
        omega: np.ndarray,
        *,
        dt: float | None = None,
    ) -> np.ndarray:
        """Deferred (no parity predecessor)."""
        raise NotImplementedError(
            "IMEXMultistep defines no discrete-dispersion analysis "
            "(deferred — no parity predecessor)")

    def __repr__(self) -> str:
        """Compact host-side summary."""
        return (f"IMEXMultistep(dt={self.dt!r}, "
                f"scheme={self._scheme!r})")


# ================================================================
#  Thin factories (deviation D-5)
# ================================================================
def CNAB2(dt: float | np.timedelta64) -> IMEXMultistep:  # noqa: N802 — spec spelling
    """
    ``fr.time_steppers.CNAB2(dt=...)`` — Crank-Nicolson / AB2.

    Parameters
    ----------
    dt : float | np.timedelta64
        The signed step size.

    Returns
    -------
    IMEXMultistep
        The configured driver (``scheme="cnab2"``).
    """
    return IMEXMultistep(dt, scheme="cnab2")


def SBDF2(dt: float | np.timedelta64) -> IMEXMultistep:  # noqa: N802 — spec spelling
    """
    ``fr.time_steppers.SBDF2(dt=...)`` — semi-implicit BDF2.

    Parameters
    ----------
    dt : float | np.timedelta64
        The signed step size.

    Returns
    -------
    IMEXMultistep
        The configured driver (``scheme="sbdf2"``).
    """
    return IMEXMultistep(dt, scheme="sbdf2")


# ================================================================
#  Internals
# ================================================================
def _prognostic(
    state: VectorField, names: tuple[str, ...],
) -> VectorField:
    """Extract the PROGNOSTIC subset of a full state, in order."""
    from fridom.spatial.fields.vector_field import (  # noqa: PLC0415 — deferred: avoid a field-core import cycle at module load
        VectorField,
    )
    return VectorField({name: state[name] for name in names})


def _zero_vector(vector: VectorField) -> VectorField:
    """Zero twin of a PROGNOSTIC vector (stored-array zeros).

    Spelled on the storage frame: ``pad`` zero-fills the ghost lanes
    of ``with_data(zeros(true))`` and ``zeros_like(storage)`` is zero
    everywhere, so the two are bitwise-identical on every slot (not
    just the true DOFs), without the ``unpad``/``pad`` round trip.
    """
    return vector.map(
        lambda field: field.with_storage(jnp.zeros_like(field.storage)))


def _forward_applies(
    operators: tuple[BoundImplicitOperator, ...],
    state: VectorField,
    ctx: StepContext,
    template: VectorField,
) -> VectorField:
    """
    Sum the merged operators' forward applies into a PROGNOSTIC vector.

    Description
    -----------
    The CN forward-apply term ``sum_op L @ X`` recomputed fresh each
    step (never buffered — the solve-only trick reads the
    pre-projection state, an O(dt) error). Built on a zero
    PROGNOSTIC template so uncovered fields carry an exact zero.
    """
    applies = _zero_vector(template)
    for operator in operators:
        applies = applies.add(**dict(operator.apply(state, ctx)))
    return applies


def _scaled(vector: VectorField, weight: jax.Array) -> VectorField:
    """
    Scale one PROGNOSTIC vector by a traced scalar, componentwise.

    Description
    -----------
    The traced-scalar scaling spelled on the **storage frame**:
    scaling commutes with the unpad slice, so the true DOFs are
    bitwise-identical to ``with_data(weight * field.data)`` (probe P1)
    while skipping the ``unpad``/``pad`` round trip; the fresh field
    claims zero ghost validity, so its scaled ghost lanes are re-synced
    before any consumer reads them. The multiplication order is
    ``weight * data`` — part of the parity op sequence.
    """
    return vector.map(
        lambda field: field.with_storage(weight * field.storage))
