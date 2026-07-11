"""
Explicit Runge-Kutta steppers.

Description
-----------
Wave 5 B: ``ButcherTableau``, the ``tableaus`` presets,
``ExplicitRungeKutta`` (fixed-step, over a tableau), and
``LowStorageRK3`` (the flagship 2N-register RK3). Owning class spec:
``design/specs/model/classes/time_steppers.md`` ("The explicit RK
family"); design source ``design/specs/model/03_time_stepping.md``
sections 5.2/5.3/5.6/5.7.

Explicit fixed-step RK carries the unit pytree ``()`` — only multistep
memory earns carry; stage values ``k_i`` are locals, recomputed per
step. Projection is project-the-state (section 5.6): RK runs
``constrain`` per produced stage state AND on the final combination.
Stage times are ``clock.shifted(c_i * dt)`` so ``eval_params`` (P0)
resolves a Ramp forcing at the correct sub-stage time with zero extra
machinery. The final combination is a separate substage (section 5.2).
"""
# Wave 5 B: ButcherTableau, tableaus, ExplicitRungeKutta, LowStorageRK3
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, ClassVar, Final

from fridom.framework.utils import jaxify
from fridom.framework2.model.stages import StageKind
from fridom.framework2.model.terms import Treatment
from fridom.framework2.model.time_steppers.base import TimeStepper

if TYPE_CHECKING:  # pragma: no cover
    import jax
    import numpy as np

    from fridom.framework2.grid.fields.vector_field import VectorField
    from fridom.framework2.model.clock import Clock
    from fridom.framework2.model.schedule import BoundSchedule


# ================================================================
#  ButcherTableau
# ================================================================
@dataclass(frozen=True)
class ButcherTableau:

    """
    Static explicit-RK tableau; tuples for hashability.

    Description
    -----------
    A frozen, hashable coefficient bundle (nested tuples, not arrays)
    so it enters the restart fingerprint through its stepper's treedef
    aux. ``b_error`` retains the embedded-error weights as
    designed-for data (the adaptive ``AdaptiveRungeKutta``); it is
    unused by fixed-step stepping, and a tableau carrying it is
    REJECTED by :class:`ExplicitRungeKutta` (adaptive stepping is a
    class, not a flag).

    Parameters
    ----------
    a : tuple[tuple[float, ...], ...]
        The strictly-lower-triangular stage-coupling matrix.
    b : tuple[float, ...]
        The final-combination weights (``len(b)`` is the stage count).
    c : tuple[float, ...]
        The stage times (fractions of dt).
    b_error : tuple[float, ...] | None, optional
        Embedded-error weights — designed-for data, unused fixed-step
        (default: None).
    """

    a: tuple[tuple[float, ...], ...]
    b: tuple[float, ...]
    c: tuple[float, ...]
    b_error: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        """Normalize the coefficient containers to nested tuples."""
        object.__setattr__(
            self, "a", tuple(tuple(row) for row in self.a))
        object.__setattr__(self, "b", tuple(self.b))
        object.__setattr__(self, "c", tuple(self.c))
        if self.b_error is not None:
            object.__setattr__(
                self, "b_error", tuple(self.b_error))

    @property
    def stages(self) -> int:
        """Number of stages, ``len(b)``."""
        return len(self.b)

    @property
    def is_embedded(self) -> bool:
        """Whether this tableau carries embedded-error weights."""
        return self.b_error is not None


# ================================================================
#  tableaus — the module-level presets (old RKMethods, as data)
# ================================================================
class tableaus:  # noqa: N801 — a namespace of preset constants

    """The explicit-RK tableau presets (parity with old RKMethods)."""

    EULER: Final[ButcherTableau] = ButcherTableau(
        a=((0.0,),),
        b=(1.0,),
        c=(0.0,),
    )

    RK2: Final[ButcherTableau] = ButcherTableau(
        a=((0.0, 0.0),
           (1 / 2, 0.0)),
        b=(0.0, 1.0),
        c=(0.0, 1 / 2),
    )

    RK3: Final[ButcherTableau] = ButcherTableau(
        a=((0.0, 0.0, 0.0),
           (1 / 2, 0.0, 0.0),
           (-1.0, 2.0, 0.0)),
        b=(1 / 6, 2 / 3, 1 / 6),
        c=(0.0, 1 / 2, 1.0),
    )

    RK4: Final[ButcherTableau] = ButcherTableau(
        a=((0.0, 0.0, 0.0, 0.0),
           (1 / 2, 0.0, 0.0, 0.0),
           (0.0, 1 / 2, 0.0, 0.0),
           (0.0, 0.0, 1.0, 0.0)),
        b=(1 / 6, 1 / 3, 1 / 3, 1 / 6),
        c=(0.0, 1 / 2, 1 / 2, 1.0),
    )

    RK4_38: Final[ButcherTableau] = ButcherTableau(
        a=((0.0, 0.0, 0.0, 0.0),
           (1 / 3, 0.0, 0.0, 0.0),
           (-1 / 3, 1.0, 0.0, 0.0),
           (1.0, -1.0, 1.0, 0.0)),
        b=(1 / 8, 3 / 8, 3 / 8, 1 / 8),
        c=(0.0, 1 / 3, 2 / 3, 1.0),
    )

    # -- adaptive (designed-for) b_error carriers --------------------
    HEUN_EULER: Final[ButcherTableau] = ButcherTableau(
        a=((0.0, 0.0),
           (1.0, 0.0)),
        b=(1 / 2, 1 / 2),
        c=(0.0, 1.0),
        b_error=(1 / 2, -1 / 2),
    )

    BOGACKI_SHAMPINE: Final[ButcherTableau] = ButcherTableau(
        a=((0.0, 0.0, 0.0, 0.0),
           (1 / 2, 0.0, 0.0, 0.0),
           (0.0, 3 / 4, 0.0, 0.0),
           (2 / 9, 1 / 3, 4 / 9, 0.0)),
        b=(7 / 24, 1 / 4, 1 / 3, 1 / 8),
        c=(0.0, 1 / 2, 3 / 4, 1.0),
        b_error=(2 / 9 - 7 / 24, 1 / 3 - 1 / 4, 4 / 9 - 1 / 3, -1 / 8),
    )

    RKF45: Final[ButcherTableau] = ButcherTableau(
        a=((0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
           (1 / 4, 0.0, 0.0, 0.0, 0.0, 0.0),
           (3 / 32, 9 / 32, 0.0, 0.0, 0.0, 0.0),
           (1932 / 2197, -7200 / 2197, 7296 / 2197, 0.0, 0.0, 0.0),
           (439 / 216, -8.0, 3680 / 513, -845 / 4104, 0.0, 0.0),
           (-8 / 27, 2.0, -3544 / 2565, 1859 / 4104, -11 / 40, 0.0)),
        b=(16 / 135, 0.0, 6656 / 12825, 28561 / 56430, -9 / 50,
           2 / 55),
        c=(0.0, 1 / 4, 3 / 8, 12 / 13, 1.0, 1 / 2),
        b_error=(-1 / 360, 0.0, 128 / 4275, 2197 / 75240, -1 / 50,
                 -2 / 55),
    )


# ================================================================
#  ExplicitRungeKutta
# ================================================================
@partial(jaxify, dynamic=("dt",))
class ExplicitRungeKutta(TimeStepper):

    """
    Fixed-step explicit RK over a Butcher tableau.

    Description
    -----------
    A pytree whose only dynamic leaf is ``dt``; the tableau is static
    treedef aux, hashed by the restart fingerprint. Embedded tableaus
    (``b_error`` set) are rejected — adaptive stepping is the
    designed-for ``AdaptiveRungeKutta``, not a flag. ``StepperState``
    is the unit pytree ``()``.

    Parameters
    ----------
    dt : float | np.timedelta64
        The signed step size (converted once; see ``TimeStepper``).
    tableau : ButcherTableau, optional
        The fixed-step tableau (default: ``tableaus.RK4``).

    Raises
    ------
    TypeError
        If ``tableau`` is not a ``ButcherTableau``.
    ValueError
        If ``tableau`` carries embedded-error weights.
    """

    supported_treatments: ClassVar[frozenset[Treatment]] = (
        frozenset({Treatment.EXPLICIT}))

    def __init__(
        self,
        dt: float | np.timedelta64,
        tableau: ButcherTableau = tableaus.RK4,
    ) -> None:
        """Validate the tableau; see the class docstring."""
        super().__init__(dt)
        if not isinstance(tableau, ButcherTableau):
            raise TypeError(
                f"tableau must be a ButcherTableau, got {tableau!r}")
        if tableau.is_embedded:
            raise ValueError(
                "ExplicitRungeKutta is fixed-step; the embedded "
                "tableau (b_error is set) is rejected — adaptive "
                "stepping is the designed-for AdaptiveRungeKutta "
                "class, not a flag on the fixed-step driver")
        self._tableau = tableau

    @property
    def tableau(self) -> ButcherTableau:
        """The static tableau; fingerprinted."""
        return self._tableau

    def _statics(self) -> tuple:
        """Contribute the tableau to the restart fingerprint."""
        return (("tableau",
                 (self._tableau.a, self._tableau.b, self._tableau.c)),)

    def init(
        self, tendency_template: VectorField,  # noqa: ARG002
    ) -> tuple[()]:
        """Return the unit pytree ``()`` — RK carries nothing."""
        return ()

    def step(
        self,
        stepper_state: tuple[()],  # noqa: ARG002
        state: VectorField,
        stages: BoundSchedule,
        clock: Clock,
    ) -> tuple[tuple[()], VectorField, Clock]:
        """
        Advance one fixed-step RK step (statically unrolled).

        Description
        -----------
        Per stage ``i`` (static unroll): the stage clock
        ``clock.shifted(c_i * dt)`` builds ``ctx_i`` (eval_params at
        the stage time — P0), ``prepare`` runs S1/S1', the stage state
        is ``state`` plus the ``a[i][j]``-weighted ``k_j``, ``k_i`` is
        that stage's explicit tendency, and ``constrain`` projects the
        stage state (per-substage S4). The final combination is a
        SEPARATE substage: the ``b``-weighted ``k_j`` (premultiplied
        ``b_j * dt``, ascending j — the AB parity arithmetic), the
        tick, then S3' ADVANCE and S4 CONSTRAINT at the ticked clock.

        Parameters
        ----------
        stepper_state : tuple[()]
            The unit carry.
        state : VectorField
            The full assembled state vector.
        stages : BoundSchedule
            The per-step stage-group view (deviation D-1).
        clock : Clock
            The pre-step clock.

        Returns
        -------
        tuple[tuple[()], VectorField, Clock]
            The advanced carry entries.
        """
        tableau = self._tableau
        dt = self.dt
        k: list[VectorField] = []
        for i in range(tableau.stages):
            ctx_i = stages.context(
                clock.shifted(tableau.c[i] * dt), dt=dt, stage_dt=dt)
            if i == 0:
                stage_state = state
            else:
                increment = _combine(tableau.a[i][:i], k, dt)
                stage_state = state.add(**dict(increment.components))
            stage_state = stages.prepare(stage_state, ctx_i)
            k.append(stages.tendency(stage_state, ctx_i).explicit)
            # per-produced-state projection (section 5.6)
            stages.constrain(stage_state, ctx_i)
        # -- the final combination substage --------------------------
        increment = _combine(tableau.b, k, dt)
        state = state.add(**dict(increment.components))
        clock = clock.tick(dt)
        ctx = stages.context(clock, dt=dt, stage_dt=dt)
        if stages.schedule.kind_entries(StageKind.ADVANCE):
            state = stages.advance_stages(state, ctx)
        state = stages.constrain(state, ctx)
        return (), state, clock

    def time_discretization_effect(
        self,
        omega: np.ndarray,
        *,
        dt: float | None = None,
    ) -> np.ndarray:
        """Deferred with parity (the old RK had none)."""
        raise NotImplementedError(
            "ExplicitRungeKutta defines no discrete-dispersion "
            "analysis (deferred with parity — the old RK had none)")

    def __repr__(self) -> str:
        """Compact host-side summary."""
        return (f"ExplicitRungeKutta(dt={self.dt!r}, "
                f"stages={self._tableau.stages})")


# ================================================================
#  LowStorageRK3 — the flagship 2N-register RK3
# ================================================================
# The Oceananigans RK3 reference coefficients (Le & Moin 1991,
# 3-stage 2N-storage): per stage (gamma_i, zeta_i, c_i). The native
# two-register recurrence advances the running state by
# dt*(gamma_i*G_i + zeta_i*G_prev) with zeta_0 zero, evaluating each
# stage tendency G_i at t + c_i*dt and carrying G_prev forward to the
# next stage. Here gamma = (8/15, 5/12, 3/4), zeta = (0, -17/60,
# -5/12), and the evaluation offsets c = (0, 8/15, 2/3) are the
# running sub-step fractions the state reaches before each stage; the
# scheme integrates to exactly t + dt. Pinned and fingerprinted per
# deviation D-6.
_RK3_COEFFICIENTS: Final[tuple[tuple[float, float, float], ...]] = (
    (8 / 15, 0.0, 0.0),
    (5 / 12, -17 / 60, 8 / 15),
    (3 / 4, -5 / 12, 2 / 3),
)


@partial(jaxify, dynamic=("dt",))
class LowStorageRK3(TimeStepper):

    """
    Three-stage low-storage (2N-register) explicit RK3.

    Description
    -----------
    The documented recommendation for new configurations (the nh
    preset still pins ``AdamBashforth(order=3)`` at cutover, V-N3). A
    pytree whose only dynamic leaf is ``dt``; the coefficient set is
    static treedef aux (fingerprinted). ``StepperState`` is ``()`` —
    the two "registers" (the running state and the previous stage
    tendency) are step-locals; only multistep memory earns carry.

    The coefficient set is pinned against the Oceananigans RK3
    reference (deviation D-6): ``gamma = (8/15, 5/12, 3/4)``,
    ``zeta = (0, -17/60, -5/12)``, tendency-evaluation offsets
    ``c = (0, 8/15, 2/3)``.

    Parameters
    ----------
    dt : float | np.timedelta64
        The signed step size (converted once; see ``TimeStepper``).
    """

    supported_treatments: ClassVar[frozenset[Treatment]] = (
        frozenset({Treatment.EXPLICIT}))

    def __init__(self, dt: float | np.timedelta64) -> None:
        """Pin the fixed coefficient set (no tableau argument)."""
        super().__init__(dt)
        self._coefficients = _RK3_COEFFICIENTS

    @property
    def coefficients(
        self,
    ) -> tuple[tuple[float, float, float], ...]:
        """Static ``(gamma_i, zeta_i, c_i)`` triples; fingerprinted."""
        return self._coefficients

    def _statics(self) -> tuple:
        """Contribute the coefficient set to the fingerprint."""
        return (("coefficients", self._coefficients),)

    def init(
        self, tendency_template: VectorField,  # noqa: ARG002
    ) -> tuple[()]:
        """Return the unit pytree ``()``."""
        return ()

    def step(
        self,
        stepper_state: tuple[()],  # noqa: ARG002
        state: VectorField,
        stages: BoundSchedule,
        clock: Clock,
    ) -> tuple[tuple[()], VectorField, Clock]:
        """
        Advance one low-storage RK3 step (2N-register recurrence).

        Description
        -----------
        Per stage the tendency is evaluated at ``clock.shifted(c_i *
        dt)`` (P0 eval_params at the sub-step time), the running state
        is advanced by ``dt * (gamma_i * G_i + zeta_i * G_prev)`` (the
        premultiplied AB parity arithmetic), and the stage state is
        projected (per-produced-state S4). After the last stage the
        clock ticks and the S3' ADVANCE / S4 CONSTRAINT groups run at
        the ticked clock (following the ``ExplicitRungeKutta``
        algorithm verbatim).

        Parameters
        ----------
        stepper_state : tuple[()]
            The unit carry.
        state : VectorField
            The full assembled state vector.
        stages : BoundSchedule
            The per-step stage-group view (deviation D-1).
        clock : Clock
            The pre-step clock.

        Returns
        -------
        tuple[tuple[()], VectorField, Clock]
            The advanced carry entries.
        """
        dt = self.dt
        previous: VectorField | None = None
        for i, (gamma, zeta, c) in enumerate(self._coefficients):
            ctx_i = stages.context(
                clock.shifted(c * dt), dt=dt, stage_dt=dt)
            state = stages.prepare(state, ctx_i)
            tendency = stages.tendency(state, ctx_i).explicit
            increment = _scaled(tendency, gamma * dt)
            if i > 0:
                increment = increment + _scaled(previous, zeta * dt)
            state = state.add(**dict(increment.components))
            state = stages.constrain(state, ctx_i)
            previous = tendency
        clock = clock.tick(dt)
        ctx = stages.context(clock, dt=dt, stage_dt=dt)
        if stages.schedule.kind_entries(StageKind.ADVANCE):
            state = stages.advance_stages(state, ctx)
        state = stages.constrain(state, ctx)
        return (), state, clock

    def time_discretization_effect(
        self,
        omega: np.ndarray,
        *,
        dt: float | None = None,
    ) -> np.ndarray:
        """Deferred with parity (the old RK had none)."""
        raise NotImplementedError(
            "LowStorageRK3 defines no discrete-dispersion analysis "
            "(deferred with parity — the old RK had none)")

    def __repr__(self) -> str:
        """Compact host-side summary."""
        return f"LowStorageRK3(dt={self.dt!r})"


# ================================================================
#  Internals (the premultiplied AB parity arithmetic)
# ================================================================
def _combine(
    weights: tuple[float, ...],
    vectors: list[VectorField],
    dt: jax.Array,
) -> VectorField:
    """
    Weighted combine ``sum_j (weights[j] * dt) * vectors[j]``.

    Description
    -----------
    The premultiplied (``w * dt`` first) ascending-j accumulation of
    the AB/RK parity arithmetic; ``weights`` and ``vectors`` are equal
    length (the stage's ``a[i][:i]`` row and the ``k`` locals, or the
    full ``b`` row and every ``k``).

    Parameters
    ----------
    weights : tuple[float, ...]
        The (static) combine coefficients.
    vectors : list[VectorField]
        The PROGNOSTIC-only stage tendencies.
    dt : jax.Array
        The signed step size.

    Returns
    -------
    VectorField
        The combined PROGNOSTIC-only increment.
    """
    increment = _scaled(vectors[0], weights[0] * dt)
    for j in range(1, len(vectors)):
        increment = increment + _scaled(vectors[j], weights[j] * dt)
    return increment


def _scaled(vector: VectorField, weight: jax.Array) -> VectorField:
    """
    Scale one PROGNOSTIC vector by a traced scalar, componentwise.

    Description
    -----------
    The traced-scalar scaling spelled on the true-shape data (field
    dunders accept Python scalars only); the multiplication order is
    ``weight * data`` — part of the parity op sequence.
    """
    return vector.map(
        lambda field: field.with_data(weight * field.data))
