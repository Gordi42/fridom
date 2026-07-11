"""
The time-stepper protocol.

Description
-----------
``TimeStepper`` (the scan-body stepper ABC) and the ``StepperState``
conventions. Owning class spec:
``design/specs/model/classes/time_steppers.md`` ("TimeStepper",
"StepperState (conventions)"); design source
``design/specs/model/03_time_stepping.md`` section 5.3.

The stepper is **not a Module but is a pytree** whose only dynamic
leaf is ``dt`` — every concrete class applies
``@partial(fr.utils.jaxify, dynamic=("dt",))`` so all other
attributes (order, eps, tables, tableaus) land in the static treedef
aux, hashed by the restart fingerprint. The stepper is a
loop-invariant traced input to the jitted chunk, never carry; it is
host-constructed and contributes nothing host-side to a step.

StepperState conventions (per-family frozen jaxified classes, no
base class with behavior):

- rings are tuples of States, **newest first**, shifted structurally
  (``(newest, *old[:-1])`` — pure dataflow renaming, never a stacked
  array plus roll); entries are PROGNOSTIC-only tendency vectors
  shaped like ``init``'s template;
- the warm-up counter is a saturating stepper-local int32 scalar
  (``jnp.minimum(counter + 1, levels - 1)``) selecting a row of a
  dense zero-padded static table — one branch-free gather per step,
  never ``clock.it`` (``reset()`` re-warms);
- explicit rings store the **summed** explicit contribution,
  **unprojected**; the implicit side buffers nothing;
- explicit fixed-step RK carries the unit pytree ``()`` — only
  multistep memory earns carry.
"""
# Wave 4 B: TimeStepper, StepperState conventions
from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Any, ClassVar, TypeAlias

import jax.numpy as jnp
import numpy as np

from fridom.framework.utils import dtype_real, to_seconds
from fridom.model.parameters import ParameterDeclaration
from fridom.model.params import TIME_STEP

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.model.clock import Clock
    from fridom.model.schedule import BoundSchedule
    from fridom.model.terms import Treatment
    from fridom.spatial.fields.vector_field import VectorField

# Per-family carried numeric state: a convention plus per-family
# frozen jaxified classes (ABState, IMEXState, RK's ()), not a base
# class with behavior. Fully dynamic carry — flattened and donated
# with it.
StepperState: TypeAlias = Any


# ================================================================
#  TimeStepper
# ================================================================
class TimeStepper(abc.ABC):

    """
    ABC of scan-body time steppers; a pytree, not a Module.

    Description
    -----------
    The scan-body stepper protocol: ``init``/``step`` plus host-side
    analysis. Concrete classes jaxify with ``dynamic=("dt",)`` —
    ``dt`` is the one dynamic scalar leaf (the sweep parameter par
    excellence; a sign flip gives backward runs); everything else is
    static treedef aux entering the restart fingerprint via
    :meth:`fingerprint_token`.

    The stepper joins the assembly binding table as provider of
    ``fr.params.TIME_STEP`` (its ``dt`` leaf) through
    :attr:`parameter_declarations` — the read surface for cfl-style
    host diagnostics and the write surface for
    ``update_parameters`` (including the sign flip that replaces
    ``run_backward``).

    Attributes
    ----------
    supported_treatments : ClassVar[frozenset[Treatment]]
        Term treatments this stepper integrates (design 03, section
        5.1). Assembly errors on an unsupported treatment — never
        silent demotion.
    dt : jax.Array
        The dynamic signed scalar step leaf (``dtype_real``); the
        live host-side read surface — in-trace consumers read
        ``ctx.dt``/``ctx.stage_dt``.

    Parameters
    ----------
    dt : float | np.timedelta64
        The signed step size; converted once at this boundary
        (``fr.utils.to_seconds`` -> ``dtype_real`` -> ``asarray``) —
        a ``timedelta64`` dies here.

    Raises
    ------
    TypeError
        If ``dt`` is not a real number or ``np.timedelta64``.
    """

    supported_treatments: ClassVar[frozenset[Treatment]]

    def __init__(self, dt: float | np.timedelta64) -> None:
        """Convert dt once onto the dynamic leaf; see class doc."""
        if isinstance(dt, np.timedelta64):
            dt = to_seconds(dt)
        if isinstance(dt, bool) or not isinstance(
                dt, int | float | np.floating | np.integer):
            raise TypeError(
                f"dt is a signed step size in seconds (a real "
                f"number or np.timedelta64), got {dt!r}")
        self.dt = jnp.asarray(float(dt), dtype=dtype_real())

    # ================================================================
    #  The assembly provider seam (binding table, step 2)
    # ================================================================
    @property
    def parameter_declarations(
        self,
    ) -> tuple[ParameterDeclaration, ...]:
        """
        Provider rows for assembly step 2 (the binding-table seam).

        Description
        -----------
        ``ParameterBindingTable.build`` duck-types the stepper
        through this attribute: the stepper provides
        ``fr.params.TIME_STEP`` from its dynamic ``dt`` leaf.
        Subclasses publishing additional dynamic leaves extend the
        tuple.

        Returns
        -------
        tuple[ParameterDeclaration, ...]
            The provider declarations.
        """
        return (ParameterDeclaration(
            TIME_STEP, attr="dt", units="s",
            doc="the stepper's time-step leaf"),)

    @property
    def provided_parameters(self) -> Mapping[str, str]:
        """
        The provided-parameter map ``{dotted name: leaf attr}``.

        Returns
        -------
        Mapping[str, str]
            ``{fr.params.TIME_STEP: "dt"}`` (plus any subclass
            extensions of :attr:`parameter_declarations`).
        """
        return {str(declaration.name): declaration.attr
                for declaration in self.parameter_declarations}

    # ================================================================
    #  The scan-body protocol
    # ================================================================
    @property
    def scan_unroll(self) -> int:
        """
        The chunk scan's preferred unroll factor (static).

        Description
        -----------
        Steppers whose carry rotates with a period — the AB
        tendency ring — return that period: inside ``lax.scan`` the
        carry slots are fixed buffers, so a structural ring shift
        costs ``period - 1`` full-field device copies per component
        per step (~2 ms/step at 256^3 on an A100, 2026-07-12
        profile), while at ``unroll = period`` the shift is pure
        dataflow renaming and every carry slot receives a freshly
        computed value at the body boundary. Numerics are
        unchanged — unrolling repeats the identical step body. The
        default is 1 (no unroll).

        Returns
        -------
        int
            The unroll factor (>= 1).
        """
        return 1

    @abc.abstractmethod
    def init(self, tendency_template: VectorField) -> StepperState:
        """
        Return a fresh ``StepperState``.

        Description
        -----------
        Zeroed rings shaped like the PROGNOSTIC tendency template,
        warm-up counter 0. Called at assembly step 8 and by
        ``model.reset()`` — ``init`` IS the re-warm (the
        OptimalBalance ramp-leg contract); no ``reset``/``_on_setup``
        successor exists.

        Parameters
        ----------
        tendency_template : VectorField
            The zero PROGNOSTIC tendency vector.

        Returns
        -------
        StepperState
            The fresh carry entry.
        """

    @abc.abstractmethod
    def step(
        self,
        stepper_state: StepperState,
        state: VectorField,
        stages: BoundSchedule,
        clock: Clock,
    ) -> tuple[StepperState, VectorField, Clock]:
        """
        Advance one canonical step (design 03, section 5.2).

        Description
        -----------
        Per substage: P0 ctx -> S1/S1' prepare -> S2 terms -> S3
        advance -> S3' ADVANCE stages -> S4 CONSTRAINT; the stepper
        owns ``clock.tick(dt)``. S5 (NaN seam) and S6 (DIAGNOSTIC
        stages) are the chunk body's per-step epilogue, never the
        stepper's. ``stages`` arrives as a ``BoundSchedule`` — the
        static schedule closed over the carry's *current* module
        pytree by the chunk body (deviation D-1; unbound term fns
        meet live module leaves there and only there).

        Parameters
        ----------
        stepper_state : StepperState
            This stepper's carry entry.
        state : VectorField
            The full assembled state vector.
        stages : BoundSchedule
            The per-step stage-group view.
        clock : Clock
            The pre-step clock.

        Returns
        -------
        tuple[StepperState, VectorField, Clock]
            The advanced carry entries.
        """

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
        Discrete-dispersion analysis of a frequency array.

        Description
        -----------
        Host/CPU-side, never traced. ``omega`` is a plain array —
        Symbol callers materialize first (the stepper stays
        grid-free); applied by recipes (``single_wave``). The base
        raises: ``AdamBashforth`` implements it in iteration 1; the
        RK and IMEX successors are deferred with parity (the old RK
        had none).

        Parameters
        ----------
        omega : np.ndarray
            Continuous frequencies (rad/s).
        dt : float | None, optional
            Override of the live ``dt`` leaf (default: None).

        Returns
        -------
        np.ndarray
            The discrete frequencies.

        Raises
        ------
        NotImplementedError
            Always, on the base class.
        """
        raise NotImplementedError(
            f"{type(self).__name__} defines no discrete-dispersion "
            "analysis (deferred with parity — the old RK had none); "
            "AdamBashforth implements it")

    # ================================================================
    #  The restart fingerprint (stepper statics)
    # ================================================================
    def fingerprint_token(self) -> tuple:
        """
        Return the stepper-statics fingerprint contribution.

        Description
        -----------
        Hashable, human-diffable rows: the class name plus the
        per-class statics (order/eps/tableau/scheme) from
        :meth:`_statics`. The dynamic ``dt`` leaf is deliberately
        excluded (fingerprints hash structure, never leaves —
        02_rules); a mismatch diffs ("stepper statics differ: cnab2
        -> sbdf2"), never silently reuses history.

        Returns
        -------
        tuple
            The static identity token.
        """
        return (type(self).__name__, *self._statics())

    def _statics(self) -> tuple:
        """Per-class ``(name, value)`` static rows; default none."""
        return ()

    # ================================================================
    #  Introspection
    # ================================================================
    def __repr__(self) -> str:
        """Compact host-side summary (class name + dt leaf)."""
        return f"{type(self).__name__}(dt={self.dt!r})"
