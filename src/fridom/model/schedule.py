"""
The static stage schedule.

Description
-----------
``Schedule`` (the static, hashable, kind-ordered assembly artifact),
``BoundSchedule`` (the ephemeral in-trace view closed over the
carry's current module pytree), and ``TendencySums`` (the frozen
jaxified per-treatment tendency sums). Owning class spec:
``design/specs/model/classes/time_steppers.md`` ("Schedule
composition"); design source
``design/specs/model/03_time_stepping.md`` sections 5.2/5.5.

The ``Schedule`` is built once, by the ``TendencyComposer`` at
assembly step 5, and joins the hashable assembly record: two
identical module configurations produce equal, equally-hashing
schedules (the shared-jit-cache obligation). ``bind(carry_modules)``
returns the per-step ``BoundSchedule`` — deliberately NOT a pytree:
a plain closure view built inside the trace, so unbound hook ``fn``
slots meet live module leaves here and only here (the D2 aliasing
rule).

Wave-3 scope: the 2.4 stage-group surface (``context`` / ``prepare``
/ ``tendency`` / ``constrain`` / ``diagnostics``). The treatment
partition consumers, the merged ``implicit`` operators, and
``advance_stages`` land at wave 5 (2.5); the schedule already
carries their static data (per-term treatments, implicit-operator
groups, ADVANCE entries).
"""
# Wave 3 C: Schedule, BoundSchedule, TendencySums (treatment
#    partition/implicit land Wave 5 B)
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Any, Final

import jax
import jax.numpy as jnp

from fridom.framework.utils import jaxify
from fridom.model.context import StepContext
from fridom.model.errors import (
    AssemblyError,
    TermEvaluationError,
)
from fridom.model.stages import StageKind
from fridom.model.terms import Treatment
from fridom.spatial.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.model.implicit import ImplicitOperator
    from fridom.spatial.fields.scalar_field import ScalarField


# ================================================================
#  Kind ordering (03 section 5.2 — the canonical step)
# ================================================================
# terms are not stages; they slot between DIAGNOSE (S1') and the
# primary advance (S3).  ``None`` is the term pseudo-kind.
_TERM_RANK: Final[int] = 2

_KIND_RANKS: Final[dict[StageKind, int]] = {
    StageKind.SELF_UPDATE: 0,   # S1  — first in EVERY substage
    StageKind.DIAGNOSE: 1,      # S1'
    StageKind.ADVANCE: 3,       # S3'
    StageKind.CONSTRAINT: 4,    # S4
    StageKind.DIAGNOSTIC: 5,    # S6  (per step, post-NaN)
}


# ================================================================
#  ScheduleEntry
# ================================================================
@dataclass(frozen=True)
class ScheduleEntry:

    """
    One scheduled hook: a term or a stage, slot-paired and gated.

    Description
    -----------
    Built by the ``TendencyComposer`` from a collected
    ``TendencyTerm``/``Stage`` declaration. ``fn`` is the UNBOUND
    hook ``(module, state, ctx) -> dict``; the ``BoundSchedule``
    calls it with the carry's live module at ``slot``. The ``gate``
    is the precomputed static write gate of the owning kind
    (03 section 5.5); contribution keys outside it are an
    assembly-time error carrying the ``key`` attribution.

    Parameters
    ----------
    key : str
        The ``"Module/term"`` (or ``"Module/stage"``) attribution
        key.
    kind : StageKind | None
        The declared stage kind; ``None`` marks a tendency term.
    slot : int
        The owning module's tuple index.
    order : int
        Explicit intra-kind order (stages; terms are always 0).
    index : int
        Declaration index — the collection-order tie-breaker.
    fn : Callable | None
        The UNBOUND hook; ``None`` on terms whose explicit path is
        derived from ``implicit.apply`` (write-once, 03 section
        5.1).
    gate : tuple[str, ...]
        The static write gate (allowed contribution keys).
    treatment : Treatment | None
        The term's integration treatment; ``None`` for stages.
    advances : tuple[str, ...] | None
        Declared advanced PROGNOSTIC subset. For terms ``None``
        means underived (the dry run derives it); for ADVANCE
        stages this is the declared claim.
    reads : tuple[str, ...]
        SELF_UPDATE only: declared state inputs (V-H5).
    implicit : ImplicitOperator | None
        The implicit-operator slot (wave-5 solve seam; carried as
        static data now).
    """

    key: str
    kind: StageKind | None
    slot: int
    order: int
    index: int
    fn: Callable | None
    gate: tuple[str, ...]
    treatment: Treatment | None = None
    advances: tuple[str, ...] | None = None
    reads: tuple[str, ...] = ()
    implicit: ImplicitOperator | None = None

    @property
    def is_term(self) -> bool:
        """Whether this entry is a tendency term (not a stage)."""
        return self.kind is None

    @property
    def rank(self) -> int:
        """The kind rank in the canonical step (terms rank 2)."""
        if self.kind is None:
            return _TERM_RANK
        return _KIND_RANKS[self.kind]

    @property
    def sort_key(self) -> tuple[int, int, int, int]:
        """Deterministic order: (rank, order, slot, index)."""
        return (self.rank, self.order, self.slot, self.index)

    def static_token(self) -> tuple:
        """
        Return the hashable static identity of this entry.

        Description
        -----------
        Attribution key, kind, slot, order, index, treatment, gate,
        advances, and reads — deliberately excluding the ``fn``
        callable and the implicit operator object (identity-hashed
        host objects; two identical module configurations must
        produce equal tokens).

        Returns
        -------
        tuple
            The static identity token.
        """
        return (
            self.key,
            self.kind.name if self.kind is not None else "TERM",
            self.slot,
            self.order,
            self.index,
            self.treatment.name if self.treatment is not None
            else None,
            self.gate,
            self.advances,
            self.reads,
        )


# ================================================================
#  Hook evaluation (shared by BoundSchedule and the composer dry
#  run — one validation path, one attribution path)
# ================================================================
def zero_like(field: ScalarField) -> ScalarField:
    """
    Return a zero field on the same grid/space/metadata.

    Description
    -----------
    Zeros are built on the *stored* array (halos included, trivially
    valid) — never as ``0.0 * field``, which would poison the
    tendency template with NaNs from a non-finite state and pay an
    arithmetic pass.

    Parameters
    ----------
    field : ScalarField
        The template field.

    Returns
    -------
    ScalarField
        The zero field.
    """
    return type(field)(
        field.grid, field.function_space,
        jnp.zeros_like(field._data),  # noqa: SLF001 — plumbing-constructor seam
        field.metadata)


def evaluate_entry(
    entry: ScheduleEntry,
    module: object,
    state: VectorField | None,
    ctx: StepContext,
) -> Mapping[str, ScalarField]:
    """
    Call one hook with attribution and write-gate validation.

    Description
    -----------
    The hook runs under ``jax.named_scope(entry.key)`` (the profiler
    attribution the old per-module timers become). Any exception it
    raises is wrapped as ``TermEvaluationError`` carrying the
    ``"Module/term"`` key, chaining the original — in particular
    grid-layer space mismatches re-raise attributed. The returned
    contribution dict is checked against the entry's static write
    gate; a stray key is an assembly-time ``AssemblyError`` with the
    same attribution.

    Parameters
    ----------
    entry : ScheduleEntry
        The scheduled hook.
    module : object
        The live owning module (``carry.modules[entry.slot]``).
    state : VectorField | None
        The full assembled state vector.
    ctx : StepContext
        The per-substage context.

    Returns
    -------
    Mapping[str, ScalarField]
        The validated contribution dict.

    Raises
    ------
    TermEvaluationError
        On any exception inside the hook (original chained).
    AssemblyError
        On a non-dict return or a contribution key outside the
        write gate.
    """
    hook = entry.fn if entry.fn is not None else entry.implicit.apply
    try:
        with jax.named_scope(entry.key):
            result = hook(module, state, ctx)
    except Exception as exc:
        raise TermEvaluationError(
            f"{entry.key}: evaluation failed with "
            f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(result, Mapping):
        raise AssemblyError(
            f"{entry.key}: hooks return a dict of contributions "
            f"keyed by component name, got {type(result).__name__}")
    stray = tuple(name for name in result if name not in entry.gate)
    if stray:
        what = "term" if entry.is_term else f"{entry.kind.name} stage"
        raise AssemblyError(
            f"{entry.key}: {what} writes {stray} outside its write "
            f"gate; allowed keys: {entry.gate}")
    return result


def apply_replace(
    entry: ScheduleEntry,
    state: VectorField,
    result: Mapping[str, ScalarField],
) -> VectorField:
    """
    Apply a stage contribution dict via ``replace`` (attributed).

    Description
    -----------
    Every stage kind overwrites its declared subset (terms are the
    one ``add`` path). Result *spaces* are validated against the
    incumbent components — a wrong-staggering stage write would
    otherwise silently re-space a component.

    Parameters
    ----------
    entry : ScheduleEntry
        The scheduled stage.
    state : VectorField
        The state to update.
    result : Mapping[str, ScalarField]
        The gate-validated contribution dict.

    Returns
    -------
    VectorField
        The updated state.

    Raises
    ------
    AssemblyError
        If a result component's space differs from the incumbent's.
    TermEvaluationError
        If applying the replacement fails (original chained).
    """
    if not result:
        return state
    for name, field in result.items():
        incumbent = state[name].function_space.bare
        written = field.function_space.bare
        if written != incumbent:
            raise AssemblyError(
                f"{entry.key}: stage writes {name!r} on space "
                f"{written!r}, but the declared component space is "
                f"{incumbent!r}")
    try:
        return state.replace(**result)
    except Exception as exc:
        raise TermEvaluationError(
            f"{entry.key}: applying the stage write failed with "
            f"{type(exc).__name__}: {exc}") from exc


def apply_add(
    entry: ScheduleEntry,
    sums: VectorField | None,
    result: Mapping[str, ScalarField],
) -> VectorField | None:
    """
    Accumulate one term's contribution dict via ``add`` (attributed).

    Description
    -----------
    One ``sums.add(**term_dict)`` per term — the per-term merge
    grouping: a term's contributions land as one functional
    accumulate, in deterministic schedule order. The ScalarField
    join rule applies per component; a wrong-staggering
    contribution re-raises as ``TermEvaluationError`` with the
    attribution key (d1_5 risk 2).

    Parameters
    ----------
    entry : ScheduleEntry
        The scheduled term.
    sums : VectorField | None
        The running PROGNOSTIC-only sums (``None`` iff the
        composition declares no PROGNOSTIC fields).
    result : Mapping[str, ScalarField]
        The gate-validated contribution dict.

    Returns
    -------
    VectorField | None
        The accumulated sums.

    Raises
    ------
    TermEvaluationError
        If the accumulation fails (original chained).
    """
    if not result or sums is None:
        return sums
    try:
        return sums.add(**result)
    except Exception as exc:
        raise TermEvaluationError(
            f"{entry.key}: accumulating the term contribution "
            f"failed with {type(exc).__name__}: {exc}") from exc


# ================================================================
#  Schedule — the static assembly artifact
# ================================================================
class Schedule:

    """
    Static kind-ordered schedule; built once at assembly.

    Description
    -----------
    A host-side, hashable assembly artifact (it joins the assembly
    record that keys the shared jitted chunk). Entries are held in
    the canonical order — per substage SELF_UPDATE, DIAGNOSE, terms,
    ADVANCE, CONSTRAINT; per step DIAGNOSTIC — with the intra-kind
    tuple ``(order, slot, index)``; never list-positioned.

    Equality and hash are structural over the entries' static
    tokens (attribution keys, kinds, slots, orders, treatments,
    gates) plus the PROGNOSTIC name tuple: two identical module
    configurations produce equal, equally-hashing schedules. The
    hook callables, the implicit-operator groups, and the
    binding-table/stepper references are deliberately excluded
    (identity-hashed host objects).

    Parameters
    ----------
    entries : tuple[ScheduleEntry, ...]
        The collected entries (any order; sorted canonically here).
    prognostic : tuple[str, ...]
        PROGNOSTIC component names, declaration order.
    implicit_groups : tuple, optional
        Static merge groups of implicit operators, each a tuple of
        ``(key, slot, operator)`` constituents (wave-5 solve seam)
        (default: ()).
    implicit_merged : tuple, optional
        The per-group ``(merged_operator, slot)`` pairs — each group
        collapsed through ``merged_with`` by the composer; the wave-5
        solve surface (``BoundSchedule.implicit``) binds them
        (default: ()).
    binding_table : object | None, optional
        The parameter binding table; ``context()`` calls its
        ``eval_params(modules, stepper, t)`` (default: None).
    time_stepper : object | None, optional
        The stepper handed to ``eval_params`` (the TIME_STEP
        provider) (default: None).
    """

    def __init__(
        self,
        entries: tuple[ScheduleEntry, ...],
        *,
        prognostic: tuple[str, ...],
        implicit_groups: tuple = (),
        implicit_merged: tuple = (),
        binding_table: object | None = None,
        time_stepper: object | None = None,
    ) -> None:
        """Sort the entries canonically and freeze the statics."""
        self._entries: tuple[ScheduleEntry, ...] = tuple(
            sorted(entries, key=lambda entry: entry.sort_key))
        self._prognostic: tuple[str, ...] = tuple(prognostic)
        self._implicit_groups: tuple = tuple(implicit_groups)
        self._implicit_merged: tuple = tuple(implicit_merged)
        self._binding_table = binding_table
        self._time_stepper = time_stepper

    # ================================================================
    #  Read surface
    # ================================================================
    @property
    def entries(self) -> tuple[ScheduleEntry, ...]:
        """All entries in canonical kind order."""
        return self._entries

    @property
    def prognostic(self) -> tuple[str, ...]:
        """PROGNOSTIC component names, declaration order."""
        return self._prognostic

    @property
    def implicit_groups(self) -> tuple:
        """Static implicit-operator merge groups (wave-5 seam)."""
        return self._implicit_groups

    @property
    def implicit_merged(self) -> tuple:
        """Per-group ``(merged_operator, slot)`` pairs (wave-5)."""
        return self._implicit_merged

    @property
    def binding_table(self) -> object | None:
        """The parameter binding table (or None pre-wave-4)."""
        return self._binding_table

    @property
    def time_stepper(self) -> object | None:
        """The stepper (TIME_STEP provider) or None pre-wave-4."""
        return self._time_stepper

    def kind_entries(
        self, kind: StageKind | None,
    ) -> tuple[ScheduleEntry, ...]:
        """
        Return the entries of one kind, in schedule order.

        Parameters
        ----------
        kind : StageKind | None
            The stage kind; ``None`` selects the tendency terms.

        Returns
        -------
        tuple[ScheduleEntry, ...]
            The matching entries.
        """
        return tuple(entry for entry in self._entries
                     if entry.kind is kind)

    # ================================================================
    #  Binding
    # ================================================================
    def bind(self, modules: tuple) -> BoundSchedule:
        """
        Close the hooks over the carry's current module pytree.

        Description
        -----------
        Pure dataflow, called per step by the chunk body: unbound
        ``fn`` slots meet live module leaves here and only here (a
        schedule capturing assembly-time modules would be the D2
        aliasing trap by construction).

        Parameters
        ----------
        modules : tuple
            The carry's live module tuple.

        Returns
        -------
        BoundSchedule
            The per-step view (not a pytree).

        Raises
        ------
        AssemblyError
            If the module tuple is shorter than the schedule's
            highest slot index.
        """
        modules = tuple(modules)
        needed = max((entry.slot for entry in self._entries),
                     default=-1)
        if needed >= len(modules):
            raise AssemblyError(
                f"schedule references module slot {needed}, but "
                f"only {len(modules)} modules were bound")
        return BoundSchedule(self, modules)

    # ================================================================
    #  Introspection
    # ================================================================
    def describe(self) -> str:
        """
        Return the kind-ordered schedule listing for the report.

        Returns
        -------
        str
            One line per entry: kind, attribution key, ordering
            tuple, treatment/advances where applicable.
        """
        lines = ["Schedule (kind-ordered):"]
        for entry in self._entries:
            kind = ("TERM" if entry.is_term else entry.kind.name)
            extra = ""
            if entry.treatment is not None:
                extra += f", treatment={entry.treatment.name}"
            if entry.advances:
                extra += f", advances={entry.advances}"
            lines.append(
                f"  {kind:<11} {entry.key} (order={entry.order}, "
                f"slot={entry.slot}, index={entry.index}{extra})")
        return "\n".join(lines)

    def _token(self) -> tuple:
        """Return the structural identity token (eq/hash basis)."""
        return (
            tuple(entry.static_token() for entry in self._entries),
            self._prognostic,
        )

    def __eq__(self, other: object) -> bool:
        """Structural equality over the static tokens."""
        if not isinstance(other, Schedule):
            return NotImplemented
        return self._token() == other._token()

    def __hash__(self) -> int:
        """Structural hash, matching ``__eq__``."""
        return hash(self._token())

    def __repr__(self) -> str:
        """Compact summary (entry count per kind)."""
        terms = len(self.kind_entries(None))
        stages = len(self._entries) - terms
        return (f"Schedule({terms} terms, {stages} stages, "
                f"prognostic={self._prognostic!r})")


# ================================================================
#  BoundSchedule — the ephemeral per-step view
# ================================================================
class BoundSchedule:

    """
    Per-step view: stage groups as ``(state, ctx)`` callables.

    Description
    -----------
    NOT a pytree — an ephemeral closure over the carry's current
    module tuple, built by ``Schedule.bind`` inside the trace and
    discarded at the end of the step. Steppers consume the stage
    groups; the chunk body consumes ``diagnostics`` in the per-step
    epilogue.

    Parameters
    ----------
    schedule : Schedule
        The static schedule.
    modules : tuple
        The carry's live module tuple (already length-checked by
        ``Schedule.bind``).
    """

    def __init__(self, schedule: Schedule, modules: tuple) -> None:
        """Store the closure state; see the class docstring."""
        self._schedule = schedule
        self._modules = modules

    @property
    def schedule(self) -> Schedule:
        """The underlying static schedule."""
        return self._schedule

    # ================================================================
    #  P0 — context construction
    # ================================================================
    def context(
        self,
        clock: Any,
        *,
        dt: Any,
        stage_dt: Any,
        sums: TendencySums | None = None,
    ) -> StepContext:
        """
        Build the frozen ``StepContext`` for one (sub)stage (P0).

        Description
        -----------
        Parameters are evaluated at the *stage time* through the
        binding table (``eval_params(modules, stepper, t)`` — the D2
        obligation; a ramped scalar and a ramped profile see the
        same time). Without a binding table (pre-wave-4 wiring) the
        params mapping is empty. ``sums`` attaches the
        per-treatment tendency sums for post-TENDENCY stages.

        Parameters
        ----------
        clock : Any
            The (stage) clock; its ``time`` attribute — or the
            value itself for scalar clocks — is the stage time.
        dt : Any
            The full step size.
        stage_dt : Any
            The increment of the current advance (signed).
        sums : TendencySums | None, optional
            Per-treatment sums for post-TENDENCY hooks
            (default: None).

        Returns
        -------
        StepContext
            The frozen per-substage context.
        """
        table = self._schedule.binding_table
        if table is None:
            params: Mapping[str, Any] = {}
        else:
            time = getattr(clock, "time", clock)
            params = table.eval_params(
                self._modules, self._schedule.time_stepper, time)
        return StepContext(params=params, clock=clock, dt=dt,
                           stage_dt=stage_dt, tendency_sums=sums)

    # ================================================================
    #  Stage groups (kind-ordered; replace-applied)
    # ================================================================
    def prepare(
        self, state: VectorField, ctx: StepContext,
    ) -> VectorField:
        """
        Run S1 SELF_UPDATE then S1' DIAGNOSE stages, kind-ordered.

        Description
        -----------
        SELF_UPDATE-first is load-bearing (02_rules): owners
        recompute their AUXILIARY fields from the stage clock
        before any consumer — including every DIAGNOSE stage —
        reads. S1' placement is equally load-bearing: the first
        substage after ``set_state``/restart recomputes diagnosed
        fields before any term reads.

        Parameters
        ----------
        state : VectorField
            The full assembled state vector.
        ctx : StepContext
            The substage context (built at P0).

        Returns
        -------
        VectorField
            The prepared state.
        """
        state = self._run_stages(StageKind.SELF_UPDATE, state, ctx)
        return self._run_stages(StageKind.DIAGNOSE, state, ctx)

    def tendency(
        self, state: VectorField, ctx: StepContext,
    ) -> TendencySums:
        """
        Accumulate the EXPLICIT term contributions (S2).

        Description
        -----------
        Deterministic order — module tuple order, then declaration
        order within a module — accumulated via one
        ``sums.add(**term_dict)`` per term onto a zero
        PROGNOSTIC-only template. IMPLICIT terms are not evaluated
        here (their contribution enters through the wave-5 solves).

        Parameters
        ----------
        state : VectorField
            The full assembled state vector.
        ctx : StepContext
            The substage context.

        Returns
        -------
        TendencySums
            The frozen per-treatment sums (``explicit`` populated).
        """
        names = self._schedule.prognostic
        sums: VectorField | None = None
        if names:
            sums = VectorField(
                {name: zero_like(state[name]) for name in names})
        for entry in self._schedule.kind_entries(None):
            if entry.treatment is not Treatment.EXPLICIT:
                continue
            module = self._modules[entry.slot]
            result = evaluate_entry(entry, module, state, ctx)
            sums = apply_add(entry, sums, result)
        return TendencySums(explicit=sums)

    def constrain(
        self, state: VectorField, ctx: StepContext,
    ) -> VectorField:
        """
        Run the S4 CONSTRAINT stages (projection et al.).

        Parameters
        ----------
        state : VectorField
            The state produced by the preceding advance.
        ctx : StepContext
            The substage context (post-advance; carries sums).

        Returns
        -------
        VectorField
            The constrained state.
        """
        return self._run_stages(StageKind.CONSTRAINT, state, ctx)

    def diagnostics(
        self, state: VectorField, ctx: StepContext,
    ) -> VectorField:
        """
        Run the S6 DIAGNOSTIC stages (step-cadence accumulators).

        Description
        -----------
        Consumed by the chunk body's per-step epilogue, after the
        S5 NaN reduction — never by the stepper. A DIAGNOSTIC stage
        may read its own component's previous value (the
        accumulation idiom, 02_rules).

        Parameters
        ----------
        state : VectorField
            The committed post-constraint state.
        ctx : StepContext
            The per-step context (carries sums).

        Returns
        -------
        VectorField
            The state with DIAGNOSTIC writes applied.
        """
        return self._run_stages(StageKind.DIAGNOSTIC, state, ctx)

    # ================================================================
    #  Wave-5 groups (data carried; behavior lands with 2.5)
    # ================================================================
    def advance_stages(
        self, state: VectorField, ctx: StepContext,
    ) -> VectorField:
        """
        Run the S3' module-owned ADVANCE stages (kind order).

        Description
        -----------
        The by-variable ADVANCE group (barotropic subcycle et al.):
        each stage writes its declared advanced subset plus own AUX,
        in schedule order, so a later stage's reads see the earlier
        stage's writes (Gauss-Seidel by the design-03 section-5.2
        read rule — the same sequential ``replace`` walk every other
        stage kind uses). The ctx carries the per-treatment tendency
        sums (the increment-form barotropic forcing reads them).

        Parameters
        ----------
        state : VectorField
            The state produced by the primary advance.
        ctx : StepContext
            The post-advance context (carries sums).

        Returns
        -------
        VectorField
            The state with the ADVANCE writes applied.
        """
        return self._run_stages(StageKind.ADVANCE, state, ctx)

    @property
    def implicit(self) -> tuple[BoundImplicitOperator, ...]:
        """
        The merged per-field implicit operators, bound to slots.

        Description
        -----------
        One :class:`BoundImplicitOperator` per merge group of
        ``Schedule.implicit_merged`` (kappa-summed framework families
        plus at most one non-mergeable custom operator per field —
        the composer collapsed each group through ``merged_with``).
        Each is closed over its constituent module slot, exposing the
        stepper-facing ``fields`` / ``apply(state, ctx)`` /
        ``solve(rhs, dt_gamma, ctx)`` surface. An empty tuple means
        no IMPLICIT terms (the IMEX driver degenerates to its
        explicit member — a legal composition).

        Returns
        -------
        tuple[BoundImplicitOperator, ...]
            The bound merged operators, group order.
        """
        return tuple(
            BoundImplicitOperator(operator, self._modules[slot])
            for operator, slot in self._schedule.implicit_merged)

    # ================================================================
    #  Internals
    # ================================================================
    def _run_stages(
        self,
        kind: StageKind,
        state: VectorField,
        ctx: StepContext,
    ) -> VectorField:
        """Run one kind's stages in schedule order (replace)."""
        for entry in self._schedule.kind_entries(kind):
            module = self._modules[entry.slot]
            result = evaluate_entry(entry, module, state, ctx)
            state = apply_replace(entry, state, result)
        return state

    def __repr__(self) -> str:
        """Compact summary referencing the static schedule."""
        return (f"BoundSchedule({self._schedule!r}, "
                f"{len(self._modules)} modules)")


# ================================================================
#  BoundImplicitOperator — the per-step bound solve surface
# ================================================================
class BoundImplicitOperator:

    """
    A merged implicit operator closed over its live module slot.

    Description
    -----------
    The per-step binding of one ``Schedule.implicit_merged`` entry:
    the merged (kappa-summed) operator paired with the carry's live
    owning module, exposing the minimal stepper-facing surface —
    ``fields`` (the advanced PROGNOSTIC subset), ``apply(state, ctx)``
    (the forward ``L @ state``), and ``solve(rhs, dt_gamma, ctx)``
    (the ``(1 - dt_gamma * L)^{-1}`` solve). ``dt_gamma`` stays an
    explicit stepper-supplied positional (never read from ``ctx``).

    Ephemeral, built per step by ``BoundSchedule.implicit`` — the
    unbound operator meets its live module leaf here (the D2 aliasing
    rule, as for every other scheduled hook).

    Parameters
    ----------
    operator : ImplicitOperator
        The merged operator (one per merge group).
    module : object
        The live owning module (``carry.modules[slot]``).
    """

    def __init__(self, operator: object, module: object) -> None:
        """Bind the operator to its live module slot."""
        self._operator = operator
        self._module = module

    @property
    def fields(self) -> tuple[str, ...]:
        """The advanced PROGNOSTIC subset (one solve per field)."""
        return self._operator.fields

    @property
    def operator(self) -> object:
        """The underlying merged (unbound) operator."""
        return self._operator

    def apply(
        self, state: VectorField, ctx: StepContext,
    ) -> Mapping[str, ScalarField]:
        """
        Forward ``L @ state`` over ``fields`` (the CNAB rhs term).

        Parameters
        ----------
        state : VectorField
            The full assembled state vector.
        ctx : StepContext
            The per-substage context.

        Returns
        -------
        Mapping[str, ScalarField]
            Increments keyed by the advanced PROGNOSTIC names.
        """
        return self._operator.apply(self._module, state, ctx)

    def solve(
        self,
        rhs: Mapping[str, ScalarField],
        dt_gamma: Any,
        ctx: StepContext,
    ) -> Mapping[str, ScalarField]:
        """
        Solve ``(1 - dt_gamma * L) x = rhs``; keys exactly ``fields``.

        Parameters
        ----------
        rhs : Mapping[str, ScalarField]
            Right-hand sides keyed by the advanced names.
        dt_gamma : Any
            The traced scalar gamma*dt of the current scheme stage.
        ctx : StepContext
            The per-substage context.

        Returns
        -------
        Mapping[str, ScalarField]
            The solved fields, keyed exactly by ``fields``.
        """
        return self._operator.solve(self._module, rhs, dt_gamma, ctx)

    def __repr__(self) -> str:
        """Compact host-side summary."""
        return (f"BoundImplicitOperator({self._operator!r}, "
                f"module={type(self._module).__name__})")


# ================================================================
#  TendencySums — the frozen per-treatment sums
# ================================================================
# the ordered static representation of the treatment partition (W2:
# Treatment enum members are not sortable — key the partition by the
# member NAME in a fixed order instead of by the member itself)
_TENDENCY_SUM_FIELDS: Final[tuple[str, ...]] = ("explicit", "implicit")
_TREATMENT_TO_FIELD: Final[dict[str, str]] = {
    "EXPLICIT": "explicit",
    "IMPLICIT": "implicit",
}


@partial(jaxify, dynamic=("explicit", "implicit"))
class TendencySums:

    """
    Per-treatment tendency sums (frozen, jaxified).

    Description
    -----------
    The contribution-dict partition, consumed at accumulation time:
    PROGNOSTIC-only vectors, one per treatment. The EXPLICIT sum is
    always populated (``None`` only when the composition declares no
    PROGNOSTIC fields); the IMPLICIT sum is the summed forward-apply
    ``L @ X`` contribution, present iff the driving scheme computed
    the forward applies this step (CNAB2 fills it; the SBDF and
    explicit paths leave it ``None``). ``__getitem__`` keys by
    ``Treatment`` through a fixed name ordering (W2: the enum members
    are not sortable, so the partition uses an ordered static
    representation keyed by ``Treatment.name``). Handed to
    post-TENDENCY stages through ``StepContext.tendency_sums``.

    Parameters
    ----------
    explicit : VectorField | None
        The summed EXPLICIT contribution (unprojected);
        ``None`` iff the composition declares no PROGNOSTIC
        fields (CS-13).
    implicit : VectorField | None, optional
        The summed IMPLICIT forward-apply contribution
        (``sum_op L @ X``), or ``None`` when the driving scheme
        computed no forward applies this step (default: None).
    """

    explicit: VectorField | None
    implicit: VectorField | None

    def __init__(
        self,
        explicit: VectorField | None,
        implicit: VectorField | None = None,
    ) -> None:
        """Freeze the sums; see the class docstring."""
        self.explicit = explicit
        self.implicit = implicit

    def __setattr__(self, name: str, value: object) -> None:
        """Set a sums field exactly once (frozen thereafter)."""
        # write-once: __init__ and pytree unflattening set fresh
        # attributes; everything else raises
        if name in _TENDENCY_SUM_FIELDS and name not in self.__dict__:
            object.__setattr__(self, name, value)
            return
        raise AttributeError(
            f"TendencySums is frozen: cannot set {name!r} (sums are "
            "built per substage by the tendency group)")

    def __delattr__(self, name: str) -> None:
        """Raise: TendencySums is frozen."""
        raise AttributeError(
            f"TendencySums is frozen: cannot delete {name!r}")

    def __getitem__(self, treatment: Treatment) -> VectorField:
        """
        Return the summed contribution of one treatment.

        Description
        -----------
        The IMPLICIT entry exists iff the driving scheme computed the
        forward applies this step (CNAB2); otherwise a ``KeyError`` —
        the increment-form barotropic default makes the split-explicit
        ADVANCE stage independent of it (V-H4).

        Parameters
        ----------
        treatment : Treatment
            The requested treatment (keyed by ``.name``).

        Returns
        -------
        VectorField
            The summed PROGNOSTIC-only contribution.

        Raises
        ------
        KeyError
            On an unknown treatment, or on an IMPLICIT lookup when
            the scheme computed no forward applies this step.
        """
        key = getattr(treatment, "name", treatment)
        field = _TREATMENT_TO_FIELD.get(key)
        if field is None:
            raise KeyError(
                f"unknown treatment {treatment!r}; the tendency "
                f"partition keys on {tuple(_TREATMENT_TO_FIELD)}")
        value = getattr(self, field)
        if value is None:
            raise KeyError(
                f"no {key} tendency sum this step; the driving "
                "scheme computed no forward applies (only CNAB2 "
                "populates the IMPLICIT entry — V-H4)")
        return value

    def __repr__(self) -> str:
        """Compact host-side summary."""
        names = (self.explicit.component_names
                 if self.explicit is not None else ())
        implicit = "" if self.implicit is None else ", +implicit"
        return f"TendencySums(explicit over {names!r}{implicit})"
