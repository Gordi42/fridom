"""
The tendency composer.

Description
-----------
``TendencyComposer``: assembly step 5/6 machinery — term/stage
collection consumption, ``"Module/term"`` attribution, write-gate
validation, deterministic accumulation order, implicit merge
grouping, and the dry-run validation. Owning class spec:
``notes/framework2/model/classes/model.md`` ("TendencyComposer");
design source ``notes/framework2/model/03_time_stepping.md``
sections 5.1/5.2/5.5.

The composer is assembly-internal and host-side; its *products* are
the static ``Schedule`` (which joins the assembly record) and the
composed step body — a pure function over ``(state, modules, ctx)``
built from the schedule: no Model reference, no host state, no
syncs, no jit here (sync cost is the grid's, 02_rules).

Wave-3 inputs are duck-typed (the sibling wave-3 tracks own
``field_table.py`` and ``module.py``): ``field_table`` is anything
iterable over records carrying ``name``/``space``/``lifecycle``/
``owner`` and exposing the ``grid`` the fields live on; ``terms``
and ``stages`` are plain sequences of ``(slot, declaration)`` pairs
in collection order (module tuple index paired with the
``TendencyTerm``/``Stage`` record).
"""
# Wave 3 C: TendencyComposer (attribution, write-gate validation,
#    dry run, accumulation)
from __future__ import annotations

import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Any

import jax.numpy as jnp

from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.model.context import StepContext
from fridom.framework2.model.declarations import Lifecycle
from fridom.framework2.model.errors import (
    AssemblyError,
    ImplicitCollisionError,
)
from fridom.framework2.model.schedule import (
    Schedule,
    ScheduleEntry,
    TendencySums,
    apply_add,
    apply_replace,
    evaluate_entry,
    zero_like,
)
from fridom.framework2.model.stages import Stage, StageKind
from fridom.framework2.model.terms import TendencyTerm, Treatment

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Sequence

# the substage stage kinds that run before the tendency terms and
# the (post-advance) kinds that run after them, in schedule order
_PRE_TENDENCY = (StageKind.SELF_UPDATE, StageKind.DIAGNOSE)
_POST_TENDENCY = (StageKind.ADVANCE, StageKind.CONSTRAINT,
                  StageKind.DIAGNOSTIC)


# ================================================================
#  TendencyComposer
# ================================================================
class TendencyComposer:

    """
    Schedule composition, dry run, attribution (assembly-internal).

    Description
    -----------
    Collects the post-bind term/stage declarations, orders them by
    kind (SELF_UPDATE first in every substage — load-bearing),
    precomputes the per-entry static write gates, groups implicit
    operators for the wave-5 merge, and statically checks what can
    be checked without evaluation. ``dry_run()`` then validates
    every hook over real zero-valued fields (V-N2's second dry-run
    mode; the halo-tracer mode is driven by assembly step 7 through
    the composed body).

    Parameters
    ----------
    field_table : Any
        Duck-typed resolved field table: iterable over records with
        ``name`` (str), ``space`` (resolved space), ``lifecycle``
        (``Lifecycle``), ``owner`` (module tuple index), and
        exposing ``grid`` (the frozen grid fields are created on).
    modules : tuple
        The module tuple (attribution names come from the classes).
    terms : Sequence[tuple[int, TendencyTerm]]
        Collected terms as ``(slot, term)`` pairs, module tuple
        order then declaration order.
    stages : Sequence[tuple[int, Stage]]
        Collected stages as ``(slot, stage)`` pairs.
    time_stepper : Any
        The stepper; its ``supported_treatments`` (when present)
        gates the term treatments — an IMPLICIT term under a purely
        explicit stepper is an assembly error, never a silent
        demotion.
    binding_table : Any
        The parameter binding table (or ``None`` pre-wave-4); the
        schedule's ``context()`` reads parameters through it at
        stage time.
    term_filter : Callable | None, optional
        Variant mechanics (2.8): a predicate ``(key, term) ->
        bool`` over the collected terms — declarations and stages
        are never filtered; the coverage lint downgrades to a
        warning under any filter; an empty filter result is a build
        error (default: None).

    Raises
    ------
    AssemblyError
        On any static check failure (bound hooks, unknown names,
        unsupported treatments, gate violations at dry run, ...).
    ImplicitCollisionError
        On more than one non-mergeable implicit operator per field.
    """

    def __init__(
        self,
        *,
        field_table: Any,
        modules: tuple,
        terms: Sequence[tuple[int, TendencyTerm]],
        stages: Sequence[tuple[int, Stage]],
        time_stepper: Any,
        binding_table: Any,
        term_filter: Callable | None = None,
    ) -> None:
        """Collect, order, and statically check; see class doc."""
        self._modules = tuple(modules)
        self._records = tuple(field_table)
        self._grid = field_table.grid
        self._term_filter = term_filter
        self._names = tuple(r.name for r in self._records)
        self._prognostic = tuple(
            r.name for r in self._records
            if r.lifecycle is Lifecycle.PROGNOSTIC)
        own_aux = _owned(self._records, Lifecycle.AUXILIARY)
        own_diag = _owned(self._records, Lifecycle.DIAGNOSTIC)

        term_entries = self._collect_terms(terms, time_stepper)
        term_entries = self._filter_terms(term_entries, terms)
        stage_entries = self._collect_stages(
            stages, own_aux, own_diag)
        _static_advance_overlap(stage_entries)
        groups = _implicit_groups(term_entries)
        merged = _merge_groups(groups)
        self._schedule = Schedule(
            term_entries + stage_entries,
            prognostic=self._prognostic,
            implicit_groups=groups,
            implicit_merged=merged,
            binding_table=binding_table,
            time_stepper=time_stepper,
        )

    # ================================================================
    #  Products
    # ================================================================
    @property
    def schedule(self) -> Schedule:
        """
        The static kind-ordered schedule.

        Description
        -----------
        Per substage SELF_UPDATE -> DIAGNOSE -> terms -> ADVANCE ->
        CONSTRAINT; per-step epilogue DIAGNOSTIC. Within a kind:
        ``(order, module index, declaration index)`` — never
        list-positioned. (Spec concretization: the spec sketch
        types this ``tuple[ScheduleEntry, ...]``; the ``Schedule``
        object carries exactly those entries plus ``bind`` and
        ``describe``.)
        """
        return self._schedule

    def compose(
        self,
    ) -> Callable[[VectorField, tuple, StepContext],
                  tuple[VectorField, TendencySums]]:
        """
        Build the composed step body (plain Python over fields).

        Description
        -----------
        A PURE function ``(state, modules, ctx) -> (state, sums)``
        running prepare (S1/S1'), the tendency accumulation (S2),
        and the CONSTRAINT group (S4) from the schedule — the
        surface the halo tracer and ``run(jit=False)`` require;
        each hook runs under ``jax.named_scope("Module/term")``.
        The primary advance (S3, the stepper's), the S3' ADVANCE
        group, and the S5/S6 epilogue are composed around this body
        by later waves.

        Returns
        -------
        Callable
            The step body ``(state, modules, ctx) ->
            (state, TendencySums)``.
        """
        schedule = self._schedule

        def step_body(
            state: VectorField,
            modules: tuple,
            ctx: StepContext,
        ) -> tuple[VectorField, TendencySums]:
            """Run prepare, tendency, constrain from the schedule."""
            bound = schedule.bind(modules)
            state = bound.prepare(state, ctx)
            sums = bound.tendency(state, ctx)
            state = bound.constrain(state, ctx)
            return state, sums

        return step_body

    def tendency_template(self) -> VectorField | None:
        """
        Build the zero PROGNOSTIC vector (``stepper.init`` input).

        Returns
        -------
        VectorField | None
            Zero-valued PROGNOSTIC components in declaration order,
            born on the grid's negotiated layout; ``None`` iff the
            composition declares no PROGNOSTIC fields (CS-13).
        """
        records = tuple(
            r for r in self._records
            if r.lifecycle is Lifecycle.PROGNOSTIC)
        if not records:
            return None
        return VectorField({
            r.name: self._grid.create_field(r.space, name=r.name)
            for r in records})

    def tendency_fn(
        self,
        *,
        filter: Callable | None = None,  # noqa: A002 — spec-fixed keyword
        constraints: bool = True,
    ) -> Callable:
        """
        Build the read-only composed tendency — task 2.8 (wave 7).

        Description
        -----------
        Not built in wave 3: the ``model.tendency`` surface
        (SELF_UPDATE + DIAGNOSE first, implicit terms via their
        forward apply, optional CONSTRAINT application) lands with
        the state-transform wave.

        Parameters
        ----------
        filter : Callable | None, optional
            Term predicate restricting the evaluated terms
            (default: None).
        constraints : bool, optional
            Whether CONSTRAINT stages apply to the result
            (default: True).

        Raises
        ------
        NotImplementedError
            Always, in this wave.
        """
        raise NotImplementedError(
            "TendencyComposer.tendency_fn lands with model.tendency "
            "(ROADMAP 2.8, wave 7); the wave-3 surface is compose() "
            "and the schedule's stage groups")

    # ================================================================
    #  Dry run (assembly step 6)
    # ================================================================
    def dry_run(self, *, params: Any = None) -> None:
        """
        Validate every term/stage over real zero-valued fields.

        Description
        -----------
        Runs each hook once, in schedule order, on a zero-valued
        state built from the field table (V-N2's real-field mode —
        the mode that also covers halo-trace-exempted modules):
        write gates per kind, contribution-key validation (typos
        fail at assembly), term ``advances`` cross-checks, result
        spaces, and the read/write ordering (SELF_UPDATE writes are
        visible to DIAGNOSE readers). Afterwards the observed write
        sets feed the same-kind overlap lint and the PROGNOSTIC
        coverage lint.

        ``params`` carries the assembly-time evaluated parameters
        (assembly passes ``binding_table.eval_params(modules,
        stepper, 0.0)``): a term reading ``ctx.params["stepper.dt"]``
        now resolves instead of dying on an empty mapping, and an
        unbound-name read surfaces (attributed) through the
        ``TermEvaluationError`` chain rather than a bare
        ``KeyError(...)`` on ``{}``.

        Parameters
        ----------
        params : Any, optional
            The evaluated ``name -> leaf`` parameter mapping; an
            empty mapping when omitted (default: None).

        Raises
        ------
        AssemblyError
            On any validation failure, attributed "Module/term".
        TermEvaluationError
            When a hook itself raises (original exception chained).
        """
        schedule = self._schedule
        state = self._zero_state()
        param_map: Any = {} if params is None else params
        ctx = StepContext(params=param_map, clock=jnp.asarray(0.0),
                          dt=jnp.asarray(1.0),
                          stage_dt=jnp.asarray(1.0))
        writes: dict[ScheduleEntry, frozenset[str]] = {}
        for kind in _PRE_TENDENCY:
            state = self._dry_stages(kind, state, ctx, writes)
        sums = self._dry_terms(state, ctx, writes)
        ctx = StepContext(params=param_map, clock=jnp.asarray(0.0),
                          dt=jnp.asarray(1.0),
                          stage_dt=jnp.asarray(1.0),
                          tendency_sums=sums)
        for kind in _POST_TENDENCY:
            state = self._dry_stages(kind, state, ctx, writes)
        _overlap_lint(schedule, writes)
        self._coverage_lint(writes)

    # ================================================================
    #  Collection internals
    # ================================================================
    def _collect_terms(
        self,
        terms: Sequence[tuple[int, TendencyTerm]],
        time_stepper: Any,
    ) -> tuple[ScheduleEntry, ...]:
        """Validate and gate the collected terms, in order."""
        supported = getattr(
            time_stepper, "supported_treatments", None)
        entries = []
        for index, item in enumerate(terms):
            slot, term = _unpack(item, "terms", self._modules)
            if not isinstance(term, TendencyTerm):
                raise AssemblyError(
                    f"terms[{index}]: expected a TendencyTerm, got "
                    f"{term!r}")
            key = f"{type(self._modules[slot]).__name__}/{term.name}"
            if term.fn is not None:
                _check_unbound(key, term.fn)
            if (term.treatment is Treatment.IMPLICIT
                    and term.implicit is None):
                raise AssemblyError(
                    f"{key}: an IMPLICIT term requires implicit= "
                    "(the operator supplies both solve and the "
                    "derived explicit path)")
            if supported is not None and term.treatment not in supported:
                raise AssemblyError(
                    f"{key}: treatment {term.treatment.name} is not "
                    f"supported by "
                    f"{type(time_stepper).__name__} (supported: "
                    f"{sorted(t.name for t in supported)}); "
                    "treatments are never silently demoted")
            if term.advances is not None:
                self._check_prognostic(
                    key, term.advances, "advances=")
            if term.implicit is not None:
                self._check_prognostic(
                    key, term.implicit.fields, "implicit fields")
            gate = (self._prognostic
                    if term.treatment is Treatment.EXPLICIT
                    else tuple(term.implicit.fields))
            entries.append(ScheduleEntry(
                key=key, kind=None, slot=slot, order=0, index=index,
                fn=term.fn, gate=gate, treatment=term.treatment,
                advances=term.advances, reads=(),
                implicit=term.implicit))
        return tuple(entries)

    def _filter_terms(
        self,
        entries: tuple[ScheduleEntry, ...],
        terms: Sequence[tuple[int, TendencyTerm]],
    ) -> tuple[ScheduleEntry, ...]:
        """Apply the variant term filter (terms only, 08 10.4)."""
        if self._term_filter is None:
            return entries
        kept = tuple(
            entry for entry, (_, term) in zip(entries, terms,
                                              strict=True)
            if self._term_filter(entry.key, term))
        if entries and not kept:
            raise AssemblyError(
                "the term filter drops every collected term; an "
                "empty filter result is a build error (model.md, "
                "variant mechanics)")
        return kept

    def _collect_stages(
        self,
        stages: Sequence[tuple[int, Stage]],
        own_aux: dict[int, tuple[str, ...]],
        own_diag: dict[int, tuple[str, ...]],
    ) -> tuple[ScheduleEntry, ...]:
        """Validate, gate, and slot-pair the collected stages."""
        entries = []
        for index, item in enumerate(stages):
            slot, stage = _unpack(item, "stages", self._modules)
            if not isinstance(stage, Stage):
                raise AssemblyError(
                    f"stages[{index}]: expected a Stage, got "
                    f"{stage!r}")
            owner = type(self._modules[slot]).__name__
            fn = stage.fn
            if isinstance(fn, str):
                fn = getattr(type(self._modules[slot]), stage.fn,
                             None)
                if fn is None:
                    raise AssemblyError(
                        f"{owner}/{stage.fn}: the stage names a "
                        f"method {stage.fn!r} that "
                        f"{owner} does not define")
            name = (stage.name if stage.name is not None
                    else getattr(fn, "__name__", str(stage.fn)))
            key = f"{owner}/{name}"
            _check_unbound(key, fn)
            unknown = tuple(read for read in stage.reads
                            if read not in self._names)
            if unknown:
                raise AssemblyError(
                    f"{key}: reads= names unknown fields "
                    f"{unknown}; declared fields: {self._names}")
            gate = self._stage_gate(key, stage, slot, own_aux,
                                    own_diag)
            entries.append(ScheduleEntry(
                key=key, kind=stage.kind, slot=slot,
                order=stage.order, index=index, fn=fn, gate=gate,
                treatment=None, advances=stage.advances,
                reads=stage.reads, implicit=None))
        return tuple(entries)

    def _stage_gate(
        self,
        key: str,
        stage: Stage,
        slot: int,
        own_aux: dict[int, tuple[str, ...]],
        own_diag: dict[int, tuple[str, ...]],
    ) -> tuple[str, ...]:
        """Compute the static write gate of one stage (03 5.5)."""
        kind = stage.kind
        if kind is StageKind.SELF_UPDATE:
            return own_aux.get(slot, ())
        if kind in (StageKind.DIAGNOSE, StageKind.DIAGNOSTIC):
            return own_diag.get(slot, ())
        if kind is StageKind.ADVANCE:
            self._check_prognostic(key, stage.advances, "advances=")
            return stage.advances + own_aux.get(slot, ())
        # CONSTRAINT: PROGNOSTIC (role-selected at bind) + own DIAG
        return self._prognostic + own_diag.get(slot, ())

    def _check_prognostic(
        self, key: str, names: Iterable[str], what: str,
    ) -> None:
        """Require the names to be declared PROGNOSTIC fields."""
        stray = tuple(name for name in names
                      if name not in self._prognostic)
        if stray:
            raise AssemblyError(
                f"{key}: {what} names {stray}, which are not "
                f"declared PROGNOSTIC fields (PROGNOSTIC: "
                f"{self._prognostic})")

    # ================================================================
    #  Dry-run internals
    # ================================================================
    def _zero_state(self) -> VectorField | None:
        """Build the real zero-valued full state (all lifecycles)."""
        if not self._records:
            return None
        return VectorField({
            r.name: self._grid.create_field(r.space, name=r.name)
            for r in self._records})

    def _dry_stages(
        self,
        kind: StageKind,
        state: VectorField | None,
        ctx: StepContext,
        writes: dict[ScheduleEntry, frozenset[str]],
    ) -> VectorField | None:
        """Evaluate one kind's stages, recording the write sets."""
        for entry in self._schedule.kind_entries(kind):
            module = self._modules[entry.slot]
            result = evaluate_entry(entry, module, state, ctx)
            writes[entry] = frozenset(result)
            state = apply_replace(entry, state, result)
        return state

    def _dry_terms(
        self,
        state: VectorField | None,
        ctx: StepContext,
        writes: dict[ScheduleEntry, frozenset[str]],
    ) -> TendencySums:
        """Evaluate the terms; cross-check declared advances."""
        sums: VectorField | None = None
        if self._prognostic and state is not None:
            sums = VectorField({
                name: zero_like(state[name])
                for name in self._prognostic})
        for entry in self._schedule.kind_entries(None):
            if entry.treatment is Treatment.IMPLICIT:
                # not evaluated in wave 3 (solves land wave 5); the
                # operator's declared fields are its static writes
                writes[entry] = frozenset(entry.implicit.fields)
                continue
            module = self._modules[entry.slot]
            result = evaluate_entry(entry, module, state, ctx)
            observed = frozenset(result)
            writes[entry] = observed
            if (entry.advances is not None
                    and set(entry.advances) != observed):
                raise AssemblyError(
                    f"{entry.key}: declares advances="
                    f"{entry.advances} but the dry run observed "
                    f"writes {tuple(sorted(observed))}")
            sums = apply_add(entry, sums, result)
        return TendencySums(explicit=sums)

    def _coverage_lint(
        self, writes: dict[ScheduleEntry, frozenset[str]],
    ) -> None:
        """Every PROGNOSTIC field must be advanced (D1.4 lint)."""
        advanced: set[str] = set()
        for entry, observed in writes.items():
            if entry.is_term:
                advanced |= observed
            elif entry.kind is StageKind.ADVANCE:
                advanced |= set(entry.advances or ())
        uncovered = tuple(name for name in self._prognostic
                          if name not in advanced)
        if not uncovered:
            return
        message = (
            f"PROGNOSTIC fields {uncovered} are advanced by no "
            "term and claimed by no ADVANCE stage (coverage lint, "
            "D1.4)")
        if self._term_filter is not None:
            # info downgrade under a variant filter (08 10.4)
            warnings.warn(message, stacklevel=3)
            return
        raise AssemblyError(message)


# ================================================================
#  Module-level helpers
# ================================================================
def _owned(
    records: tuple, lifecycle: Lifecycle,
) -> dict[int, tuple[str, ...]]:
    """Map module slot -> owned field names of one lifecycle."""
    owned: dict[int, tuple[str, ...]] = {}
    for record in records:
        if record.lifecycle is lifecycle:
            owned[record.owner] = (
                *owned.get(record.owner, ()), record.name)
    return owned


def _unpack(
    item: tuple[int, Any], label: str, modules: tuple,
) -> tuple[int, Any]:
    """Validate one ``(slot, declaration)`` collection pair."""
    try:
        slot, declaration = item
    except (TypeError, ValueError) as exc:
        raise AssemblyError(
            f"{label} entries are (module slot, declaration) "
            f"pairs, got {item!r}") from exc
    if (not isinstance(slot, int) or isinstance(slot, bool)
            or not 0 <= slot < len(modules)):
        raise AssemblyError(
            f"{label}: module slot {slot!r} is not a valid index "
            f"into the {len(modules)}-module tuple")
    return slot, declaration


def _check_unbound(key: str, fn: object) -> None:
    """Reject bound hooks (the D2 aliasing trap)."""
    if not callable(fn):
        raise AssemblyError(
            f"{key}: the hook must be callable, got {fn!r}")
    if getattr(fn, "__self__", None) is not None:
        raise AssemblyError(
            f"{key}: the hook is a BOUND method of "
            f"{type(fn.__self__).__name__}; hooks are stored "
            "unbound and paired with a module slot at compose time "
            "(a bound method captures the assembly-time instance "
            "while live parameters ride the carry)")


def _implicit_groups(
    entries: tuple[ScheduleEntry, ...],
) -> tuple:
    """
    Group implicit operators for the wave-5 merge; check collisions.

    Description
    -----------
    Mergeable families (equal, non-``None`` ``merge_key()``) group
    together — they combine exactly at wave 5. Non-mergeable custom
    operators (``merge_key()`` is ``None``) are singleton groups;
    two of them touching one field is an ``ImplicitCollisionError``
    naming both terms.

    Parameters
    ----------
    entries : tuple[ScheduleEntry, ...]
        The collected term entries.

    Returns
    -------
    tuple
        Static merge groups, each a tuple of ``(key, slot,
        operator)`` constituents.

    Raises
    ------
    ImplicitCollisionError
        On a second non-mergeable operator per field.
    """
    groups: dict[Any, list] = {}
    custom_by_field: dict[str, str] = {}
    for entry in entries:
        op = entry.implicit
        if op is None:
            continue
        merge_key = getattr(op, "merge_key", lambda: None)()
        if merge_key is None:
            for field in op.fields:
                if field in custom_by_field:
                    raise ImplicitCollisionError(
                        f"two non-mergeable implicit operators "
                        f"advance {field!r}: "
                        f"{custom_by_field[field]} and {entry.key}; "
                        "at most one custom implicit operator per "
                        "field (mergeable families sum instead)")
                custom_by_field[field] = entry.key
            groups[("custom", entry.key, entry.index)] = [
                (entry.key, entry.slot, op)]
        else:
            groups.setdefault(("family", merge_key), []).append(
                (entry.key, entry.slot, op))
    return tuple(tuple(group) for group in groups.values())


def _merge_groups(groups: tuple) -> tuple:
    """
    Collapse each implicit merge group to one operator (merged_with).

    Description
    -----------
    The merge INVOCATION deferred from ``_implicit_groups`` (which
    only groups): each group's constituents are combined left-to-right
    through ``merged_with`` — mergeable framework families kappa-sum
    exactly (``(1 - dt_gamma * (L1 + L2))``), and singleton groups
    (one term, or a non-mergeable custom operator) pass through
    untouched. The merged operator inherits the FIRST constituent's
    module slot; framework families read their coefficients through
    ``ctx.params`` / owner leaves, so a family spanning several slots
    stays correct under the single bound module (the ``_summed_kappa``
    uniform-first-argument contract).

    Parameters
    ----------
    groups : tuple
        The static merge groups, each a tuple of ``(key, slot,
        operator)`` constituents.

    Returns
    -------
    tuple
        Per-group ``(merged_operator, slot)`` pairs.
    """
    merged = []
    for group in groups:
        (_key, slot, operator), *rest = group
        combined = operator
        for _other_key, _other_slot, other in rest:
            combined = combined.merged_with(other)
        merged.append((combined, slot))
    return tuple(merged)


def _static_advance_overlap(
    entries: tuple[ScheduleEntry, ...],
) -> None:
    """Pre-check declared ADVANCE claims for equal-order overlap."""
    advance = [entry for entry in entries
               if entry.kind is StageKind.ADVANCE]
    for left, right in combinations(advance, 2):
        if left.order != right.order:
            continue
        shared = set(left.advances or ()) & set(right.advances or ())
        if shared:
            raise AssemblyError(
                f"same-kind stages {left.key} and {right.key} "
                f"declare overlapping advances "
                f"{tuple(sorted(shared))} with equal order="
                f"{left.order}; declare an explicit order= "
                "(correctness never depends on list position)")


def _overlap_lint(
    schedule: Schedule,
    writes: dict[ScheduleEntry, frozenset[str]],
) -> None:
    """Same-kind, equal-order overlapping write(-read) sets error."""
    stage_entries = [entry for entry in schedule.entries
                     if not entry.is_term]
    for left, right in combinations(stage_entries, 2):
        if left.kind is not right.kind or left.order != right.order:
            continue
        shared = writes.get(left, frozenset()) & writes.get(
            right, frozenset())
        shared |= writes.get(left, frozenset()) & set(right.reads)
        shared |= writes.get(right, frozenset()) & set(left.reads)
        if shared:
            raise AssemblyError(
                f"same-kind stages {left.key} and {right.key} have "
                f"overlapping write(-read) sets "
                f"{tuple(sorted(shared))} with equal order="
                f"{left.order}; declare an explicit order= "
                "(correctness never depends on list position)")
