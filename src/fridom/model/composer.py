"""
The tendency composer.

Description
-----------
``TendencyComposer``: assembly step 5/6 machinery — term/stage
collection consumption, ``"Module/term"`` attribution, write-gate
validation, deterministic accumulation order, implicit merge
grouping, and the dry-run validation. Owning class spec:
``design/specs/model/classes/model.md`` ("TendencyComposer");
design source ``design/specs/model/03_time_stepping.md``
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

import dataclasses
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from fridom.model.context import StepContext
from fridom.model.declarations import Lifecycle
from fridom.model.errors import (
    AssemblyError,
    ImplicitCollisionError,
)
from fridom.model.params import RAMPING_ENVELOPE
from fridom.model.roles import Velocity
from fridom.model.schedule import (
    Schedule,
    ScheduleEntry,
    TendencySums,
    apply_add,
    apply_replace,
    evaluate_entry,
    zero_like,
)
from fridom.model.stages import Stage, StageKind
from fridom.model.terms import TendencyTerm, Treatment
from fridom.spatial.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Sequence

    from fridom.model.phases import Phases

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
    allow_unadvanced : Sequence[str], optional
        PROGNOSTIC field names the caller declares are *deliberately*
        advanced by no term — the explicit waiver of the D1.4 coverage
        lint (see :meth:`_coverage_lint`). Every name must be a
        PROGNOSTIC field of this assembly; an unknown name is an
        assembly error, so a typo in the waiver cannot silently widen
        it. Waiving a field that some term does advance is harmless
        and stays silent (default: ()).
    phases : Phases | None, optional
        The declared phase axis (``fr.model.Phases``): the PROGNOSTIC
        partition whose groups the multistep steppers loop over.
        Resolved at the END of :meth:`dry_run` — the per-entry
        membership needs the observed write sets — and installed on
        the schedule there. A partition resolving to ONE group is the
        unphased path and installs nothing, so ``phases=None`` and
        ``Phases.total()`` produce identical schedules. Refused
        outright under a stepper whose ``supports_phases`` is False
        (default: None).

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
        allow_unadvanced: Sequence[str] = (),
        phases: Phases | None = None,
    ) -> None:
        """Collect, order, and statically check; see class doc."""
        _check_stepper_phases(phases, time_stepper)
        self._phases = phases
        self._modules = tuple(modules)
        self._records = tuple(field_table)
        self._grid = field_table.grid
        self._term_filter = term_filter
        self._names = tuple(r.name for r in self._records)
        self._prognostic = tuple(
            r.name for r in self._records
            if r.lifecycle is Lifecycle.PROGNOSTIC)
        self._allow_unadvanced = self._check_waiver(allow_unadvanced)
        own_aux = _owned(self._records, Lifecycle.AUXILIARY)
        own_diag = _owned(self._records, Lifecycle.DIAGNOSTIC)

        term_entries = self._collect_terms(terms, time_stepper)
        pairs = tuple(zip(term_entries, terms, strict=True))
        pairs = self._filter_pairs(pairs)
        term_entries = self._apply_envelope(pairs)
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
        coverage lint — and, when a phase axis is declared, the
        PHASE RESOLUTION: the write sets decide which group each
        term and each unclaimed constraint belongs to, so the axis
        can only be installed here, once the dry pass has observed
        them (:meth:`_resolve_phases`).

        The whole hook pass runs under a single zero-argument
        ``jax.eval_shape`` trace: it is abstract-evaluated, with no
        device execution and no XLA compiles. This is sound because
        (a) the dry pass consumes only component *names* and
        function-space *metadata* — the ``frozenset(result)`` keys,
        the ``function_space.bare`` space comparisons, the
        ``advances`` string-set check — and never array *values*;
        (b) every hook is already trace-safe by construction, since
        ``compose()`` is jitted into the chunk executable, so
        tracing here surfaces the same errors eager execution did
        (a hook branching on a traced value raises the same
        ``TracerBoolConversionError`` it would raise inside the
        jitted step); and (c) the closure is deliberately
        ZERO-ARGUMENT so grid-derived materializations (e.g. the
        spectral eigenvalue table, which depends on grid metadata,
        not on the traced state) are captured abstractly too instead
        of compiling eagerly. The two Python-side lints run on the
        collected write sets, outside the traced closure.

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
        param_map: Any = {} if params is None else params
        writes: dict[ScheduleEntry, frozenset[str]] = {}

        # Run every hook under ONE zero-argument abstract-evaluation
        # trace instead of eager execution. Under the jax.eval_shape
        # dynamic trace, find_top_trace routes every jnp primitive to
        # the trace regardless of operand concreteness, so the whole
        # dry pass — the zero-state build, every term/stage hook, and
        # the grid-derived eigenvalue table build — is traced into one
        # jaxpr and NOTHING is compiled or run on device. All the
        # Python-side composition/shape/dtype validation still executes
        # and raises: writes[] records the contribution KEYS (concrete
        # strings even when the values are tracers) and the
        # gate/advances/space checks compare metadata, not values (see
        # the docstring for the correctness rationale).
        def _evaluate() -> None:
            state = self._zero_state()
            ctx = StepContext(params=param_map, clock=jnp.asarray(0.0),
                              dt=jnp.asarray(1.0),
                              stage_dt=jnp.asarray(1.0))
            for kind in _PRE_TENDENCY:
                state = self._dry_stages(kind, state, ctx, writes)
            sums = self._dry_terms(state, ctx, writes)
            ctx = StepContext(params=param_map, clock=jnp.asarray(0.0),
                              dt=jnp.asarray(1.0),
                              stage_dt=jnp.asarray(1.0),
                              tendency_sums=sums)
            for kind in _POST_TENDENCY:
                state = self._dry_stages(kind, state, ctx, writes)

        jax.eval_shape(_evaluate)
        resolved = self._resolve_phases(writes)
        membership = None if resolved is None else resolved[1]
        _overlap_lint(schedule, writes, membership)
        self._coverage_lint(writes)
        if resolved is not None:
            (groups, membership, implicit_groups, implicit_merged,
             implicit_phases) = resolved
            self._phase_coverage_lint(writes, membership, groups)
            self._schedule = schedule.with_phases(
                tuple(dataclasses.replace(
                    entry, active_phases=membership.get(entry))
                    for entry in schedule.entries),
                groups, implicit_groups, implicit_merged,
                implicit_phases)
        self._time_dependent_field_lint()

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
            fn = term.fn
            if fn is not None:
                fn = _normalize_owned_hook(
                    key, fn, self._modules[slot])
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
                fn=fn, gate=gate, treatment=term.treatment,
                advances=term.advances, reads=(),
                implicit=term.implicit, linear=term.linear,
                per_phase=term.per_phase))
        return tuple(entries)

    def _filter_pairs(
        self,
        pairs: tuple[tuple[ScheduleEntry, tuple[int, TendencyTerm]],
                     ...],
    ) -> tuple[tuple[ScheduleEntry, tuple[int, TendencyTerm]], ...]:
        """Apply the variant term filter (terms only, 08 10.4)."""
        if self._term_filter is None:
            return pairs
        # module-aware predicates (fr.model.term_predicates,
        # wants_module=True) receive the owning module for owned_by's
        # isinstance check; legacy two-argument callables keep the
        # (key, term) signature.
        wants_module = getattr(self._term_filter, "wants_module", False)
        kept = tuple(
            (entry, (slot, term)) for entry, (slot, term) in pairs
            if (self._term_filter(entry.key, term, self._modules[slot])
                if wants_module
                else self._term_filter(entry.key, term)))
        if pairs and not kept:
            raise AssemblyError(
                "the term filter drops every collected term; an "
                "empty filter result is a build error (model.md, "
                "variant mechanics)")
        return kept

    def _apply_envelope(
        self,
        pairs: tuple[tuple[ScheduleEntry, tuple[int, TendencyTerm]],
                     ...],
    ) -> tuple[ScheduleEntry, ...]:
        """Wrap the envelope-matched term hooks (C1 mechanism).

        Description
        -----------
        Detects (at most) one term-envelope module — any module whose
        ``envelope_terms`` capability attribute is a term predicate
        (``fr.model.modules.TendencyEnvelope``) — and wraps every filter-
        surviving matched term's hook so its contribution dict is
        scaled by the stage-time ``ctx.params["ramping.envelope"]``
        (never a host-captured value — the D2 rule). Taught refusals:
        a second envelope module, a matched ``IMPLICIT`` term (the
        solve inverts ``1 - dt*gamma*L``, not ``rho*(...)``), a
        matched ``linear=True`` term (ramping ``L`` is the
        parameter-deformation path's job), and an empty match — an
        error without a ``term_filter``, a warning under one (a
        filtered variant, e.g. an all-linear backward leg, keeps an
        inert envelope and must still build).
        """
        entries = tuple(entry for entry, _ in pairs)
        envelopes = tuple(
            (slot, module)
            for slot, module in enumerate(self._modules)
            if getattr(module, "envelope_terms", None) is not None)
        if not envelopes:
            return entries
        if len(envelopes) > 1:
            named = ", ".join(
                f"modules[{slot}] ({type(module).__name__})"
                for slot, module in envelopes)
            raise AssemblyError(
                f"more than one term-envelope module in this "
                f"assembly: {named}; one envelope module per "
                "assembly — compose one predicate with | instead")
        predicate = envelopes[0][1].envelope_terms
        wrapped: list[ScheduleEntry] = []
        matched = False
        for entry, (slot, term) in pairs:
            if not predicate(entry.key, term, self._modules[slot]):
                wrapped.append(entry)
                continue
            matched = True
            if entry.treatment is Treatment.IMPLICIT:
                raise AssemblyError(
                    f"the term envelope matches the IMPLICIT term "
                    f"{entry.key}: an enveloped implicit solve is "
                    "unsound — the solve inverts (1 - dt*gamma*L), "
                    "not rho*(...); narrow the envelope predicate "
                    "(e.g. & fr.model.term_predicates.explicit) or "
                    "treat the term "
                    "explicitly")
            if entry.linear:
                raise AssemblyError(
                    f"the term envelope matches the linear=True term "
                    f"{entry.key}: ramping the linear operator is "
                    "the parameter-deformation path's job — ramp its "
                    "coefficient instead (ramps={...}, the f0-ramp "
                    "idiom), or exclude it "
                    "(~fr.model.term_predicates.linear)")
            fn = (entry.fn if entry.fn is not None
                  else entry.implicit.apply)
            wrapped.append(dataclasses.replace(
                entry, fn=_wrap_envelope(fn), enveloped=True))
        if not matched:
            message = (
                f"the term envelope "
                f"({predicate.fingerprint_token()}) matches no "
                "collected term; an inert envelope silently skips "
                "the ramp")
            if self._term_filter is not None:
                # info downgrade under a variant filter: a filtered
                # leg (e.g. linear-only backward) must still build
                warnings.warn(message, stacklevel=3)
            else:
                raise AssemblyError(message)
        return tuple(wrapped)

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
                reads=stage.reads, writes=stage.writes,
                implicit=None, phase=stage.phase))
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

    def _check_waiver(
        self, allow_unadvanced: Sequence[str],
    ) -> frozenset[str]:
        """Validate the ``allow_unadvanced=`` coverage-lint waiver.

        Description
        -----------
        The waiver is only meaningful for PROGNOSTIC fields of *this*
        assembly, so an unknown name is refused rather than ignored: a
        typo would otherwise leave the lint armed on the field the
        caller meant to waive while reading as if it were disarmed.
        Duplicates are collapsed.

        Parameters
        ----------
        allow_unadvanced : Sequence[str]
            The caller's declared-inert PROGNOSTIC field names.

        Returns
        -------
        frozenset[str]
            The validated waiver set (empty when not used).

        Raises
        ------
        AssemblyError
            If a name is not a PROGNOSTIC field of this assembly.
        """
        waived = frozenset(allow_unadvanced)
        unknown = tuple(sorted(waived - set(self._prognostic)))
        if unknown:
            raise AssemblyError(
                f"allow_unadvanced names {unknown}, which are not "
                f"PROGNOSTIC fields of this assembly; the PROGNOSTIC "
                f"fields are {self._prognostic}")
        return waived

    def _coverage_lint(
        self, writes: dict[ScheduleEntry, frozenset[str]],
    ) -> None:
        """Every PROGNOSTIC field must be advanced (D1.4 lint).

        Description
        -----------
        A PROGNOSTIC field is "advanced" iff a term writes it, or an
        ADVANCE **or CONSTRAINT stage claims it** via ``advances=``
        (the spec §5.4 lint amendment; the barotropic subcycle and
        the implicit free surface's ``ps``, HY-D4). A CONSTRAINT
        stage's unclaimed writes earn no credit: the nonhydro
        pressure projection replaces the velocities without
        advancing them, and silencing the lint for those would hide
        a genuinely term-free prognostic.

        The lint exists to catch a field frozen by an **assembly
        mistake**, which it cannot tell apart from one frozen on
        purpose — so a deliberately inert field is declared, not
        guessed: ``allow_unadvanced=`` drops the named PROGNOSTIC
        fields from the check. That is the supported spelling for
        e.g. a non-rotating linear nonhydrostatic slice, whose
        horizontal velocities carry no term at all (only the pressure
        constraint rewrites them); before the waiver existed the only
        way through was a zero-valued Coriolis module, which declares
        the terms with a zero leaf and therefore *hides* the very
        condition the lint reports. The waiver is host-side only: it
        changes no schedule, no term, and no number.
        """
        advanced: set[str] = set()
        for entry, observed in writes.items():
            if entry.is_term:
                advanced |= observed
            elif entry.kind in (StageKind.ADVANCE,
                                StageKind.CONSTRAINT):
                advanced |= set(entry.advances or ())
        uncovered = tuple(name for name in self._prognostic
                          if name not in advanced
                          and name not in self._allow_unadvanced)
        if not uncovered:
            return
        message = (
            f"PROGNOSTIC fields {uncovered} are advanced by no "
            "term and claimed by no ADVANCE/CONSTRAINT stage "
            "(coverage lint, D1.4). Usually a missing module: "
            "install the term that should advance them — an "
            "advection scheme, a closure targeting them, a Coriolis "
            "or buoyancy module. If they are deliberately inert (a "
            "non-rotating linear slice leaves the horizontal "
            "velocities term-free), say so explicitly with "
            f"Model(..., allow_unadvanced={uncovered}) rather than "
            "declaring a zero-valued Coriolis module to fake the "
            "coverage")
        if self._term_filter is not None:
            # info downgrade under a variant filter (08 10.4)
            warnings.warn(message, stacklevel=3)
            return
        raise AssemblyError(message)

    # ================================================================
    #  Phase resolution (the phase axis; assembly step 6b)
    # ================================================================
    def _resolve_phases(
        self, writes: dict[ScheduleEntry, frozenset[str]],
    ) -> tuple | None:
        """Resolve the declared phase axis against the dry run.

        Description
        -----------
        Returns ``(groups, membership, implicit_groups,
        implicit_merged, implicit_phases)``, or ``None``
        when there is no axis to install — no declaration at all, or
        a partition that resolves to ONE group (``Phases.total()``),
        which IS the unphased schedule: nothing is installed, the
        entries keep their default (phase-free) static tokens, and
        the steppers run their literal one-pass body.

        ``membership`` maps each entry to the phase indices it runs
        in; ``None`` means "every phase" (SELF_UPDATE, DIAGNOSE, the
        per-step DIAGNOSTIC epilogue, and constraints that write no
        PROGNOSTIC field). The rules are the ``Stage``/
        ``TendencyTerm`` docstrings'; the refusals are taught.
        """
        if self._phases is None:
            return None
        groups = self._phases.resolve(
            self._prognostic,
            velocity=self._velocity_names(),
            claimed=self._claimed_names())
        if len(groups) <= 1:
            return None
        owner = {name: index for index, group in enumerate(groups)
                 for name in group}
        membership = {
            entry: self._entry_phases(entry, writes, owner, groups)
            for entry in self._schedule.entries}
        return groups, membership, *self._implicit_phases(owner)

    def _velocity_names(self) -> tuple[str, ...]:
        """PROGNOSTIC names carrying the ``Velocity`` role."""
        prognostic = set(self._prognostic)
        return tuple(
            record.name for record in self._records
            if record.name in prognostic
            and any(isinstance(role, Velocity)
                    for role in getattr(record, "roles", ())))

    def _claimed_names(self) -> tuple[str, ...]:
        """PROGNOSTIC names claimed by an ADVANCE/CONSTRAINT stage."""
        claimed: list[str] = []
        for entry in self._schedule.entries:
            if entry.kind in (StageKind.ADVANCE,
                              StageKind.CONSTRAINT):
                claimed.extend(entry.advances or ())
        return tuple(claimed)

    def _entry_phases(
        self,
        entry: ScheduleEntry,
        writes: dict[ScheduleEntry, frozenset[str]],
        owner: dict[str, int],
        groups: tuple[frozenset[str], ...],
    ) -> tuple[int, ...] | None:
        """Resolve one entry's phase membership (or None = every)."""
        if entry.phase is not None:
            if not 0 <= entry.phase < len(groups):
                raise AssemblyError(
                    f"{entry.key}: phase={entry.phase} is out of "
                    f"range — this assembly resolves "
                    f"{len(groups)} phase(s) "
                    f"(valid indices 0..{len(groups) - 1})")
            return (entry.phase,)
        if entry.kind is StageKind.DIAGNOSTIC:
            # S6 runs once per STEP, outside the phase loop
            return None
        written = self._written_prognostic(entry, writes, owner)
        if entry.is_term:
            return self._term_phases(entry, written)
        if entry.kind in (StageKind.SELF_UPDATE, StageKind.DIAGNOSE):
            # the per-substage refresh IS the point of the axis
            return None
        claim = tuple(entry.advances or ())
        if claim:
            phases = tuple(sorted({owner[name] for name in claim}))
            if len(phases) > 1:
                raise AssemblyError(
                    f"{entry.key}: the stage claims advances="
                    f"{claim}, whose names live in different phase "
                    f"groups {phases}; a stage advances one group "
                    "per phase. Split the claim into one stage per "
                    "group, put the names in one group "
                    "(fr.model.Phases((...), (...))), or pin the "
                    "stage with Stage(..., phase=k) if it really "
                    "advances both in one go")
            return phases
        return written or None

    def _written_prognostic(
        self,
        entry: ScheduleEntry,
        writes: dict[ScheduleEntry, frozenset[str]],
        owner: dict[str, int],
    ) -> tuple[int, ...]:
        """Phase indices of the entry's observed PROGNOSTIC writes."""
        observed = writes.get(entry, frozenset())
        return tuple(sorted({owner[name] for name in observed
                             if name in owner}))

    @staticmethod
    def _term_phases(
        entry: ScheduleEntry, written: tuple[int, ...],
    ) -> tuple[int, ...] | None:
        """Phase membership of one term (the straddle refusal)."""
        if not written:
            return None
        if (len(written) == 1 or entry.per_phase
                or entry.treatment is Treatment.IMPLICIT):
            # an IMPLICIT term straddles as its MERGE GROUP, which
            # carries the precise "coupled blocks are atomic" refusal
            return written
        raise AssemblyError(
            f"{entry.key}: the term writes across the phase groups "
            f"{written}, but declares per_phase=False. A term is "
            "summed into ONE group's increment, so a straddling "
            "write set would land in a group it is not integrated "
            "with. Declare @fr.model.term(per_phase=True) — the "
            "term is then evaluated once per phase it touches and "
            "masked to that phase's keys (read ctx.phase.fields to "
            "skip the other groups' work) — or split it into one "
            "term per group")

    def _implicit_phases(
        self, owner: dict[str, int],
    ) -> tuple[tuple, tuple, tuple[int | None, ...]]:
        """Assign each implicit merge group to its phase.

        Description
        -----------
        A group whose ``fields`` lie in one phase is assigned to it
        untouched. A group that STRADDLES is split into one operator
        per phase — but only if the operator declares itself
        FIELD-SEPARABLE by exposing ``restricted_to(fields)``
        (``VerticalDiffusion`` does: its ``apply``/``solve`` are a
        per-field loop over independent column bands, so restricting
        the field tuple is exact, not an approximation). A straddling
        group WITHOUT that seam is the taught refusal: a genuinely
        coupled implicit block is atomic under by-variable splitting.

        Returns the (possibly lengthened) ``implicit_groups``,
        ``implicit_merged`` and the parallel owning-phase row.
        """
        constituents: list = []
        merged: list = []
        resolved: list[int | None] = []
        for (operator, slot), group in zip(
                self._schedule.implicit_merged,
                self._schedule.implicit_groups, strict=True):
            fields = tuple(getattr(operator, "fields", ()))
            phases = tuple(sorted({owner[name] for name in fields
                                   if name in owner}))
            if len(phases) <= 1:
                constituents.append(group)
                merged.append((operator, slot))
                resolved.append(phases[0] if phases else None)
                continue
            restrict = getattr(operator, "restricted_to", None)
            if restrict is None:
                keys = ", ".join(key for key, _s, _op in group)
                raise AssemblyError(
                    f"the implicit merge group ({keys}) solves "
                    f"{fields}, whose names live in different phase "
                    f"groups {phases}: coupled implicit blocks are "
                    "atomic under by-variable splitting (spec 5.1) "
                    "— one solve cannot be split across two phases. "
                    "An operator whose solve is independent per field "
                    "may declare restricted_to(fields) and is then "
                    "split automatically; otherwise put the coupled "
                    "fields in one group (fr.model.Phases((...), "
                    "(...))), or split the operator yourself")
            for index in phases:
                subset = tuple(name for name in fields
                               if owner.get(name) == index)
                constituents.append(group)
                merged.append((restrict(subset), slot))
                resolved.append(index)
        return tuple(constituents), tuple(merged), tuple(resolved)

    def _phase_coverage_lint(
        self,
        writes: dict[ScheduleEntry, frozenset[str]],
        membership: dict[ScheduleEntry, tuple[int, ...] | None],
        groups: tuple[frozenset[str], ...],
    ) -> None:
        """Every PROGNOSTIC is advanced IN ITS OWN phase.

        Description
        -----------
        The phase-aware half of the D1.4 coverage lint: the global
        lint only asks whether SOME term writes a field, but under
        the phase axis the writer must also RUN in the phase that
        adds that field's increment — a stage pinned with ``phase=``
        to the wrong group, or a claim pinned away from the names it
        claims, would otherwise freeze the field silently. The
        ``allow_unadvanced=`` waiver and the variant-filter warning
        downgrade apply exactly as they do to the global lint.
        """
        for index, group in enumerate(groups):
            advanced: set[str] = set()
            for entry, observed in writes.items():
                active = membership.get(entry)
                if active is not None and index not in active:
                    continue
                if entry.is_term:
                    advanced |= (observed & group if entry.per_phase
                                 else observed)
                elif entry.kind in (StageKind.ADVANCE,
                                    StageKind.CONSTRAINT):
                    advanced |= set(entry.advances or ())
            uncovered = tuple(
                name for name in self._prognostic
                if name in group and name not in advanced
                and name not in self._allow_unadvanced)
            if not uncovered:
                continue
            message = (
                f"PROGNOSTIC fields {uncovered} belong to phase "
                f"{index} ({tuple(sorted(group))}) but nothing "
                "advances them THERE: every term writing them, and "
                "every stage claiming them, runs in another phase. "
                "Check the Stage(phase=...) pins, or partition the "
                "fields so each one shares a group with its writer "
                "(fr.model.Phases((...), (...)))")
            if self._term_filter is not None:
                warnings.warn(message, stacklevel=3)
                continue
            raise AssemblyError(message)

    def _time_dependent_field_lint(self) -> None:
        """Every ``time_dependent`` AUX field has an owner SELF_UPDATE stage.

        Description
        -----------
        The TDF-D3 lint: a ``time_dependent``-marked AUXILIARY field
        (``FieldDeclaration.time_dependent``) must be rewritten every
        substage by a SELF_UPDATE stage of its OWNING module — nothing
        else structurally guarantees a field that "evolves in time" ever
        updates (the coverage lint covers PROGNOSTIC fields only, so this
        is the lint gap the roadmap entry names). Resolution is per field
        through the stage ``writes=`` declaration; an undeclared
        (``writes=None``) SELF_UPDATE stage satisfies any of its owner's
        marked fields — the documented fallback ("owner has at least one
        SELF_UPDATE stage").
        """
        su_writes: dict[int, list[tuple[str, ...] | None]] = {}
        for entry in self._schedule.kind_entries(StageKind.SELF_UPDATE):
            su_writes.setdefault(entry.slot, []).append(entry.writes)
        offenders = tuple(
            f"{record.name} ({record.owner_type})"
            for record in self._records
            if record.lifecycle is Lifecycle.AUXILIARY
            and getattr(record, "time_dependent", False)
            and not _covered_by_self_update(
                record.name, su_writes.get(record.owner)))
        if offenders:
            raise AssemblyError(
                f"time_dependent AUXILIARY field(s) {offenders} are not "
                "written by a SELF_UPDATE stage of their owning module "
                "(TDF-D3): a field marked time_dependent evolves in time, "
                "so its owner must recompute it every substage with an "
                "fr.model.self_update stage (declare writes= naming "
                "the field, or add the stage)")


# ================================================================
#  Module-level helpers
# ================================================================
def _check_stepper_phases(
    phases: Phases | None, time_stepper: Any,
) -> None:
    """Refuse a phase axis under a stepper that cannot loop.

    Description
    -----------
    The phase loop lives inside the MULTISTEP steppers
    (``AdamBashforth``, ``IMEXMultistep``): they commit one
    post-advance state per step, so running the substage chain once
    per group is a single extra pass. A Runge-Kutta or exponential
    stepper would have to run the loop inside EVERY tableau
    substage — three barotropic solves per step for LowStorageRK3 —
    which no production model does; the refusal is designed-for,
    never a silent demotion.
    """
    if phases is None or time_stepper is None:
        return
    if getattr(time_stepper, "supports_phases", False):
        return
    raise AssemblyError(
        f"phases={phases!r} under "
        f"{type(time_stepper).__name__}, which does not support the "
        "phase axis (supports_phases=False): the staggered step is "
        "a loop over the variable groups inside ONE multistep "
        "advance, and a substage-based stepper would repeat that "
        "loop (and its barotropic solve) per tableau stage. Use a "
        "multistep driver — fr.model.time_steppers.AdamBashforth or "
        "IMEXMultistep (CNAB2 / SBDF2) — or drop phases=")


def _covered_by_self_update(
    name: str, stages: list[tuple[str, ...] | None] | None,
) -> bool:
    """Whether an owner SELF_UPDATE stage rewrites ``name`` (TDF-D3).

    ``stages`` is the owner's SELF_UPDATE ``writes=`` declarations
    (``None`` = the owner has none). An undeclared (``writes=None``)
    stage covers any of its owner's marked fields (the fallback).
    """
    return stages is not None and any(
        writes is None or name in writes for writes in stages)
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


def _normalize_owned_hook(key: str, fn: object, owner: object) -> object:
    """
    Accept a bound method of the OWNING module; normalize to unbound.

    Description
    -----------
    A term ``fn`` is stored unbound and paired with a module slot at
    compose time (a bound method would pin the assembly-time instance
    while live parameters ride the carry — the D2 aliasing trap). The
    natural authoring spelling ``fn=self._advect`` is nonetheless
    accepted when ``fn`` is a bound method of exactly the owning
    module: it is normalized to its ``__func__`` so the composer's
    ``fn(carry.modules[slot], state, ctx)`` call is identical to the
    unbound spelling. A bound method of any OTHER object stays the
    aliasing error.

    Parameters
    ----------
    key : str
        The ``"Module/term"`` attribution key (error prefix).
    fn : object
        The declared hook (callable; unbound or bound).
    owner : object
        The owning module instance (the term's slot).

    Returns
    -------
    object
        The unbound hook (``fn.__func__`` for a bound owner method,
        else ``fn`` unchanged).

    Raises
    ------
    AssemblyError
        If ``fn`` is a bound method of a different object (the D2
        aliasing trap).
    """
    bound_self = getattr(fn, "__self__", None)
    if bound_self is None:
        return fn
    if bound_self is not owner:
        raise AssemblyError(
            f"{key}: the hook is a BOUND method of "
            f"{type(bound_self).__name__}; hooks are stored "
            "unbound and paired with a module slot at compose time "
            "(a bound method captures the assembly-time instance "
            "while live parameters ride the carry)")
    return fn.__func__


def _wrap_envelope(fn: Callable) -> Callable:
    """
    Wrap a term hook in the stage-time envelope multiply.

    Description
    -----------
    The returned hook scales every contribution of ``fn`` by
    ``ctx.params["ramping.envelope"]`` — the envelope value is read
    from the per-substage context (the ``TendencyEnvelope`` module's
    live leaf, stage-time resolved), never closed over as a host
    value (the D2 no-host-capture rule: a captured value would bake
    the ramp endpoints into the jitted step as constants).

    Parameters
    ----------
    fn : Callable
        The unbound term hook ``(module, state, ctx) -> dict``.

    Returns
    -------
    Callable
        The enveloped hook, same signature.
    """
    def enveloped(
        module: object, state: object, ctx: StepContext,
    ) -> dict:
        """Scale the wrapped hook's contributions by rho(t)."""
        rho = ctx.params[RAMPING_ENVELOPE]
        return {name: value * rho
                for name, value in fn(module, state, ctx).items()}

    return enveloped


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

    A non-mergeable custom operator whose fields overlap a mergeable
    **family** group is the same footgun (the driver solves the two
    groups independently and ``updates.update(...)``s each, so the
    second solve silently clobbers the first on the shared field): it
    is a taught ``ImplicitCollisionError`` naming both parties and the
    reason, in either declaration order. Two *different* mergeable
    families overlapping is out of this lint's scope (unchanged).

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
        On a second non-mergeable operator per field, or a custom
        operator overlapping a mergeable family's field.
    """
    groups: dict[Any, list] = {}
    custom_by_field: dict[str, str] = {}
    family_by_field: dict[str, str] = {}
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
                if field in family_by_field:
                    raise _custom_family_collision(
                        field, entry.key, family_by_field[field])
                custom_by_field[field] = entry.key
            groups[("custom", entry.key, entry.index)] = [
                (entry.key, entry.slot, op)]
        else:
            for field in op.fields:
                if field in custom_by_field:
                    raise _custom_family_collision(
                        field, custom_by_field[field], entry.key)
                family_by_field.setdefault(field, entry.key)
            groups.setdefault(("family", merge_key), []).append(
                (entry.key, entry.slot, op))
    return tuple(tuple(group) for group in groups.values())


def _custom_family_collision(
    field: str, custom_key: str, family_key: str,
) -> ImplicitCollisionError:
    """Build the taught custom-overlaps-family collision error.

    Description
    -----------
    A non-mergeable custom implicit operator and a mergeable family
    both advancing ``field`` become two separate merge groups; the
    IMEX driver solves each independently from the raw rhs and
    ``updates.update(...)``s the result, so the second solve clobbers
    the first on the shared field (the latent footgun, spec §2). The
    message names both parties and the reason.

    Parameters
    ----------
    field : str
        The clobbered PROGNOSTIC field.
    custom_key : str
        The custom (non-mergeable) operator's attribution key.
    family_key : str
        A constituent of the mergeable family group.

    Returns
    -------
    ImplicitCollisionError
        The taught error (raise it at the call site).
    """
    return ImplicitCollisionError(
        f"the custom implicit operator {custom_key} and the mergeable "
        f"implicit family {family_key} both advance {field!r}: the "
        "driver solves the merged family and the custom operator "
        "independently and overwrites each result, so one solve would "
        "silently clobber the other on the shared field. A custom "
        "implicit operator may not overlap a mergeable family's "
        "fields; treat the coupling in one operator (or move it to a "
        "CONSTRAINT stage that runs after the merged solve)")


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
    membership: dict[ScheduleEntry, tuple[int, ...] | None] | None
    = None,
) -> None:
    """Same-kind, equal-order overlapping write(-read) sets error.

    Phase-aware: two same-kind, equal-order stages that never run in
    the same phase cannot collide, so the lint skips them (that is
    what lets a geometry declare one pinned SELF_UPDATE stage per
    phase, both rewriting the same AUXILIARY field).
    """
    stage_entries = [entry for entry in schedule.entries
                     if not entry.is_term]
    for left, right in combinations(stage_entries, 2):
        if left.kind is not right.kind or left.order != right.order:
            continue
        if membership is not None and _disjoint_phases(
                membership.get(left), membership.get(right)):
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


def _disjoint_phases(
    left: tuple[int, ...] | None, right: tuple[int, ...] | None,
) -> bool:
    """Whether two resolved memberships never share a phase.

    ``None`` means "every phase", which intersects everything.
    """
    if left is None or right is None:
        return False
    return not set(left) & set(right)
