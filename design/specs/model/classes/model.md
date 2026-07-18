---
status: normative
date: 2026-07-13
---

# Model layer redesign — Class designs: Model

Part of the model-layer class designs; see [`README.md`](README.md)
for the document map and the shared template. This file owns the
**Model** cluster: `fr.Model` itself (constructor = the nine-step
assembly; the full lifecycle/run-loop surface), the assembly
internals (`FieldTable`/`VelocitySelector`, `TendencyComposer`, the
binding and re-materialization tables, `AssemblyRecord`,
`AssemblyReport`, the fingerprint), the `State` vocabulary-class
contract, the run-loop result types
(`AdvanceResult`/`RunResult`/`PanicError`), preset-factory rules,
and the model-layer error registry.

Normative sources (all signed; nothing here re-decides):
[`../04_run_loop_io.md`](../04_run_loop_io.md) §6.1–6.7 (primary),
[`../01_concepts.md`](../01_concepts.md) D1.3/D1.5/D2.4,
[`../02_rules.md`](../02_rules.md),
[`../03_time_stepping.md`](../03_time_stepping.md) §5.2/§5.5,
[`../08_state_transforms.md`](../08_state_transforms.md) §10.4, and
the coupling constraints **CS-2..13**
([`../09_coupling_designfor.md`](../09_coupling_designfor.md)
§11.2–11.3). The grid lifecycle + frozen-grid verify path this
cluster consumes is normative in
[`../../grid/classes/grid.md`](../../grid/classes/grid.md) ("Grid lifecycle" +
the amended "Merge call site" entry). Declarations, `Module`,
`StepContext`, steppers, transforms, and IO/ops classes are owned by
the sibling docs ([`declarations.md`](declarations.md),
[`module.md`](module.md), [`time_steppers.md`](time_steppers.md),
[`transforms.md`](transforms.md), [`io_ops.md`](io_ops.md)).

Deviations called out in this file (spec-level detail the notes did
not fix; each also listed under Open questions where residual):
the `State.prognostic` backing mechanism (the FieldTable-as-static-
aux proposal — **resolved by rejection**, 2026-07-08, in favor of the
model-mediated `FieldTable.subset` spelling; see the `State`
vocabulary contract and Open question 5), the `model.clock` host read
(needed
by the Session's CS-18 clock assertion and progress reporting; §6.5's
read list does not name it), the error name `RunTargetError` (the
design says only "named error"), and `set_aux` leaving the panic
flag untouched (a clarification consistent with the §6.5 rule that
only resume-path operations clear it).

---

## 1. Package layout

```
fridom/framework2/model/
    __init__.py        # re-exports: Model (as fr.Model), errors, results
    model.py           # Model, ModelState (the carry), ParameterView,
                       #   DiagnosticsNamespace, step_chunk
    field_table.py     # FieldTable, FieldRecord, VelocitySelector
    assembly.py        # the nine-step pipeline function, AssemblyRecord,
                       #   ParameterBindingTable, Params,
                       #   RematerializationTable, Fingerprint
    composer.py        # TendencyComposer
    report.py          # AssemblyReport
    results.py         # RunStatus, AdvanceResult, RunResult, PanicError
    errors.py          # the model-layer error registry (§7)
```

Vocabulary `State` subclasses live per model package
(`fridom/nonhydro/state.py`, ...). Presets are per-package factory
functions (`fridom/nonhydro/model.py: Model(...) -> fr.Model`),
never classes (§6). `fr.ops.Session` and the `OutputStream`
implementations live in the IO/ops cluster; this file only fixes the
Model-side binding rules they attach to.

---

## 2. fr.Model

Composition root: grid + modules + stepper. `__init__` **is**
assembly; the instance is a **host-side driver holding assembly
artifacts plus the single mutable slot `_carry`** — lifecycle
methods are mutating spellings over pure carry transformers (new
carry built functionally, reference swapped).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final (presets are factories, never subclasses) |
| Pytree | **host object, not a pytree** (a Model-as-pytree would double-flatten the carry — the D2 aliasing bug by construction) |
| Task | 2.3 (assembly, lifecycle); 2.4 (advance/run/panic); 2.6 (snapshot/io binding); 2.7 (`constrain` and its consumers; `blank_state`/`state_space` **not built**); 2.8 (`tendency`/`variant`) |
| Design refs | 04 §6.1–6.7; 01 D1.3/D1.5/D2.4; 02_rules (set_aux, exemptions, fingerprint scope, no-pickled-models); 08 §10.4; CS-2..13 |

```python
"""fr.Model — the composition root and host driver."""
from __future__ import annotations

import fridom.framework2 as fr


class Model:
    """Grid + modules + stepper; owns assembly, the carry, the run
    loop. Host-side driver — nothing traces through it."""

    def __init__(                                              # 2.3
        self,
        *,
        grid: fr.grid.Grid,
        modules: tuple[fr.Module, ...],
        time_stepper: fr.time_steppers.TimeStepper,  # REQUIRED, no default
        io: tuple[OutputStream, ...] = (),
        state_type: type[fr.VectorField] | None = None,
        name: str | None = None,
    ) -> None:
        """Assemble (the nine-step pipeline, §6.2): pure and
        deterministic in its inputs — same inputs, identical carry
        treedef. No ``setup()`` exists."""
        ...

    # ================================================================
    #  Read surface (host, chunk boundaries)
    # ================================================================

    @property
    def state(self) -> fr.VectorField:                         # 2.3
        """The carry's state vector, read-only (the old ``.z``
        setter dies). Donation caveat: copy-on-read decision at
        implementation (open thread)."""
        ...

    @property
    def parameters(self) -> ParameterView:                     # 2.3
        """Read-only mapping over the binding table; values read
        live from the carry; Ramp-valued slots return the Ramp
        object (evaluate via ``.at_time(t)``). Hinted
        MissingParameterError on unprovided names."""
        ...

    def module(                                                # 2.3
        self,
        module_type: type[M],
        *,
        name: str | None = None,
    ) -> M:
        """Typed lookup of a live carry module (unpublished knobs);
        ambiguous matches error listing candidates."""
        ...

    @property
    def diagnostics(self) -> DiagnosticsNamespace:             # 2.3
        """Bound package diagnostics (``model.diagnostics.pot_vort()``);
        parameters resolved lazily through ``self.parameters``."""
        ...

    @property
    def grid(self) -> fr.grid.Grid:                            # 2.3
        """The (frozen) grid this model was assembled on."""
        ...

    @property
    def clock(self) -> fr.time_steppers.Clock:                 # 2.4
        """Host read of the carry clock (start/elapsed/it).
        [Spec addition: needed by the Session's cross-model clock
        assertion (CS-18) and progress reporting.]"""
        ...

    @property
    def name(self) -> str | None:                              # 2.3
        """Report/log attribution (two models, one process)."""
        ...

    @property
    def panicked(self) -> bool:                                # 2.4
        """Cheap host-readable panic flag (CS-6)."""
        ...

    @property
    def carry(self) -> ModelState:                             # 2.4
        """Opaque in-memory carry snapshot — the disk-free twin of
        ``snapshot()`` (CS-12). Read-only; the sanctioned setter is
        a leave-open (``debug_nan`` builds on this)."""
        ...

    # ================================================================
    #  State factories (the surface IC recipes and transforms build on)
    # ================================================================

    def blank_state(self) -> fr.VectorField:                   # NOT BUILT
        """PROGNOSTIC subset at declared defaults, born sharded
        (D5 amendment S2). Never implemented: IC recipes compose
        ``grid.create_field(space, ...)`` with spaces read off
        ``model.field_table[name].space``. Build or strike —
        07_open_threads §9.1."""
        ...

    def state_space(self, name: str) -> TensorProductSpace:    # NOT BUILT
        """The negotiated (laid-out) space of a declared field —
        feeds ``grid.create_field`` / ``grid.random`` (sketch 7.3).
        Never implemented: read it off ``model.field_table[name].space``
        (07_open_threads §9.1)."""
        ...

    # ================================================================
    #  Lifecycle mutators (host, chunk boundaries; §6.5)
    # ================================================================

    def set_fields(                                            # 2.3
        self,
        **fields: Callable | np.ndarray | fr.ScalarField,
    ) -> None:
        """User ICs, PROGNOSTIC-only (non-PROGNOSTIC names error,
        pointing to ``set_aux``); incoming values re-homed per
        ``decomposition.sharding(space)``. Clears the panic flag."""
        ...

    def set_state(self, state: fr.VectorField) -> None:        # 2.3
        """Overwrite the PROGNOSTIC subset; **missing components are
        left untouched** (amended V-C12); AUX/DIAG components in the
        input are ignored with a debug log. Re-homes; clears the
        panic flag."""
        ...

    def set_aux(                                               # 2.3
        self,
        *,
        rewarm: bool = False,
        **fields: fr.ScalarField | np.ndarray,
    ) -> None:
        """The declaration-consented host write (V-C1, 02_rules):
        AUXILIARY **or DIAGNOSTIC** components with
        ``host_writable=True`` only — the coupler exchange path and
        the S6-accumulator reset. Accepts device-resident Fields and
        re-homes without a host round-trip (CS-8). No warm-up
        re-ramp by default (exchange data is forcing). Leaves the
        panic flag untouched [clarification: not a resume path]."""
        ...

    def update_parameters(                                     # 2.3
        self,
        updates: Mapping[str | ParamName, object],
        *,
        rewarm: bool = True,
    ) -> None:
        """Resolve through the binding table (incl.
        ``fr.params.TIME_STEP`` — the stepper is a provider);
        validate structure preservation (same treedef -> no
        recompile; scalar<->Ramp is a treedef change -> re-assemble);
        write functionally; re-materialize owner-derived AUX fields
        (host-writable components exempt, CS-2); re-ramp the
        multistep warm-up by default (old-physics buffers; dt sign
        flips require it). Does **not** clear the panic flag."""
        ...

    def reset(self) -> None:                                   # 2.3
        """Exactly the OptimalBalance need: re-init stepper state
        (re-warm), reset the clock (**which is what restarts a Ramp
        leg**), clear panic, DIAGNOSTIC -> defaults. PROGNOSTIC and
        AUXILIARY untouched. Invariant: ``reset(); set_state(z)`` ==
        fresh assembly + ``set_state(z)``, bitwise."""
        ...

    def constrain(                                             # 2.7
        self, state: fr.VectorField, *, t: float | None = None,
    ) -> fr.VectorField:
        """Apply the CONSTRAINT stages to a state and return it —
        the opt-in initial projection of non-divergence-free ICs
        (optional because project-the-state self-corrects within one
        substage). Shipped *pure* rather than as the in-place
        ``apply_constraints()`` this doc first named: the caller feeds
        the result back through ``set_state``, and ``tendency(...,
        constraints=True)`` and the transforms reuse the same path."""
        ...

    # ================================================================
    #  Tendency and variants (D5 amendments to the D4 surface)
    # ================================================================

    def tendency(                                              # 2.8
        self,
        state: fr.VectorField,
        *,
        t: float | None = None,
        filter: TermPredicate | None = None,
        constraints: bool = True,
    ) -> fr.VectorField:
        """Host-callable, jitted, read-only wrapper over the
        composed tendency. SELF_UPDATE and DIAGNOSE stages run
        first, at ``t`` (amended V-H8 — the result reflects
        recomputed diagnostics, not the input's); implicit terms via
        their forward apply; CONSTRAINT stages applied to the result
        iff ``constraints``. Never advances the carry. ``t=None``
        reads the carry clock. Consumers: per-term budgets
        (``filter=fr.terms.named(...)``), term unit tests,
        linear-stability matvecs, the future TangentPropagator."""
        ...

    def variant(                                               # 2.8
        self,
        *,
        term_filter: TermPredicate | None = None,
        updates: Mapping[str | ParamName, object] | None = None,
        name: str | None = None,
    ) -> Model:
        """Derived model (08 §10.4): re-assembly on the parent's
        frozen grid (verify path, passes by the ⊆ lemma).
        Declarations never filtered — same FieldTable/State treedef;
        stages survive. ``updates=`` is assembly-time and may change
        value **specs** (scalar -> Ramp, ``TIME_STEP`` sign), unlike
        post-assembly ``update_parameters``. Filter token + updates
        specs enter the fingerprint; coverage lint downgrades to
        info; empty filter result / unknown ``named`` key = build
        error."""
        ...

    # ================================================================
    #  The run loop (§6.3)
    # ================================================================

    def advance(                                               # 2.4
        self,
        steps: int,
        *,
        sync: bool = True,
    ) -> AdvanceResult:
        """The public IO-free primitive (what a coupler loop
        interleaves). Chunked scan over the donated carry via the
        shared ``step_chunk``; the per-chunk panic read is the
        **only** host synchronization (no-hidden-sync, CS-4 — the
        committed carry may hold pending arrays). Raises
        ``PanicError`` at the abort boundary and on entry with a
        panicked carry. ``sync=False`` is reserved: returns a
        ``PendingAdvance`` handle (CS-5, semantics leave-open)."""
        ...

    def run(                                                   # 2.4
        self,
        steps: int | None = None,
        *,
        runlen: float | np.timedelta64 | None = None,
        end_time: float | np.timedelta64 | None = None,
        outputs: tuple[OutputStream, ...] = (),
        snapshots: Snapshots | None = None,
        max_chunk: int | None = None,
        progress: bool | ProgressReporter = True,
        jit: bool = True,
        profile: str | None = None,
        debug_nan: bool = False,
        raise_on_nan: bool = False,
    ) -> RunResult:
        """Sugar over ``advance`` + IO — reimplemented as a
        single-model Session loop (bitwise test: ``run()`` == a
        user-written Session loop). Exactly one of
        steps/runlen/end_time; sign-agnostic reduction
        ``steps = ceil((end - t0)/dt - eps)`` with precondition
        ``(end - t0)·dt > 0`` (``RunTargetError``). Repeated
        ``run()`` continues from the carry, bitwise identical to one
        uninterrupted run — given the per-step scan body compiles
        identically across chunk lengths; the equivalence test
        asserts a body-jaxpr/HLO match, degrading to
        identical-chunk-plan comparison if XLA ever specializes the
        body on trip count (phase-1 finding 1: bitwise claims hold
        only between identically-compiled paths). Converts
        ``PanicError`` to
        ``RunResult(NAN_ABORT)`` unless ``raise_on_nan``; never
        exits the process."""
        ...

    # ================================================================
    #  Persistence (§6.4; persistence = script re-assembly + snapshots)
    # ================================================================

    def snapshot(self, path: str | Path) -> None:              # 2.6
        """Dumb leaf blob: versioned manifest + true-shape gathered
        leaf arrays (device-count-portable); manifest records the
        fingerprint digest **and its source record** plus
        provided-parameter values (at least TIME_STEP)."""
        ...

    def load_snapshot(self, path: str | Path) -> None:         # 2.6
        """Assemble-then-load: fingerprint check
        (``SnapshotMismatchError`` **diffs**, never silent reuse; dt
        *sign* mismatch errors, magnitude warns), overwrite all
        carry leaves, re-home per device count. Clears the panic
        flag. ``Model.restore(path)`` classmethod rejected (the dill
        pattern reborn)."""
        ...

    # ================================================================
    #  Introspection
    # ================================================================

    @property
    def report(self) -> AssemblyReport:                        # 2.3
        """The assembly report (§3, AssemblyReport); run-start
        addendum appended at first ``run()``."""
        ...

    @property
    def fingerprint(self) -> Fingerprint:                      # 2.6
        """The restart fingerprint (02_rules scope: structure,
        never leaves; bare declared spaces, never Layout)."""
        ...

    def __repr__(self) -> str:                                 # 2.3
        """The assembly report's header."""
        ...
```

### The carry and the shared jitted entry

```python
@fr.utils.jaxify                    # all slots dynamic subtrees
class ModelState:
    """The full dynamic carry (successor of the old ModelState;
    variable spelled ``model_state``). No coupling/run-loop state
    lives outside it (CS-11)."""

    state: fr.VectorField           # all declared components, all lifecycles
    modules: tuple[fr.Module, ...]  # live module leaves (parameters, own AUX)
    stepper_state: StepperState     # multistep buffers + warm-up counter
    clock: Clock                    # traced float64 start/elapsed/it
    panic: PanicState               # (flag: sticky bool, it: first-failure int32)


def step_chunk(                                                # 2.4
    record: AssemblyRecord,         # hashable static (jit cache key)
    carry: ModelState,              # donated
    stepper: TimeStepper,           # loop-invariant, non-donated argument
    n: int,                         # static chunk length
) -> ModelState:
    """THE framework-level jitted entry — one per process, never a
    per-assembly closure (which would silently defeat the shared jit
    cache; implementation rule + compilation-count regression test).
    ``lax.scan`` of the composed step; AOT ``lower().compile()``
    replaces the first-step compile-timing hack and reports peak
    memory. Compiled lengths: {C, 1} (remainders via the lazily
    compiled ``chunk(1)``)."""
    ...
```

### ParameterView and DiagnosticsNamespace

```python
class ParameterView(Mapping[str, object]):
    """Read-only host view over the binding table (D2.4 tier 1)."""

    def __getitem__(self, name: str | ParamName) -> object:    # 2.3
        """Live leaf read from the carry; Ramp-valued slots return
        the Ramp object, never a silently-evaluated value."""
        ...

    def __iter__(self) -> Iterator[str]: ...                   # 2.3
    def __len__(self) -> int: ...                              # 2.3

    def at_time(self, t: float) -> Mapping[str, jax.Array]:    # 2.3
        """Explicitly evaluate every TimeDependent slot at ``t``
        (the D2.4 rule: no implicit clock evaluation host-side)."""
        ...

    def info(self, name: str | ParamName) -> ParameterDeclaration:
        """Units/doc/provider of a bound name."""               # 2.3
        ...

    # designed-for: as_field(name, space) — host-side helper, never
    # table unification (d2_1 risk 4).


class DiagnosticsNamespace:
    """``model.diagnostics`` — bound diagnostic expressions."""

    def __getattr__(self, name: str) -> BoundDiagnostic:       # 2.3
        """A callable wrapping the package-level pure function with
        parameters resolved through ``model.parameters`` —
        **lazily**: hinted MissingParameterError only when an
        absent-provider diagnostic is actually called."""
        ...

    def __dir__(self) -> list[str]: ...                        # 2.3


class BoundDiagnostic:
    def __call__(                                              # 2.3
        self, state: fr.VectorField | None = None,
    ) -> fr.ScalarField | jax.Array:
        """Evaluate on ``state`` (default: the current carry state).
        Pure ``(model_state) -> Field | scalar`` — exactly the
        writer-expression contract (``derived={"pv": ...}``)."""
        ...
```

Semantics, invariants, error behavior:

- **Constructor rules (§6.1)**: `time_stepper` has no default (no
  physics-free dt exists; presets supply the package default — nh:
  `AdamBashforth(order=3)` at cutover, V-N3, with low-storage RK3
  the documented recommendation). Rejected kwargs, each replaced:
  `halo=` (traced), `progress_bar=`/`nan_checker=`/`restart_module=`
  (run-time policy), verbosity (global log level + the report).
  All parameters are keyword-only (presets forward kwargs; no
  positional ambiguity between grid and modules).
- **The nine-step assembly (§6.2, normative)** — `__init__` runs:
  (1) fields: collect declarations/references, resolve patterns via
  `("declared_space", mesh)` resolvers, build the `FieldTable`
  (checks: `FieldCollisionError`, `MissingFieldError` with hints,
  lifecycle/role sanity, dot-free names, `require=` outcomes;
  empty-PROGNOSTIC compositions are **legal** — stage-only
  schedules, the mediator-as-model guard, CS-13);
  (2) parameters: build the `ParameterBindingTable` — **the stepper
  joins as provider of `fr.params.TIME_STEP`** (checks:
  one-provider, no-default names, provided-must-be-dynamic, the
  duplicate-module aliasing lint, explicit-wins/`USE_PROVIDED`);
  (3) **dispatch merge** — resolve `Module.dispatch` pattern keys
  through the step-1 resolvers, `grid.merge_overrides` exactly once
  (first model on the grid); same resolved key from two modules is
  `DispatchCollisionError`; `("declared_space", ...)` never
  module-mergeable; lazy-factory rows expose requirements unbound;
  (4) `bind(table)` in module order (role selections frozen to
  static tuples; merged registry visible; `TimeDependent` reads
  raise unless `at_time(0.0)`);
  (5) terms + stages: kind-ordered schedule, static checks
  (treatment vs stepper, implicit merges/`ImplicitCollisionError`,
  coverage lint with stage advances-claims, same-kind overlap lint,
  IMEX-RK × split-explicit error), compose the step body, collect
  `extra_halo`, build the re-materialization table;
  (6) composer dry run on halo tracers, per term/stage, attributed;
  (7) `grid.negotiate(state_spaces, tendency=composed_step,
  halo=extra_halo)` → `ReshardingReport`; `grid.freeze()` — **or, on
  an already-frozen grid, the verify path** (satisfiability-relaxed:
  new zero-demand ConstantSpace-family spaces are adopted; genuinely
  larger demands raise `GridFrozenError` with "assemble the most
  demanding model first");
  (8) allocate the carry — fields **born in the negotiated layout**
  (defaults evaluated with assembly-time parameters through the
  same code path as re-materialization), `stepper.init(template)`,
  float64 clock, `panic=(False, 0)`; the `ReshardingReport` walk
  applies only to pre-built leaves entering later;
  (9) emit `model.report` and compute `model.fingerprint`.
  **Ordering audit** (load-bearing): the dry run and the halo trace
  must see the registry *as merged* — merge is step 3, before bind,
  dry run, and negotiate. ICs remain post-assembly.
- **Amended (Phase-2 reconciliation, 2026-07-08) — step 7, combined
  halo semantics (REQUIRED)**: the negotiated `HaloSpec` is
  `trace(tendency)` **merge_max** the union of the exempted modules'
  declared `extra_halo` — trace ∨ extra_halo, never either-or (an
  exempted WENO module must not shrink the halo a traced advection
  chain demands). Reality note: the landed grid's `_negotiated_halo`
  treats `halo=` as an *exclusive override* that silently discards
  `tendency=` (and the `Grid.negotiate` docstring states yet a third
  precedence) — filed as a grid work item in
  [`../../../plans/done/phase2_grid_followups.md`](../../../plans/done/phase2_grid_followups.md).
- **Amended (2026-07-08) — step 8, the `ReshardingReport` walk**: the
  landed report is **layout-only** — a halo-only renegotiation reads
  `changed=False` — while halo changes alter *storage* shapes; the
  sanctioned re-home path for any pre-built leaves entering the
  negotiated layout is the true-shape re-pad (what
  `set_fields`/`set_state` do), never a bare `device_put`.
- **Lifecycle table (§6.5)**, condensed: post-assembly host
  operations at chunk boundaries only — `set_fields`, `set_state`,
  `set_aux`, `constrain`, `update_parameters`, `reset`,
  `snapshot`/`load_snapshot`, reads, `advance`/`run`. Canonical
  per-sweep-point order: **`update_parameters → reset →
  set_fields`** (V-C7; safe because re-materialization at the stale
  clock is masked by SELF_UPDATE-first, and rewarm is idempotent
  under the pair). Forbidden: attribute pokes
  (`ImmutableParameterError` teaching shim), frozen-grid ops,
  anything treedef-changing (module add/remove/enable/disable,
  Ramp-spec changes, treatment flips) → re-assemble. In-run: only
  the sanctioned traced dynamics. The three-operation matrix
  (02_rules): `set_aux` writes consented components;
  `update_parameters` skips them; `reset()` zeroes DIAGNOSTIC
  (including consented) but never AUXILIARY.
- **Panic-flag ledger**: cleared by `set_fields`/`set_state`/
  `load_snapshot`/`reset`; kept by `update_parameters` (changing ν
  after a NaN is not a resume path) and by `set_aux`
  [clarification].
- **Run-loop mechanics (§6.3)**: chunk boundaries = union of all
  trigger step sets ∪ snapshot steps ∪ last step, subdivided by
  `max_chunk` (auto ~256; documented meaning: host-sync
  granularity); per-step S5 `isfinite` reduction into
  `panic.(flag, it)` + chunk-boundary abort — **no `lax.cond` no-op
  wrapper** (per-step GPU sync tax; recorded as opt-in retrofit;
  per-step default *pending benchmark*, 2.4 item);
  `debug_nan=True` keeps a chunk-start carry copy and replays to
  the exact first-bad step with `chunk(1)`; Ctrl-C: the in-flight
  chunk completes, carry consistent, zero steps lost, no exception
  on first interrupt (`RunResult(INTERRUPTED)`); boundary sequence:
  sync → panic check → writer flush → progress → walltime check
  (predictive: elapsed + predicted chunk + snapshot margin);
  `run(jit=False)` eager mode works by construction (the halo
  tracer already requires a plain-Python step) and prints the
  per-term timing table — the old timer's true successor;
  `jax.named_scope("module/term")` stamped by the composer.
- **IO binding split (§6.4, amended V-C5)**: constructor `io=`
  holds standing config naming **unbound** diagnostic primitives,
  name-resolved at assembly; `run(outputs=...)` adds per-run streams
  accepting bound `model.diagnostics.*` spellings; the union binds
  (`OutputStream.bind()` — store creation, dry evaluation) at **run
  start**, never at assembly (presets never create files for models
  that never run). Streams dedupe by resolved path; two distinct
  streams on one path is `IOCollisionError`. **`Snapshots` is
  run-config only** — `io=` rejects it (one resume path, never
  two). Walltime policy lives in `fr.io.Snapshots(...,
  on_walltime=fr.io.resubmit())`; trigger lowering is a pure
  reusable planner function (CS-7, io_ops cluster).
- **`model.tendency` composition**: shares the composer's term
  closures and the binding table with the step body; jitted
  separately (its own cache entry keyed by the assembly record +
  filter token).
- **`fr.ops.Session` pointer**: multi-model driving, walltime
  prediction, panic polling, graceful Ctrl-C, and
  `s.advance(atm=10, ocn=1)` are the Session's (io_ops cluster);
  the Model-side commitments it builds on are exactly `advance`'s
  typed contract (CS-3), the no-hidden-sync invariant (CS-4), and
  `model.panicked`/`model.clock`.
- **Rejected on this surface** (pointers, never re-argued):
  enable/disable flags (d4_1 §5 — re-assembly + Ramp-to-zero
  replace them), `Model.save/load`/dill (02_rules
  no-pickled-models), `run_backward` (flip the `TIME_STEP` leaf),
  `Model.restore` classmethod (d4_3), in-trace `io_callback`
  outputs (§6.8 reconciliation 1), Model-as-pytree (d4_1 §4).

---

## 3. Assembly internals

### FieldTable (+ FieldRecord)

The Model-owned resolution table built at assembly step 1;
declaration order = component order. What `bind(table)` receives.

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen after step 1 (MOM6-style lock before halo tracing/negotiation) |
| Pytree | static host data (hashable; part of the AssemblyRecord) |
| Task | 2.2/2.3 |
| Design refs | 01 D1.4/D1.5; 04 §6.2 step 1; V-H2/V-N1/V-S3; CS-13 |

```python
"""FieldTable — name -> owner -> pattern -> space resolution."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldRecord:
    """One resolved declaration row."""

    name: str                          # component key; collision unit
    owner: int                         # module tuple index
    owner_type: str                    # qualified class name (report/fingerprint)
    pattern: SpacePattern              # as declared (fingerprint uses this)
    space: TensorProductSpace          # resolved bare interned space
    lifecycle: Lifecycle               # PROGNOSTIC | AUXILIARY | DIAGNOSTIC
    roles: frozenset[Role]
    host_writable: bool                # consent flag (AUX ∪ DIAG; CS-1/CS-2)
    metadata: fr.FieldMetadata         # grid-layer annotation that survives


class FieldTable:
    """Declaration-ordered resolution table; the bind-time query
    surface (role selections resolve once, into static tuples)."""

    def __init__(self, records: tuple[FieldRecord, ...]) -> None: ...  # 2.2

    @property
    def names(self) -> tuple[str, ...]: ...                    # 2.2
    def __getitem__(self, name: str) -> FieldRecord: ...       # 2.2
    def __contains__(self, name: str) -> bool: ...             # 2.2
    def __iter__(self) -> Iterator[FieldRecord]: ...           # 2.2
    def __len__(self) -> int: ...                              # 2.2

    @property
    def prognostic(self) -> tuple[str, ...]:                   # 2.2
        """PROGNOSTIC names, declaration order (may be empty — CS-13)."""
        ...

    @property
    def auxiliary(self) -> tuple[str, ...]: ...                # 2.2
    @property
    def diagnostic(self) -> tuple[str, ...]: ...               # 2.2

    @property
    def host_writable(self) -> tuple[str, ...]:                # 2.3
        """Consented names (listed in model.report; set_aux gate)."""
        ...

    def select(self, role: Role) -> tuple[str, ...]:           # 2.2
        """By-role query: open set, may match zero fields (a no-op,
        never an error — the reference/role distinction)."""
        ...

    def velocity(self) -> VelocitySelector:                    # 2.2
        """The Velocity-family selector (replaces positional
        ``self[:3]`` slices)."""
        ...

    def subset(                                                # 2.8
        self, state: fr.VectorField, lifecycle: Lifecycle,
    ) -> fr.VectorField:
        """Lifecycle-subset view of an assembled state (backs the
        model-mediated ``m.state.prognostic`` read — the adopted
        spelling, 2026-07-08; Tier-2 read-back)."""
        ...

    def fingerprint_token(self) -> tuple: ...                  # 2.6
```

Notes:

- Checks discharged while building: one owner per name
  (`FieldCollisionError` naming both modules), unsatisfied
  `FieldReference` (`MissingFieldError` with the reference hint),
  dot-free names (parameter names must contain a dot — the two
  tables stay visually disjoint, lint-enforced), lifecycle/role
  sanity (`ADVECTED`/`TRACER` strictly PROGNOSTIC; `Velocity` alone
  may sit on DIAGNOSTIC, V-H2), `require=` pattern outcomes logged.
- The `name → owner → pattern → space` table is the report's field
  section and the typo'd-coordinate mitigation (silent non-match is
  the dimension-generality feature; the log makes it visible).
- Declarations are transient assembly inputs; only AUX `default=`
  closures survive, in the re-materialization table — the table
  rows here keep everything *structural*.

### VelocitySelector

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen dataclass |
| Pytree | static (frozen into module bind state) |
| Task | 2.2 |
| Design refs | 01 D1.4 + V-H2 (DIAGNOSTIC velocities), V-N1 (transverse components) |

```python
@dataclass(frozen=True)
class VelocitySelector:
    """The Velocity family, resolved once at bind time."""

    components: tuple[str, ...]     # all Velocity-role fields, declaration order
                                    #   (PROGNOSTIC ∪ DIAGNOSTIC — reads span both)
    labels: tuple[tuple[str, str], ...]   # (name, component label) pairs
    directional: tuple[str, ...]    # labels ∈ grid.names: enter div/grad/flux directions
    transverse: tuple[str, ...]     # labels ∉ grid.names: slaved (∂ ≡ 0 by construction);
                                    #   full family members for friction/CFL/energy/advection
    prognostic: tuple[str, ...]     # the write-target subset (friction targets this;
                                    #   diagnosed w has no momentum equation)

    def label_of(self, name: str) -> str: ...                  # 2.2
```

Notes: labels are **opaque keys, never derived from spaces** (B-grid
`u`,`v` share one interned space; A-grid has no signal); on
tensor-product grids labels are coordinate names, and a label absent
from `grid.names` marks the transverse component (the 2D (x,z)
slice's collocated `v`). Assembly warns — never errors — on
label-vs-staggering contradictions. Role-driven write-targeting
intersects PROGNOSTIC automatically and is listed in `model.report`.
Multi-velocity futures (`group=` qualifier under split-explicit /
coupling) stay designed-for, not built — split-explicit `U, V` carry
no Velocity role at all.

### ParameterBindingTable (+ Params)

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen after step 2 |
| Pytree | static host data (part of the AssemblyRecord); `Params` is a thin frozen in-trace mapping |
| Task | 2.2/2.3 |
| Design refs | 01 D2.1/D2.4; 04 §6.2 step 2 + §6.5 (TIME_STEP provider); d2_1 |

```python
"""The static binding table: dotted name -> (slot, attr)."""


class ParameterBinding(NamedTuple):
    name: str                          # canonical dotted name
    slot: int | Literal["stepper"]     # module tuple index, or the stepper
    attr: str                          # dynamic-leaf attribute on the provider
    declaration: ParameterDeclaration  # units/doc (report; ParameterView.info)


class ParameterBindingTable:
    """Assembly-frozen resolution of provides/requires."""

    def __init__(self, entries: tuple[ParameterBinding, ...]) -> None: ...

    def __getitem__(self, name: str | ParamName) -> ParameterBinding: ...
    def __contains__(self, name: str | ParamName) -> bool: ...
    def __iter__(self) -> Iterator[ParameterBinding]: ...

    def eval_params(                                           # 2.2
        self,
        modules: tuple[fr.Module, ...],
        stepper: TimeStepper,
        t: jax.Array,
    ) -> Params:
        """The in-trace read, composed once at assembly: fresh live
        leaves per **stage time** (`fr.resolve_at` on TimeDependent
        values). Called at P0 of every substage; feeds
        ``StepContext.params``."""
        ...

    def host_view(self, model: Model) -> ParameterView: ...    # 2.3

    def fingerprint_token(self) -> tuple: ...                  # 2.6


class Params(Mapping[str, jax.Array]):
    """Thin frozen mapping delivered to every traced entry point;
    unknown-name lookups are caught by the assembly dry run."""
```

Notes: consumers never hold provider objects or frozen values (the
jax aliasing rule — pytrees are trees, not DAGs); an assembly lint
errors on any jaxified module appearing twice in the carry.
Identity-defaulted references (`scaling.rossby` → 1.0) bind to a
constant entry and are listed in the report ("identity defaults in
effect"); registry-marked no-default names raise
`MissingParameterError` with the registry hint. The `TIME_STEP` row
is what makes dt sweeps and backward legs go through
`update_parameters` with no private stepper pokes.

### RematerializationTable

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen at step 5 |
| Pytree | static host data (part of the AssemblyRecord) |
| Task | 2.3 |
| Design refs | 04 §6.5 (`update_parameters` step 4); 02_rules (defaults-one-path, host-writable exemption); CS-2; d4_4 §1.1 |

```python
@dataclass(frozen=True)
class RematerializationEntry:
    field: str                       # AUX component name
    owner: int                       # module slot
    default: Callable | float | None # the RETAINED declaration default;
                                     #   may be an UNBOUND owner method
    space: TensorProductSpace
    host_writable: bool              # exempt from re-runs (CS-2)


class RematerializationTable:
    """AUX declarations with callable defaults, retained in the
    static assembly record (the D1.1 softening)."""

    entries: tuple[RematerializationEntry, ...]

    def materialize(                                           # 2.3
        self,
        modules: tuple[fr.Module, ...],
        grid: fr.grid.Grid,
        *,
        owners: frozenset[int] | None = None,
    ) -> dict[str, fr.ScalarField]:
        """Run the owners' defaults with their **current** leaves —
        the ONE code path shared by allocation (step 8, all owners)
        and ``update_parameters`` (changed owners only,
        host-writable entries skipped). Defaults read only the
        owner's own leaves (02_rules)."""
        ...
```

Notes: invalidation is per-owner and conservative (all
non-host-writable AUX declarations of any module whose leaves
changed). Ramp-fed AUX allocation values are placeholders —
`self_update` owns them in-run (documented). Cross-module-derived
AUX values are `self_update` territory or a re-assembly, never a
default.

### TendencyComposer

Owns everything about turning collected terms + stages into the
step body: attribution, validation, deterministic accumulation.

| Aspect | Value |
|--------|-------|
| Kind | concrete (assembly-internal; not user-facing) |
| Pytree | host object; its *product* (the composed step) is a pure function over the carry |
| Task | 2.3 (compose + dry run); the schedule kinds/steppers it drives land with 2.5 |
| Design refs | 03 §5.1–5.2/§5.5 (kind order, write gates, per-treatment sums); 04 §6.2 steps 5–6; V-N2 (extra_halo second mode); d1_5 §4 |

```python
"""TendencyComposer — schedule composition, dry run, attribution."""


class TendencyComposer:

    def __init__(                                              # 2.3
        self,
        *,
        field_table: FieldTable,
        modules: tuple[fr.Module, ...],
        terms: tuple[TendencyTerm, ...],       # collected post-bind
        stages: tuple[Stage, ...],
        time_stepper: TimeStepper,
        binding_table: ParameterBindingTable,
        term_filter: TermPredicate | None = None,   # variant mechanics (2.8)
    ) -> None:
        """Collect, order, and statically check; raises the
        assembly-error family (treatment-vs-stepper,
        ImplicitCollisionError, coverage lint, same-kind overlap
        lint, IMEX-RK × split-explicit)."""
        ...

    @property
    def schedule(self) -> tuple[ScheduleEntry, ...]:           # 2.3
        """Kind-ordered (§5.2): per substage SELF_UPDATE -> DIAGNOSE
        -> terms -> ADVANCE(s) -> CONSTRAINT; epilogue S5 NaN seam
        -> S6 DIAGNOSTIC. Within a kind: (order, module index,
        declaration index). Never list-positioned. (Entry internals:
        time_steppers.md.)"""
        ...

    def compose(self) -> Callable[[ModelState], ModelState]:   # 2.3
        """The step body — plain Python over fields (the halo
        tracer and ``run(jit=False)`` require it), stamped with
        ``jax.named_scope("module/term")`` per hook."""
        ...

    def dry_run(self) -> None:                                 # 2.3
        """Assembly step 6, on halo tracers, per term/stage,
        attributed: write gates per kind (terms: add-only, keys
        PROGNOSTIC-gated; stages: replace on the declared subset),
        contribution-key validation (typos fail at assembly),
        ``advances`` cross-check, result spaces, read/write ordering
        sanity. Terms of ``extra_halo``-declaring modules are
        exempted from the halo trace and validated in a **second
        dry-run mode over real zero-valued fields** (V-N2)."""
        ...

    def tendency_fn(                                           # 2.8
        self,
        *,
        filter: TermPredicate | None = None,
        constraints: bool = True,
    ) -> Callable[..., fr.VectorField]:
        """The read-only composed tendency behind ``model.tendency``
        (SELF_UPDATE + DIAGNOSE first; implicit via forward apply)."""
        ...

    def tendency_template(self) -> fr.VectorField:             # 2.3
        """Zero PROGNOSTIC vector for ``stepper.init``."""
        ...
```

Notes:

- **Accumulation order is deterministic and attributed**: module
  tuple order, then declaration order within a module; the composer
  accumulates contribution dicts via `VectorField.add` (terms only
  ever add; overwrites are stages, applied per the §5.5 write-gate
  table). Per-treatment tendency sums are kept separate and handed
  to post-tendency stages through `StepContext` (the partition IMEX
  and the barotropic slow forcing consume).
- **Amended (Phase-2 reconciliation, 2026-07-08) — the halo-trace
  entry must be NAME-KEYED**: the dry run's tracer state has to carry
  the declared component names (attribution, per-field negotiation,
  and the FieldTable cross-checks all key on them), but the landed
  `trace_halo` builds anonymous positional components (`c0`, `c1`,
  ...). The model therefore either calls `VectorTracer` with a name
  mapping directly or demands the `trace_halo` Mapping extension —
  grid work item in the same followups file
  ([`../../../plans/done/phase2_grid_followups.md`](../../../plans/done/phase2_grid_followups.md)).
- **Attribution duties**: trace-time exceptions from a term/stage
  are wrapped as `TermEvaluationError` carrying the "Module/term"
  key (in particular `SpaceMismatchError` from a wrong-staggering
  contribution is re-raised with the module/term name — d1_5 risk 2,
  else error quality regresses vs the old in-module `+=`).
- **Variant mechanics**: `term_filter` acts on collected terms only
  (declarations/stages never filtered); its canonical token joins
  the fingerprint; the coverage lint downgrades error → info under
  any filter; an empty implicit partition is graceful (the solve
  loop is simply not emitted — CNAB2 degenerates to textbook AB2).
- `eval_params` is the binding table's; the composer calls it at P0
  of every substage at stage time (a ramped scalar and a ramped N²
  profile see the same time).

### AssemblyRecord

The hashable static that keys the shared `step_chunk` entry — the
jit-cache obligation made a class.

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen |
| Pytree | static argument to `step_chunk` (hashable; `__eq__`/`__hash__` **structural**, not identity) |
| Task | 2.4 |
| Design refs | 04 §6.3 ("identical re-assemblies share the jit cache"); d4_4 §3; 07 residual (shared-runner lint) |

```python
@dataclass(frozen=True)
class AssemblyRecord:
    """Everything static the traced step depends on."""

    grid: fr.grid.Grid                  # identity-hashed static (same grid
                                        #   object required for cache sharing)
    field_table: FieldTable
    binding_table: ParameterBindingTable
    remat_table: RematerializationTable
    schedule: tuple[ScheduleEntry, ...]
    stepper_statics: tuple              # order/tableau/eps — dt excluded (leaf)
    state_type: type[fr.VectorField]
    extra_halo: HaloSpec | None
    term_filter_token: str | None       # variant provenance
    name: str | None = None             # excluded from __eq__/__hash__

    def __hash__(self) -> int:                                 # 2.4
        """Structural: grid identity + the component tables'
        fingerprint tokens — identical re-assemblies compare equal,
        so sweep members share the compiled chunk."""
        ...

    def __eq__(self, other: object) -> bool: ...               # 2.4

    def step_fn(self) -> Callable[[ModelState], ModelState]:   # 2.4
        """The composed step, derived and memoized keyed by self
        (the composed callable itself never enters a jit closure —
        a per-assembly closure would silently defeat the cache)."""
        ...

    def fingerprint(self) -> Fingerprint: ...                  # 2.6
```

Notes: an **assembly lint errors on unhashable statics**, and a
**compilation-count regression test** asserts one compile across a
re-assembly sweep (both 07 residual obligations, owned by 2.4).
Different module tuples are different records — module-*type* sweeps
recompile by design (and use a fresh grid per composition, V-C2).

### AssemblyReport (+ Fingerprint)

| Aspect | Value |
|--------|-------|
| Kind | concrete host objects; `Fingerprint` frozen dataclass |
| Pytree | none (host) |
| Task | 2.3 (report); 2.6 (fingerprint consumers: snapshot manifest) |
| Design refs | 04 §6.7/§6.2 step 9; 02_rules (fingerprint scope, refined scope); d4_1 §6 (the mock is the format reference) |

```python
class AssemblyReport:
    """model.report — printable without device sync; logged at INFO."""

    # Section structure (the d4_1 mock is the reference), in order:
    #   1. header            grid / stepper / modules / name
    #   2. fields            name -> owner -> pattern -> space -> lifecycle
    #                        -> roles -> default provenance; matched tags,
    #                        require= outcomes, host-writable listing,
    #                        Velocity write-target intersections (V-H2)
    #   3. parameters        the binding table; identity-defaults in effect
    #   4. dispatch          merged overrides (or verify-path outcome)
    #   5. schedule          kind-ordered, with WHY self-updates are
    #                        scheduled and which implicit merges happened
    #   6. halo/layout       negotiation results; which chain set the max
    #   7. lint              warnings (aggregated untransported-ADVECTED
    #                        line, label-vs-staggering, filter downgrades)
    #   8. run-start addendum  defaults-vs-user-initialized provenance
    #                        table, appended at first run() (D1.1's
    #                        logging promise)

    @property
    def header(self) -> str: ...                               # 2.3
    def __str__(self) -> str: ...                              # 2.3
    def section(self, name: str) -> str: ...                   # 2.3


@dataclass(frozen=True)
class Fingerprint:
    """The restart fingerprint: digest + diffable source record."""

    digest: str
    source: tuple[tuple[str, str], ...]   # (key, token) rows, human-diffable

    def diff(self, other: Fingerprint) -> str:                 # 2.6
        """Human-readable structural diff — what
        SnapshotMismatchError prints ("stepper statics differ:
        cnab2 -> sbdf2"); never silent reuse."""
        ...
```

Fingerprint scope (02_rules, restated): hashes **structure, never
leaves** — field declarations (names / declared `SpacePattern`s /
bare interned spaces / lifecycles; never `Layout` or device
topology), the module tuple (types + order), per-term treatments,
stepper statics (order/eps/tableau) and module-owned ADVANCE-stage
integrator statics, parameter *specs* (a Ramp's shape is structure,
endpoints are leaves), and the variant filter token + updates specs.
IC/state differences are deliberately invisible (a restart
overwrites them). The snapshot **manifest** additionally records
provided-parameter *values* (at least `TIME_STEP`) and a
machine-readable header without leaf IO (CS-10; io_ops cluster).

---

## 4. State — the vocabulary-class contract

Extends the grid notes' `State` section
([`../../grid/classes/fields.md`](../../grid/classes/fields.md)): what a model
package's `State` subclass is and is not. A vocabulary class is
**not a structural entity** — no declared fields, no assembly logic,
inherited constructor; it bundles curated accessors and
parameter-free diagnostics for the names canonical in that model
family, whether or not the current module list produced them.

| Aspect | Value |
|--------|-------|
| Kind | concrete per model package, subclass of `fr.VectorField`; one registered class, never synthesized |
| Pytree | inherited from `VectorField` (components dynamic, names in the treedef) |
| Task | 2.7 (nonhydro/shallowwater ports); `_component` on `VectorField` is a fields.md amendment owed |
| Design refs | 01 D1.5; d1_5 §1; 08 §10.3 (`prognostic` read-back); CS-14 |

```python
"""fridom.nonhydro.state — the nonhydro vocabulary class."""
from __future__ import annotations

import fridom.framework2 as fr


class State(fr.VectorField):
    """Vocabulary class: curated accessors + parameter-free
    diagnostics. Supplied by the core module (state_type),
    Model(state_type=...) as the fallback spelling."""

    @property
    def u(self) -> fr.ScalarField:                             # 2.7
        """Zonal velocity (declared by a dynamical-core module)."""
        return self._component(
            "u", hint="declared by a dynamical-core module, "
                      "e.g. nh.DynamicalCore")

    # v, w analogous; every canonical name gets one property with a
    # one-line curated hint. No setters (ImmutableStateError). User
    # tracers and module-private fields get NO properties: z["dye"].

    @property
    def b(self) -> fr.ScalarField:                             # 2.7
        """Buoyancy (present when a stratification module is
        assembled)."""
        return self._component(
            "b", hint="add a stratification module, e.g. "
                      "nh.modules.ConstantStratification")

    # ``prognostic`` as a *state-side* property is superseded
    # (2026-07-08): the FieldTable-as-aux backing it required is
    # rejected — see the contract bullet below. The Tier-2 read-back
    # of 08 §10.3 is model-mediated sugar over ``FieldTable.subset``;
    # plain states carry no table.

    # Parameter-free diagnostics only (rel_vort_z here; ekin carries
    # dsqr and is a model.diagnostics function instead), written in
    # the field algebra — explicit .to conversions, dispatched
    # products, integrate.
```

The `VectorField._component` helper (fields.md amendment owed —
fields.md already names `MissingComponentError` for `add`; this adds
the property-side raise path):

```python
def _component(self, name: str, *, hint: str = "") -> fr.ScalarField:
    """Return the named component or raise MissingComponentError:
    name what's missing, append the curated hint, list what's
    present ('Components present: u, v, w, dye.'). Host-side
    difflib close-match suggestions in __getitem__ are optional
    (the error path never traces)."""
```

Contract, restated normatively:

- **`state["b"]` is primary and the only form module code uses**
  (modules stay generic over State classes; the halo trace stays
  State-class-independent). No dynamic `__getattr__` fallback
  (masks `AttributeError`s inside real properties; typing opacity);
  no generated accessors (per-instance class synthesis breaks
  pytree registration).
- **No mutation surface**: property setters and `__setitem__` are
  raising teaching shims (`ImmutableStateError`: "use
  ``z = z.replace(u=...)`` / ``z = z.add(u=...)``"); the raising
  `.data` setter is fields.md's. Ports go fully functional
  immediately.
- **Treedef stability is part of every model port**: required test
  `tree_structure(step(state)) == tree_structure(state)`
  (componentwise arithmetic preserves metadata precisely for this).
- **The `prognostic` backing — RESOLVED BY REJECTION (2026-07-08)**:
  the earlier spec deviation attaching the (static, identity-hashed)
  `FieldTable` as aux data on model-built states is **dropped**, on
  two independent grounds from the reconciliation audit. First,
  Phase-1 `VectorField` rebuilds via `type(self)(dict(...))` on
  every functional op (`replace`, `add`, `map`, componentwise
  arithmetic), which would silently drop the attached table — a
  treedef break of exactly the class the metadata amendment
  (fields.md, 2026-07-08) just closed. Second, a state-attached
  table makes treedefs differ across models sharing one grid,
  weakening the one-grid-many-models jit-cache story for *every*
  state consumer other than `step_chunk`. Adopted spelling:
  `FieldTable.subset` (already specified above) — the
  `m.state.prognostic`-style read becomes **model-mediated sugar**
  (the model, which owns the table, does the subsetting); plain
  states carry no table. A lifecycle-subset read on a bare state
  gets a taught error pointing at the model-mediated spelling.
- **CS-14 guard**: nothing in the transform algebra may
  `isinstance(state, State)` — signatures are opaque compared
  values; the vocabulary class is sugar, not a type gate. A bare
  generic `Model` (no state_type provider) uses plain `VectorField`
  and loses nothing but sugar; >1 provider is an assembly error.

---

## 5. Run-loop result types

| Aspect | Value |
|--------|-------|
| Kind | frozen dataclasses + one enum + one exception |
| Pytree | none (host values) |
| Task | 2.4 |
| Design refs | 04 §6.3 coupling-round amendments; 09 §11.2 decision 3a; CS-3..6 |

```python
"""results.py — typed run-loop returns (CS-3)."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class RunStatus(Enum):
    """Shared status vocabulary (RunResult, Session aggregation)."""

    COMPLETED = "completed"
    NAN_ABORT = "nan_abort"
    WALLTIME = "walltime"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True)
class AdvanceResult:
    """advance()'s minimal frozen return — everything else stays
    RunResult's."""

    steps_done: int
    panicked: bool                  # panic flag as read at the last boundary
    panic_it: int | None            # first-failure iteration (S5 record)
    wall_seconds: float


@dataclass(frozen=True)
class RunResult:
    """run()'s plain, aggregatable return; scripts and SLURM
    drivers branch on .status — run() never exits the process."""

    status: RunStatus
    steps_done: int
    final_it: int
    final_time: float
    compile_seconds: float          # AOT lower().compile() accounting
    run_seconds: float
    steps_per_second: float


class PanicError(RuntimeError):
    """Typed abort raised by advance() at the chunk boundary where
    the panic flag reads true, and on entry with a panicked carry
    (with resume guidance: set_fields/set_state/load_snapshot/
    reset clear the flag). Carries attribution for multi-model
    sessions; carries are left in memory for autopsy."""

    model_name: str | None
    first_bad_it: int | None
    partial: AdvanceResult | None   # steps completed before the abort


class PendingAdvance:
    """RESERVED (CS-5): the advance(sync=False) handle. Name and
    slot fixed; semantics leave-open — not built."""
```

Notes: `advance` raises; `run` catches and converts to
`RunResult(NAN_ABORT)` unless `raise_on_nan=True` (notebooks want
the carry; context managers need exceptions — both contracts hold).
The host-side NaN report at abort includes first-failure it/time and
per-component non-finite counts (chunk-end caveat documented).

---

## 6. Preset-factory rules

No class — a normative checklist (D1.3 commitment 2, sharpened by
d4_1). A preset (`nh.Model(...)`, per-package factory function):

- **may**: build the module tuple, forward kwargs, pick the default
  stepper (nh: `AdamBashforth(dt=..., order=3)` at cutover, V-N3;
  low-storage RK3 documented as the recommendation) and default
  `io=`, set `name=`;
- **may not**: subclass `Model`, hold parameters or fields, register
  grid-level resolvers, create files (binding is run-start), or
  mutate the model post-construction;
- **normative test (the treedef-identity test)**: preset and
  explicit assembly produce **identical carry treedefs** — asserted
  in CI for every shipped preset;
- when a composition genuinely lacks a natural owner for a shared
  field, the preset adds a minimal declaring module
  (`fr.modules.Tracer("T")` one-liners), never a reference that
  auto-creates.

---

## 7. The model-layer error registry

All concrete model-layer errors in one table (module
`framework2/model/errors.py` unless noted). Assembly-time errors
subclass a common `AssemblyError`; teaching shims subclass
`TypeError`-flavored bases so they read as API misuse. Errors listed
as *pointer* are owned (defined) by another cluster and only raised
or re-raised here.

| Error | Raise site(s) | Contract |
|---|---|---|
| `FieldCollisionError` | assembly step 1 | two declarations of one field name; names **both** modules; no silent merge |
| `MissingFieldError` | assembly step 1 | unsatisfied `FieldReference`; carries the reference hint |
| `MissingComponentError` | `VectorField._component` (vocabulary properties), `VectorField.add`/`__getitem__` unknown key (defined with `VectorField`, grid pkg; re-exported here) | name the missing component, append the curated hint, list present components |
| `MissingParameterError` | assembly step 2 (REQUIRED reference unsatisfied); host `model.parameters[...]`; bound diagnostic **call** (lazy) | attributed to the requiring module/diagnostic; registry hint + provided-parameter list |
| `ParameterCollisionError` | assembly step 2 | two providers of one dotted name; names both modules |
| `DispatchCollisionError` | assembly step 3 | same resolved dispatch key from two modules (module order never silently selects an operator) |
| `ImplicitCollisionError` | assembly step 5 | more than one non-mergeable custom implicit operator per field (mergeable families sum instead) |
| `TimeDependentParameterError` | bind-time parameter reads without `at_time(0.0)`; `from_model` eigenmode/transform constructors without `at_time=` | error text teaches the split: grid factor at bind, parameter factor in-step |
| `TermEvaluationError` | composer trace-time wrapper (and dry run) | wraps term/stage exceptions with the "Module/term" attribution key; space mismatches re-raised with attribution |
| `ImmutableStateError` | `State`/`VectorField` property setters, `__setitem__` (teaching shim; the `.data` setter twin is fields.md's) | guidance: `replace`/`add` |
| `ImmutableParameterError` | post-assembly module attribute pokes (teaching shim) | guidance: `update_parameters` (leaves) or re-assemble (structure) |
| `PanicError` | `advance()` abort boundary; `advance`/`run` entry on a panicked carry | model `name=`, first-failure iteration, partial result; carries left for autopsy |
| `RunTargetError` [named here] | `run()` planning | `(end − t0)·dt > 0` precondition violated; also inconsistent steps/runlen/end_time combinations |
| `IOCollisionError` | run-start stream binding | two distinct streams resolving to one path (streams dedupe by resolved path); also `io=` handed a `Snapshots` (run-config only) |
| `SnapshotMismatchError` | `load_snapshot` | prints `Fingerprint.diff` ("stepper statics differ: cnab2 → sbdf2"); dt **sign** mismatch errors here, magnitude warns |
| `GridFrozenError` | *pointer* (grid pkg): assembly step 7 verify path | genuinely larger demands on a frozen grid; "assemble the most demanding model first"; satisfiability relaxation applies first |
| `SignatureMismatchError` | *pointer* (transforms cluster): compose/call-time signature checks | composition-tree path + componentwise diff (owned by [`transforms.md`](transforms.md)) |
| `SpaceMismatchError`, `GridMismatchError` | *pointer* (grid pkg) | reached through field algebra inside terms; composer re-wraps with attribution |

Lint-level (warn, never raise): untransported-`ADVECTED` (one
aggregated line), Velocity label-vs-staggering contradictions,
`P @ P` on idempotent transforms (transforms cluster), coverage-lint
info downgrade under a variant filter.

---

## Open questions

Genuinely unresolved residuals only (owners as parked in
[`../07_open_threads.md`](../07_open_threads.md)); decided questions
are not reopened.

1. ~~**`state_type`-under-jaxify mechanics**~~ — **closed by
   implementation**: the core module publishes its State class as a
   plain `Module.state_type` class attribute, the assembly record
   lints it as a hashable static, and `Model(state_type=...)` is the
   override. No jaxify interaction survived.
2. ~~**Donation vs live `model.state` views**~~ — **closed by
   implementation: copy-on-read**. The carry's buffers are donated to
   the next `advance()`, so a live view would reference deleted
   buffers; `model.state` copies its leaves (one device-to-device pass
   per read). The cost benchmark is still nominally owed.
3. **Shared-jitted-runner discipline** — the assembly lint for
   unhashable statics and the compilation-count regression test are
   implementation obligations (2.4), not yet designs; the
   `AssemblyRecord.step_fn` memoization spelling is theirs to fix.
4. **NaN-check cadence** — per-step S5 default *pending benchmark*
   (2.4 item); a cadence knob (every-k mask or chunk-boundary-only)
   is sanctioned if profiling demands.
5. **`State.prognostic` backing — RESOLVED BY REJECTION
   (2026-07-08)**: the FieldTable-as-static-aux proposal is dropped.
   Phase-1 `VectorField` rebuilds via `type(self)(dict(...))` on
   every functional op and would silently drop the attached table
   (treedef break), and a state-attached table makes treedefs differ
   across models on one grid — weakening the one-grid-many-models
   jit-cache story for every state consumer other than `step_chunk`.
   Adopted: the already-specified `FieldTable.subset(state,
   lifecycle)` spelling; `m.state.prognostic`-style reads are
   model-mediated sugar (the model owns the table and does the
   subsetting); plain states carry no table. The `_component`
   amendment for fields.md remains owed.
6. **Diagnostics-namespace population channel** — specified here as
   the core-module contribution paired with `state_type` (the D1.3
   commitment-4 channel, precedented by the default-projector hook);
   the exact `Module` hook spelling is owed to
   [`module.md`](module.md).
7. **`set_aux` and the panic flag** — specified here as
   non-clearing (not a resume path); confirm when the coupler
   protocol page (CS-18, doc-level) is written at 3.2.
8. **`RunTargetError` naming** — the design mandates a named error
   for the run-target precondition; the name is coined here.
9. **Post-assembly writer attach + capture streams** — 2.6 API
   (io_ops cluster); the Model-side binding split above must remain
   sufficient for it.
10. **Multi-process walltime/interrupt consensus** — rank-0
    broadcast at chunk boundaries; 3.2/3.3 (Session-side; the Model
    surface is untouched).
