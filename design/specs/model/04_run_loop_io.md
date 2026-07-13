---
status: normative
date: 2026-07-13
---

# Model layer redesign — Composition, the run loop, and IO

Part of the model redesign notes; see
[`00_overview.md`](00_overview.md) for the document map. Concept
frame: decision D4 in [`01_concepts.md`](01_concepts.md); research
reports in [`research/`](../../research/README.md) (d4_1–d4_4).

Status: **resolved (signed off 2026-07-08).** This file is the
full D4 design (ROADMAP 2.3 + 2.4, and the 2.6 seams); the D4
section of `01_concepts.md` is the summary. Sign-off note on the
NaN check: the per-step `isfinite` write was the default **pending
benchmark** — if it costs measurably on real models, a cadence knob
(every k steps via a carried counter mask, or chunk-boundary-only
mode) is sanctioned; the trade is documented in §6.3.
**Benchmark landed (2026-07-12, A100, nonhydro 256³):** the
per-step reduction was a measurable share of the GPU step, and the
owner directed the sanctioned chunk-boundary-only mode; S5 now runs
once per chunk on the scan's final state (non-finite values
propagate, so mid-chunk blow-ups are still caught at the boundary;
`panic.it` records the detecting boundary, and the exact first-bad
step remains `debug_nan`/`replay_nan` territory). No cadence knob
was added — chunk-boundary-only is the single behavior.

---

## 6.1 The Model object

([d4_1](../../research/d4_1_assembly_model.md)) The constructor:

```python
model = fr.Model(
    grid=grid,
    modules=(core, coriolis, stratification, advection, closure),
    time_stepper=fr.time_steppers.LowStorageRK3(dt=60.0),  # REQUIRED, no default
    io=(),                # standing output config (successor of diagnostics=)
    state_type=None,      # fallback/override for the core-supplied State class
    name=None,            # report/log attribution (two models, one process)
)
```

`time_stepper` has no default (no physics-free dt exists; presets
supply the package default — **pinned at validation sign-off
(V-N3): the nh preset defaults to `AdamBashforth(order=3)` at
cutover for parity** — note eps is order-2-only, 02_rules — **with
low-storage RK3 as the documented recommendation**). Rejected kwargs, each replaced:
`halo=` (traced), `progress_bar=`/`nan_checker=`/`restart_module=`
(run-time policy), verbosity (global log level + the report).

**`fr.Model` is a host-side driver, not a pytree**: nothing traces
through it (the traced unit is the step over the carry); a
Model-as-pytree would create a second flatten path to the carry's
leaves — the D2 aliasing bug by construction. It holds the assembly
artifacts (FieldTable, binding table, schedule, composed step,
report, io bindings) plus the single mutable slot `_carry`.
Lifecycle methods are **mutating spellings over pure carry
transformers** (new carry built functionally, reference swapped);
`model.state` is a read-only property.

**Preset rules** (D1.3 sharpened): a preset builds the module tuple,
forwards kwargs, picks default stepper/io, sets name — and nothing
else. Normative test: preset and explicit assembly produce
**identical carry treedefs**.

**D5 amendments to the Model surface (applied on D5 sign-off)**:
- **`model.tendency(state, *, t=None, filter=None,
  constraints=True) -> State`** — a host-callable, jitted,
  read-only wrapper over the composed tendency (**SELF_UPDATE and
  DIAGNOSE stages run first, at `t`** — amended V-H8: the
  hydrostatic tendency is meaningless without recomputed
  p_hyd/w; the result reflects recomputed diagnostics, not the
  input's; implicit terms via their forward apply; CONSTRAINT
  stages applied to the result when asked; never advances the
  carry). Consumers: per-term budget
  diagnostics (`filter=fr.terms.named(...)`), term unit tests
  against analytic solutions, linear-stability matvecs, the future
  `TangentPropagator` (`jax.jvp` of exactly this function).
- **`model.blank_state()`** (PROGNOSTIC subset at declared
  defaults, born sharded) and **`model.state_space(name)`** — the
  State factory every IC recipe and transform builds on (used but
  never named by D1.1). **Not built**: the ported IC recipes compose
  `grid.create_field(space, ...)` with spaces taken from
  `model.field_table[name].space` or from the eigenmode surface. The
  two methods are sugar over the field table, and remain the intended
  spelling — build them or strike them
  ([`07_open_threads.md`](07_open_threads.md) §9.1 item 1).
- **`model.variant(term_filter=..., updates=..., name=...)`** — the
  derived-model constructor (full semantics in
  [`08_state_transforms.md`](08_state_transforms.md) §10.4); its
  `updates=` is assembly-time and may change value specs
  (scalar → Ramp, `TIME_STEP` sign), unlike post-assembly
  `update_parameters`.

## 6.2 The assembly pipeline (normative)

([d4_1](../../research/d4_1_assembly_model.md)) `fr.Model.__init__` **is**
assembly — no `setup()`, one pass, a pure deterministic function of
its inputs (same inputs → identical treedef):

1. **Fields**: collect declarations/references; resolve
   `SpacePattern`s via the grid-level `("declared_space", mesh)`
   resolvers → interned `state_spaces`; build the `FieldTable`;
   record the name→owner→pattern→space table. Checks: collisions,
   unsatisfied references (hints), lifecycle/role sanity, dot-free
   names. The optional pattern `require=("x",)` kwarg is adopted
   (typo mitigation, with the table).
2. **Parameters**: collect declarations/references; build the
   binding table `{name: (slot, attr)}` — **the stepper joins it as
   provider of `fr.params.TIME_STEP`** (its dt leaf). Checks:
   one-provider, no-default names, provided-must-be-dynamic, the
   duplicate-module aliasing lint, explicit-wins.
3. **Dispatch merge** — *the resolved grid-notes call site*:
   `Module.dispatch` is a constructor-frozen mapping keyed by `kind`
   or `(kind, SpacePattern)`; the model resolves pattern keys via
   the step-1 resolvers and calls `grid.merge_overrides` **exactly
   once**; same resolved key from two modules is an error (module
   order never silently selects an operator);
   `("declared_space", ...)` is never module-mergeable; values may
   be lazy factories (the transform-row mechanism) for operators
   binding to negotiated context. No `Module.setup()` exists.
4. **`bind(table)`** (module order): role selections frozen to
   static tuples; grid-factor precomputes; the merged registry is
   visible; time-dependent reads raise unless `at_time(0.0)`.
5. **Terms + stages**: collect (post-bind); build the kind-ordered
   schedule; static checks (treatment vs stepper, implicit
   merges/collisions, coverage lint with stage advances-claims,
   same-kind overlap lint, IMEX-RK × split-explicit error); compose
   the step body; collect `extra_halo`; build the
   **re-materialization table** (AUX declarations with callable
   defaults — retained in the static assembly record, §6.5).
6. **Dry run** on halo tracers, per term/stage, attributed
   (write gates, keys, advances, spaces, ordering sanity).
7. **`grid.negotiate(state_spaces, tendency=composed_step,
   halo=extra_halo)`** → `ReshardingReport`; **`grid.freeze()`** —
   or, on an already-frozen grid, the **verify path** (§6.5).
8. **Allocate the carry** `(state, modules, stepper_state, clock,
   panicked)` — fields **born in the negotiated layout** (defaults
   evaluated with assembly-time parameters), `stepper.init`, float64
   clock, `panicked=False`. The `ReshardingReport` walk applies only
   to pre-built leaves entering later (`set_fields`/`set_state`/
   `load_snapshot` re-home per `decomposition.sharding(space)`).
9. **Emit `model.report`** (§6.7) and compute **`model.fingerprint`**
   (the 02_rules restart rule’s source record).

**Ordering verification** (fixes the draft pipeline): the dry run
and the halo trace must see the registry **as merged** (the
decomposition notes are explicit; an override's wider stencil must
be what the tracer intercepts, and dry-run space validation must see
the operator that will actually run) — hence merge at step 3, before
bind, the dry run, and negotiate. ICs remain the post-assembly step,
plus the opt-in initial projection of non-divergence-free ICs
(optional because project-the-state self-corrects within one
substage). **Shipped spelling**: `model.constrain(state, *, t=None) ->
State` — a pure, host-callable application of the CONSTRAINT stages to
a state, rather than the in-place `apply_constraints()` this section
originally named; the caller feeds the result back through
`set_state`. The pure form is what `model.tendency(...,
constraints=True)` and the transforms reuse.

## 6.3 The run loop

([d4_2](../../research/d4_2_run_loop.md)) **`advance(steps)` is the public
IO-free primitive; `run()` is sugar** over it plus IO — the hook an
outer coupler loop (Phase 3) interleaves.

**Coupling-round amendments (signed 2026-07-08, design-for §11.2 +
CS-3..6):** `advance()` returns a minimal frozen **`AdvanceResult`**
(steps done, panic `(flag, it)`, wall seconds — everything else
stays `RunResult`'s) and raises a typed **`PanicError`** (carrying
the model `name=`) at the abort boundary; `model.panicked` is a
cheap host-readable property; the **no-hidden-sync invariant**
holds (the panic read is `advance`'s only host synchronization; the
committed carry may hold pending arrays) and
`advance(steps, *, sync=True)` reserves the `sync=False` →
`PendingAdvance` spelling. **The ops packaging is the hybrid**:
protocol-level utilities (walltime predictor, progress renderer,
stream binding — normative) composed by **`fr.ops.Session`**, a
context manager: `__enter__` plans triggers/binds outputs/
resume-checks snapshots (enforcing the coupler protocol's
window-boundary and cross-model-clock rules mechanically);
`s.active` = predictive walltime + panic polling + graceful first
Ctrl-C; **`s.advance(atm=10, ocn=1)`** is the sanctioned
multi-model spelling (dispatch-then-sync built in); `__exit__`
guarantees wait-for-in-flight → writer flush → triggered snapshot →
`on_walltime` on every exit path (on `PanicError`: flush, no
auto-snapshot, carries left for autopsy). **`run()` is reimplemented
as a single-model Session loop** — the bitwise test extends
(`run() ≡ user-written Session loop`); notebooks keep `run()` and
never see the `with`. Session state is config + transient counters,
discarded at exit (CS-11). The Session's exact signature is
provisional until 2.4/3.2; the protocols are the commitment.

```python
result = model.run(steps=N | runlen=... | end_time=...,   # exactly one
                   outputs=(...), snapshots=fr.io.Snapshots(...),
                   max_chunk=None, progress=True, jit=True,
                   debug_nan=False, raise_on_nan=False) -> RunResult
```

- `runlen`/`end_time` reduce to steps **sign-agnostically**
  (amended, V-S1): `steps = ceil((end − t0)/dt − eps)` with the
  precondition `(end − t0)·dt > 0` (named error otherwise);
  `runlen` is an unsigned duration, direction from the dt sign —
  the old overshoot semantics preserved in both directions.
  Repeated `run()` continues from the carry, **bitwise identical to
  one uninterrupted run** (warm-up included) — given the per-step
  scan body compiles identically across chunk lengths; the
  equivalence test asserts a body-jaxpr/HLO match, degrading to
  identical-chunk-plan comparison if XLA ever specializes the body
  on trip count. `run_backward` stays
  dead (flip the `fr.params.TIME_STEP` leaf; the snapshot manifest
  guards the sign across restarts — 02_rules). `RunResult` carries
  status/steps/compile-vs-run seconds/rates; `run()` never exits the
  process; running a panicked carry raises with guidance.
- **What is jitted**: one framework-level
  `step_chunk(assembly_record, carry, n)` — donated carry
  (`donate_argnums`), loop-invariant stepper/schedule as non-donated
  arguments; the assembly record is hashable static, so identical
  re-assemblies **share the jit cache** (a per-assembly closure
  would silently defeat it — implementation rule + a
  compilation-count regression test). AOT `lower().compile()`
  replaces the old first-step compile-timing hack, adding a peak-
  memory report.
- **Chunking**: boundaries = union of all trigger step sets ∪
  snapshot steps ∪ last step (§6.4), subdivided by `max_chunk`
  (auto ~256); remainders via a lazily compiled `chunk(1)` (two
  trace shapes in the common regular-cadence case; pad-and-mask
  rejected). Documented meaning of the knob: host-sync granularity.
- **NaN mechanism**: once-per-chunk S5 `isfinite` reduction (on
  the scan's final state — per-step was priced out by the
  2026-07-12 GPU benchmark, see the sign-off note) into
  `panic.(flag, it)` (catches Inf; records the detecting chunk
  boundary) + chunk-boundary abort with a host report; **no `lax.cond` no-op wrapper** (a data-dependent
  conditional in the scan forces per-step GPU pipeline syncs and
  blocks whole-loop fusion — it taxes every healthy step; recorded
  as an opt-in retrofit). `debug_nan=True` keeps a chunk-start copy
  and replays to the exact first-bad step.
- **Ctrl-C contract**: the in-flight chunk completes, the carry is
  consistent, zero steps lost, no exception on first interrupt.
- **Walltime**: `fr.io.Snapshots(trigger=fr.every(walltime=...))`
  with a **predictive** boundary check (elapsed + predicted chunk +
  snapshot margin) — stops before the chunk that would blow the
  budget; `on_walltime=fr.io.resubmit()` keeps the old
  `scontrol`→`sbatch` auto-detection as a plain function.
- Boundary sequence: sync → panic check → writer flush → progress →
  walltime check.

## 6.4 The IO seams (what 2.6 builds behind)

([d4_3](../../research/d4_3_io_seams.md)) **Outputs are evaluated
host-side on the synced carry at chunk boundaries; no `io_callback`
in the traced step.** The decisive seam argument: under in-trace
triggers every writer set becomes part of the step trace — adding a
diagnostic to a writer would recompile the physics; chunk-boundary
evaluation keeps the compiled step IO-free forever. (Supporting:
ordered-callback D2H pinning vs donation, traced gathers on
multi-device, no vmap composability.) The frozen contract is
D2.3's: an output is a pure `(model_state) -> Field | scalar`
callable + a trigger + a sink; the in-trace *capture* variant is
the designed-for 2.6 escape for every-step output, reachable
without contract changes.

- **Triggers** are declarative frozen data (`fr.every(steps=|hours=
  |walltime=)`, `fr.at(times=...)`, unions, windows), lowered at run
  planning to step-index sets **in step space** (amended, V-S1:
  `k = (t − t0)/dt`, snap = `ceil(k)` — sign-agnostic by
  construction; `fr.at` times outside the run's time interval error
  at planning; realized times logged and written); `every()`
  includes step 0 (the `execute_at_start` successor); walltime
  triggers are snapshot/action-only. **Binding split (amended,
  V-C5)**: diagnostic-*name* resolution happens at assembly (for
  constructor `io=`) or immediately (bound spellings);
  `OutputStream.bind()` — store creation, dry evaluation — happens
  at **run start**, never at assembly (presets never create files
  for models that never run). Streams dedupe by resolved path; two
  distinct streams targeting one path is an `IOCollisionError`.
  **`Snapshots` is run-config only** — `io=` rejects it (one resume
  path, never two).
- **Writer seam**: `fr.io.Writer(path, fields, derived, trigger,
  mode, chunks, attrs)`; `fields=` resolves against the FieldTable
  (default: PROGNOSTIC + DIAGNOSTIC; AUXILIARY opt-in); bind-time
  dry evaluation validates and creates stores eagerly; the layout
  contract makes written zarr xarray/xgcm-openable (staggered dims
  from spaces, FieldMetadata attrs, CF time axis + iteration
  coordinate, fingerprint digest as provenance); **sinks receive
  Fields, not numpy** (gather inside the it-1 sink; decomposed-slice
  output is a sink swap keyed by `local_slice`). Resume contract:
  writers open `mode="a"` and **`truncate_after` keyed on the
  iteration coordinate** (amended, V-S1: monotone in both time
  directions — backward runs invert the time axis, and
  decreasing-time CF axes are legal) — else a crashed segment forks
  the axis.
- **Snapshots are a dumb leaf blob** (directory: versioned manifest
  + **true-shape gathered** leaf arrays — device-count-portable
  restarts; orbax-shaped, no orbax dependency; the manifest
  additionally records **provided-parameter values** — at least
  `TIME_STEP`, guarding the backward-run sign across restarts, and
  the fingerprint's "spaces" are the **bare declared spaces**,
  never `Layout`/device topology — 02_rules, amended V-S1/V-C3),
  *not*
  zarr-via-the-writer — writers are for humans, snapshots for the
  machine. The manifest stores the fingerprint digest **and its
  source record**, so mismatches *diff* ("stepper statics differ:
  cnab2 → sbdf2"), never silently reuse. API: assemble-then-load —
  `model.snapshot(path)` / `model.load_snapshot(path)`;
  `Model.restore(path)` classmethod rejected (the dill pattern
  reborn). `fr.io.Snapshots(path, trigger, keep, resume=True,
  on_walltime=...)` is the run config; resume finds the newest
  complete snapshot (atomic tmp-dir + rename), checks, loads,
  re-plans remaining steps against the absolute target.
- **The dill successor**: `Model.save/load` dies. Persistence =
  **script re-assembly + leaf snapshots** (02_rules entry: "no
  pickled models; anything not reconstructible from
  (script, snapshot) is a design bug").
- **`OutputStream` protocol** (the one interface D4 owns):
  `trigger; bind(model); write(model_state); truncate_after(time);
  close()` — Writer, TimeSeries (scalar rows, replicated-reduction
  fetch, tail-able CSV), and future capture streams all implement
  it.

## 6.5 Post-assembly lifecycle

([d4_4](../../research/d4_4_lifecycle_coupling.md)) The mutation surface,
one table: pre-assembly free; assembly snapshots derived data;
post-assembly (host, chunk boundaries) allows `set_fields`
(PROGNOSTIC-only, incoming fields re-homed), `set_state` (PROGNOSTIC
subset; AUX/DIAG in the input ignored; **missing components are
left untouched** — amended V-C12, so `rest="zero"`-trimmed
transform outputs feed larger models cleanly),
**`set_aux` (amended V-C1: the declaration-consented host write for
module-owned AUX — the coupler exchange path; no rewarm by default;
02_rules)**, `constrain` (+ `set_state`), `update_parameters`,
`reset()`,
`snapshot`/`load_snapshot`, reads, `advance`/`run` — with the
canonical per-point sweep order **`update_parameters → reset →
set_fields`** (amended V-C7; safe because re-materialization at the
stale clock is masked by SELF_UPDATE-first, and rewarm is
idempotent under the pair); forbidden: attribute pokes (teaching
error),
frozen-grid ops, anything treedef-changing (module add/remove/
enable/disable, Ramp-spec changes, treatment flips) → re-assemble.
In-run: only the sanctioned traced dynamics.

- **`update_parameters(updates, *, rewarm=True)`**: resolve through
  the binding table (incl. `TIME_STEP`); validate structure
  preservation (same treedef → no recompile); write functionally;
  **re-materialize owner-derived AUX fields** — *host-writable
  components are exempt* (coupling sign-off: the host write is
  their source of truth, the default is initialization-only;
  02_rules) — by re-running the
  owners' declaration defaults — `FieldDeclaration.default` may be
  an **unbound method** of the owner (the D3 aliasing pattern),
  reading **only the owner's own leaves**; allocation and
  update_parameters share one code path (the defaults rule becomes
  "evaluated with the owner's *current* leaves, same path both
  times" — correct by construction). Solver precomputes need no
  hook (D2.1's bind/in-step split already forbids baking parameter
  values). Does not clear the panic flag.
- **Stale multistep buffers: auto re-ramp by default**
  (`rewarm=True` zeroes the warm-up counter; buffers need no
  zeroing — warm-up rows never weight old entries before they shift
  out). Rationale: buffered tendencies embody *old physics* (an O(1)
  discontinuity beats the O(dt) warm-up cost), and it buys the
  invariant `update_parameters(p); run ≡ fresh-assembly(p) +
  set_state; run`. Sign-flipping dt *requires* it. `rewarm=False`
  for knowingly-epsilon changes.
- **`reset()`** — exactly the OptimalBalance need: re-init stepper
  state (re-warm), **reset the clock (which is what restarts a Ramp
  leg)**, clear panic, DIAGNOSTIC → defaults; PROGNOSTIC/AUXILIARY
  untouched. Invariant: `reset(); set_state(z)` ≡ fresh assembly +
  `set_state(z)`, bitwise.
- **Sweeps and the grid ruling**: shared jit cache requires the
  **same grid object** (fields carry the grid as identity-hashed
  static aux — a fresh grid per point recompiles), identical
  treedef, same stepper statics, swept values as leaves.
  **One-grid-many-models is the sanctioned idiom, via the
  frozen-grid verify path**: at freeze the grid records a
  negotiation fingerprint (state-space set, override keys, HaloSpec,
  layouts); later assemblies on the frozen grid verify their demands
  against it — pass by construction for identical composition,
  `GridFrozenError` with "assemble the most demanding model first"
  otherwise. The cheaper sweep needs no re-assembly at all:
  `update_parameters` + `reset` + `set_fields` per point.
  **Amended at validation sign-off (V-C2)**: *module-type* sweeps
  (FPlane→BetaPlane) use a **fresh grid per composition** — free at
  the step-cache level, since different module tuples are different
  assembly records anyway — and verify is relaxed to
  **satisfiability**: a new state space with zero halo/layout/
  override demands (the R2 ConstantSpace family: `Profile("y")`
  after `Profile()`) is adopted into the record rather than
  refused; only genuinely larger demands raise. Also amended
  (V-C11): coupled runs derive one model's dt from the other by
  exact division (per-model float64 clocks otherwise dephase by
  ulps per exchange window).

## 6.6 Multi-device and coupling proofing

([d4_4](../../research/d4_4_lifecycle_coupling.md)) Multi-device: fields
are born sharded (allocation post-negotiation); entry points re-home
external leaves; gather lives at the host/IO boundary — module code
has zero device awareness; CI reruns the lifecycle tests under
forced host devices (AGENTS.md pattern), including snapshot
portability across device counts. Coupling (Phase 3 stays a pure
addition): `advance` as the interleavable primitive; exchanged data
enters as an AUX field owned by the receiving model's coupler module
(owner-write rule preserved); clocks stay per-model with dt ratios
as exact per-window step counts; no process-global mutable state
(audited: registries grid-private, interning per-mesh, `fr.params`/
Role registries immutable, jit cache keyed by statics, `name=` for
log attribution); cross-grid arithmetic already raises, so the
Phase-3 `Regrid` operator slots in cleanly. The OptimalBalance
workflow composes end-to-end with the resolved pieces (Ramp legs via
`reset()`'s clock reset; dt sign via `update_parameters`; base-point
projections host-side; `from_model` needs f0/n2/dsqr, not Ro — no
`at_time=` friction).

## 6.7 Reports, debugging, profiling, logging

- **`model.report`** (assembly): header, the field resolution table,
  the parameter binding table (identity-defaults listed), dispatch
  overrides, the kind-ordered schedule (with *why* self-updates are
  scheduled and which implicit merges happened), halo/layout results,
  lint warnings; plus a **run-start addendum** — the
  defaults-vs-user-initialized provenance table (discharges D1.1's
  logging promise). `model.__repr__` is its header.
- **The old timer dies**; three tiers replace it: always-on
  per-chunk stats (RunResult); `jax.named_scope("module/term")`
  stamped by the composer so profiler rows carry module names;
  **`run(jit=False)`** eager mode — works by construction (the halo
  tracer already requires plain-Python steps) — with a per-term
  timing table, pdb, and `jax_debug_nans`.
- **Logging**: rich at assembly, chunk-boundary run stats,
  `jax.debug.print` as the documented in-trace escape hatch (never
  in shipped modules — step-cadence monitoring is a DIAGNOSTIC
  accumulator or writer expression).

## 6.8 Reconciliations (where the reports disagreed)

1. **IO evaluation point** (d4_2 assumed in-trace `io_callback`;
   d4_3 argued chunk-boundary): chunk-boundary adopted — the
   IO-free-step seam argument is decisive; d4_2's stated fallback
   (boundaries derived from trigger cadences) is exactly what
   applies, and its donation/remainder/interrupt analysis survives
   unchanged.
2. **Grid reuse** (d4_1: one Model per Grid, hard error; d4_4:
   one-grid-many-models required for sweep cache sharing): d4_4
   adopted — fields carry the grid as an identity-hashed static, so
   a fresh grid per sweep point recompiles, defeating D2.4's
   promised shared-cache sweeps; the frozen-grid **verify path**
   replaces d4_1's hard error, and d4_1's amendment text is adapted
   (merge call site = assembly step 3 of the *first* model on the
   grid).
3. **Output attachment** (d4_1: constructor `io=`; d4_3:
   `run(outputs=...)`): both — the constructor slot holds standing
   config naming *unbound* diagnostic primitives (bound at
   assembly; resolves the D2.3 chicken-and-egg), `run(outputs=...)`
   adds per-run streams accepting bound `model.diagnostics.*`
   spellings; the union binds at run start.
4. **Walltime spelling** (d4_2: `run(walltime=, on_walltime=)`;
   d4_3: inside `fr.io.Snapshots`): the Snapshots config owns it —
   one home for trigger + snapshot + resubmit.
5. **Constraint-application naming** (d4_1 `enforce_constraints`):
   `apply_constraints()` was adopted here, then superseded at
   implementation by the **pure** `model.constrain(state, *, t=None)
   -> State` (§6.2), which composes with `tendency` and the
   transforms instead of mutating the carry.

## 6.9 Residual open points

Settled at implementation: `state_type` ships (Module attribute +
`Model(state_type=)` kwarg, linted hashable); **`model.state` is
copy-on-read** — the carry's buffers are donated to the next
`advance()`, so a live view would reference deleted buffers (the copy
is one device-to-device pass per read); `truncate_after` ships on
every sink, CSV included. The sign-off amendments (grid-notes merge
call site, frozen-grid fingerprint/verify path, the D1.1 softening,
the three 02_rules entries) are all applied.

Still carried in [`07_open_threads.md`](07_open_threads.md): the
GPU-conditional and S5-fusion cost claims (the NaN-cadence benchmark);
the post-assembly writer-attach API and the capture-stream variant;
multi-process walltime/interrupt consensus (3.2/3.3); the
shared-jitted-runner lint and compilation-count regression test.
