---
status: normative
date: 2026-07-07
---

# Model layer redesign — Coupled models (design-for constraints)

Part of the model redesign notes; see
[`00_overview.md`](00_overview.md) for the document map. Research
reports in [`research/`](../../research/README.md) (c1 precedents, c2 the
concrete walk, c3 architecture).

Status: **fully signed and applied (2026-07-08).** Decisions 1 and
2: 02_rules gained the extended `set_aux`, the re-materialization
exemption + three-operation matrix, and the S6 accumulation idiom;
D2.3's escape-hatch pointer re-aimed; the §5.5 gate annotated.
Decision 3: **the hybrid** — 3a (typed `AdvanceResult` +
`PanicError`, no-hidden-sync, reserved `sync=`) and 3b
(**`fr.ops.Session`** as the context-manager front door over
normative protocol-level utilities; `run()` reimplemented as a
single-model Session loop, bitwise test extended; notebooks keep
`run()` and never see the `with`) — folded into 04 §6.3. The
CS-1..18 constraint list is final and carried by the class-spec
briefs. Scope: this is **not** the
ROADMAP 3.2 coupling design — it is the *design-for validation* the
class-specification phase needs, produced the way the grid notes
validated against future grid types: a precedent survey, an
adversarial walk of a concrete atmosphere–ocean configuration
through the resolved D1–D5 design, and an architecture pre-design,
distilled into the constraint list below. 3.2 remains the full
design, done with 2.x implementation experience in hand.

---

## 11.1 The verdict, and the recommended Phase-3 shape

**The resolved design's coupling hooks are fundamentally right** —
`advance` as the interleavable primitive, receiver-owned AUX via
`set_aux`, per-model clocks with exact-division dt, host-held
grid-pair `Regrid` — and they match where production coupling has
converged: **sequential single-program coupling is a deliberate
production choice, not a fallback** (ECMWF moved *off* OASIS
concurrency to NEMO-as-subroutine for correctness *and* speed; FMS
serial default; ClimaCoupler sequential-only). The classic coupler
machinery that motivated MPMD infrastructures (handshakes, LAG
choreography, deadlock detection, PE-layout balancing, offline
weight files) **evaporates** in a single-program jax world; what
remains irreducible — windowed flux accumulation, per-field
time policies, lag/ordering error, coupler state in restarts, clock
divisibility, conservative flux remapping, initialization shock —
is exactly what the constraint list targets ([c1](../../research/c1_coupling_precedents.md)).

The recommended Phase-3 shape ([c3](../../research/c3_coupling_architecture.md)):

- **A `CoupledModel` facade layered on the driver pattern** — sugar
  over `advance()` + `set_aux()`, owning N models + a **declarative
  exchange table** (CMEPS-style frozen data:
  `(source read, transform, time policy, target AUX)`) + the ops
  surface; normative test: facade run ≡ hand-written driver,
  bitwise — chunk for chunk, both sides hitting the same jit-cache
  entries per model (compile-counter asserted); bitwise holds only
  between identically-compiled paths (phase-1 finding 1).
  Coupler-owning-stages-in-both-models is rejected *by
  construction* (it would contradict one-carry-per-model, the stage
  signature, and Model-not-a-pytree at once).
- **Flux computation lives in a fast-side module, inside the
  trace** — recomputed every fast step against the held slow-state
  AUX (the production pattern), with the window mean accumulated in
  **carry-resident state** — which makes mid-window coupled restart
  free via the existing snapshot machinery. The one **new
  normative rule** this demands: *no coupling state may live as
  host-side Coupler/driver attributes — anything with time-integral
  semantics is a declared field in some model's carry.*
- **Direct first-order conservative `Regrid` for fluxes** (on
  integer-ratio tensor grids this ≈ the grid layer's cell-average
  restriction), linear for states, declared **per field** in the
  exchange table alongside the two-policy time vocabulary
  (`ACCUMULATE/AVERAGE` fast→slow fluxes; `INSTANT`-and-hold
  slow→fast states — OASIS `LOCTRANS` reduced to its two
  survivors). No exchange grid for idealized tensor grids — but the
  flux-host stays a switch (CPL7's decade-long bake-in regret), and
  a **mediator-as-degenerate-2D-`fr.Model`** remains possible (the
  module-only-models property) as the designed-for exchange-grid
  answer.
- **Concurrency**: sequential is the primary semantic; overlap on
  disjoint devices exists *for free* via jax async dispatch — but
  **only under dispatch-then-sync ordering** (with a blocking
  `advance`, the per-chunk panic-flag read serializes the two
  models entirely). The class specs keep overlap possible with two
  sentences (constraints 13–14); nothing more is built.
- **Iterated/Schwarz coupling**: design-for only (no production
  consumer anywhere; 2 iterations double cost). It is expressible
  later as D5 algebra over a **product state space**
  (`FixedPoint(CoupledPropagator)`), which §10.1 already permits;
  the enabling hooks are the in-memory carry read (constraint 20)
  and product-ready transform bases (constraint 21). Window
  rollback is structurally free (immutable carries).
- **Initialization shock** is a data problem, mitigated by the
  existing transforms (balanced ICs via `OptimalBalance` /
  projections + short coupled spin-up) — one doc line, no
  machinery. Fraction merging, area corrections, and global
  conservation fixers are **out of scope**: ship the budget
  *diagnostic* (a paired-diagnostics expression), never the fixer.

## 11.2 Decisions needed before the class specs

From the adversarial walk ([c2](../../research/c2_coupled_walk.md)) —
three, of which the first is the one the class specs cannot proceed
without:

1. **The windowed-accumulation bundle (walk F1+F2).** The gap: an
   S6 DIAGNOSTIC stage can accumulate the window-mean flux in-trace
   and the host can read it at chunk boundaries — but **nothing can
   reset it** (`set_aux` is AUX-only; `reset()` nukes clock and
   warm-up); meanwhile D2.3's blessed accumulator route
   (`self_update`) is **cadence-broken** (per-substage → RK3
   multi-counts at stage times; correct under AB3 by accident).
   Resolution: (a) `FieldDeclaration.host_writable=True` becomes
   **lifecycle-polymorphic (AUX ∪ DIAGNOSTIC)** and `set_aux`
   writes any consented non-PROGNOSTIC component; (b) the **S6
   accumulation idiom** (read-own-previous + `replace`) is blessed
   normatively; (c) **D2.3's escape-hatch pointer is re-aimed from
   `self_update` to S6 DIAGNOSTIC stages**, with `cadence=STEP`
   reserved on self_update and the per-substage hazard documented;
   (d) optionally a `fr.modules.WindowAccumulator` preset
   (host-reset and in-trace modulo-reset variants).
2. **The re-materialization exemption (walk F4).** As signed,
   `update_parameters` re-runs declaration defaults for all AUX of
   a changed owner — **silently zeroing host-written exchange
   fields** (τ, SST; an AUX accumulator's partial sum
   unrecoverably). Resolution: host-writable components are
   **exempt** — the host write is their source of truth; the
   default is initialization-only.
3. **The `advance` surface (walk F5 + c3 H1–H4)** — *refined after
   the context-manager investigation (2026-07-08)* into two parts:
   - **3a (signed with 1 and 2): the primitive's contract.**
     `advance()` returns a minimal frozen `AdvanceResult` (steps
     done, panic `(flag, it)`, wall seconds) and raises a typed
     `PanicError` (carrying the model `name=`) at the abort
     boundary; `RunResult` stays plain/aggregatable with a shared
     status enum. Needed regardless of packaging: a context
     manager's cleanup only works if failures are exceptions, and a
     session can only attribute stats if the primitive returns
     them.
   - **3b: the ops packaging — `fr.ops.Session`, a context
     manager as the front door, over normative protocol-level
     utilities.** `__enter__` plans triggers (the CS-7 reusable
     planner), binds outputs, resume-checks snapshots (including
     the CS-18 cross-model clock assertion); `s.active`
     encapsulates the predictive walltime check + panic polling +
     graceful first-Ctrl-C; **`s.advance(atm=10, ocn=1)` is the
     sanctioned multi-model spelling** (dispatch-then-sync built
     in — the correct concurrency order becomes the easy one);
     `__exit__` guarantees wait-for-in-flight → writer flush →
     triggered snapshot → `on_walltime`, on *every* exit path
     (normal, `PanicError`, `KeyboardInterrupt`) — dissolving the
     forgotten-`try/finally` contra. `run()` is reimplemented as a
     single-model Session loop (the bitwise run≡driver test
     extends to run≡Session-loop); notebooks keep `run()` —
     interactive use never sees the `with` block. The
     protocol-level utilities (walltime predictor, progress
     renderer, stream binding) remain the normative spec; the
     Session's exact signature is provisional until 2.4/3.2.
     Session state is config + transient counters only, discarded
     at exit (CS-11 holds: everything time-integral lives in
     carries). On `PanicError`, `__exit__` flushes writers and
     leaves carries in memory for autopsy (no automatic crash
     snapshot; an `on_error=` policy is a possible later option).

## 11.3 The consolidated class-spec constraint list

Merged from c2's 12 and c3's H1–H11 (deduplicated); **CS-1..3 are
§11.2's decisions**, the rest are one-sentence spec items:

| # | Constraint | Source |
|---|---|---|
| CS-1 | `host_writable` lifecycle-polymorphic; `set_aux` spec not hard-coded to AUXILIARY; S6 accumulation idiom normative; D2.3 pointer re-aimed; `cadence=` reserved on self_update | F1/F2 |
| CS-2 | The re-materialization table carries the `host_writable` flag; host-writable components exempt | F4 |
| CS-3 | `advance()` typed status return + typed `PanicError`; `run()` implementable purely on top of `advance` + free-standing ops utilities (walltime predictor, progress renderer, resubmit) | F5, H4 |
| CS-4 | **No-hidden-sync invariant**: the panic read is `advance`'s only host synchronization; the committed carry may hold pending (future-valued) arrays | H1 |
| CS-5 | Reserve `advance(steps, *, sync=True)`; `sync=False` returns a `PendingAdvance` handle (full semantics leave-open) | H2 |
| CS-6 | `model.panicked` as a cheap host-readable property | H3 |
| CS-7 | Trigger lowering / run planning as a pure reusable function (triggers → step sets), importable by drivers | H5 |
| CS-8 | `set_aux` accepts device-resident Fields and re-homes without a host round-trip (composes with CS-4) | H6 |
| CS-9 | Master-clock wording: one designated master dt; every coupled model's dt derived by **exact division**; windows are exact step counts in every participant (generalizes V-C11 beyond pairwise) | H7 |
| CS-10 | Snapshot **manifest header** machine-readable without leaf IO (clock time, it, dt, fingerprint digest) — the cross-model consistency assertion | F6 |
| CS-11 | Snapshots never embed run-loop/driver state (the "no coupling state outside the carry" rule, stated) | c1, H11 |
| CS-12 | Public in-memory carry **read** (`model.carry` as an opaque snapshot value, the disk-free twin of `snapshot()`); the sanctioned *setter* leave-open (`debug_nan` builds on it) | F7, H8 |
| CS-13 | Empty-PROGNOSTIC models are legal (stage-only schedules) — the mediator-as-model lint guard | H9 |
| CS-14 | `StateTransform` stays product-ready: signatures are opaque compared values, no `isinstance(state, State)` in the base, endo-ness asserted only where required | H10 |
| CS-15 | `Regrid` (Phase-3 slot): grid-pair-bound, built after both freezes, host-held; constructor admits `conservative=True` with the discrete-integral-preservation contract stated | F8, c1 |
| CS-16 | Surface/trace restriction operator (3D → boundary 2D) on the Phase-3 operator list; coupler-AUX space policy pinned: Profile-broadcast + owner-declared indicator AUX for it-1, §3.6 trace-space designed-for (with `set_aux` re-homing specified for both) | F3, F10 |
| CS-17 | RESOLVED (2026-07-08, Silvano): **global precision only** — no per-space width axis; `scalars` stays Körper-only. Accumulator precision is covered by the S6 chunk-cadence idiom: in-trace sums span at most one chunk, the chunk-boundary host read accumulates in float64 on the host (`fr.modules.WindowAccumulator`). See [`classes/declarations.md`](classes/declarations.md), Open questions item 3 | F8 |
| CS-18 | The **coupler protocol page** (doc, not a class): exchange-then-advance order (primes window 0; self-heals staleness), snapshots of coupled runs only at window boundaries, cross-model clock assertion on resume, window index derived from clocks | F6, F9 |

## 11.4 Reconciliations (where the reports differed)

1. **The accumulator mechanism** (c1 recommended `self_update`-
   into-own-AUX; c2 proved `self_update` is cadence-broken and
   S6-DIAGNOSTIC is the correct write slot): **c2's mechanics with
   c1's placement principle** — the fast-side module owns the
   accumulator as carry state; the write is an S6 stage; the reset
   is the extended `set_aux`. c1's headline consequence (coupled
   restart free because carry-resident) holds unchanged.
2. **Sequential vs concurrent** (c1: sequential is the production
   answer; c3: dispatch-then-sync overlap is nearly free):
   compatible — sequential is the primary semantic and nothing
   concurrent is *built*; CS-4/5 keep the overlap reachable for the
   day profiling asks for it.
3. **Exchange grid** (c1: direct conservative regrid, no exchange
   grid for idealized tensor grids; c3: mediator-as-model as the
   eventual conservative answer): both — direct regrid is the 3.2
   default; the mediator stays *possible* (CS-13) as the
   designed-for escalation for genuinely non-nested grids.

## 11.5 Explicitly not promised / deferred to 3.2

Not promised: multi-host (`jax.distributed`) coupling; fused
cross-model super-steps (traceable Regrid's named preconditions:
cross-mesh resharding over a super device mesh, Regrid demands in
both negotiations pre-freeze, an exchange seam in the traced
schedule); threads/MPMD; automatic conservative-flux machinery;
Schwarz iteration (design-for only). Deferred to 3.2: the
`CoupledModel` surface (exchange-spec object, ensemble
report/panic policy, coordinated `Snapshots`); the Regrid spec and
its demand timing vs frozen grids; the Coupler-module template and
mediator preset; parallel vs Gauss-Seidel window schedules;
`CoupledPropagator`/`ProductSignature` + the product norm; the
coupled manifest format; whether the mediator needs a stepper;
multi-host program-divergence discipline and walltime/interrupt
consensus (with 3.3).
