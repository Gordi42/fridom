---
status: normative
date: 2026-07-13
---

# Model layer redesign — Rules

Part of the model redesign notes; see
[`00_overview.md`](00_overview.md) for the document map.

Status: **complete** — the rules accumulated from the D1–D5 sign-offs,
the validation walks, and the coupling design-for. They bind the
implemented model layer (`fridom.model`); the first five entries name
the section that carries each rule's full statement.

## 4. Rules

- **Lifecycle / setup order**: the normative assembly pipeline
  (D4), including the dispatch-merge call site owed to the grid
  notes ([`../grid/02_rules.md`](../grid/02_rules.md) section 3.4) and the
  exact interaction with `negotiate`/`freeze`.
- **Field registration rules** (D1): declaration validity,
  collision handling, role vocabulary, treedef stability, when the
  component set freezes.
- **Parameter resolution rules** (D2): provides/requires namespace
  discipline, one-provider-per-name, auxiliary-field parameters,
  the diagnostics story.
- **Purity rules**: what code runs inside the traced step (pure
  modules, no Python-state mutation, no branching on traced
  values) vs. at the run-loop boundary (host callbacks); the
  successor of the old "stateful modules belong in
  `mset.diagnostics`" rule.
- **Halo rules**: tendency traceability requirements
  (`trace_halo`), `Module.extra_halo` declaration duty for raw
  `.data` escapes.
- **Read/write gating** (decided with D1, restate normatively):
  tendency terms receive the full state vector (all lifecycles);
  contribution dicts may only key PROGNOSTIC components; AUXILIARY
  components are written only by their owning module; DIAGNOSTIC
  reads see the most recent write in stage order.
- **Naming** (decided with D1): the state vector is always `state`
  (never `z` — collides with the vertical coordinate); the carry
  variable is `model_state`; `mz`/`dz` spellings are retired.
- **Restart fingerprint** (consolidated, decided with D3): the
  snapshot fingerprint hashes everything a restart's numerics depend
  on — field declarations (names/spaces/lifecycles), the module
  tuple (types + order), per-term treatments, stepper statics
  (order/eps/tableau), and parameter *specs* (a Ramp's shape is
  structure; its endpoints are leaves). Mismatch = named error,
  never silent reuse of incompatible stepper history.
- **Clock precision** (decided with D3; reconciled 2026-07-08 with
  the global-precision ruling): the *authoritative* clock is
  host-side float64 always — the calendar, run targets, trigger
  times, and snapshot times live in numpy, unaffected by the x64
  flag. The traced `Clock.start/elapsed` leaves take the global
  width: float64 under the default x64-on run, where the original
  guarantee holds verbatim. In an x64-off (float32) run a float64
  traced leaf is impossible (JAX downcasts), so the rule becomes:
  in-trace elapsed is **re-anchored from the host float64 clock at
  every chunk boundary** (accumulation drift bounded by one chunk,
  ~256 steps — the same chunk-cadence pattern as the S6
  accumulator), and the exact integer `it` remains the primary key
  for triggers/schedules. Residual float32 quantization of
  *absolute* stage time (~eps·t) still reaches in-trace
  `TimeDependent` evaluation — float32 runs carry that phase error
  by choice; run float64 (the default) where Ramp/forcing phase
  fidelity matters, or evaluate phases in per-chunk-anchored shifted
  time (designed-for refinement, not required for 2.4).
- **Lagged implicit coefficients** (decided with D3):
  state-dependent implicit coefficients (CATKE-like) are evaluated
  on the state passed into the stage (predictor/lagged values); the
  solve itself stays linear.
- **Diagnostics metadata** (decided with D2/D3): diagnostic
  functions end with `with_metadata(...)` — binary field algebra
  returns default metadata by the grid-layer rule.
- **Defaults, one path** (decided with D4): `FieldDeclaration.default`
  is evaluated with the owner's *current* leaves, through the same
  code path at allocation and at `update_parameters` (that identity
  is what makes re-materialization correct by construction); a
  default closure may read **only the owner's own leaves** —
  cross-module-derived AUXILIARY values are `self_update` territory
  or a re-assembly.
- **No pickled models** (decided with D4): persistence = script
  re-assembly + leaf snapshots; anything not reconstructible from
  (script, snapshot) is a design bug. The snapshot manifest carries
  the fingerprint digest *and its source record* so mismatches diff.
- **Fingerprint scope** (decided with D4): the restart fingerprint
  hashes structure, never leaves — IC/state differences are
  deliberately invisible (a restart overwrites them); do not expect
  IC mismatches to be caught.
- **SELF_UPDATE-before-consumers is load-bearing** (decided with
  D5): `reset()` leaves AUXILIARY fields untouched, so a
  time-dependent AUX field still holds a previous run's end-time
  value after a reset; Tier-2 transform determinism (`T(state)` a
  pure function of its input) relies on the D3 schedule running
  SELF_UPDATE first in every substage, so the owner recomputes from
  the reset clock before any consumer reads. Regression test: two
  consecutive transform calls on one input, bitwise-equal outputs,
  with a Ramp-valued AUX field in the twin.
- **`model.set_aux` — the declaration-consented host write**
  (validation sign-off V-C1; **extended at the coupling sign-off,
  2026-07-08**): host-side, chunk-boundary-only writes to
  **AUXILIARY or DIAGNOSTIC** components are legal **iff** the
  owning module declared consent on the `FieldDeclaration`
  (`host_writable=True` — lifecycle-polymorphic); incoming values
  are re-homed per `decomposition.sharding(space)` like
  `set_fields` inputs; **no warm-up re-ramp by default** (exchange
  data is forcing; `rewarm=` available). The declaration is the
  owner's sanction. `host_writable` means **externally sourced
  data** (exchange fields, accumulator resets, assimilation
  increments) — never "user-tunable parameter field" (user-supplied
  profiles go through constructor values feeding the declaration
  default). Every host-writable component is listed in
  `model.report`. This is the Phase-3 coupler's exchange path.
- **Host-writable components are exempt from re-materialization**
  (coupling sign-off, walk F4): `update_parameters` re-runs
  declaration defaults only for fields that are *owner-derived*; a
  host-written field's source of truth is the host write — its
  default is initialization-only. (Without this, a routine
  `update_parameters` would silently zero exchanged fluxes and
  destroy partial accumulator sums.) The three-operation matrix,
  stated: `set_aux` writes consented components;
  `update_parameters` skips them; `reset()` zeroes DIAGNOSTIC
  (including consented ones) but never AUXILIARY.
- **The S6 accumulation idiom** (coupling sign-off, walk F1/F2): a
  DIAGNOSTIC-kind stage may read its own component's previous value
  and `replace` with the updated sum — step-cadence, post-NaN-seam,
  carry-resident (restart-exact), host-read at chunk boundaries,
  host-reset via the extended `set_aux`. **This — not
  `self_update` — is the sanctioned home for step-frequency
  accumulation** (time means, window-mean fluxes, budgets):
  `self_update` runs per *substage* and multi-counts under
  multi-stage steppers (RK3: three unweighted stage-time samples
  per step — correct under AB3 only by accident). `cadence=STEP`
  is reserved on `self_update`, not built; the hazard is documented
  in its docstring. (No `fr.modules.WindowAccumulator` preset is
  promised; a windowed-accumulation preset can be introduced with
  coupling if wanted — see
  [`07_open_threads.md`](07_open_threads.md) §9.1.)
- **`extra_halo` mechanics** (validation sign-off, V-N2): a module
  declaring `Module.extra_halo` has its terms **exempted from the
  halo trace** (the declared spec substitutes); contribution-key
  and write-gate validation for those terms runs in a second
  dry-run mode over real zero-valued fields. The sanctioned
  closure-author escape for clamps/branches is a **custom pointwise
  Operator (halo 0)** registered like any stencil kernel — never
  raw `.data` in a term; `Where`-style conditions are built through
  the operator layer's `("select", ...)` kind, not field
  comparisons (fields define none).
- **Sync cost is the grid's, never the modules'** (D3 corollary,
  2026-07-08): terms and stages are plain Python over fields; no
  hook sees, places, or elides a halo sync, and the term signature
  `(self, state, ctx) -> dict` is sync-policy-neutral. The placement
  is **grid-owned**, and the shipped strategy is **consumption-side
  sync with trace-time halo-validity tracking** (ROADMAP task 1.8):
  an operator exchanges iff its operand's validity is below the
  application's requirement, so the composed step pays roughly one
  exchange per state component per step, and field arithmetic and
  pointwise products exchange not at all. The model layer neither adds
  nor removes any exchange — it only has to hand the *whole* step to
  `grid.negotiate(tendency=...)` at assembly step 7, which it does, so
  the halo trace observes every operator application in program order.
  Contract:
  [`../grid/classes/decomposition.md`](../grid/classes/decomposition.md).
- **Provides implies constancy** (validation, V-S walk): a module
  *provides* a scalar parameter only when that scalar is the whole
  truth — `BetaPlaneCoriolis` holds an `f0` leaf but must **not**
  provide `coriolis.f0` (its Coriolis parameter is `f(y)`, the aux
  field); analytic consumers (`from_model` eigenmodes) rely on the
  provide's existence as a constancy check.
- **Restart-fingerprint scope, refined** (validation): "spaces"
  means the **declared `SpacePattern`s / bare interned spaces** —
  `Layout`, negotiation fingerprints, and device topology never
  enter the fingerprint (device-count-portable snapshots depend on
  it). Module-owned **ADVANCE-stage integrator statics** (substep
  count, filter spec) join the fingerprint alongside stepper
  statics. The snapshot **manifest** additionally records
  provided-parameter *values* (at least `fr.params.TIME_STEP`);
  `load_snapshot` errors on a dt **sign** mismatch and warns on
  magnitude (the successor of `run_backward`'s sign re-forcing).
- **Ramp endpoints are signed times** (validation, V-S2): backward
  legs run over negative clock times, so a backward Ramp spans
  `[−T, 0]`; **`Ramp.reversed()` reflects the time domain** (a
  naive value-endpoint swap over `[0, T]` clips to a constant for
  t ≤ 0 — no ramp at all).
- **eps is order-2-only** (sign-off note): `AdamBashforth` accepts
  `eps` only at `order=2`; order ≥ 3 warm-up uses textbook AB2
  (deliberate one-step startup delta vs the old code — §8.8).
- **Bitwise-equality umbrella** (2026-07-08, phase-1 finding 1):
  bitwise-equality claims compare **identically-compiled paths
  only** — eager-vs-eager (the operator layer's tested contract),
  or one executable against itself (shared jit-cache entry,
  compile-counter asserted). Comparisons across compilations — old
  framework vs framework2, 1-vs-N devices, changed chunk plans,
  info-augmented vs plain jitted programs — are tolerance-based at
  ≤ a few ulp per step, accumulation-aware. See
  [`06_validation.md`](06_validation.md) §8.8.
