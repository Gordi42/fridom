---
status: normative
date: 2026-07-07
---

# Model layer redesign — Staged / split time stepping

Part of the model redesign notes; see
[`00_overview.md`](00_overview.md) for the document map. Concept
frame: decision D3 in [`01_concepts.md`](01_concepts.md); research
reports in [`research/`](../../research/README.md) (d3_1–d3_4).

Status: **resolved (signed off 2026-07-07).** This file is the
full D3 design (ROADMAP 2.5, designed before 2.4 freezes the run
loop); the D3 section of `01_concepts.md` is the summary.

---

## 5.1 The term surface

([d3_1](../../research/d3_1_term_surface.md)) A module contributes
tendency terms as **declared frozen objects** — the FieldDeclaration
pattern applied to behavior:

```python
TendencyTerm(
    name="vertical",                   # "Module/term" is the attribution key
    fn=type(Module).method,            # UNBOUND; None iff implicit set (derived path)
    treatment=EXPLICIT | IMPLICIT,     # closed enum, it-1
    advances=(...),                    # optional-declared, dry-run-verified
    transports=(...),                  # declared-only intent -> D1.4 coverage lint
    implicit=ImplicitOperator | None,  # required iff IMPLICIT
    linear=False,                      # strict; debug-lint verifiable
)
```

- `@fr.term` decorator sugar for the trivial case;
  `Module.tendency_terms()` override for constructed cases (runs
  after `bind(table)`, so `advances` may come from role selections).
- **`fn` is stored unbound** and paired with a module *slot* at
  compose time — a bound method would capture the assembly-time
  module instance while live parameters ride the carry (the D2
  aliasing trap). The composer calls
  `term.fn(carry.modules[slot], state, ctx)`.
- **Treatment is author-declared; the user override lives on the
  module constructor** (`VerticalMixing(kv=..., treatment=
  fr.IMPLICIT)` — the Oceananigans `time_discretization=`
  precedent). Model-level override dicts rejected. Steppers declare
  `supported_treatments`; an implicit term under a purely explicit
  stepper is an **assembly error, never silent demotion**.
- **Write-once**: a term with `implicit=op` may omit `fn` — the
  explicit path is derived from `op.apply`, so flipping treatment
  cannot desynchronize the two.
- Terms only ever **add** (composer accumulates via
  `VectorField.add`); anything that overwrites is a *stage* (§5.2).
- The **TendencyComposer** owns all attribution: assembly dry run
  per term on halo tracers (write-gate/key/space validation with
  module+term names, `advances` cross-check, implicit-collision and
  treatment/stepper checks), trace-time `TermEvaluationError`
  wrapping, and deterministic accumulation order (module order,
  declaration order).

**The implicit operator** (two capabilities, nothing more — the
minimal surface every IMEX family in §5.4 needs):

```python
class ImplicitOperator(Protocol):
    fields: tuple[str, ...]                        # advanced subset, mandatory
    def apply(self, module, state, ctx) -> dict    # L·state (forward; CNAB rhs, derived path)
    def solve(self, module, rhs: dict, dt_gamma, ctx) -> dict
        # (1 - dt_gamma·L)^{-1} rhs; γ-agnostic — the scheme owns γ; dt_gamma traced
```

- **Mergeable framework families**:
  `fr.implicit.VerticalDiffusion(axis, fields, kappa=fn)` — same-axis
  operators touching a field merge by **summing κ** into one
  tridiagonal solve (sequential opaque solves would be Lie splitting
  inside an IMEX stage — order-degrading); boundary rows from the
  field's declared space BCs; flux BCs are explicit forcing.
  `fr.implicit.SpectralDiagonal(symbol=...)` merges by summing
  eigenvalues (iteration-2). **At most one non-mergeable custom
  implicit operator per field** (`ImplicitCollisionError`).
- The solve is a grid-bound registry **Operator** declaring
  `layout local along the solve axis` — negotiated like transforms,
  intercepted generically by `HaloTracer`, no `.data` bypasses.
  Retires the `RFFTPressureSolver` hand-sharding pattern.
- Coupled blocks (`fields=("u","v")` semi-implicit Coriolis) are
  **atomic** under by-variable splitting (§5.4).
- The **CN solve-only trick is unsound** here (recovering `L·Xⁿ`
  from the previous solve reads the pre-projection state — an O(dt)
  error every step): the forward apply is mandatory.

## 5.2 The stage schedule

([d3_3](../../research/d3_3_stage_schedule.md),
[d3_4](../../research/d3_4_imex_splitting.md)) The canonical step:

```
for each stepper substage i (incl. a separate final combination):
  P0  ctx_i = StepContext(eval_params(modules, t_i), clock@t_i, dt, stage_dt_i)
  S1  SELF_UPDATE stages     (scheduled iff an input is time-dependent; own AUX)
  S1' DIAGNOSE stages        (pre-tendency DIAGNOSTIC writes: p_hyd = ∫b dz,
                              and — validation V-H1 — hydrostatic diagnosed
                              w = −∫∇·u dz; S1'-placement is load-bearing:
                              the first substage after set_state/restart
                              recomputes them before any term reads)
  S2  tendency terms         (EXPLICIT contributions -> composer -> add)
  S3  primary ADVANCE        (the stepper: explicit combine + IMPLICIT solves)
  S3' additional ADVANCE stages   (module-owned, by-variable: barotropic subcycle)
  S4  CONSTRAINT stages      (replace velocities; write own DIAGNOSTIC p)
per-step epilogue:
  S5  NaN check -> carried panicked flag        (mechanism owned by D4)
  S6  DIAGNOSTIC stages      (step-cadence accumulators)
  clock tick; host boundary (IO/progress/restart) is D4's
```

- **Kind vocabulary (closed, it-1)**: `SELF_UPDATE`, `DIAGNOSE`,
  `ADVANCE`, `CONSTRAINT`, `DIAGNOSTIC` — plus terms, which are not
  stages. A **stage body is an arbitrary pure function
  `(module, state, ctx) -> dict`** writing its declared subset (the
  generalization the split-explicit subcycle and a Phase-3 Coupler
  need); dicts are applied per the kind's write gate (`replace`;
  terms: `add`).
- **Ordering discharges D1.3 commitment 5**: schedule position is a
  pure function of declared *kind*; within a kind
  `(order: int = 0, module tuple index, declaration index)`.
  **Correctness never depends on list position** — an assembly lint
  errors on same-kind stages with overlapping write (or write-read)
  sets and equal `order=`. Bitwise determinism *may* tie-break by
  module order (permuting the tuple already changes the treedef).
- **User extension**: the old `add_module`-before-the-trio hack
  becomes a theorem (terms precede constraints by kind); users own
  stages by declaring a kind (positivity clamp → CONSTRAINT;
  accumulators → DIAGNOSTIC; geometry → SELF_UPDATE). No
  "insert before X" API; no open kind set.
- **Read rule (discharges fed-forward d)**: any read sees the
  **nearest preceding write in this schedule, crossing substage and
  step boundaries** — `state["p"]` in a term sees the previous
  substage's (or step's) projection; Gauss-Seidel across ADVANCE
  stages is this rule applied to PROGNOSTIC components.
- **The `div` ruling**: declare DIAGNOSTIC **iff read outside the
  producing stage** (later stage, step-cadence IO, warm start
  across steps). `p` qualifies (warm start + IO); **`div` is not
  declared** — it is a local variable of the projection stage;
  incompressibility monitoring is `nh.diagnostics.divergence(state)`
  at output cadence. *(Amendment owed on sign-off: D1.3/D1.5's
  "(p, div)" wording → p only.)*
- **self_update runs per substage at substage time** — free in the
  micro-schedule, stage-consistent with `eval_params` (a ramped
  scalar and a ramped N² profile see the same time; once-per-step
  would inject an O(dt²·dN²/dt) error). A `cadence=STEP` opt-in for
  expensive updates is recorded, not built. **Scheduling trigger
  (amended, validation V-H5)**: a module's self_update is scheduled
  iff one of its inputs is time-dependent **or** its declaration
  names state inputs (`reads=("eta",)`, assembly-checked like a
  `FieldReference`) — covering state-derived AUX (z*-geometry
  following `eta`), which the Ramp-shaped rule alone would never
  schedule.
- **NaN seam at S5**: once per step, post-constraint (the committed
  state is checked), pre-DIAGNOSTIC (keeps garbage out of
  accumulators); reduces into the carried `panicked` flag consumed
  by D4's `lax.cond` no-op wrapper and chunk-boundary abort.

## 5.3 The stepper core

([d3_2](../../research/d3_2_stepper_core.md)) The stepper is **not a
Module but is a pytree** whose only dynamic leaf is `dt`; evolving
state lives in a `StepperState` carry entry. Carry =
`(state, modules, stepper_state, clock)`; the stepper object is a
loop-invariant traced input.

```python
class TimeStepper:
    def init(self, tendency_template) -> StepperState
    def step(self, stepper_state, state, schedule, clock)
        -> (stepper_state, state, clock)          # owns clock.tick(dt)
    def time_discretization_effect(self, omega, *, dt=None)   # host-side analysis
```

- **Warm-up under scan**: a dense zero-padded (order × order)
  coefficient table row-indexed by a **carried saturating int32
  counter** — the direct traced transcription of the old
  `update_coeff_AB`. Unroll-first-K-steps is rejected (special first
  chunk; breaks bitwise mid-warm-up restart). The counter is
  stepper-local, not `clock.it` (`reset()` re-warms — OptimalBalance
  depends on it). Ring buffers are **tuples of States shifted
  structurally** (dataflow renaming, newest first).
- **AB parity** (cutover, 2.7) and **the eps ruling (amended at
  validation sign-off)**: `eps` is an **order-2-only parameter** —
  the quasi-AB2 computational-mode damper; `AdamBashforth(order=2,
  eps=0.01)` accepts it, any other order rejects it, and **order ≥ 3
  warm-up uses textbook AB2** `[3/2, -1/2]`. This is a deliberate
  startup-only delta vs the old code (which applied its eps'd AB2
  row as every order's warm-up row — one step, O(eps·dt),
  tolerance-based cutover; recorded in the §8.8 parity list).
  Bitwise rules otherwise unchanged: `weights = row * dt`
  premultiplied, ascending-j accumulation, tendency evaluated at
  the pre-tick time.
- **RK**: `StepperState = ()` (stage values are locals — only
  multistep memory earns carry); stage clocks via
  `clock.shifted(c_i·dt)`; adaptive/embedded RK excluded from it-1
  (designed-for: dt into StepperState + bounded `while_loop`,
  diffrax as reference; keep the `b_error` tableau data).
- **dt is a dynamic scalar leaf on the stepper** — not static (dt is
  the sweep parameter par excellence; sign flips give backward
  runs), not carry (adaptive-only). `float | np.timedelta64`
  converted once at the constructor boundary. Read surfaces:
  `ctx.dt`/`ctx.stage_dt` in-trace; the live leaf host-side
  (registrable as `fr.params.TIME_STEP` for `cfl`).
- **Clock**: traced `start/elapsed/it` (**float64 even in float32
  runs** — 02_rules entry; float32 seconds lose sub-dt resolution
  within hours); `time = start + elapsed`; calendar
  (`start_date: datetime64`) strictly host-side.
  **`run_backward` is dropped as a method, kept as a capability**:
  the primitive is signed dt (verified: OptimalBalance never called
  run_backward — it flips the dt leaf and drives steps).
- **`time_discretization_effect`** survives as a host-side stepper
  method (full-order row incl. eps; np.roots on CPU) with an
  explicit `dt=` override; `omega` is a plain array — Symbol callers
  materialize first; applied by recipes (`single_wave`), per D2.4.
- **Run shape (assumption handed to D4)**: chunked scan; chunk
  boundary = host sync (writer flush, snapshot — the carry is the
  checkpoint, NaN abort, progress); steps-count runs primary,
  time-target reduces to steps at fixed dt; the carried warm-up
  counter makes the first chunk the same trace as every other.

## 5.4 Split integrators: IMEX by term, Gauss-Seidel by variable

([d3_4](../../research/d3_4_imex_splitting.md)) Unifying form
`∂t X = F(X,t) + L·X`. What each family needs from the §5.1 surface:

| Family | solve(γ) | forward `L·X` | F history | X history | eval_params |
|---|---|---|---|---|---|
| AB1–4 | — | only if forced explicit | s−1 | — | 1/step |
| CNAB2 | γ=½ | yes (one/step) | 2 | — | 1/step |
| SBDF2/3 | γ=⅔ / 6⁄11 | no | s | **s−1 states** | 1/step |
| IMEX-RK (ARS) | diagonal γ | optional | within-step | — (self-starting) | per stage |

- **Buffers partition by treatment**: the explicit ring buffer
  stores the **summed** explicit contribution (per-term history is
  never needed — the partition is consumed at accumulation time);
  the implicit side buffers nothing; SBDF adds the one new buffer
  class (past states). Buffers are **owned per ADVANCE stage**
  (the barotropic inner integrator owns its own). Warm-up switches
  whole (weights, γ) tuples — γ changes across warm-up levels, so
  `dt_gamma` is necessarily traced.
- **By-variable stages**: module-owned ADVANCE stages advance named
  PROGNOSTIC subsets reading the latest state (Gauss-Seidel via the
  §5.2 read rule). **Split-explicit free surface = ordinary
  PROGNOSTIC fields, decisively**: the FreeSurface module declares
  `eta, U, V` (2D, constant-along-z spaces) and owns
  ADVANCE({eta,U,V}) — a `lax.scan` over N static substeps with
  time-filtering, slow forcing read from `ctx`'s per-treatment
  tendency sums — plus CONSTRAINT({u,v}) for the depth-mean
  correction. The nested-mini-model alternative is rejected (hidden
  η, nested restart, broken treedef discipline); a genuinely
  separate model (Phase-3 coupling) is where nesting earns its keep.
  **`U, V` carry no Velocity role** (diagnostically-slaved
  transports; `table.velocity()` keeps returning the baroclinic
  trio) — resolving the D1 residual without the `group=` qualifier.
- **StepContext carries the per-treatment tendency sums** to
  post-TENDENCY stages — the second consumer of the
  contribution-dict partition. **The barotropic slow forcing
  (amended, V-H4)**: the default is the **increment form**
  `G = ∫(X* − Xⁿ) dz / dt` computed by the stage from the
  substage-start state (buffered via the ADVANCE own-AUX gate) —
  automatically consistent with the outer scheme's weights and
  warm-up row, and including the implicit-mixing increment by
  construction; the raw per-treatment-sums variant is the
  constructor knob (`forcing="tendency_sums"`).
- **Lint amendment (load-bearing)**: stage `advances` claims count
  as "advanced" in the D1.4 coverage lint (else `eta, U, V` fail
  assembly).
- **Where composition honestly ends**: IMEX-RK × split-explicit has
  no production precedent → assembly error (multistep outer drivers
  only); which pieces enter the barotropic fast forcing is a module
  constructor knob; ROMS-style pipelined predictor-correctors are a
  custom integrator against the same term surface — never a Model
  subclass.
- The **non-split alternative** falls out of the same surface: an
  implicit free surface is `grad η` as an IMPLICIT term whose solve
  is a 2D Helmholtz solve.

## 5.5 Signatures: the StepContext

([d3_3](../../research/d3_3_stage_schedule.md), confirming d3_1's
position; discharges fed-forwards f and g) Every in-trace hook is
`(self, state, ctx) -> dict`:

```python
@fr.utils.jaxify                    # frozen, all-scalar leaves
class StepContext:
    params:   Mapping[str, Array]   # eval_params(modules, stage_time)  [D2]
    clock:    Clock                 # traced; .time == stage time
    dt:       Array                 # full step size        (cfl's read surface)
    stage_dt: Array                 # increment of the current advance
    # post-TENDENCY stages additionally see the per-treatment tendency sums
```

| hook | returns | applied via |
|---|---|---|
| term `fn` | `{prognostic: increment}` | `add` (accumulated) |
| implicit `solve` | `{field: solved}` | stepper (γΔt positional) |
| `self_update` | `{own aux: field}` | `replace` |
| module-owned ADVANCE stage | `{declared advanced PROGNOSTIC subset} ∪ {own aux}` | `replace` (amended V-H3: own-AUX is the declared home of stage-owned cross-step integrator state; the stage's integrator *statics* — substep count, filter spec — join the restart fingerprint) |
| CONSTRAINT stage | `{prognostic/own diag: field}` | `replace` |
| DIAGNOSE/DIAGNOSTIC stage | `{own diag: field}` | `replace` (a DIAGNOSTIC stage may read its own previous value — **the accumulation idiom**, blessed at the coupling sign-off; the sanctioned home for step-cadence sums, replacing `self_update` in that role — 02_rules) |

One validation path, one halo-trace path (the tracer wraps *state*;
ctx is scalars — zero mimicry cost); no hook mutates or returns
`self` (step-evolving module data that isn't a parameter leaf is a
declared AUXILIARY/DIAGNOSTIC field). D2.3 reconciliation stands:
**kwargs at the notebook boundary, ctx inside the trace**; the
binding layer converts. This amends D1.5's `tendency(self, state)`
as D2 reconciliation 5 anticipated.

## 5.6 The projection: project-the-state

([d3_3](../../research/d3_3_stage_schedule.md) §1, verified against
Oceananigans source; [d3_4](../../research/d3_4_imex_splitting.md) §5)
The old code projects the *tendency* (the trio acts on `dz`); the
new design **projects the state, after every state-producing
advance**:

- **Exactly equivalent** to the old scheme for explicit AB/RK when
  the state starts divergence-free (P linear + idempotent) — cutover
  parity is safe;
- **self-correcting** (IC/restart/roundoff divergence removed within
  one substage — the old scheme preserved it forever);
- **the only coherent IMEX arrangement** (an implicit solve does not
  preserve divergence-freeness; solve, then project — standard
  fractional-step). IMEX-RK projects per stage; multistep once per
  step; history buffers store unprojected tendencies.
- The written diagnostic is `p := φ / stage_dt` (the physical
  pressure to O(dt) — Oceananigans does exactly this, verified);
  document for pressure-budget users.

## 5.7 Iteration-1 scope

Freeze the abstractions now (treatment tags, the two-capability
implicit surface, stage kinds + advances claims, per-stage
eval_params, StepContext, per-ADVANCE buffers); ship incrementally:

- **Ship in 2.5**: AB1–4 + eps at parity, explicit RK at parity
  (low-storage RK3 as the flagship default), **one generic
  `IMEXMultistep` driver with CNAB2 + SBDF2 coefficient sets**, and
  a reference vertical-diffusion implicit consumer validated against
  an exact 1D decay solution and a stiff-κ column test — debugging
  the implicit surface on a toy before the hydrostatic port depends
  on it.
- **Design-freeze only**: IMEX-RK tableau slots (ARS(2,2,2)/(4,4,3)
  — build 2.7/3.1); the split-explicit stage (build at 3.1);
  SBDF3; the `group=` velocity qualifier; adaptive dt
  (dt into StepperState + bounded while_loop).

## 5.8 Reconciliations (where the reports disagreed)

1. **Stage kinds** (d3_3: SELF_UPDATE/CONSTRAINT/DIAGNOSTIC with the
   advance as bare stepper math; d3_4: + DIAGNOSE + module-ownable
   ADVANCE): union adopted — DIAGNOSE is needed by the hydrostatic
   `p_hyd` (a pre-tendency diagnostic write d3_3's vocabulary had no
   slot for), and ADVANCE-as-kind is what lets the barotropic
   subcycle be module-owned. d3_3's ordering/determinism/lint rules
   apply unchanged to the enlarged vocabulary.
2. **Implicit-collision strictness** (d3_4: one implicit term per
   field; d3_1: mergeable families + one non-mergeable custom):
   d3_1 adopted — coefficient-sum merging is exact, so the lint
   applies to non-mergeable operators only.
3. **solve signature** (d3_3 wanted γΔt from `ctx.stage_dt`):
   `dt_gamma` stays a separate positional (d3_1/d3_4) — it is
   γ-specific (CN: dt/2; SBDF2: 2dt/3), *not* the stage increment;
   `ctx.stage_dt` exists for the pressure normalization.
4. **self_update cadence** (d3_2 assumed once per step; d3_3 argued
   per substage): per substage adopted; resolves d3_2's
   aux-staleness risk in the accurate direction.
5. **StepContext** confirmed D3-wide (d3_1 proposed, d3_2 compatible,
   d3_3 argued, d3_4 extended with per-treatment sums).

## 5.9 Residual open points

Carried in [`07_open_threads.md`](07_open_threads.md): the
explicit-`order=` vs topological-sort upgrade for DIAGNOSTIC chains;
`add_prognostic` (key-aligned add of PROGNOSTIC-only vectors —
small fields.md follow-up); the restart-fingerprint rule
consolidation (per-term treatments + stepper statics + Ramp specs +
module tuple); backward-run sign conventions through `stage_dt`;
initial host-side projection of non-divergence-free ICs (D4/IC);
`update_parameters` leaving s−1 stale buffered tendencies (D4
lifecycle note; optional counter re-ramp); chunked-scan buffer
donation; the float64-clock rule and the diagnostics-metadata rule
(02_rules).
