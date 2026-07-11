# Model layer redesign — The state-transform algebra

Part of the model redesign notes; see
[`00_overview.md`](00_overview.md) for the document map. Concept
frame: decision D5 in [`01_concepts.md`](01_concepts.md); research
reports in [`research/`](research/README.md) (d5_1–d5_3).

Status: **resolved (signed off 2026-07-08), with one carve-out:
NNMD is descoped** — it needs a proper rewrite anyway and the future
version will *not* contain a model propagator (per sign-off); the
§10.5 NNMD port spec is retained as archaeology only, and the old
`nnmd.py` is not ported at cutover (2.7). This file is the full D5
design; the D5 section of `01_concepts.md` is the summary.
Requirement (raised at D4 sign-off): first-class composable
`State -> State` objects — projections, time-averaging, ramped
propagation — with an algebra mirroring the operator algebra, e.g.
`optimal_balance_cycle = forward @ vortical @ backward`.

---

## 10.1 The base: `fr.StateTransform`

([d5_1](research/d5_1_algebra.md))

```python
class StateTransform:
    domain:   StateSignature      # (grid identity, mapped (name, space) subset)
    codomain: StateSignature
    traceable: bool               # Tier 1 iff True; ANDs under composition
    idempotent: bool = False      # declared, never detected

    def __call__(self, state) -> State
    def call_with_info(self, state) -> tuple[State, TransformInfo]
    def cost(self) -> TransformCost           # internal model steps; sums
    def __repr__(self) -> str                 # the composition tree
    @property
    def complement(self) -> StateTransform    # Identity() - self; idempotent-gated
```

- **Two tiers**: Tier 1 (closed-form — eigenmode projections,
  filters, `Shift`) are **jaxify-registered frozen pytrees**
  (structure static, numeric captures dynamic leaves — the
  `fr.Ramp` pattern), jit- and vmap-able (IC ensembles:
  `jax.jit(jax.vmap(P))(batch)`); Tier 2 (dynamical — run a model
  internally) are host objects, never pytrees, with a **trace
  guard** raising a taught error on tracer-valued inputs.
- **Info: `call_with_info`, never a mutating attribute** (the
  D1.5-shim argument: silently wrong under jit, order-dependent
  calls). Law: `T(s) == call_with_info(s)[0]` bitwise; info composes
  structurally (a tree mirroring the composition tree); fields:
  iterations, errors, model_steps, elapsed_model_time, extra. The
  old `return_details=True` dies. Progress hooks (`on_progress`,
  `on_iteration`) are host-side observers, forbidden from
  influencing results.
- `cost()`/`repr` answer the cost-opacity problem: the repr is the
  annotated composition tree (tier + step counts).

## 10.2 The algebra

All signatures concrete at construction → **eager checks at
compose time**, plus a call-time recheck (trace-time under jit —
zero steady cost).

| Spelling | Rule |
|---|---|
| `A @ B` — `(A@B)(s) = A(B(s))`, right-to-left | `B.codomain == A.domain` |
| `A + B`, `A - B` — pointwise on outputs | domains AND codomains equal |
| `a * A`, `-A` | plain scalars only (**no Ramp coefficients** — transforms are autonomous maps; no clock in the surface) |
| `A ** n` | `n==0 → Identity(A.domain)`; endo required for n≥2; n<0 raises |
| `Identity()` | signature-polymorphic until composed; elided in chains |
| `Shift(state0)` | `s ↦ s + state0`; Tier 1; the affine piece |

Normalization is **structural only** (flatten nested
Compose/Sum, elide Identity) — no rewriting, no idempotent folding:
what you wrote is what runs. `P @ P` on an idempotent transform
emits an info-level lint hint. No `Zero` transform (no consumer).

**`FixedPoint(T, *, tol=1e-9, max_it=3,
norm=fr.transforms.relative_l2, on_divergence="stop_best",
on_iteration=None)`** — `T` a transform **or a factory
`State -> StateTransform`** evaluated on the current iterate per
iteration (this absorbs OB's `update_base_point` cleanly).
Divergence (`err > prev`): keep the old stop rule, **fix the old
bug** — old OB returned the *worse* iterate; `"stop_best"` returns
the argmin-error one (`"raise"`/`"ignore"` alternatives). Host-level
in it-1 (traced `while_loop` variant designed-for, no consumer).

**The norm**: `fr.transforms.relative_l2` — the old `norm_of_diff`
formula (`2‖a−b‖₂/(‖a‖₂+‖b‖₂)`, volume-weighted over PROGNOSTIC
components) relocated to the transforms layer (respecting D1's
eviction of norms from fields), shipped as the **explicit-kwarg
default** (relative, dimensionless, parameter-free — which is
exactly why the energy norm cannot be default);
`norm=nh.diagnostics.energy_norm(model)` one kwarg away for
dimensional stratified runs. A model-supplied default norm is
rejected (invisible criterion, re-coupling).

## 10.3 The laws

1. **Determinism.** `T(state)` is a pure function of its input given
   frozen config. Tier-2 call semantics, normative:
   `m.reset(); m.set_state(state); m.advance(N); return
   m.state.prognostic` — PROGNOSTIC-only read-back; the final clock
   goes to *info*, never into the returned State; config = parameter
   leaves snapshotted at construction (later parent updates don't
   propagate — documented; `T.with_parameters(...)` designed-for).
   **New load-bearing invariant (02_rules entry + regression
   test)**: `reset()` leaves AUX untouched, so Tier-2 determinism
   relies on **SELF_UPDATE running first in every substage** (the
   owner recomputes time-dependent AUX from the reset clock before
   any consumer reads).
2. **Signatures.** Compared: grid **object identity** + the mapped
   `(name, space)` subset (names by equality, spaces by identity,
   order included). Checked at compose AND call time;
   `SignatureMismatchError` carries the composition-tree path, a
   componentwise diff, and hinted grid verdicts. **Signature ≠
   treedef** — accumulator-augmented variants and
   explicit-params transforms interoperate with model-built ones.
   **The extra-component rule** (reconciled, see §10.7): a
   family-built projection declares its mapped subset (u,v,w,b) plus
   a **`rest` policy** — default `rest="zero"` (components outside
   the family map to zero: completeness holds restricted to the
   family, and `state − P(state)` carries the full tracer);
   `rest="pass"` as the explicit opt-out.
3. **Isolation by construction.** A Tier-2 transform **never stores
   the model you pass it** — the constructor treats it as an
   assembly spec and builds its own internal model (via
   `model.variant(...)`, or — amended V-C6 — a **fresh assembly on
   the parent's frozen grid** when the twin adds declarations, e.g.
   the carry-resident accumulator upgrade; accumulator AUX
   declarations must reuse spaces already in the parent's
   state-space set, or the satisfiability-relaxed verify covers
   them); no injection path exists, so two transforms never share a
   model. Note (amended V-C8): transforms constructed on either
   side of an `update_parameters` embed **different** frozen
   configs — the staleness rule is per-transform, so cross-transform
   consistency is the caller's to ensure (build both, then update
   neither; or `resync()` when it lands).
   (Refinement for presets: OB's exposed leg propagators share the
   *preset's own* two models — sequential, reset-prefixed use is the
   documented exception within one owner.) Passing a pre-built model
   is ownership transfer. Internal models run `io=()`, named
   `"{transform}/internal"` for log attribution.
4. Transforms map the PROGNOSTIC subset; outputs are
   `set_state`-compatible.

**Idempotency**: declared flag; three consumers only — the
`complement` sugar, the `P @ P` lint hint, and a generic
`assert_idempotent` validation check.

**Stages vs Tier-1 transforms** (ruled): a use relation, not
competition — the pressure projection **stays a stage** (not
autonomous; consumes `stage_dt`), but any stage body may call a
Tier-1 transform held as bind-time static structure
(same model, `traceable=True`). Example: a relaxation CONSTRAINT
stage nudging only the balanced part.

## 10.4 Model variants: `model.variant(term_filter=..., updates=..., name=...)`

([d5_2](research/d5_2_variants.md)) Re-assembly on the parent's
**frozen grid** (the verify path) with two ingredients: the term
filter at assembly step 5, and parameter seeding at step 8.

- **Declarations are never filtered** — same FieldTable, State
  treedef, shapes, halos, layouts as the parent: parent↔variant
  state exchange is copy-free (the property every Tier-2 inner loop
  leans on). Stages survive (the linear nonhydro variant still
  projects).
- **Parameter snapshot at variant creation** (live sharing is
  *impossible* — carries are functionally replaced every chunk, so
  "shared" module objects go stale after the parent's first step;
  within one pytree it's the D2 double-flatten bug). Deferred
  creation rejected (assembly in OB's hot loop; frozen config
  becomes a moving target). Refresh is a one-liner;
  `Transform.resync()` designed-for.
- **`updates=` is assembly-time and may change value specs**
  (scalar → `fr.Ramp`, `TIME_STEP` sign) — unlike post-assembly
  `update_parameters`; OB's backward variant requires it. The carry
  treedef may then differ; the load-bearing identity is the State
  treedef.
- **The verify contract is ⊆/≤, stated as a lemma**: filters only
  remove terms → variant demands are a subset → the frozen-grid
  verify path passes **by construction** (variants never hit
  `GridFrozenError`); the variant inherits the parent's
  layouts/halos — possibly oversized, always correct, and what makes
  the exchange copy-free. (Grid-notes amendment owed if the verify
  wording read as equality.)
- **Lint/fingerprint**: coverage lint downgrades error → info under
  any filter (the audit trail stays in the report); double-transport
  stays an error; the filter's canonical token + updates specs enter
  the assembly fingerprint (variant snapshots never silently load
  into the parent — mismatches diff). Empty filter result or unknown
  `named` key = build-time error. Stage filtering: designed-for
  only.
- **Empty implicit partition is graceful**: filtering the mixing
  term under CNAB2 leaves an empty operator set — the solve loop is
  not emitted at trace time and the scheme degenerates to *textbook*
  AB2 (documented: without the AB stepper's eps — same order, not
  bitwise). No stepper change, no demotion.

**The term predicates** (`fr.terms`): five leaves —
`linear`, `explicit`/`implicit`, `owned_by(Type)` (isinstance;
**`fr.closures.ClosureBase` is introduced** as the framework closure
base, earning its keep by hosting D1.4's role-target boilerplate),
`named(*keys)` ("Module/term" attribution keys, validated at build —
unknown keys error), `advancing(*fields)` (splits one module's
terms: Smagorinsky stress vs κ mixing) — combined with `& | ~`.
Frozen reprable expression trees with stable `fingerprint_token()`s
(qualified class names for `owned_by`). Rejected: `transporting`
(suggests per-field masking that term-granular filtering cannot
deliver — "freeze one tracer" is a term rewrite, documented), bare
lambdas (unfingerprintable; a tokened `where(fn, token=)` escape is
designed-for). `fr.linearize(model)` ≡
`model.variant(term_filter=fr.terms.linear)`.

## 10.5 The ported family (`fr.transforms` / `nh.transforms`)

([d5_3](research/d5_3_family_ports.md)) The old `fr.projection`
namespace dissolves. **Homing (amended at validation sign-off,
V-S4)**: the Tier-2 presets (`Propagator`, `TimeAverage`,
`OptimalBalance`, `FixedPoint`, `Shift`) live in **`fr.transforms`**
— shallowwater is a first-class OB consumer — with `base=` resolved
through a **core-module-supplied default-projector hook** (the
D1.3 commitment-4 channel); packages ship thin aliases
(`nh.transforms.*`) and the package-specific Tier-1 projections:

- **`VorticalProjection` / `WaveProjection` /
  `DivergenceProjection`** (Tier 1): thin wrappers over
  `em.projector`-machinery with **dual source constructors** —
  `(model, *, discrete=True, at_time=None)` via `from_model`
  (inheriting all D2.4 validation, incl. the Ramp `at_time=` rule)
  or `(em)` explicit-params. `WaveProjection ≡ P(+1) + P(−1)` — the
  algebra at work. **Nyquist zeroing lives inside `em.q/p`
  materialization** (grid-level), with the documented caveat that
  completeness holds on the Nyquist-free subspace.
- **`fr.transforms.Propagator(model_or_variant, steps=|runlen=)`**
  (Tier 2): run-as-transform; the forward/backward building block.
- **`nh.transforms.TimeAverage(model, *, period=None, n_ave=2,
  equidistant=True, backward_forward=False,
  filter=fr.terms.linear)`** (Tier 2): owns
  `model.variant(term_filter=filter)`; old algorithm ported
  faithfully — flat endpoint-inclusive means, nested passes with
  descending periods (staggered sinc zeros), optional
  backward-forward symmetrization via `TIME_STEP` sign flips;
  `period=None` reads `coriolis.f0` (implementing what the old
  docstring only promised). The user's inviscid averaging is
  `filter=fr.terms.linear & ~fr.terms.owned_by(ClosureBase)`.
  Accumulation it-1: host `advance(1)` loop (old cost, zero
  machinery); upgrades: a carry-resident accumulator module
  (self_update — why signature ≠ treedef matters), then the 2.6
  capture stream. **Parity delta pinned for cutover**: the old
  "linear" twin kept Smagorinsky (nonlinear!) running —
  `fr.terms.linear` correctly drops it.
- **`nh.transforms.OptimalBalance(model, *, ramp_period, base=None,
  ramp="exp", max_it=3, tol=1e-9, update_base_point=True,
  filter=None, backward_filter=None)`** (Tier 2): two owned variants
  with Ramp-valued `scaling.rossby` (up/down) and sign-flipped
  `TIME_STEP` on the backward one; `filter=` threads to both (the
  user's requirement); `backward_filter` is the `mset_backwards`
  successor — recommended content
  `~owned_by(ClosureBase) & ~fr.terms.implicit` (drop dissipation
  rather than sign-flip it; backward diffusion is ill-posed — the
  old "negative viscosity" escape was, grep-verified, never used).
  **Public pieces**: `ob.base`, the two leg Propagators, and
  `ob.ramp_cycle = forward @ base @ backward`; the base-point
  exchange is FixedPoint-factory policy
  (`Shift(z_base) @ (Identity − P) @ ramp_cycle`); details via
  `call_with_info`. Old piecewise-constant θ=n/N ramping becomes
  continuous stage-time Ramp evaluation (tolerance-based cutover,
  expected slight improvement).
- **NNMD: descoped at sign-off** — the old `nnmd.py` is not ported;
  a future rewrite (its own design exercise) will not contain a
  model propagator. The d5_3 archaeology (eigenpair table, the
  `N(z)`-via-variant-tendency mechanism, the quadraticity caveat)
  is retained in the research report for that rewrite's benefit;
  the `model.variant(term_filter=...).tendency(z)` spelling of a
  restricted tendency remains generally available regardless.

## 10.6 Surface additions to resolved decisions

| # | Addition | Owner | Consumers |
|---|---|---|---|
| S1 | **`model.tendency(state, *, t=None, filter=None, constraints=True) -> State`** — host-callable jitted read-only wrapper over the composed tendency (implicit terms via forward apply; constraints optional; never advances the carry) | D4 surface (**applied on sign-off**) | **per-term budget diagnostics** (`filter=fr.terms.named(...)`); term unit tests; linear-stability matvecs; the future TangentPropagator (jvp of exactly this). (NNMD dropped as a consumer — descoped.) |
| S2 | **`model.blank_state()`** + `model.state_space(name)` | D4 surface (**applied on sign-off**) | every IC recipe (D1.1 never named the factory recipes build on); transforms |
| S3 | `em.omega_field(s) -> Field` | ~~2.7 eigenmode surface~~ **deferred with the NNMD rewrite** (its only consumer) | — |
| S4 | `variant(updates=)` may change value specs (assembly-time) | D5 normative text | OB |
| S5 | the variant-verify ⊆ lemma | grid notes (verify wording) | all Tier-2 |
| S6 | signature ≠ treedef | D5 normative text | all compose checking |

Explicitly not needed: trajectory access as D4 surface;
`time_discretization_effect` on transforms (stays recipe-side).

## 10.7 Reconciliations

1. **Info spelling**: d5_3's `ob.trace` attribute sketch is
   superseded by d5_1's `call_with_info` (the trace-attribute is
   exactly the rejected `last_info`).
2. **Signature strictness vs tracers**: d5_1's strict full-table
   equality (with an `OnComponents` adapter designed-for) is
   refined by d5_3's mapped-subset + `rest` policy — adopted: the
   call-time check requires the input to *contain* the mapped
   components; extra PROGNOSTIC components follow the declared
   `rest` policy (`"zero"` default). This subsumes the adapter for
   the common case.
3. **Divergence spelling**: `on_divergence="stop_best"` (d5_1) is
   the kwarg; d5_3's `"rollback"` is the same semantics.
4. Registry-constant naming: `fr.params.SCALING_ROSSBY`
   (`"scaling.rossby"`), per D2.1's dotted scheme.

## 10.8 Residual open points

Signed off 2026-07-08: the `rest="zero"` default, the
`on_divergence="stop_best"` behavior change, and the TimeAverage
Smagorinsky parity delta all stand; NNMD is descoped (§10.5).
Remaining, picked up where noted: the backward-dissipation warning
question (old kept closures active backward — warn when closures
survive onto a backward variant? decide at the OB port, 2.7); the
blessed Ramp endpoint-reversal spelling (`Ramp.reversed()`? — API
sketch item); the JVP linear-tag lint priority (2.5 — `linearize`
now consumes the tag); OB memory (2 carries; the
shared-internal-model alternative conflicts with independent leg
Propagators — revisit if it bites); `fr.terms.where(fn, token=)`,
stage filtering, `OnComponents`, `Transform.resync()`, batched
Tier-2 ensembles, traceable FixedPoint, `em.omega_field` — all
designed-for / deferred.
