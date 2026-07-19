---
status: done
date: 2026-07-18
---

# General time-dependent fields — implementation plan

**Goal (owner, 2026-07-14 request; direction ratified in chat
2026-07-18):** close the general half of the roadmap entry
"Time-dependent parameters and time-dependent fields"
([`../../roadmap/open.md`](../../roadmap/open.md)): a *profile* that
itself evolves — `f(y, t)`, `csqr(y, t)` — with **non-affine** time
dependence, beyond the affine-blend subset shipped 2026-07-17
([`../done/adiabatic_ramping.md`](../done/adiabatic_ramping.md) R1/R2).
Also in scope, because the same honesty seam serves both: the open
AR-D7 follow-up (`dsqr`'s frozen-`L` report is cross-module and
unwired — a live silent-wrongness hole under `ETDRK4` today), and the
mandatory answer for what `ETDRK4` does about time dependence in `L`.

Companion records:
[`../../roadmap/open.md`](../../roadmap/open.md)
§"Time-dependent parameters and time-dependent fields" (the driving
entry); [`../done/adiabatic_ramping.md`](../done/adiabatic_ramping.md)
(R1 scalars, R2 `FieldBlend`, AR-D7);
[`../../research/exponential_stepper.md`](../../research/exponential_stepper.md)
§5 (the measured frozen-`L` fallback);
[`../../research/d2_2_representation.md`](../../research/d2_2_representation.md)
R1 (field-vs-scalar representation rule);
[`../../research/ale_on_fv.md`](../../research/ale_on_fv.md) and
`src/fridom/model/modules/moving_geometry.py` (the shipped
`(coords, t)`-recompute precedent this plan generalizes);
[`../../specs/model/03_time_stepping.md`](../../specs/model/03_time_stepping.md)
§5.2/§5.5 (schedule slots, `StepContext`);
[`../../specs/model/02_rules.md`](../../specs/model/02_rules.md)
(V-H5, V-N2, accumulation idiom);
[`../../specs/model/classes/declarations.md`](../../specs/model/classes/declarations.md)
(the `TimeDependent`/`resolve_at` contract).

## 1. The design space, and why this shape

Four options were investigated (research session 2026-07-18; evidence
below is file:line-verified on dev):

- **(A) SELF_UPDATE rewrite** — the owning module re-evaluates a user
  law at the substage clock and rewrites its carry-resident AUXILIARY
  field. Shipped precedent: `MovingGeometry`
  (`src/fridom/model/modules/moving_geometry.py`) does exactly this
  for mapping parameters (`H`, `Y_N`): one AUXILIARY field per
  parameter plus a `<name>_dot` companion, a
  `Stage(kind=StageKind.SELF_UPDATE, ...)` (S1, first in every
  substage, `stages.py:51-53`) re-evaluating a static user callable at
  `ctx.clock.time` (traced, exact per RK substage —
  `model.py:611-625` builds each substage context from
  `clock.shifted(c_i*dt)`), one `jax.jvp` in `t` yielding value and
  derivative together (`moving_geometry.py:293-340`), write-back via
  `with_data` claiming **zero ghost validity** so the carry treedef is
  step-stationary (`moving_geometry.py:27-33`), `extra_halo =
  HaloSpec({})` per V-N2, halo fill left to consumption (GAP-B).
- **(B) declaration-level recompute contract** — same lowering,
  synthesized by the composer from a `FieldDeclaration` attribute.
  Rejected as machinery (declarations are static data; synthesizing
  stages crosses a layering line for three consumers' worth of need)
  but its *marker* is adopted: the declaration carries a
  `time_dependent` flag so lints and guards can see field-carried time
  dependence structurally.
- **(C) read-side stage-time evaluation** (generalize `FieldBlend`'s
  `_stage_blend_f` pattern; carry field stays a t=0 snapshot).
  Rejected for the general case: it only works when the owner is the
  sole consumer. `csqr` has four cross-module consumers reading
  `state["csqr"]` directly (sadourny mass flux
  `shallowwater2/modules/sadourny.py:648`, thickness-weighted Coriolis
  `shallowwater2/modules/coriolis.py:293`, energy metric
  `model/energy.py:342-344`, diagnostics
  `shallowwater2/diagnostics.py:119`) — each would silently read the
  stale snapshot or need bespoke override plumbing. The known
  "`f_coriolis` IO shows the t=0 snapshot" wart from the R2 landing is
  this failure mode in miniature.
- **(D) separable subset** — `f(y,t) = sum_i w_i(t) P_i(y)` with
  non-affine `TimeFunction`/`TimeSeries` weights already works
  mechanically through `FieldBlend.evaluate`
  (`field_blend.py:216-255` resolves any `TimeDependent` weight at
  stage time); the AR-D1/D2 "affine" restriction is about the
  adiabatic-path guarantee, not the machinery. Documented as the cheap
  escape hatch; not the general answer (non-separable profiles do not
  fit; tabulated time bases cap temporal convergence order).

**Chosen: A with B's marker.** Every consumer — tendencies, energy
metric, diagnostics, IO, restart — sees stage-time values with zero
changes, because they already read the field live from `state[...]`;
the constant-in-time assumption lives entirely in the setup-baked
layer (§5). The per-substage recompute is idempotent in `t`, so the S1
accumulation hazard (`stages.py:194-206`) does not bite, and the cost
is trivial for `Profile` fields (1-D), which is what the d2_2 R1
representation rule prescribes for these consumers anyway.

## 2. Decisions

**TDF-D1 — Mechanism: owner-declared SELF_UPDATE recompute; the
MovingGeometry sampling discipline becomes shared machinery.** The
value/derivative sampling core of `MovingGeometry._sample`
(arity-dispatched coords, one `jax.jvp` in `t`, `with_data` write-back
with zero ghost claim) is extracted into a reusable helper in
`src/fridom/model/` (suggested home: a new `scheduled_field.py`, or
`time_dependent.py` if it stays small). `MovingGeometry` is
re-expressed on the helper **bitwise** (its tests are the regression
gate). The `_dot` companion is opt-in (`MovingGeometry` needs it;
Coriolis/`csqr` do not).

**TDF-D2 — User surface: `ProfileFunction`, a static law with dynamic
leaf parameters.** The field analogue of `TimeFunction`
(`time_dependent.py:329-377`): `ProfileFunction(fn, params=())` where
`fn(*coords, t, *params)` is a **static descriptor** (change of law =
one recompile) and `params` are jaxified dynamic leaves (sweeps never
recompile; `jax.grad` through them works). This deliberately improves
on `MovingGeometry`'s identity-hashed closures, matching the Ramp
compile-stability invariants
(`tests/model/test_time_dependent.py::test_endpoint_and_timing_sweeps_do_not_recompile`).
Coordinates are the field's own nodes from its declared space
(`Profile("y")` -> the 1-D `y` array), pointwise, zero stencil.
Like the scalar `TimeDependent` family, `ProfileFunction` supports no
field/array arithmetic — a consumer forgetting to sample fails loudly.

**TDF-D3 — Declaration marker + lint.** `FieldDeclaration` gains
`time_dependent: bool = False` (AUXILIARY-only; default keeps every
existing declaration untouched). The composer lints: every
`time_dependent`-marked field must be written by a SELF_UPDATE stage
of its owner. To make the lint exact, `self_update`/`Stage` gain an
optional `writes=` tuple (default `None` = undeclared; the lint then
falls back to "owner has at least one SELF_UPDATE stage" and a
docstring note). `MovingGeometry` adopts `writes=` and the marker.
Note the coverage lint (`composer.py:583-617`) covers PROGNOSTIC
fields only — without this marker nothing structurally guarantees a
"time-dependent" AUXILIARY field ever updates; that is the lint gap
the roadmap entry names.

**TDF-D4 — Frozen-`L` guard becomes structural: linear terms declare
what their linearity depends on.** Today's guard
(`_check_time_dependent_linear`, `assembly.py:115-139`) unions a
per-module hook `time_dependent_linear_parameters()` (module.py
default, overridden in `coriolis.py:515-525`, `:701-714`,
`stratification.py:111`) — per-module `isinstance` bookkeeping that
cannot see cross-module ownership (the `dsqr` hole: `DynamicalCore`
owns it, `ConstantStratification.buoyancy_force` consumes it in a
`linear=True` term at `stratification.py:128-132`, neither reports)
nor field-carried time dependence. Rework:

- `@fr.model.term(..., linear=True)` gains `linear_params:
  tuple[str, ...] = ()` and `linear_fields: tuple[str, ...] = ()` —
  the declared parameter names / field names the term's linear
  operator depends on. Annotating a name that resolves to a plain
  constant is free (the guard only fires on actual time dependence),
  so annotations list every *potentially* time-dependent dependency.
- The guard resolves each `linear_params` name through the binding
  table to its provider leaf (same seam as `eval_params`,
  `assembly.py:524-558`) and reports `isinstance(leaf,
  TimeDependent)`; each `linear_fields` name resolves to its
  `FieldDeclaration` and reports the TDF-D3 marker.
- **The sweep runs over declared terms of present modules, NOT
  schedule-included terms.** Under the ETDRK4 contract the linear
  terms are exactly the ones filtered *out* of the assembled N-model
  (`term_filter=~terms.linear`, `exponential.py:39-72`), so a
  schedule-scoped sweep would see nothing. This preserves today's
  behavior (the beta-plane refusal test asserts the raise on a model
  assembled with the filter). A module whose linear term is
  deliberately absent from both L and the run can hit a false
  positive; the error message explains the situation. Accepted.
- Suggested lowering: keep the assembly call-site and the
  `time_dependent_linear_parameters()` hook name, but make the
  **base-class default implementation structural** (walk the class's
  declared terms' annotations, resolve leaves/declarations); delete
  the hand-written overrides in `coriolis.py` and
  `stratification.py`. Error type, message format, and the
  `exponential_stepper.md` §5 citation stay exactly as shipped
  (`errors.py:197-247`).
- Annotations to add in this pass: `ConstantStratification.
  buoyancy_force` -> `(N2, DSQR)` (closes the hole);
  the Coriolis rotation terms -> their `f0`/`beta` params and the
  `f_coriolis` field; shallow-water linear terms consuming `csqr` ->
  the `csqr` field; sweep every `linear=True` term in
  `model/`, `nonhydro2/`, `shallowwater2/` and annotate.
- `DynamicalCore` **additionally** reports `dsqr` itself: `dsqr`
  enters the frozen eigenbasis through the pressure projection
  (`core.py:608,659,712`), which is a CONSTRAINT stage, not a term,
  so the term sweep alone would miss a ramped `dsqr` in a model
  assembled without stratification. Owner-side report, one comment
  explaining why.

**Lowering note (Wave 1, as shipped).** The structural sweep lives in
the **base-class default of `Module.time_dependent_linear_parameters`**
(module.py), not in the assembly guard: the guard call-site
(`assembly.py:1702`), the per-module `(module, name)` offender
attribution, `TimeDependentLinearOperatorError`, its message format, and
the `exponential_stepper.md` §5 citation are all **unchanged** — the
error surface stays byte-compatible. Chosen because the hook is called
directly (unassembled) by the module unit tests
(`FPlaneCoriolis(f0=Ramp).time_dependent_linear_parameters()`), which
requires the hook itself to resolve time dependence, and because the
guard runs at assembly **step 2, pre-bind** — the binding table exists
but the modules are not yet bound, so a `tendency_terms()` override that
needs bind cannot be walked there. Consequences:

- Resolution is **module-local**: the default walks the class's
  `@fr.term` declarations (bind-free) and resolves each `linear_params`
  name to the module's own leaf (via `parameter_declarations`, falling
  back to the leaf whose attribute is the name's last dotted segment —
  this reaches the beta-plane's *consumed-but-not-provided* `f0`
  offset), and each `linear_fields` name to the module's own
  `FieldDeclaration`.
- **Cross-module** dependencies (a term that consumes a leaf owned by
  another module) are out of the local default's reach, so the
  annotation there is inert; the **owning** module reports them. This is
  exactly the `dsqr` case: the stratification term's `DSQR` annotation
  is inert, and `DynamicalCore` reports a ramped `dsqr` from its own
  leaf (which also covers the no-stratification model).
- Terms built in a **`tendency_terms()` override** (diffusion, vertical
  mixing, moving geometry, perturbation-advection background) rather than
  a `@fr.term` method are not walked by the default and stay unguarded,
  exactly as before this change; a module that needs the guard for such
  a term overrides the hook (`DynamicalCore` is the precedent).
- Annotations are placed **per term, truthfully**, except
  `ConstantStratification.buoyancy_force`, which carries both `N2` and
  `DSQR` per the decision above (the `restoring` term physically reads
  `n2`, but the module-level report is what matters and this keeps the
  two coupling parameters declared on one term).

**TDF-D5 — ETDRK4 semantics: refuse, never auto-split.** A
time-dependent `L` has no fixed eigenbasis (the discrete eigenanalysis
is undefined in that regime), and the owner's standing preference is
explicit honest flags over auto-magic that could silently change
physics. `TimeDependentLinearOperatorError` stays the answer; its
message already cites the measured fallback (gravity-only eigenbasis,
rotation in the tendency, still 52.7x AB3 —
`exponential_stepper.md` §5).

**TDF-D6 — Setup-baked consumers get the same honesty.**
`PolarizedWaveMaker` extends its existing Ramp-refusal
(`polarized_wave_maker.py:232-252`) to `ProfileFunction`-valued
parameters and `time_dependent`-marked fields. The eigen/analysis
tools (`eigenbasis`/`ChannelEigenmodes` `at_time` snapshot,
`Eigenmodes._read` `at_time`, energy weights, `pot_vort`) stay
deliberately time-frozen analysis surfaces; their docstrings state the
snapshot semantics. (The energy-weight snapshot is no longer the
*default*: state-sourced field weights replaced it 2026-07-19, TDF-D10,
[`td_fields_followups.md`](td_fields_followups.md) — the eigen tools
themselves stay `at_time`-frozen, unchanged, and keep the explicit
`snapshot=True` metric.) No re-diagonalization contract in this plan.

**TDF-D7 — Consumers in scope: `f` and `csqr`.** The Coriolis family
gains a law-valued `f` path (the profile-owning class whose
`f_coriolis` lives on `Profile("y")`; `FPlaneCoriolis`'s scalar and
`BetaPlaneCoriolis`'s affine-blend paths stay bit-for-bit untouched —
the new path activates only when a `ProfileFunction` is passed).
Shallow-water `DynamicalCore` gains the same for `csqr` (both the
1-DOF and the meridional-profile variants,
`shallowwater2/modules/core.py:275-310`). The t=0 materialization
(`default=` builder) samples the law at `t=0.0`, mirroring
`_f_default`'s `resolve_at(self.f0, 0.0)` spelling. Nonhydro `n2(z,t)`
is a named follow-up, not in this plan.

**TDF-D8 — Differentiability policy applies.** The recompute is
step-path code: each consumer wave ships the standard autodiff
regression (grad of a quadratic loss through a short `_chunk_body`
run w.r.t. a `ProfileFunction` param leaf, finite and matching a
central finite difference to rtol 1e-4). The sampling helper must not
introduce masked singularities (plain evaluation; no clipping).

**TDF-D9 — FieldBlend is NOT unified onto the rewrite path.**
(Reversed by owner 2026-07-19, see
[`td_fields_followups.md`](td_fields_followups.md) TDF-D11.) AR-D2
(term-side evaluation for the affine blend) stands; re-basing
`f_coriolis` on a SELF_UPDATE rewrite would fix the IO-staleness wart
there too but relitigates an owner-ratified ruling and changes tested
behavior. Logged as an open question for the owner, nothing more.

## 3. Stages

**Wave 1 — guard rework (`refactor/linear-term-guard`).** TDF-D4 in
full: term annotations, structural default hook, delete hand-written
overrides, `DynamicalCore` dsqr report, tests. Gate: existing refusal
tests stay green with unchanged messages
(`test_coriolis.py::test_beta_plane_etdrk4_refuses_a_ramped_beta`,
`test_exponential.py:296-313`, `test_assembly_pipeline.py:520`); NEW
test: `nh.DynamicalCore(dsqr=Ramp(...))` + ETDRK4 raises, naming
`dsqr`; a ramped `dsqr` under `AdamBashforth` still assembles and
runs. Landable alone; fixes a live silent-wrongness bug.

**Wave 2 — mechanism + consumers (`feat/time-dependent-fields`).**
TDF-D1/D2/D3/D7/D8: helper extraction (MovingGeometry bitwise),
`ProfileFunction`, declaration marker + lint (+ `writes=`), the
Coriolis `f(y,t)` and shallow-water `csqr(y,t)` paths, and the
TDF-D4 `linear_fields` wiring so ETDRK4 refuses a scheduled `f`.
Tests mirror the R1/R2 suite shapes (`test_field_blend.py` is the
template): stage-time correctness against a hand-stepped AB3 oracle,
carry-treedef stability across steps, one-compile law / zero-recompile
param sweeps, device-count invariance (forced devices), the autodiff
regression, the lint (marked field without an update stage raises),
and the ETDRK4 refusal.

**Wave 3 — honesty sweep + records (`chore/td-fields-honesty`).**
TDF-D6: `PolarizedWaveMaker` extension + eigen-tool docstrings.
Roadmap hygiene in the same change: trim the open.md entry to what
remains (the `n2(z,t)` follow-up, the FieldBlend-unification open
question, the R6 docs deferral stays in its own entry), move the
shipped substance to done.md, move this plan to `plans/done/`.

Each wave: mirrored tests for every edited file + `uv run ruff check
src tests` before merging; short-lived branch, `git merge --no-ff`
onto dev, branch and worktree deleted in the same session. Re-check
dev movement before merging (parallel sessions are active).

## 4. Cost and risk

Sizing against the closest landed analogue (R2/`FieldBlend`,
`20bd375e`): wave 1 small (the linear-term population is a handful),
wave 2 the bulk (~comparable to R2: a few hundred lines source, a few
hundred test), wave 3 small. Main risks: (i) treedef/ghost-claim
regressions in the scan — covered by the treedef-stability test and
the bitwise MovingGeometry gate; (ii) guard-rework behavior drift —
covered by keeping the error surface and the existing refusal tests
byte-compatible; (iii) per-substage cost — a 1-D profile FMA-scale
recompute, negligible; full-3-D laws are possible but priced at one
field evaluation per substage (documented, not optimized).

## Landed

All three waves shipped 2026-07-18.

- **Wave 1 — structural frozen-`L` guard** (`refactor/linear-term-guard`,
  merge `bb2fb96f`; 11 files, +636/−68). TDF-D4 in full.
- **Wave 2 — mechanism + consumers** (`feat/time-dependent-fields`,
  merge `ceb9db75`; 14 files, +1424/−78). TDF-D1/D2/D3/D7/D8.
- **Wave 3 — honesty sweep + records** (`chore/td-fields-honesty`,
  this merge). TDF-D6 + roadmap hygiene + this move to `plans/done/`.

**Lowering notes (as shipped).**

- *Wave 1:* the structural frozen-`L` sweep lives in the base-class
  default of `Module.time_dependent_linear_parameters`, not in the
  assembly guard — the guard call-site, per-module offender
  attribution, `TimeDependentLinearOperatorError` and its message stay
  byte-compatible; resolution is module-local, so cross-module
  dependencies (the `dsqr` case) are reported by the **owning** module
  (`DynamicalCore`), and `tendency_terms()`-built terms stay unguarded
  unless the owner overrides the hook. See the "Lowering note (Wave 1,
  as shipped)" section above.
- *Wave 2:* the `(coords, t)` recompute core and `ProfileFunction` were
  extracted into `model/scheduled_field.py` and `MovingGeometry` was
  re-expressed on the shared helper **bitwise** (its tests are the
  regression gate); the `time_dependent` marker rides
  `FieldDeclaration` → `FieldRecord`, the `SELF_UPDATE` coverage lint
  keys off `Stage(writes=)` with the "owner has at least one
  `SELF_UPDATE` stage" fallback, and the law-valued `f`/`csqr` paths
  materialize their `t=0` snapshot via the `default=` builder while the
  per-substage stage rewrites the field at stage time. (Wave 2 did not
  add a separate lowering-note block to this plan; this is that
  one-line summary.)

**Open remainders** (trimmed into
[`../../roadmap/open.md`](../../roadmap/open.md)):

1. Nonhydro `n2(z, t)` law-valued path (TDF-D7) — `ConstantStratification`
   does not yet accept a `ProfileFunction`; a named follow-up.
2. `FieldBlend` unification (TDF-D9) — the affine-blend `f_coriolis`
   still evaluates term-side (AR-D2); re-basing it on the `SELF_UPDATE`
   rewrite path is an owner decision, logged as an open question.
3. A re-diagonalization contract for the analysis tools — explicitly
   **out of scope** per TDF-D6; the eigen/energy surfaces stay
   time-frozen with documented `at_time` snapshot semantics.
