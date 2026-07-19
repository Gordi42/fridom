# Time-dependent fields — follow-ups: state-sourced energy metric + FieldBlend unification

Status: **active** (owner-approved 2026-07-19 in chat: "implement both …
I go with your recommendations").

Parent records:
[`time_dependent_fields.md`](time_dependent_fields.md) (TDF-D1..D9, the
general `(coords, t)` rewrite machinery, shipped 2026-07-18);
[`adiabatic_ramping.md`](adiabatic_ramping.md) (AR-D1..D8, the affine
`FieldBlend`, shipped 2026-07-16/17). Rulings here continue the TDF
numbering (TDF-D10, TDF-D11).

Two independent work items, both closing warts the TDF campaign
documented rather than fixed:

- **A** — the energy metric bakes its field weights at build time, so a
  time-dependent `csqr`/`N²` run gets a `t = 0` metric at every analysis
  sample: silently mislabeled physics (the same defect family as the
  dsqr/ETDRK4 hole TDF-D4 closed).
- **B** — the affine `FieldBlend` evaluates term-side while the carried
  `f_coriolis` stays a frozen `t = 0` snapshot, so I/O, restart and any
  cross-module reader see stale values; and the repo carries two
  time-dependent-field mechanisms where one now suffices.

## 1. Rulings

**TDF-D10 — Energy-metric field weights are state-sourced by default;
the frozen-snapshot spelling becomes explicit.** A weight whose source
field is declared `time_dependent=True` is stored as a *state-sourced
descriptor* ("read component `csqr` off the operand; weight is the
field / its reciprocal"), resolved inside `apply`/`inner`/`norm` from
the operand state — which, post-TDF, carries its own stage-time values,
so the metric is automatically evaluated at the measured state's own
time with zero clock plumbing. `inner(a, b)` resolves from **`b`** (the
weighted operand, matching the existing `a† · (M b)` spelling; a
cross-time inner product is not an energy and is documented
out-of-scope). Applying a state-sourced metric to a container lacking
the source component (an eigenmode basis vector, a bare `(u,v,p)`
bundle) is a taught error naming the snapshot spelling.
`from_model` gains `snapshot: bool = False`; `snapshot=True` reproduces
today's build-time baking (the eigen/channel family passes it — a
frozen basis needs the *matching* frozen metric, TDF-D6 unchanged: no
re-diagonalization). Non-marked (static-profile) field weights and all
scalar weights keep today's baked path bit-identically; `Ramp`-valued
*scalars* stay frozen at `at_time` (explicit, documented, and the
deliberate semantics of the adiabatic-ramping reference metrics — a
clock is not readable off a bare component bundle). The hydrostatic
terrain depth weight `H(x, y)` is grid-metric-derived, not
state-resident, and keeps snapshot semantics (a MovingGeometry-tracking
`H` is a noted follow-up, not scope).

**TDF-D11 — `FieldBlend` lowers onto the SELF_UPDATE rewrite path;
AR-D2's mechanism half is superseded, its generality half preserved
(TDF-D9 reversed by owner, 2026-07-19).** A blend-active module marks
its blended field `time_dependent=True` and rewrites it each substage
in a SELF_UPDATE stage computing the same
`p(t) = Σ w_i(t)·P_i` from the assembly-materialized ingredient fields
(unchanged: static AUXILIARY, halos exchanged once) and the module's
leaves resolved at the substage clock. The tendency term reads the
carry field plainly — the term-side `evaluate`-in-term seam
(`_stage_blend_f`) is deleted. `FieldBlend` itself survives as the
declaration-level descriptor (ingredients, `is_active`,
`field_declarations`) plus the stage's compute core: AR-D2's ruling
*for generality* (stratification/topography blends as plain future
consumers) is untouched; only "no SELF_UPDATE rewrite" is superseded,
now that the rewrite machinery exists and the term-side path is the
redundant second mechanism with a known staleness wart.

## 2. Work item A — state-sourced `EnergyMetric` (`src/fridom/model/energy.py`)

Mechanics:

- New frozen host descriptor (module-level, not a pytree), e.g.
  `StateSourcedWeight(field: str, fn: Callable)` with module-level
  `_identity` / `_reciprocal` transforms (keeps it hashable/static).
- Resolution seam at the top of `apply`/`inner`: descriptor → read
  `operand[field]`, apply `fn`, then flow into the *existing* branches
  (`_weigh` lifts via `.to` per component exactly as today; the
  Parseval leg's field-weight `NotImplementedError` fires on the
  resolved `ScalarField` unchanged). Missing component → taught error.
- `from_model`: the varying-coefficient branches (`csqr` field on
  shallow water → `{u: csqr, v: csqr, p: 1}`; `n2` field on nonhydro →
  `b: 1/n2`) consult the composed **`FieldRecord.time_dependent`** flag
  via `model.field_table[name]` (duck-typed
  `getattr(record, "time_dependent", False)` — the composer-lint
  spelling reads a `FieldRecord`, `field_table.py:94`; `model.state`
  fields carry no such attribute) and emit the descriptor instead of
  the baked array — still behind the `allow_field_weights` gate, still
  validated by `_profile_field` (which keeps its existence /
  taught-rejection checks and only stops returning the captured array
  on the marked path). `snapshot=True` short-circuits to today's
  baking.
- Resolution/skip interplay (review-pinned): `apply` keeps its
  `if name in state` skip on the *weighted component* and gates the
  source read on it — the taught error fires only when the weighted
  component is present but the *source* field is absent; `apply`
  resolves from the operand, `inner`/`norm` from `b` (= `a` for
  `norm`), matching where the weight multiplies.
- Callers swept — exactly **three** exist in `src`
  (`EnergyMetric(...)` is never constructed directly):
  `eigen.py:184` (`allow_field_weights` False — `snapshot=True` is
  defensive, pass it for explicitness), `eigen_channel.py:401` and
  `transforms/balance_expansion.py:370` (both
  `allow_field_weights=True` — `snapshot=True` is **load-bearing**: a
  state-sourced descriptor would break their weight extraction and the
  frozen-basis consistency, TDF-D6). The per-model `eigenmodes.py`
  builders reference `from_model` in prose only — not callers.
  `from_model` docstring rewritten to state the new default truthfully
  (the TDF-wave-3 "does NOT track" paragraph inverts).

Gates (mirrored `tests/model/test_energy.py` + consumers' mirrors +
ruff + `tests/nonhydro/test_linear_model.py` smoke):

- Static model: weights and numbers bit-identical to dev.
- Tracking: a shallow-water model with `csqr=ProfileFunction(...)`,
  states at two times (short run) — `norm`/`inner` match an oracle
  metric baked per-time (`snapshot=True` on a model whose state sits at
  that time); the two times genuinely differ.
- Taught error on a component bundle lacking the source field.
- `snapshot=True` on a time-dependent model reproduces dev behavior.
- Eigen path: existing eigen/channel tests stay green untouched.
- One `jax.grad` through `norm` of a state-sourced metric is finite
  (the reciprocal is a genuine divide, no seal — TDF-D8 spirit).

## 3. Work item B — `FieldBlend` on the rewrite path

Files: `src/fridom/model/field_blend.py`,
`src/fridom/model/modules/coriolis.py` (`BetaPlaneCoriolis`),
`src/fridom/shallowwater2/modules/coriolis.py` (conserving rotation),
sw2 energy-correction module (see below), mirrored tests.

Mechanics:

- Blend-active declarations: `f_coriolis` gains `time_dependent=True`
  (still materialized at `t = 0` by `_f_default` — valid static
  treedef; the marker is repr-participating only when True, so the
  static path's assembly fingerprint is untouched), ingredient
  declarations unchanged.
- `BetaPlaneCoriolis.stages` emits a **second** gated stage: the
  existing law stage for `_profile_active`, a new blend stage for
  `_blend_active` (S1, `Stage(kind=SELF_UPDATE, fn=...,
  reads=(target, *ingredients), writes=(target,))`), whose fn computes
  `_BETA_BLEND`'s affine combination at the substage clock
  (`getattr(ctx.clock, "time", ctx.clock)` — the same seam
  `_stage_blend_f` reads today, so the rewrite changes *where* the
  blend happens, not *when*: SELF_UPDATE rank 0 and terms rank 2 share
  one substage `StepContext`) and returns `{target: blended}`. Put the
  reusable stage/rewrite helpers on `FieldBlend` so future consumers
  (stratification/topography) wire two thin lines.
- **Halo intent (review-confirmed 2026-07-19, no fallback needed):**
  the blend is pointwise field arithmetic over halo-valid ingredients,
  so the rewritten field is halo-valid by construction — written back
  as a full field, **no** `extra_halo`, stays halo-traced. Evidence:
  the tracer walks SELF_UPDATE stages of non-exempt modules
  (`assembly.py:2136`; exemption is `extra_halo is not None`, which
  stays `None` here); `_seal_carry_ghosts` normalizes every field to
  the full decomposition halo at each carry boundary
  (`model.py:698,720`), so the carry treedef is byte-stable exactly as
  on the shipped law path; the demand accounting is byte-identical to
  the term-side blend (the pointwise blend records no reach; the
  term's `.to` reach-1 lands on `f_coriolis`); `apply_replace` /
  `VectorField.replace` is a pure component swap re-attaching only
  annotation metadata, no bit-changing op.
- Term simplification: `_coriolis` drops the `f_field` branch and
  `_stage_blend_f` is deleted. **Both routes**: the shared `_coriolis`
  term *and* sw2's route-B `_ConservingRotation.coriolis`
  (`shallowwater2/modules/coriolis.py:715`) read the carry;
  `NonlinearBetaPlaneCoriolis` inherits the blend stage + marked
  declaration through the `_blend_active`-gated properties (assert in
  a test, incl. that its route-B guard report stays `()`). The now
  always-`None` `f_field=` parameters of `linear_rotation` and
  `conserving_rotation`/`_conserving_chart` are removed with their
  branches. The f-plane scalar override `_stage_scalar_f` (R1) is
  untouched.
- sw2 energy-correction: the blend-active refusal
  (`shallowwater2/modules/coriolis.py:547`) is **lifted**
  (review-confirmed): the correction computes
  `conserving_rotation − linear_rotation`, both reading
  `state["f_coriolis"]`, so on the stage-time carry the total
  telescopes to the exact conserving rotation (the correction adopts
  the linear module's `metric_weight` at bind; nothing else is
  load-bearing). Gate: a conservation test on a ramped-`f` run.
- Guard/lint: no *refusal* behavior change — ETDRK4 already refuses
  ramped `f0`/`beta` via `linear_params`; the marker adds the
  `linear_fields` route (double-covered). But the **offender tuple**
  of `time_dependent_linear_parameters()` for a blend-active module
  gains `"f_coriolis"` — update any exact-tuple/message asserts
  (`tests/model/modules/test_coriolis.py`,
  `tests/shallowwater2/test_coriolis.py`,
  `tests/model/time_steppers/test_exponential.py`). The composer lint
  is satisfied by the new `writes=`; no other stage writes
  `f_coriolis`, so the overlap lint stays clean. Assert in tests.
- Docstrings: the "carry stays a `t = 0` snapshot; the term reads the
  fresh blend" narrative (module docstring, `field_declarations`,
  `_BETA_BLEND` comment, `field_blend.py` header — including its
  "we still do NOT build general time-dependent fields" paragraph,
  now false) rewritten truthfully.

Gates (mirrored `tests/model/test_field_blend.py`,
`tests/model/modules/test_coriolis.py`,
`tests/shallowwater2/test_coriolis.py`
+ ruff + `tests/nonhydro/test_linear_model.py` smoke):

- Static (plain-float) path bit-identical to dev.
- Blend-active tendencies: same arithmetic order as the old term-side
  evaluate → assert bitwise on cpu against dev values (tolerance
  fallback only if a genuine reassociation shows up — then record it).
- The wart-fix test: after a step, `state["f_coriolis"]` equals the
  stage-time blend, not the `t = 0` snapshot.
- Adiabatic-ramping integration tests (leakage decay, `Propagator`
  legs) stay green — `model.variant(updates=...)`-seeded ramps drive
  the stage exactly as they drove the term (`_blend_active` stays a
  freshly-evaluated predicate).
- Scan/carry: treedef stable across chunks (existing carry-stability
  pattern); forced-4 invariance on the blend path.
- One autodiff regression: `jax.grad` through `Model.propagator` of a
  short blended run w.r.t. a `Ramp` leaf (or the IC) is finite and
  matches central finite differences to rtol 1e-4 (TDF-D8 pattern).

## 4. Sequencing, branches, records

Independent — run in parallel, disjoint files:
**A** on `feat/energy-metric-state-weights`, **B** on
`refactor/field-blend-self-update`; merge gates per item as above;
`merge --no-ff` onto dev with dev-moved-under-us reconciliation
(re-merge dev into the branch, re-run gates, then land).

Records, after both land: this plan moves to `design/plans/done/`; the
roadmap's FieldBlend-unification remainder and the TDF-D6 metric-
staleness note move `open.md → done.md`; `adiabatic_ramping.md` AR-D2
gets a one-line forward pointer ("mechanism half superseded 2026-07-19,
see this plan"); `time_dependent_fields.md` TDF-D9 likewise.

## 5. Landed

Both items shipped 2026-07-19.

- **A — state-sourced `EnergyMetric`** (`feat/energy-metric-state-weights`,
  merge `2d222425`). A field weight whose source is `time_dependent=True`
  becomes a `StateSourcedWeight` host descriptor resolved at apply time —
  off the operand in `apply`, off `b` in `inner`/`norm` — so the metric is
  evaluated at the measured state's own stage-time with zero clock
  plumbing. `from_model` gains `snapshot: bool = False`; the three `src`
  callers were swept (`eigen.py` defensive; `eigen_channel.py` and
  `transforms/balance_expansion.py` **load-bearing** — both pass
  `snapshot=True` to keep their `allow_field_weights=True` weight
  extraction and frozen-basis consistency, TDF-D6). Gates: 48 energy
  tests (5 new — tracking, taught error, `snapshot=True` parity, the
  state-sourced `jax.grad`), the eigen/channel mirrors, and the
  `test_linear_model` smoke all green. Note: the nonhydro `n2`
  reciprocal branch is wired and tested but not reachable through a real
  model until an `n2` law lands — `MeridionalStratification` takes only a
  static callable — so when the TDF-D7 `n2(z, t)` follow-up wires that
  law, its metric side is already free.
- **B — `FieldBlend` on the rewrite path** (`refactor/field-blend-self-update`,
  merge `27790fdf`). The blend now rewrites its carried field each
  substage through a SELF_UPDATE stage built by the reusable
  `FieldBlend.stage()`/`rewrite()` helpers; the term reads the carry
  plainly and `_stage_blend_f` plus the `f_field=` parameters are
  deleted. One lowering note vs §3: the stage fn is passed as a **string
  method name**, because the composer rejects a bound-method stage fn.
  Blend-active tendencies held **bitwise** against dev. The sw2
  energy-correction blend-active refusal is **lifted** — on the
  stage-time carry the `conserving − linear` total telescopes to the
  exact conserving rotation (production residual <1e-13, route A ==
  route B <1e-12). The `time_dependent_linear_parameters()` offender
  tuple for a blend-active module now includes `"f_coriolis"` via the
  `linear_fields` route (exact-tuple asserts updated). Gates: forced-4
  blend-path invariance <1e-11, the `jax.grad`-through-`propagator`
  autodiff regression FD-matched to rtol 1e-4, 155 passed + 4 forced-4.
