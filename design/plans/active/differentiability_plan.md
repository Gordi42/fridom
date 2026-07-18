# Differentiability — closure plan

Status: **proposed** (2026-07-18, not yet started). Produced by a
three-track investigation in one session: a verification of every
claim in the roadmap's `model.propagator()` entry against the code, a
step-path VJP hazard sweep, and a full design-record inventory of
open differentiability items. All file:line citations below were
verified against dev `b773e119`. Owner calls collected in §8.

## 1. Scope — what this plan covers

The roadmap carries one headline differentiability item and a tail of
scattered residuals. This plan owns:

- **The public differentiable run surface `model.propagator()`**
  (roadmap "Differentiable run surface", incl. the `remat` knob and
  the frozen-operator taught error) — §5.
- **The last unsealed step-path VJP hazards** — the three coriolis
  `metric_weight` divides — plus their missing autodiff coverage — §4.
- **Record hygiene**: four stale/contradictory statements about
  already-closed hazards — §3.
- **A disposition for every other differentiability mention** in the
  records, so nothing is silently dropped — §6/§7.

Explicitly **out of scope** (owned elsewhere, listed in §7): the
time-dependent-fields campaign (its own ratified plan, in flight),
immersed graded advection's VJP seal (its own plan), multigrid
autodiff gates (its plans), and the **mapped+advection chunked-scan
forward non-finite on GPU** — that one is a forward-primal
compilation fault, not autodiff, and stays its own roadmap entry.

## 2. Verified current state (2026-07-18)

The differentiability invariant is in good shape; what remains is
narrow and well-bounded.

- **The kernel path is differentiable by construction.**
  `_chunk_body` (`model/model.py:638`) is a pure `lax.scan`; donation
  lives only in the jit wrapper (`_lower_chunk`, `model.py:792-794`,
  `donate_argnums=(2,)`), the panic pair rides the returned carry
  (host sync is in `advance`, `model.py:2029`), the stepper is a
  loop-invariant non-carry argument, and there is no io in-scan
  (DIAGNOSTIC stages do run in-scan and must stay pure — they do, by
  the module contract). The CG solve differentiates as the truncated
  algorithm (`krylov.py:167-190`, scan+cond, no implicit-function
  machinery) — exact to FD by design.
- **19 test files** already exercise `jax.grad` through
  `_chunk_body` per the AGENTS.md policy (canonical pattern
  `tests/model/test_model_autodiff.py`: identity-splice of the
  differentiation leaf, quadratic loss, central FD at rtol 1e-4).
- **Exactly three unsealed masked-singularity divides remain in the
  step path** (full sweep, this session): the coriolis
  `metric_weight` divides (§4). The `MetricScaled` reciprocal divides
  (`spatial/operators/mapped.py:218-222`) share the structure but are
  pre-audited reverse-safe in every live composition (all current
  compositions keep strictly-positive `sqrt_g` on addressable slots).
  Everything else is sealed or structurally safe: `_safe_ratio`
  (`model/modules/advection.py:481`), `_divide_by_jacobian`
  (`nonhydro2/modules/mapped_pressure.py:1112`), the staggering
  measure-divide seal (commit `2ab4c4a0`), `_biharmonic_root`
  (`model/closures/diffusion.py:471`), the exp-stepper phi functions
  (safe-denominator + select, `exponential.py:143-166`), and the
  Smagorinsky `custom_jvp` `_sqrt_clipped`.
- **House cure is uniform**: inline double-`jnp.where`
  (`bad = den==0; safe = where(bad,1,den); out = where(bad,0,num/safe)`),
  exemplar `_potential_vorticity` (`shallowwater2/modules/sadourny.py:251-276`);
  `custom_jvp` (primal untouched) only when the guard costs step time
  (one precedent: Smagorinsky); `custom_vjp` banned. Seals that run
  under halo-trace accounting must keep the plain quotient for
  storage-less tracers (the `_safe_ratio` / `_biharmonic_root` escape
  hatch) or the numeric halo tracer rejects the re-promotion.

## 3. Phase 0 — record hygiene (direct-to-dev, hours)

Four statements are stale and would misdirect the next reader:

1. `design/research/jax_grad_run_investigation.md:190-192` lists the
   `mapped_pressure.py:748 / jacobian` divide as "audited, unfixed" —
   it is the `velocity_correction` divide **fixed** by `ee350bda`
   (now `_divide_by_jacobian`, ~`:1112`). The record is frozen: add a
   dated addendum, do not rewrite. The addendum also points at this
   plan and records the H1-H3 status (§4).
2. `design/plans/active/fv_nonhydro_scoping.md:906-914` calls that
   same mapped-pressure reverse-NaN an open roadmap follow-up —
   correct in place with a dated note.
3. `design/plans/active/hydrostatic_model_plan.md:372-375` still
   lists nonlinear terrain advection as "not reverse-safe (Z/J
   unguarded)" — sealed via `_safe_ratio` (done.md records grad=FD at
   9e-10). Correct in place.
4. `src/fridom/spatial/operators/banded.py:28-30` claims the
   tridiagonal kernel's autodiff/batching support is "backend-uneven"
   — refuted by `multigrid_kernel_study.md:288-297` (all four kernels
   natively reverse-differentiable, jax 0.10.2). Docstring fix rides
   the Phase 1 branch (it touches `src/`).

Also fix the roadmap propagator entry's stale citation
(`model.py:1628` → `_replace_leaf` `model.py:1822`,
`update_parameters` `model.py:1835`) and add a pointer to this plan.

## 4. Phase 1 — seal the last hazards, close the coverage gaps

**H1-H3, `model/modules/coriolis.py` (real hazards, latent):**

| # | Line | Expression | Fires when |
|---|---|---|---|
| H1 | 185 | `-((w.to(u)*f_u*u).to(v)) / w.to(v)` (`linear_rotation`) | `metric_weight` set (e.g. `"csqr"`); denominator 0 in never-valid padding and immersed dry cells |
| H2 | 243 | `flux_weight * v.to(u) / w_1` (`chart_rotation`) | chart grids; `w_1 = sqg_u*g_uu` is 0 in padding |
| H3 | 244 | `-((flux_weight*u).to(v)) / w_2` (`chart_rotation`) | same, `w_2` |

Forward values are finite (masked 0/0); the VJP is `-num/w²` →
`0·inf = NaN`. Same class as the cured Sadourny PV divide. No
autodiff test currently differentiates a weighted coriolis or
`chart_rotation` at all — the hazard is invisible to the suite.

Work (one branch, small):

- Seal H1-H3 with the double-`jnp.where` pattern, replicating the
  halo-trace escape hatch (see §2). No `custom_jvp` — these divides
  are not step-time-critical.
- Policy tests in the mirrored file(s): grad-finite + FD-matched
  through a short `_chunk_body` run for (a) `metric_weight="csqr"`
  rotation, (b) `RotationCoriolis` on a chart grid; plus the
  Sadourny-style proof assertion (bare quotient NaNs, guarded is
  finite) where cheap.
- `banded.py` docstring correction (§3 item 4) rides along.

**H4/H5, `MetricScaled` (`mapped.py:218-222`) — keep as watch-items.**
Pre-audited reverse-safe; a guard has real cost and no live firing
composition exists. Add a short in-code comment naming the exposure
condition (a composition that feeds the reciprocal a bounded-axis
metric whose exact-0 ghost ring survives to the divide) and pointing
at the audit. Optional cheap canary: a direct grad test of the
reciprocal branch on a mapped grid pinning today's safety (§8, owner
call D4).

**Sequencing constraint:** the in-flight `refactor/linear-term-guard`
worktree (time-dependent-fields wave 1) is editing `coriolis.py`.
Land Phase 1 **after** that branch merges (or rebase over it); the
seal is small and rebases trivially.

## 5. Phase 2 — `model.propagator()` (the headline)

### 5.1 What the investigation confirmed

The roadmap entry is accurate on the mechanics: build the pure
callable from `model._carry` + `record` (static) + stepper, jit
`_chunk_body` **without** `donate_argnums` (the autodiff tests prove
this path), splice `wrt` leaves by identity exactly as
`test_model_autodiff.py` does, default to a fresh stepper state
(`model._fresh_stepper_state()`, `model.py:1801` — gradients include
the warm-up ramp, documented). `TIME_STEP` resolves through the
binding table (`ParameterBinding`, `assembly.py:151`; stepper
provides `dt`, `time_steppers/base.py:151-173`); identity-defaulted
constants (`slot is None`) are already refused by `update_parameters`
(`model.py:1871-1875`) and the propagator refuses them identically.
Single `_chunk_body` call per invocation (no chunking — chunking
exists for host io, which the propagator excludes; the x64 clock
re-anchor in `advance` is a no-op under default x64-on).

### 5.2 Gaps the entry missed (found this session)

1. **Name collision.** `fr.transforms.Propagator`
   (`model/transforms/propagator.py:38`) already exists and is the
   opposite of this surface: a Tier-2 host `StateTransform`
   (`traceable=False`, tracer-guarded, runs `advance` internally;
   consumed by optimal balance and adiabatic ramping). The lowercase
   method `model.propagator()` is distinguishable, but the docs must
   cross-reference both ways. Owner call D1 (§8).
2. **Materialized-parameter trap (silent-zero gradients).** Params
   like `f0`/`csqr` are materialized at assembly into AUXILIARY
   fields (`jnp.full(space.shape, self.f0)`); splicing the scalar
   leaf alone leaves the AUX field stale, so `grad` w.r.t. them would
   be **silently zero** — exactly the silent-wrongness class this
   surface must not ship. The detection mechanism already exists: the
   `RematerializationTable` (`assembly.py:963`, `model.py:1896`)
   knows which owners are materialized. v1: **refuse** `wrt` names
   whose owner appears in the remat table (taught error naming the
   limitation). Follow-on: probe whether the remat entries are
   traceable and can run inside the transformer (in-trace
   rematerialization); the time-dependent-fields recompute contract
   (TDF wave 2) generalizes exactly this and may subsume it. Owner
   call D3.
3. **Frozen-L refusal needs a map that does not exist yet.**
   `freezes_linear_operator` exists (`base.py:124`, ETDRK4
   `exponential.py:231`), but the only existing hook
   (`time_dependent_linear_parameters` +
   `_check_frozen_linear_operator`, `assembly.py:113-139`) reports
   only *time-dependent* linear params — a time-independent param
   frozen into the eigenbasis is invisible to it, and gradients
   w.r.t. it are silently stale. The in-flight **TDF-D4** design
   (structural `linear_params`/`linear_fields` declarations on
   `linear=True` terms, resolved through the binding table) builds
   precisely the needed map; the propagator's refusal predicate
   differs only in firing on *any* `wrt` in the L-feeding set, not
   just time-dependent ones. Sequence the refusal after TDF wave 1
   merges and reuse the declarations. Fallback if TDF stalls:
   conservative v1 — under a `freezes_linear_operator` stepper,
   refuse every bound-parameter `wrt` (IC-field `wrt` stays allowed),
   taught error naming TDF-D4 as the unlock.
4. **`remat` interacts with `unroll`.** `_chunk_body` unrolls by
   `stepper.scan_unroll` (AB3 → 3). Wrap `one_step` in
   `jax.checkpoint` via a small hook and force `force_unroll=1` under
   `remat` (the parameter already exists, `model.py:643`). No
   `jax.checkpoint` exists anywhere in `src/` today; the similarly
   named `remat_table` is unrelated (AUX rematerialization — see gap
   2). Record the measured tape baseline (~1.4 MB/step at 8³) and the
   post-remat number in the outcome record.

### 5.3 Work breakdown (one branch, medium)

1. **Name resolution → carry transformer**: `wrt` tuple → binding
   lookups (reuse `_read_leaf`, `assembly.py:339-349`) → a pure
   `theta -> (modules, stepper, carry)` splice by identity; PROGNOSTIC
   names splice `carry.state[name].storage`. Refusals: unknown name,
   identity-defaulted constant, materialized owner (gap 2), frozen-L
   set (gap 3).
2. **The surface**: `model.propagator(*, wrt, steps, remat=None)`
   returning `(theta, state=None) -> ModelState` (return-type
   decision D2, §8). Fresh stepper state by default; `state=None`
   means the current committed carry's state.
3. **`remat` hook** in `_chunk_body` (gap 4).
4. **Tests** (`tests/model/test_model_propagator.py`, prefix-shard of
   the model tests): grad vs FD for each `wrt` class (closure param,
   `dt`, IC field); remat on/off gradient equality; chunk-splitting
   invariance vs `advance`; every taught error (`pytest.raises` with
   match); ETDRK4 refusal both ways (L-param refused, N-param and IC
   allowed).
5. **Policy migration**: AGENTS.md's differentiability-policy pattern
   text switches its canonical recipe from the private `_chunk_body`
   splice to `model.propagator()`; the 19 existing `_chunk_body`
   test files stay as they are (mechanical migration is optional and
   not worth the churn — new tests use the public surface).
6. Roadmap: move the propagator entry to `done.md`, trim residuals
   here (hygiene rule).

### 5.4 Phase 3 — D5 `TangentPropagator`: defer

Forward-mode `jax.jvp` of `model.tendency` (spec
`04_run_loop_io.md:69-81`). `model.tendency` itself is built and
jitted (`model.py:2252`); the tangent wrapper shares only the
name-resolution piece. Its one named consumer (NNMD) was descoped.
Defer under the house "who wants it" test; the Phase 2 name-resolution
work leaves it a small lift when a consumer appears. Move its mention
into the roadmap's sized-deferred section.

## 6. Dispositions — every other differentiability mention

- **CG floor trap** (tolerance below the ~1e-14 residual floor NaNs
  the reverse pass): documented convention in the docstring; no work.
- **`physical_diff` grad-unsafe** (hydrostatic): documented design
  constraint; H2b uses the guarded `_slope_gradient`; no work unless
  the verb enters a step path.
- **Robin 2e α-differentiability**: deferred with its parent (no
  consumer); the constraint "α/g arrive as traced module params" is
  already recorded in the boundary plan.
- **FV↔nodal mapped-divide `custom_jvp` perf lever**: sealed and
  safe; the lever stays unexercised until a multi-GPU cost signal.
- **sw2 `metric ** 0.5` siblings** (`state.py`/`diagnostics.py`):
  audited safe (static positive metric, outside the tendency path).

## 7. Owned elsewhere (do not duplicate here)

- **Time-dependent `f`/`csqr` in L × ETDRK4** — the ratified
  time-dependent-fields plan (TDF-D4/D5: structural guard,
  refuse-never-auto-split; TDF-D8: policy autodiff tests). This
  plan's only coupling is reusing TDF-D4's map (§5.2 gap 3).
- **Immersed graded advection** — GA-D2 pre-masked-operand VJP seal +
  G4 autodiff gate, in its plan.
- **Multigrid** — per-phase autodiff gates + smoother dry-cell
  double-`where` guards, in its plans.
- **Mapped+advection chunked-scan GPU non-finite** — forward primal,
  separate roadmap entry, untouched by this plan.

## 8. Owner calls

- **D1 — naming.** Keep `model.propagator()` next to
  `fr.transforms.Propagator` (recommended: yes, lowercase
  method vs transform class, with two-way doc cross-references), or
  pick another name (`model.pure_run()`, `model.kernel()`).
- **D2 — return type.** The roadmap wrote `-> State` but also wants
  the panic pair "riding the returned carry"; recommended: return the
  final `ModelState` (state, stepper state, clock, panic — all
  functionally inspectable), with `.state` as the State.
- **D3 — materialized params.** v1 taught-error refusal via the
  remat-table owner set (recommended), in-trace rematerialization as
  a follow-on/probe — or require in-trace remat in v1.
- **D4 — `MetricScaled` canary.** Comment-only (recommended) or also
  a direct grad canary test pinning today's reverse-safety.
- **D5 — Phase 2 sequencing.** Recommended: land 5.3 items 1-4
  independent of TDF, with the frozen-L refusal in its conservative
  fallback form if TDF wave 1 has not merged yet, upgraded to the
  TDF-D4 map when it lands.

## 9. Sequencing summary

```
Phase 0 (records)  — anytime, direct-to-dev; hours.
Phase 1 (seals)    — after refactor/linear-term-guard merges; small.
Phase 2 (surface)  — anytime after Phase 1 branches cleanly;
                     frozen-L refusal upgraded post-TDF-D4; medium.
Phase 3 (tangent)  — deferred (no consumer).
```

Phases 1 and 2 are ordinary `<type>/<topic>` branches with mirrored
tests and the merge gate; Phase 0 is design-only except the
`banded.py` docstring (rides Phase 1).
