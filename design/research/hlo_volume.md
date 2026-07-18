---
status: frozen
date: 2026-07-18
---

# Cold-compile HLO volume — census refresh and closure (negative)

The roadmap carried "Cold-compile HLO volume" as the remaining
cold-start lever after the 2026-07-16 time-to-first-step campaign
([`time_to_first_step.md`](time_to_first_step.md)): step-body compile
scales with HLO volume, so shrink the traced graph. This campaign asked
the follow-up directly: **can the step body's HLO volume be reduced
without any step-time regression?** Four parallel investigations
(census refresh, frame plumbing, advection batching, mapped/CG body)
answer **no** — every reducible-looking category is either already
reduced, folds away before it costs anything, or can only shrink by
buying back a measured runtime regression. The item is closed as a
negative and retired from [`../roadmap/open.md`](../roadmap/open.md).

Setup: dev `26023478`, jax/jaxlib 0.10.2, one A100 (+ CPU compile
checks — unoptimized HLO is backend-identical and CPU compile ≈ GPU/2,
both verified), `FRIDOM_DISABLE_COMPILE_CACHE=1`, fresh process per
data point, chunk compile read from `fr.model.model._CHUNK_COMPILE_LOG`.
Harnesses, raw HLO dumps, and the one prototype patch:
`~/work/fridom/hlo-volume-2026-07-18/{census,seal-dedup,
advection-batch,mapped-cg}/`.

## 1. Census refresh — and two corrections to the 07-16 record

Chunk executable, bench-step configs (flat 64³, mapped/walled 32³):

| config | unroll | unopt | opt | compile s (GPU) |
|---|---|---|---|---|
| flat linear | 3 | 1134 | 3843 | 1.15 |
| flat centered | 3 | 1371 | 4671 | 1.65 |
| flat upwind5 | 3 | 1875 | 7112 | 2.98 |
| flat weno5 | 3 | 2510 | 10939 | 4.37 |
| mapped centered (spectral precond) | 3 | 4521 | 14712 | 5.02 |
| walled linear | 3 | 1296 | 4558 | 1.41 |

(unroll=1 rows and per-op-kind breakdowns in the census raw data.)

**Correction 1 — compile tracks *optimized* HLO, not unoptimized.**
Unoptimized/StableHLO counts are unroll-invariant (centered 1385 sHLO
at unroll 1 and 3): XLA expands the AB3 `scan_unroll=3` replication
during optimization (centered opt 1644 → 4671). Any volume argument
must be made on the optimized graph; the scaling exponent re-measures
at ~1.1–1.3 (roadmap said ~1.35).

**Correction 2 — the "~70 pads" were never a compile driver.** Of the
69 jaxpr `pad` eqns per centered body, only 20 are storage↔true
round-trips (48 are FV-reconstruction frame realigns), and jax's own
jaxpr→HLO lowering folds them into **9** shared `%_pad` sub-computations
before XLA ever sees them. The 60 seal `dynamic_update_slice` survive;
the 15 gathers are consumption-side halo fills.

Attribution (centered flat body, deep jaxpr eqns): pressure 32%,
advection 32% (07-16 read 45%; "~⅓" is the durable statement),
seal/frame plumbing 16%, stepper combine 6%, coriolis 6%.

## 2. The motivating numbers were stale

- **weno5 "8.5–10 s" → 4.37 s** chunk compile (−38%) — already
  delivered by the 07-16 selected-input lowering; weno5's unopt volume
  fell from 3.0× to 1.8× centered. Nobody had re-measured.
- **mapped "16–18 s vs 2–3 s flat" → 5.0 s vs 1.65 s (~3×).** The old
  figure is best reconciled as the first-advance-wall metric artifact
  §1 of `time_to_first_step.md` already debunked (compile + whole
  first-chunk execution): today mapped-spectral shows
  `first_advance_wall` ≈ 6 s against pure compile ≈ 2 s (CPU). The
  production (spectral-preconditioned) mapped step is
  ~1.9 s CPU / ~3.8 s GPU, exactly size-independent.

## 3. Frame plumbing — irreducible (measured negative)

- The 60-DUS carry ghost seal is the *deliberately* runtime-optimal
  spelling: O(halo) in-place writes at the carry materialization point
  (2 disjoint slabs per periodic axis × 3 axes × 5 fields); the
  concatenate/index-map alternatives are the documented +2.8 ms/step
  regression (`tensor.py` design notes, `8c940666`). It is
  load-bearing: 27 of 48 stencil consumptions skip their halo fill
  because the sealed carry enters at full validity.
- Application-level trace dedup has nothing to collapse: 1 duplicate in
  27 operator applications, 0 duplicate syncs — interning plus the
  composer already keep the trace tight.
- Demand-aware seal skipping (e.g. `p` in the flat spectral config) is
  worth ~0.9% unopt for a silent stale-ghost-read hazard on
  mapped/walled — rejected on risk/reward.
- Stub-bounded ceiling: deleting *all* ghost plumbing would buy −25%
  compile (1369→1163 unopt, centered chunk); the subset achievable with
  zero step-time regression is ≈ 0.

## 4. Advection batching — structurally blocked (measured negative)

The 12 flux traces (4 fields × 3 axes) look like batch fodder; per
axis 3 primal reconstructions are structurally identical (one field is
dual per axis, and only the q-reconstruction is shareable — velocity
face, product, and flux-diff differ per field). Two measured findings
kill every batching form:

- **Call-dedup is free but useless.** Tracing one kernel and calling it
  12× cuts *unoptimized* HLO −37% — and changes compile time by
  exactly nothing: XLA's `CallInliner` inlines the calls before the
  compile-dominating passes, so the optimized graph is op-for-op
  identical (verified at n=32/64, chunk 1/50). It is also not a
  numeric no-op: the inner-jit boundary reassociates the WENO combine
  at roundoff and breaks 12 `selected-vs-Where` bitwise invariants.
  Prototype preserved as
  `advection-batch/call-dedup-prototype.patch` (branch discarded).
- **True batching needs a stacked state.** The only form that shrinks
  the *optimized* graph (micro: −3×) operates on an already-stacked
  `(3, …)` array. The state stores separate per-field leaves — with
  genuinely different shapes on walled/mapped/immersed grids, so the
  flat-periodic shape match is a coincidence, and stacking at use
  costs temp 0 → 571 MB and 2.6–8× kernel runtime (the
  `stencil_lowering.md` §5 temp-bytes disqualifier; at weno5 512³ the
  step already runs ~24.7 GB temp near the memory ceiling).

## 5. Mapped/multigrid body — already optimal (no lever)

- Default mapped (spectral precond): ~73% of its 3517 unopt ops are the
  projection — the genuinely larger mapped flux operator (2.3× flat
  un-preconditioned) plus the spectral inverse traced 3× by the peeled
  PCG. Neither shrinks without changing the operator or the
  preconditioner.
- Opt-in multigrid (17–37 s compile): ~89% is V-cycle plumbing — each
  level adds exactly 4312 unopt ops regardless of shape, levels cannot
  be `lax.scan`-ed (per-level shapes differ), the V-cycle traces 3× per
  solve structurally, and interning cannot fire across distinct
  coarse-grid operators. The tridiagonal kernel is secondary: the
  cuSPARSE swap (`26023478` era) already made the GPU auto-default the
  smallest-compile option (−19% compile, −27% opt-HLO vs scan) *and*
  the fastest at runtime. Fewer levels would save ~7 s compile but is
  a step-time regression (convergence tuned at levels=5) — rejected.
- One wrinkle, deliberately not acted on: CPU auto resolves the
  tridiagonal method to PCR, the *largest* HLO (+34% vs scan). CPU
  multigrid is not a production path; do not flip the default without
  a CPU step-time measurement.

## 6. Where this leaves cold compile

The remaining volume is intrinsic: stencil arithmetic, the mapped flux
operator, V-cycle levels, and the ×3 unroll replication (whose
runtime-neutral alternative — the async two-tier chunk compile — is
its own open roadmap item and the only honest cold-start lever left).
Persistent-cache reruns and the eval_shape dry run (07-16) already
cover everything else. **Do not revisit** without new evidence:
inner-jit/call-dedup of repeated kernels (CallInliner), field-stacked
advection batching (temp/runtime), seal-DUS re-spelling (8c940666),
demand-aware seal skipping (stale-ghost hazard), multigrid level
scanning (shape mismatch), CPU tridiagonal default flips (unmeasured
step time).
