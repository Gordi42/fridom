---
status: active
date: 2026-07-16
---

# Roadmap — open work

The single open-work tracker for FRIDOM, ordered: **next steps** first,
then **long-term goals**. Task numbers are historical and stable — other
records cite them ("ROADMAP 3.5") — so they do not run in order.

**Hygiene rule (binding): this file holds only open work.** The moment
something ships, move its record to [`done.md`](done.md) — in the same
change that reports it shipped — and trim the open entry here to what
actually remains. Status narrative ("shipped", "landed", "resolved",
"update: ... now exists") must not accumulate in this file; where
context about finished work is needed, one pointer to the `done.md`
entry is enough.

# Next steps

## Performance guard — final A100 validation

The closure design and its whole buildable surface shipped
2026-07-18 (entry in [`done.md`](done.md); plan + owner rulings:
[`../plans/active/perf_guard_plan.md`](../plans/active/perf_guard_plan.md)
— PR CI gates *structure*, the DKRZ A100 node gates *time*,
manual-trigger only, **no automated cluster submissions**).
Remaining (plan §6): **one green, manually submitted**
`benchmarks/ci/step_guard.sbatch` run on the A100 node.

The first run (2026-07-18, job 26346485; log in plan §7) was RED by
**true positive**: it caught the tiny-nodal-case shift from the FV
storage-frame spelling, since attributed and re-baselined on `dev`
(`f87ea9d7`) by the owning session — the guard mechanics, the
`uo0780_gpu` account, and the gpu-marked cusparse legs (2 passed on
a real A100, job 26346504) are all validated. The next quiescent
guard run is expected green against the re-recorded baselines; when
it is, move this entry to [`done.md`](done.md) and log it in the
plan.

## Gaps against the Oceananigans reference comparison

The 2026-07 matched-protocol comparison against Oceananigans.jl
0.105.3 (suite and full report live in the untracked
`benchmarks/comparison` — out-of-tree by design, see its README)
confirmed the new stack wins where it is pressure-solve-bound. The
three trailing areas it identified are largely closed — single-GPU
memory ceiling, time-to-first-step, WENO throughput (entries in
[`done.md`](done.md)). Still open:

- **Re-run the full comparison suite** on post-fix dev. The
  2026-07-17 single-GPU recheck already re-measured the changed rows
  (weno5 512³ 130.5 ms/step, oc edge 1.57×); what remains is the
  full-table refresh — including the multi-GPU scaling rows — blocked
  on a 4-GPU allocation. New runs report the honest `compile_s`
  metric (chunk metric fixed 2026-07-18; entry in
  [`done.md`](done.md)).
- **Async two-tier chunk compile** (optional, interactive-UX). Measured
  (first advance −24..31%, steady state bitwise-unchanged), default-off
  patch preserved, unlanded.
  [`../research/time_to_first_step.md`](../research/time_to_first_step.md)
- **Storage-halo width probe.** Biased order-5 pads storage to `n+8`
  per axis where the nominal reach needs `n+6` (centered: `n+4` vs
  `n+2`) — ~6% inflation on every upwind5 buffer, est. 2–3 ms/step
  @192³ on RTX-3060-class hardware. A core staggering-policy
  question, parity-sensitive, unprobed
  ([`../research/upwind5_revisit.md`](../research/upwind5_revisit.md)
  §6).
- **Hydro surface-flux correction: slice-only `A(1)`.** The H7
  constancy-preserving surface advective flux (owner-ratified
  default, [`../plans/active/hydrostatic_model_plan.md`](../plans/active/hydrostatic_model_plan.md)
  §H7) costs +18–49% on `se_centered` and +11–36% on `se_weno5`
  hydro steps in the comparison suite (worst at 2048²×64, resweep
  2026-07-17), flipping centered hydro from ~break-even to
  0.79–0.88 oc/fridom; `*_linear` configs unaffected. The correction
  `−q·A(1)` is mathematically nonzero **only in boundary-adjacent
  cells**, yet is evaluated as full-3D flux divergences (one per
  advected field per axis, memory-bound; face-velocity reuse is
  already XLA-CSE'd — measured perf-neutral). Lever: evaluate `A(1)`
  on the boundary-adjacent 2D slice only; needs a DSL
  slice/restriction path on the advecting-velocity faces.
  **Plan accepted 2026-07-18** (owner picked the same-mesh
  non-broadcasting `TraceSpace` architecture over ConstantSpace reuse
  / PointMesh activation): design, decision record, and phasing in
  [`../plans/active/boundary_trace_plan.md`](../plans/active/boundary_trace_plan.md);
  implementation not started.

## Channel eigenmodes on multi-device — remaining gaps

The projection now **runs** multi-GPU: the fused distributed
contraction shipped 2026-07-18 (merge `e60259de`, entry in
[`done.md`](done.md)). Still open:

- **File the two upstream jax issues** (owner go-ahead required — the
  drafts are ready). Both faults are jax/jaxlib 0.10.2:
  - **GPU (T5).** XLA:GPU/GSPMD lowers a sharded-transform-axis FFT
    through its distributed Cooley-Tukey decomposition whose
    twiddle-factor constants are `complex64` against `complex128`
    data; the HLO verifier rejects `multiply c64[] c128[]`. Repro +
    issue:
    [`../research/artifacts/channel_fftnorm_gpu/`](../research/artifacts/channel_fftnorm_gpu/).
    (Fridom no longer hits it on the 3-D channel — the fused lowering
    keeps FFT axes local — but the unsupported remainder below and any
    naive consumer still would.)
  - **CPU (T5b).** A batched-`eigh` heap corruption in jaxlib's CPU
    LAPACK on many-core hosts. Repro + issue:
    [`../research/artifacts/channel_sort_segfault/`](../research/artifacts/channel_sort_segfault/).
- **Unsupported sharded-periodic remainder** (kept on the narrowed
  taught `NotImplementedError`) — solution paths investigated
  2026-07-18
  ([`../research/eigen_remainder_investigation.md`](../research/eigen_remainder_investigation.md));
  the half-axis-sharded 3-D case **shipped** the same day
  (layout-aware half-axis re-designation, merge `feade7fa` — entry in
  [`done.md`](done.md)). Still the remainder:
  - *2-D channel* (highest exposure — the **default** for any 2-D
    channel on >1 device): recommend a gather path scoped to 2-D
    (exact, negligible cost at every size the dense engine can build;
    `em.q` is already replicated). A bounded-partner psum kernel was
    proven exact but shelved — its ×P basis-slicing edge only pays in
    a regime the dense `eigh` cannot reach. Needs owner ratification,
    then implementation.
  - *Non-1-D meshes*: **unreachable today** (the decomposition
    negotiates only single-axis layouts; a hand-built 2-axis mesh dies
    at decomposition build) — keep the defensive decline. The pencil
    primitive (per-mesh-axis `all_to_all` in one 2-D-mesh `shard_map`)
    is proven composable for the day a 2-D backend lands.
- **Pre-existing multi-device eigenbasis faults surfaced by the
  2026-07-18 validation** (both reproduce on the pre-merge dev; the
  existing multi-device eigen tests hit them before reaching the
  projection):
  - **Setup `GridFrozenError`:** on small sharded grids the
    `linearize(model)` probe's halo demand exceeds the frozen halo, so
    `channel_eigenpairs` cannot even build the basis.
  - **`mode()` / synthesis sharded-FFT crash:** the backward-only
    synthesis path (`mode`, `channel_random_state`) still crashes on a
    sharded periodic axis; it does not route through the fused
    contraction (out of its scope — a candidate follow-on).

Evidence, provenance probes, and the full re-attribution history:
[`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md).

## Mapped + advection + chunked scan goes non-finite on GPU

Found 2026-07-17 while GPU-measuring the CG tolerance (A100, jax
0.10.2, dev `b77f8582` — predates the tolerance work). A
terrain-following mapped nonhydro2 run with advection **on** goes
non-finite whenever `chunk_size >= 2`, at 256³ and even at dt=0.005
(2.5e-4 physical) — while the *same* steps run finite one at a time
(`chunk_size = 1`). Flat + advective + chunked is fine; mapped +
linear + chunked is fine; only the mapped advective step inside the
scanned chunk breaks, which points at a scanned-chunk
compilation/fusion fault, not physics. The known
`--xla_disable_hlo_passes=multi_output_fusion` workaround does **not**
fix it (so it is not jax#39100). Distinct from the mapped
reverse-mode NaN (that is a VJP-only masked singularity; this is the
forward primal). Work: bisect the module set (advection scheme ×
mapped metric terms) to a minimal repro, check CPU vs GPU and
chunk-length sensitivity, then either a fridom-side restructuring or
an upstream repro. Evidence: the CG-tolerance GPU measurement
(research record
[`../research/cg_stopping_criterion.md`](../research/cg_stopping_criterion.md),
GPU addendum).

## Finite-volume nonhydro — decisions and validation

All FV stages (F0–F6) are shipped — every non-immersed grid serves
`family="fv"`, unmapped/unimmersed grids are FV by *default*, moving
geometry now runs on FV too (ALE-on-FV closed, scoping §13 addendum 2),
and the FV nonhydro is feature-complete against nodal except cut cells
(out of scope by decision; entries in [`done.md`](done.md), records in
the scoping §10–§13). Open:

- **Stretched + terrain-following combined — residuals.** The
  correctness question and the implementation campaign shipped
  2026-07-17 (entry in [`done.md`](done.md); research + rulings in
  [`../research/stretched_terrain_combined.md`](../research/stretched_terrain_combined.md)).
  Open:
  - **`EnergyMetric`/eigenmodes weight `ps` by the flat extent on
    charts** — terrain energy diagnostics are physically
    inconsistent (model-layer, outside the hydrostatic package;
    flagged by the terrain build).
  - **Variable-depth implicit + split-explicit free surfaces**
    (H3): taught errors on charts today; need the variable-csqr 2D
    solve (hydrostatic plan §7). The barotropic volume-vs-energy
    tension is documented in the plan §8 (energy chosen; exact
    volume needs variable-`c²`).
  - **`MetricScaled` divides** (`mapped.py:219-222`) share the
    masked-singularity structure but are empirically reverse-safe;
    guard only if a composition exposes them (VJP-fix audit).
  - **GPU validation** of the new stretched+terrain paths (the
    standing 4-GPU baseline re-record shipped 2026-07-17 without a
    stretched+terrain-combined bench case, so this stays open — validate
    separately). Single-GPU leg done 2026-07-17 (gpu4 campaign
    wrap-up): `test_mapped_pressure_stretched.py` +
    `test_stretched_mesh.py` green on a real A100 (CUDA, fusion
    workaround set). Open remainder: the multi-GPU leg.
[`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md)

## Immersed partial cells — residuals

The immersed grid works in all three models with genuine partial
cells in every dimension (stages I0–I4 shipped 2026-07-17; entry in
[`done.md`](done.md), record + per-stage corrections in
[`../plans/active/immersed_partial_cells_plan.md`](../plans/active/immersed_partial_cells_plan.md)).
Open, none blocking:

- **Biased/upwind/WENO advection on immersed grids** — taught error
  today; needs the graded-fallback closure keyed on masks (the wall
  precedent, `graded.py`).
- **Mapped + immersed composition** — taught error; the mapped and
  masked PCGs are not composed.
- **Partial-bottom-cell hydrostatic pressure gradient** — the
  Pacanowski–Gnanadesikan refinement; the current unweighted `p_hyd`
  cumsum is 2nd-order away from partial bottom cells only.
- **Fraction-weighted Sadourny momentum** (sw2) and **masked
  closures** (diffusion/Smagorinsky/VerticalMixing self-reject on
  immersed grids today).

## Docs & examples rebuild

The bulk of the rebuild: **12 example ports** (only
`shallowwater/barotropic_instability.py` is on the new stack) and the
**entire prose page tree** — `docs/source/` still holds the old-stack
pages. Also retires the pre-rendered-media machinery
(`@skip_on_doc_build`, the git-LFS videos).

Two upstream gaps it needs: an `fr.io` root alias, and `pot_vort` on
shallowwater2 (documented, absent). Reader-facing content is
owner-reviewed privately before it reaches `dev` (see AGENTS.md).
[`../plans/active/docs_examples_plan.md`](../plans/active/docs_examples_plan.md)

## Cutover — retire the old stack

Gated on the docs rebuild (above) being far enough along not to break the
build, plus owner sign-off of the intentional-deltas table. **Physics
parity is closed**; what remains is mechanical. The old packages are still
on disk (136 modules, 107 test files) and still exported from
`src/fridom/__init__.py`.

- Rehome the two survivors (`framework/utils/`, `framework/logger.py`) —
  everything else depends on this decision (likely a top-level
  `fridom.utils`); then rewrite the 45 source + 16 test imports.
- Delete the old packages and their tests; rename `nonhydro2` /
  `shallowwater2`; fix the root exports, `tests/conftest.py`, the CI
  multi-device path, coverage config, benchmarks.

The old "rename `framework2` → `framework`" plan is obsolete: the split
already landed the new stack as `spatial` + `model`, so only the two model
packages still carry a `2`.
Records: [`../plans/active/cutover_parity_plan.md`](../plans/active/cutover_parity_plan.md)
(parity, sign-off) and
[`../plans/active/cutover_checklist.md`](../plans/active/cutover_checklist.md)
(the executable swap list).

## Time-dependent parameters **and time-dependent fields**

Requested by Silvano, 2026-07-14 (while landing `ETDRK4`).

Today a `Ramp` reaches only the *scalar* parameter leaves. `f` and
`csqr` are not scalars: they are materialized at assembly into
AUXILIARY **fields** (`FPlaneCoriolis._f_default` /
`DynamicalCore._csqr_default` call `jnp.full(space.shape, self.f0)`),
so `FPlaneCoriolis(f0=Ramp(...))` raises a bare `TypeError` from
`jnp.full`. Two levels are wanted, and the second is the real ask:

1. **Time-dependent scalars** — `f0`, `csqr`, ... accept a
   `TimeDependent` and resolve at stage time (`resolve_at`). Mostly a
   matter of routing the declaration's `default=` through the
   time-dependent path instead of freezing it once, plus a taught error
   where a consumer needs a frozen snapshot.
2. **Time-dependent fields** — `f(y, t)`, `csqr(y, t)`: a *profile* that
   itself evolves. This is not just a leaf swap. An AUXILIARY field is
   carry-resident and its treedef must stay scan-stable, so the
   time-dependence has to enter either as a stage that rewrites the
   field (a `SELF_UPDATE`/`DIAGNOSE` kind) or as a declaration-level
   "recompute from `(coords, t)` each step" contract. Which of those is
   right is the design question; the coverage lint and the halo/GAP-B
   rules both bear on it.

Level 1 and the *affine-blend* subset of level 2 (a field that moves
along an affine path in declared scalars, e.g.
`f(y,t) = f0(t) + beta(t) * y`) **shipped 2026-07-17** as stages
R1/R2 of
[`../plans/done/adiabatic_ramping.md`](../plans/done/adiabatic_ramping.md).
What stays open **here**: the general case (profiles with non-affine
time dependence), plus one small follow-up from that landing —
`dsqr`'s AR-D7 report is cross-module (owned by `DynamicalCore`,
consumed by `ConstantStratification.buoyancy_force`) and is
documented but not wired.

**Interaction with the exponential stepper** (the reason this surfaced):
`ETDRK4` freezes `L` in an eigenbasis snapshot. Anything time-dependent
that lives in `N` is already correct (the `Ramp` on `scaling.rossby`
is tested). But a time-dependent `f`/`csqr` lives in **L**, and
`L(t1)`/`L(t2)` do not commute, so `exp(L dt)` stops being the
propagator — the stepper would silently integrate a stale operator. Any
design here must say what `ETDRK4` does about it. The measured fallback
is recorded in
[`../research/exponential_stepper.md`](../research/exponential_stepper.md)
§5: keep the stiff time-independent part (gravity) in the eigenbasis,
leave the time-dependent part in the tendency — still 52.7x AB3, capped
by the inertial rather than the gravity CFL. Note also that a
time-dependent `L` has no fixed eigenbasis at all, so the discrete
eigenanalysis is itself undefined in that regime.

## Adiabatic-ramping docs — example review (deferred at landing)

ROADMAP 3.8 shipped 2026-07-17
([`done.md`](done.md) §3.8;
[`../plans/done/adiabatic_ramping.md`](../plans/done/adiabatic_ramping.md)),
with R6 (the double-ramp example + Advanced Topics docs page) merged
**on owner instruction without the private content-review pass**.
Open work: the owner review of
`examples/shallowwater/adiabatic_double_ramp.py` and
`docs/source/advanced/adiabatic_ramping.rst` (projection onto the
working tree, `REVIEW:` markers, sweep-and-apply per AGENTS.md), plus
the style-guide tensions flagged at preparation: citation
infrastructure for the unpublished JFM draft (bibtex vs the current
prose citation), whether the one-chapter Advanced Topics scaffold
stands or folds into the docs rebuild, doctest wiring for inline
snippets, and API cross-refs as literals until the new-stack API
reference lands.

---

# Sized, deferred — real work, but nothing is waiting on it

Real work, but nothing is waiting on any of it. The first two were
sized on 2026-07-13 against real diffstats of comparable landed work;
none is hard to justify *technically*, all fail the "who wants it" test
today. Promote an item the moment a consumer appears.

## Boundary closures, stage 2e — the Robin dynamic `(α, g)` path

*Medium (1-2 weeks; ~400-700 LOC source, ~300-400 test, against the
`9a95202a` analogue of 21 files / +1054).* **No consumer exists**: `BC.ROBIN`
appears only in `bc.py`, two `NotImplementedError` raises, and the tests
asserting those raises. Both models use only Dirichlet/Neumann walls.
Note (2026-07-17): flux-type boundary forcing shipped as a *tendency*
module (`BoundaryFlux`, [`done.md`](done.md)) and deliberately does
**not** consume this path (plan §BF-D1) — 2e's consumers remain
inhomogeneous boundary *values* (moving lids, prescribed wall
buoyancy) and dynamic Robin data, and none exists yet. Do it when a
model needs one — otherwise it ships untested-by-use.

Two genuine design questions, unspecified in the record, are why this is
not days: **(1)** "seeded at assembly" fights "compiles once across an α
sweep" — `Grid` is not a pytree and the registry carries no dynamic
fields, so a row seeded at assembly makes α a captured constant and a
swept α *does* retrace; α/g must arrive as traced module params read at
apply time. **(2)** Under sharding the fill runs *inside* `shard_map`, so
an inhomogeneous `g` on a boundary face is a transverse-sharded operand
needing its own `in_specs`/masking.

Stage 2f (the mirror/halo-claim widening) is small but pure perf, gated on
profiling nobody has done. Defer.
[`../plans/active/boundary_plan.md`](../plans/active/boundary_plan.md)

## Diffusion/friction closures at walls and on terrain — residuals

Stages 0–4 shipped 2026-07-17 (walls free/no-slip on the nodal
family, implicit no-slip rows, mapped along-σ, the `VerticalMixing`
stretched/terrain gates, the measure-divide VJP seal), and the FV
walled lift shipped 2026-07-18 (walled `CellAvg` targets take the
same flux-retag closure — both families now covered; entries in
[`done.md`](done.md), record + §9 addendum
[`../research/diffusion_walls_terrain_scoping.md`](../research/diffusion_walls_terrain_scoping.md)).
Open:

- **Measure-aware implicit column** — the `VerticalMixing`
  stretched/terrain gates stand until the banded column learns
  `grid.measure` widths + the terrain Jacobian (the multigrid V-cycle
  already consumes measure widths on stretched columns — N3, entry in
  [`done.md`](done.md) — this is its implicit-diffusion twin; pairs
  with the flagged variable-kappa follow-up, `implicit.py`).
- **Stage 5** — the geopotential-correct full-metric (then rotated)
  diffusion tensor; deferred, separate plan (record §3.6 A/C).
- **Owner ratification** — shipped on the record's RECs, unreviewed:
  `slip="free"` default, biharmonic same-treatment-both-passes
  (incl. no-slip), along-σ as the first terrain deliverable
  (record §6 calls 1–3).
- **`grid.measure` pre-assembly ordering (lead)** — querying a
  mapped mesh's measure before `Model` assembly freezes the
  decomposition early; the stale cached measure then
  shape-mismatches the step frame. Normal build→run ordering is
  unaffected (probe artifact; matters for diagnostics workflows).
- Partial slip (NEMO `shlat`-style) = a Robin wall flux — a
  boundary-closure 2e consumer (entry above).

## Open boundaries — sponge is small, through-flow is large

*Sponge tier: SMALL (days). Genuine through-flow: LARGE (3–6 weeks,
four shippable stages; scoped 2026-07-17).* **No consumer exists** —
nothing on the roadmap needs net inflow/outflow today, and a
sponge-emulated open boundary (`Relaxation` + coordinate mask next to
a wall) is already possible with zero code. The scoping record
([`../research/open_boundaries_scoping.md`](../research/open_boundaries_scoping.md))
sizes the tiers: a packaged `SpongeLayer` module (days); prescribed
through-flow — open-side space structure (the wall face becomes a
DOF; widest blast radius), boundary data via the FV wall-flux slot or
stage 2e ghost-fill rows, an MITgcm-style volume-flux balance pass
(the Poisson solvability condition; the spectral solver itself needs
**no** surgery — predictor-only inflow, the Oceananigans route), and
Orlanski/perturbation-advection outflow as CONSTRAINT-stage
overwrites. A shallowwater2 Flather module (~1 wk) is the cheap pilot
— no elliptic solve. Five owner calls are listed in the record (§4)
before Tier 2 starts. Doing Tier 2 finally gives boundary-closure 2e
its consumer.

## Eigenmode phase I — `Banded` as a first-class realized map

*Medium (1-2 weeks; ~500-700 LOC source, 8-12 files) for the FD/nodal
vertical route — but **large** if the `Fourier ⊗ Chebyshev` headline is
taken literally.* **No consumer exists**: the walled nonhydro pressure
solve is already purely diagonal (`Fourier×Fourier×Cosine`);
variable-coefficient wall-bounded eigenmodes are served by the shipped
channel engine (`model/eigen_channel.py`); and the terrain-mapped pressure
has cross terms, so it is not block-diagonal and `Banded` would not serve
it either. The one plausible win — a stretch-only vertical map, where a
per-mode Thomas solve would replace fixed-iteration CG exactly — is
speculative and unrequested.

**Landmine:** `d/dz` in *Chebyshev coefficient* space is dense-triangular,
not banded. A real Fourier⊗Chebyshev solve needs the Shen/Galerkin BC
basis and Clenshaw–Curtis measures, neither of which is built. Only the
FD/nodal-vertical variant is actually in reach.
[`../plans/active/projection_eigenmode_roadmap.md`](../plans/active/projection_eigenmode_roadmap.md)

## High-order mapped stencils — the full lift (spike answered)

*Medium (1-2 weeks).* Held back from "next steps" on payoff, not
difficulty. The C-grid biased advection tendency is only 2nd order on
**any** mesh once the advecting velocity varies, so the mapped divisor
does **not** buy asymptotic order back for `UpwindAdvection` /
`WENOAdvection`. The real payoff is ENO/dispersion behaviour restored on
stretched meshes (the reason those schemes exist), honest design order
for standalone `WenoReconstruction` and `FiniteDifference` order > 2,
and mapped one-sided FD. That is quality on stretched meshes, not a new
capability headline — worth doing, not urgent.

The plan's blocker is answered: the 2026-07-16 Jacobian spike identified
the **same-row discrete Jacobian** as the divisor (restores design order
*and* satisfies the discrete metric identity exactly; entry in
[`done.md`](done.md)). All that remains is the lift itself, on the
recorded route
([`../plans/active/high_order_mapped_plan.md`](../plans/active/high_order_mapped_plan.md)
§3, numbers in
[`../research/mapped_jacobian_spike.md`](../research/mapped_jacobian_spike.md)).

## Coefficient-space product/power rows — needs an owner call

*Small in code, but a semantics decision, not a missing row.*
Elementwise multiplication of two Fourier-coefficient fields is *not*
the product of the represented functions (it is a convolution), so
registering it under the same `("multiply", space)` kind invites silent
nonsense. Needs an owner ruling first; it blocks nothing. The last open
item of the Phase-2 grid follow-ups (the rest landed — see
[`done.md`](done.md)).
[`../plans/active/phase2_grid_followups.md`](../plans/active/phase2_grid_followups.md)

## Mapped-solve residual levers — measured, none currently worth taking

*The parent line — "multi-device compile and execution cost", formerly
roadmap 3.9 — closed 2026-07-16 (entry in [`done.md`](done.md)). What
survives is a short list of measured, deliberately-not-taken levers —
promote one only when its trigger appears:*

- **Single-precision distributed solve** — `single_precision_solve` is
  a no-op on multi-device walled/mapped grids (the full-precision
  distributed solve takes precedence; documented at
  `spectral_solve.py`). Worth roughly the single-device −10% if a
  multi-device user ever asks.
- **Distributed-transform planner size floor** — small problems pay
  unamortized collective latency (32³ walled/mapped +5.5%); a size
  floor on the planner is the lever if toy-size multi-device runs ever
  matter
  ([`distributed_transform_plan.md`](../plans/active/distributed_transform_plan.md)).
- **Surplus staggered reblock leg (`n = n_cells + 1`)** — deliberately
  still on the global reblock path (needs a `P*(cells+1)` frame plus
  one realigning collective-permute; gate documented in
  `decomposition/tensor.py`). No hot-loop consumer exists — the Neumann
  pressure sibling keeps the pressure-space shape. Revisit only if a
  Neumann-outer field enters a hot loop.
## Multigrid preconditioner — follow-up measurements

The semicoarsened V-cycle preconditioner shipped 2026-07-17 and the
V-cycle kernel swap it called for shipped 2026-07-18 (merge
`0ece46b1`; both entries in [`done.md`](done.md), measurements in
[`../research/multigrid_kernel_study.md`](../research/multigrid_kernel_study.md)
§Addendum). Open, none blocking:

- **cuSPARSE kernel under GSPMD (multi-device) — unvalidated.** The
  line smoother's `method="auto"` resolves to the batched
  `lax.linalg.tridiagonal_solve` (cuSPARSE) on any GPU backend,
  including sharded multi-GPU runs. A custom call's GSPMD
  partitioning is not guaranteed: XLA may all-gather the batch axes
  instead of partitioning them (correct but slow). Validate on real
  4×A100 (parity + no unexpected all-gathers in the HLO); until
  then a multi-device run that sees them should set
  `multigrid_tridiagonal_method="pcr"` (pure jax, partitions
  cleanly). Caveat documented in `banded.py`.
- **Mapped GPU wall-clock still behind spectral after the swap** —
  measured in-model: parity at 128³ (0.975×), 0.67× at 512³; the
  1.5× GB-2 bar stays unmet and the 128³-study projection "likelier
  at larger n" is refuted at 512³. The study-ranked residual levers
  (fewer coarse sweeps; cheaper mapped operator applies — now the
  dominant V-cycle cost) are unclaimed; take only with a concrete
  mapped-GPU production driver. Immersed remains the case where
  multigrid wins (study projection 1.3–2.0×, in-model post-swap
  standing unmeasured).

## Differentiable run surface — `model.propagator()`

Reverse-mode `jax.grad` through a run is exact and policy-tested
(record:
[`../research/jax_grad_run_investigation.md`](../research/jax_grad_run_investigation.md);
AGENTS.md "Differentiability policy"), but the supported spelling is
still the private kernel recipe (`_chunk_body` + leaf splicing by
identity). The public surface is one composition away — nothing new
has to be invented, only assembled:

- **Surface**: `model.propagator(*, wrt=("friction.nu", ...), steps,
  remat=None)` returning a pure `(theta, state=None) -> State`.
  Name resolution reuses the `update_parameters` machinery verbatim
  (binding table -> `(slot, attr)` -> `_replace_leaf`,
  `model.py:1628`) but builds a carry *transformer* instead of
  committing; `wrt` names bound parameters (incl. `TIME_STEP`) or
  PROGNOSTIC fields (IC differentiation splices
  `state[name].storage`).
- **Kernel**: `_chunk_body` without donation; `record` static,
  stepper loop-invariant; no io, no `bool(panic)` host sync — the
  panic pair rides the returned carry for functional inspection.
- **`remat`**: optional `jax.checkpoint` on the scan body. Reverse
  mode tapes O(steps) (measured ~1.4 MB/step at 8^3); production
  adjoints need this knob. Requires a small hook in `_chunk_body`.
- **Warm-up semantics**: default to a fresh stepper state (gradients
  include the multistep warm-up ramp); document.
- **Taught errors**: steppers with `freezes_linear_operator`
  (ETDRK4) refuse `wrt` names owned by the frozen operator —
  gradients w.r.t. a stale `exp(L dt)` snapshot are silently wrong.
- **Follow-ons unlocked**: the D5 `TangentPropagator` (`jax.jvp` of
  `model.tendency`,
  [`../specs/model/04_run_loop_io.md`](../specs/model/04_run_loop_io.md))
  shares the name-resolution piece; the AGENTS.md test-policy pattern
  migrates from the private kernel to the public surface once it
  exists.
- **Residual hazards to sweep when first exercised under grad**: the
  `metric_weight` divisions in `model/modules/coriolis.py`
  (`/ w.to(v)`, `/ w_1`, `/ w_2`) share the masked-0/0 class the
  Sadourny PV division was cured of; guard like
  `_potential_vorticity` when that path meets an adjoint.
---

# Long-term goals

| #   | Task | Notes |
|-----|------|-------|
| 3.1 | **Hydrostatic model — external comparison legs** | The model itself shipped 2026-07-17 (entry in [`done.md`](done.md); record [`../plans/active/hydrostatic_model_plan.md`](../plans/active/hydrostatic_model_plan.md) §8). The **Oceananigans leg executed 2026-07-17** (out-of-tree `benchmarks/comparison` harness, single A100; machine-precision linear parity, full HY-D6 ladder — results in the bench repo's `results/HYDRO_REPORT.md`; it also surfaced the implicit+advection surface-closure instability, root-caused and fixed same day, plan §H7). Still open: the **Veros and pyOM3 legs** (**pyOM3 source access needs the owner**) — and the **owner review of `examples/hydrostatic/comparison_baseline.py`** — landed on `dev` 2026-07-17 by owner authorization *before* review (deviation from the examples-review workflow, owner instruction in chat); the review itself is still owed — sweep `REVIEW:` markers / direct edits when it happens. Designed-fors (T/S + EOS, topography / variable-`csqr` CG solve, z*/ALE, spherical) stay in plan §7. |
| 3.7 | **Spherical nonhydro** | The 3D spherical chart (`X(lon, lat, h)`, so the metric comes out diagonal and `w = dh/dt` is already physical) needs the C2 chart metrics and the C3 elliptic machinery to meet: the pressure operator becomes the Laplace–Beltrami on the chart — still SPD under the sqrt(g)-weighted product, so the PCG structure carries over, but the operator assembly must be written. Not the first 3D-spherical consumer: a hydrostatic model needs no pressure solve and is the likelier first use (3.1). |
| 3.2 | **Coupled models — design** | `jax.distributed`, field exchange between models on different meshes/devices/processes, a `Coupler` module plus regridding operators, a synchronization schedule. **Pre-designed** in [`../specs/model/09_coupling_designfor.md`](../specs/model/09_coupling_designfor.md) (precedent survey + adversarial walk + architecture; the class specs carry its CS-1..18 constraints, so 3.2 stays a pure addition). |
| 3.3 | **Coupled models — implementation** | Same-process multi-device, then multi-host. Depends on 3.2. Its old cost prerequisite (3.9/3.10 — "decomposed runs must be affordable before coupling them is credible") is met: the multi-device execution-cost line closed 2026-07-16 ([`done.md`](done.md)). |

---

## Known gaps in the shipped surface

Small, but they are promises the specs make that the code does not keep
(surfaced by the 2026-07-13 spec audit):

- `model.blank_state()` / `model.state_space(name)` — specified, never
  built. Either add the two one-liners or strike them from the spec.
- `fr.modules.WindowAccumulator` — named as shipping in four places;
  does not exist.
- `add_prognostic` — every stepper family's combine step wants it.

## Cross-cutting rules

- Mirrored tests (95% branch coverage gate), ruff-clean.
- Benchmarked with the 0.1 infrastructure (runtime, compile, memory).
- The old `framework` stays runnable until the cutover; new work does not
  go into it.
- Unstructured grids stay out of scope; the designed-for grid extensions
  must not be precluded by the iteration-1 core.
