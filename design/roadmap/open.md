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
- **Hydro surface-flux correction — weno re-measure + multi-host
  remaining.** The centered §8 criteria are **met** (arm sweep
  2026-07-18, owner-requested; record in
  [`../plans/active/boundary_trace_plan.md`](../plans/active/boundary_trace_plan.md)
  §9): overhead vs `surface_flux=False` single-digit-to-negative
  on `se_centered`, oc/fridom 0.92–1.04 top rungs, `im_centered`
  0.99–1.12 with all rungs stable; per-scheme lowering default
  landed (`893e82a8`). Same-day slice-exactness audit fixed a
  real top-row error (~15–17% u/v) for order-5 biased staggered
  momentum (`81995781` — biased momentum now takes the exact
  full-3D correction; constancy-oracle record in plan §9).
  Real multi-host validation (plan §4 gate) is **met** (2026-07-18
  evening, owner-requested: `srun -n 4` bitwise/1e-15 vs 1-GPU,
  both schemes; record in plan §9 — including the multi-process
  compile-cache deadlock it exposed and fixed, `94786a7c`).
  Remaining (sharpened by the 2026-07-19 analysis of the
  remainders job 26350823 — that job raced parallel dev merges in
  the shared checkout, so its weno5 arms ran at three commits and
  the scatter arm lost 4/5 rungs to a mid-merge conflict at
  child-import time; centered arms clean at `33707661`):
  (a) post-reroute weno5 ladder re-measure — the only clean
  same-commit A/B pair (rf=28) has embed +1.0% over scatter, so
  the provisional biased `"embed"` default stands, but a
  definitive verdict needs a clean re-run (owner-gated GPU,
  ~20 min); (b) the `surface_flux=False` opt-out is now measured
  SLOWER than the scatter default (up to −16% for the default at
  big rungs) and +48% over the pre-H7 point (default: +24%) —
  the linear rungs are bitwise-stable across every measurement,
  so it is advective-path only; either the dirty-tree pre-H7
  baseline is invalid or a real change entered
  `c669ec6f..565eaa51` — owner decision: one-rung bisect (small
  GPU job) or won't-chase (oc parity 0.92–1.04 holds either
  way). The (c) barotropic-IC finding was **ruled an IC gap and
  fixed** 2026-07-19 (entry in [`done.md`](done.md)). Step-guard
  checkpointing stays on Silvano's own batch cadence (never
  agent-initiated).

## Channel eigenmodes on multi-device — remaining gaps

The projection now **runs** multi-GPU: the fused distributed
contraction shipped 2026-07-18 (merge `e60259de`, entry in
[`done.md`](done.md)). Still open:

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
Both pre-existing eigenbasis faults surfaced by the 2026-07-18
validation are **fixed** (entries in [`done.md`](done.md)): the setup
`GridFrozenError` was a negotiate/verify cap asymmetry (record:
[`../research/halo_sharding_invariants.md`](../research/halo_sharding_invariants.md)
§1) and the `mode()`/synthesis crash is closed by the fused
backward-only synthesis on 3-D channels (the 2-D channel is the
ratification item above).

Evidence, provenance probes, and the full re-attribution history:
[`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md).

## Naive GSPMD transform path — phased illegality (phases 2+)

Owner-approved 2026-07-18; phases 0–1 shipped the same day (Tier-1
taught guard at the `Transform` seam; 3-D channel synthesis reroute —
entries in [`done.md`](done.md), record:
[`../research/gspmd_naive_transform_illegality.md`](../research/gspmd_naive_transform_illegality.md)).
Remaining, per
[`../plans/active/gspmd_transform_illegality_plan.md`](../plans/active/gspmd_transform_illegality_plan.md):

- **Phase 2** — the 2-D channel gather path (the ratification item in
  the section above; covers projection *and* synthesis).
- **Phase 3** — a standalone distributed `Transform` apply consuming
  the unconsumed `distributed_*_plan`s (L): re-legalizes eigenmode and
  state transforms, the exponential stepper, and the Krylov spectral
  apply on sharded grids; converts the residual marked-test debt.
- **Tier-2 decision (owner)** — whether all-local naive transforms on
  a multi-device mesh (silent all-gather) also become illegal, with an
  allow-replicated escape for Chebyshev/mismatched-layout solves.
- **Ratifications (owner)** — verify-side capping of explicit `halo=`
  (shipped behavior, consistent with negotiate) and the
  `_cap_for_sharding` over-reach onto non-sharded axes.
- **weno5 momentum z-seam (~1e-5) on z-sharded layouts** — the
  residual of the 2026-07-19 seam fix (entry in
  [`done.md`](done.md)): `WenoReconstruction`'s vertical
  footprint exceeds the negotiated z-halo of 2 on a z-sharded
  layout, leaving a ~1e-5 seam error in u/v (buoyancy is fixed;
  a z-halo >= 3 cures it in probe runs). This lives in the
  owner-governed halo negotiation/cap machinery (the same family
  as the two ratification items above), so it is deliberately
  left for an owner call rather than patched around; the
  z-shard parity test pins the current behavior (b tight,
  momenta finite-only) and documents the residual in-code.

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
  - **Terrain diagnosed "w" output labeling** — the slope-advection
    buoyancy term shipped (`fix/terrain-buoyancy-slope-term`, entry in
    [`done.md`](done.md)), so `b` now couples to the physical
    `w = Jω + u·Zₓ + v·Z_y` internally; the diagnosed **output** field
    `w` still carries the contravariant flux `Jω`, not the physical
    vertical velocity — an output/documentation question only.
  - **nonhydro2 mapped energy leak — un-diagnosed cousin** (research
    §1.6 of
    [`../research/energy_metric_asymmetry.md`](../research/energy_metric_asymmetry.md)):
    the bare mapped nonhydro2 operator leaks the physical energy
    pairing on a divergence-free random state at −6.5e-3 (a = 0.2;
    CG-iteration-independent, nodal ≡ fv bitwise; energy bounded in
    time integration). Whether it is the same class as the hydrostatic
    slope-term gap (the mapped buoyancy/w coupling convention) or
    ordinary interpolation-transpose truncation is **untested** — the
    n-scaling probe (research §1.4 recipe: fixed resolved broadband
    state, skew vs n) has not been run. Follow-up: run the n-scaling
    probe; if resolution-independent, audit the mapped w/buoyancy
    convention like the hydrostatic case.
  - **Variable-depth split-explicit free surface** (H3 residual):
    still a taught error on charts. The *implicit* half shipped
    2026-07-18 (multigrid_generalization_plan phase B: the
    volume-exact variable-csqr solve, resolving the plan §8
    volume-vs-energy tension for the implicit variant); the
    subcycle's terrain transport form is the remaining half.
  - **Terrain + walled-horizontal** (the one remaining layer of the
    walled-horizontal gap; the flat/immersed gap itself is closed —
    entry in [`done.md`](done.md)): a hydrostatic model on a
    sigma-chart terrain grid with a walled *horizontal* axis still
    fails to assemble. This layer is a genuine missing conversion,
    not a tag relabel: the mapped slope gradient
    (`hydrostatic/modules/core.py` `_slope_gradient` →
    `spatial/coordinate_mapping.py` `_at_space` → `field.to`)
    needs to *interpolate* a wall-normal-face quantity along the
    walled axis, and the BC-free `('interpolate', Inner(x))` row
    (correctly) does not exist. Likely spelling: retag the
    face quantity onto its Dirichlet sibling first (the odd-parity
    claim of `_dirichlet_mid`), so the registered tagged
    interpolate row resolves — but the seam sits in the shared
    `coordinate_mapping` machinery, so the claim needs a
    per-call-site justification, not a blanket arm.

[`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md)

## Immersed partial cells — residuals

The immersed grid works in all three models with genuine partial
cells in every dimension (stages I0–I4 shipped 2026-07-17; entry in
[`done.md`](done.md), record + per-stage corrections in
[`../plans/active/immersed_partial_cells_plan.md`](../plans/active/immersed_partial_cells_plan.md)).
Open, none blocking:

- **Mapped + immersed composition** — **shipped** (M0–M4 nonhydro2
  composed solve + multigrid + advection proof; M5 hydrostatic
  wet-column terrain barotropic solve, masked contravariant continuity,
  order≥2 chart-mask guard —
  [`../plans/active/mapped_immersed_composition_plan.md`](../plans/active/mapped_immersed_composition_plan.md) §6).
  Residuals: the terrain barotropic **multigrid** preconditioner is not
  yet wet-aware (taught error on a terrain + immersed grid — the masked
  spectral default converges in ≤17 iters, so no lever open at these
  sizes); split-explicit + terrain stays a taught error (pre-existing);
  genuine-chart (J≠1) physical column-equivalence twin is ambiguous
  (algebraic gates substitute, M2-M4 correction-6 precedent).
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

## Time-dependent fields — remaining follow-ups

The general non-affine time-dependent field mechanism
(`ProfileFunction` + the `SELF_UPDATE` recompute), the `dsqr` AR-D7
cross-module wiring, and the `ETDRK4` answer (refuse a time-dependent
`L`) shipped across three waves — see the done.md entry and
[`../plans/done/time_dependent_fields.md`](../plans/done/time_dependent_fields.md).
What remains:

1. **Nonhydro `n2(z, t)` law-valued path (TDF-D7).** The Coriolis
   `f(y, t)` and shallow-water `csqr(y, t)` law paths shipped;
   `ConstantStratification` does not yet accept a `ProfileFunction`
   for a non-affine `n2(z, t)` profile. A named follow-up, deliberately
   out of the shipped plan's scope.
2. **`FieldBlend` unification (TDF-D9, owner decision).** The
   affine-blend `f_coriolis` still evaluates term-side (AR-D2), which
   leaves the known IO-staleness wart (`f_coriolis` IO shows the `t=0`
   snapshot). Re-basing it on the `SELF_UPDATE` rewrite path would fix
   that too, but relitigates an owner-ratified ruling and changes
   tested behavior — logged as an open question for the owner, nothing
   more.
3. **A re-diagonalization contract for the analysis tools — declined
   (TDF-D6).** The eigen/analysis surfaces (`eigenbasis` /
   `ChannelEigenmodes` / `Eigenmodes.from_model`, the `EnergyMetric`
   weights) stay deliberately time-frozen: they snapshot `f`/`csqr`/
   `dsqr` at a fixed `at_time` (docstrings now say so). A contract that
   makes them track a time-dependent `L` is explicitly out of scope —
   a time-dependent `L` has no fixed eigenbasis, so the discrete
   eigenanalysis is undefined in that regime. Recorded here only so the
   decision is not silently lost.

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
§§Addendum, Addendum 2). Open, none blocking:

- **Semicoarsening coarse-level layout — perf/hardening follow-ups.**
  The multi-device parity break itself is **closed** (root-caused,
  cured, verified on real 4x A100 — entry in [`done.md`](done.md),
  record
  [`../research/semicoarsen_multidevice_regression.md`](../research/semicoarsen_multidevice_regression.md)),
  but the layout that exposed it remains the negotiated choice: every
  real semicoarsening config still **sigma-shards its coarsest level**
  (correct to machine precision since `b57e3e78`, yet one coarse sweep
  lowers to ~181 collective-permutes on the smallest, most
  latency-bound grid). Follow-ups, none blocking: (a) owner call —
  prefer replication for coarse levels (demote/exclude the hierarchy
  `vertical` in the coarse-level shardability ranking, or replicate
  below a cell-count floor); measure the coarse-level timing first;
  (b) a hierarchy-builder warning when a level's layout shards the
  line-smoother axis (future negotiation-policy drift fails loudly);
  (c) stretched-base eager hierarchy pre-warm so stretched columns
  take the full-coarsening default (its coarsest level replicates
  naturally, and the `MappedIntervalMesh`-ctor jit limit is bypassed
  via the `Grid.coarsened` memo); (d) `multi_device` markers for the
  parity-test victims (unmarked 4-device-only failures are invisible
  to single-device CI).
- **Coarse-level agglomeration — remaining follow-ups.** The mechanism
  (replicate coarse levels below a per-shard-extent threshold, knob
  `multigrid_agglomerate` default OFF) **shipped** 2026-07-19 (merge
  `9e08493f`; entry in [`done.md`](done.md), record
  [`../plans/active/multigrid_agglomeration_plan.md`](../plans/active/multigrid_agglomeration_plan.md)
  §4). CPU forced-device HLO confirms the census's flagged latency
  collectives vanish (coarse z-halo permutes 24→0, column-transpose
  all-to-alls 6→0). Open, none blocking:
  - **GPU wall-clock leg** — the census's projected ~9–14 ms/step
    recovery at 128³ is **not** wall-clock-validated: the named 4×A100
    allocation was dead at run time and per AGENTS.md no new GPU job
    was submitted. Needs a live 4-GPU allocation (owner-provided).
  - **`tau` sweep** — `tau ∈ {2,4,8}` unrun; `tau = 4` is the default
    on structural grounds (catches the 1–2-plane coarse levels), to be
    pinned by the GPU sweep above.
  - **Replicated-reduction folding** — on CPU, XLA GSPMD re-partitions
    the replicated coarse levels' projection sums into all-reduces
    rather than folding them to local sums, so all-reduce/all-gather
    counts *rose* and the total collective count is net flat (the win
    is removing the largest-payload all-to-alls and tiniest sub-KB
    permutes, not the raw count). Whether a `with_sharding_constraint`
    hint folds them on GPU is open.
  - **Default-on decision (owner)** — whether agglomeration becomes the
    multi-device mg default, gated on the GPU sweep.
- **Residual mapped-GPU levers, unclaimed** — fewer coarse sweeps;
  cheaper mapped operator applies (the finest level dominates the
  post-swap V-cycle: one sweep = 15.7 ms cuSPARSE solve + 12.0 ms
  operator apply at 512³). Take only with a concrete driver toward
  the 1.5× GB-2 bar. Immersed in-model post-swap standing is now
  measured (2026-07-18, kernel study Addendum 2): mg 1.09×/1.12× at
  128³/256³ at the production budget=100, where spectral also converges
  (71–73 iters) — the 1.3–2.0× projection was a budget=30 artifact, and
  mg is the only converged option below budget ≈70. On 4 GPUs
  mg-cuSPARSE is 1.11× at 512³ (bandwidth-amortized) — the large-n end
  where the collective count is a smaller fraction, so any remaining
  lever hunt there is large-n / multi-GPU-aware.

## TangentPropagator — the D5 forward-mode surface

*Small.* `jax.jvp` of `model.tendency` (spec
[`../specs/model/04_run_loop_io.md`](../specs/model/04_run_loop_io.md));
the shared name-resolution piece shipped with `Model.propagator` (the
public reverse-mode surface — entry in [`done.md`](done.md)). **No
consumer exists** (NNMD descoped); build when one appears. Plan §5.4:
[`../plans/active/differentiability_plan.md`](../plans/active/differentiability_plan.md).
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
