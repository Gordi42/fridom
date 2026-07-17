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

## Guiding target (end state)

- **No ModelSettings**: a `Model` is assembled from a grid and modules;
  every physical parameter (`f0`, `n2`, `csqr`, ...) lives in a module.
- **Everything is one pytree**: modules can modify anything during
  `update`, and the whole run is a single `jax.jit` call (no Python time
  loop).
- **Grid = function spaces**: fields live on function spaces, operators
  map between spaces, mesh arrays are lazy, and new grid types
  (stretched, spherical) slot into the same abstraction.
- **Fields are ergonomic**: init from callables, dimension reduction
  (`g = f.sel(x=a)`).
- **Models**: nonhydro, shallowwater, and coupled multi-model runs
  (multi-device, later multi-host).

Phases 1 and 2 delivered the first four; the grid is `fridom.spatial`
and the model layer is `fridom.model`.

---

# Next steps

## Performance guard — wire the benchmark harness as a CI gate

The A/B harness exists (`benchmarks/model/bench_step.py`, **committed**
baselines `benchmarks/baselines/step-gpu{1,4}.json`,
`--fail-on-regression`, the `nh_flat_prime` / `nh_flat_walled_x` guard
cases — see [`done.md`](done.md)), but the CI benchmark job still only
smoke-runs ("No timing assertions", `.github/workflows/tests.yml`), so
a silent perf regression on the untimed CI path stays green.

Also open: **fast-path assertions beyond the solve.**
`tests/nonhydro2/test_distributed_projection.py` asserts the
distributed fast path for all four solve geometries (periodic,
walled-z, walled-x, prime), and the halo-claim assertions guard
storage-frame arithmetic — but the other fast paths remain unasserted.
A change that pushes production off one of them would pass the suite
and quietly cost ~2x at scale.
[`../plans/active/perf_geometry_merge_plan.md`](../plans/active/perf_geometry_merge_plan.md)

## Gaps against the Oceananigans reference comparison

The 2026-07 matched-protocol comparison against Oceananigans.jl
0.105.3 (suite and full report live in the untracked
`benchmarks/comparison` — out-of-tree by design, see its README)
confirmed the new stack wins where it is pressure-solve-bound. The
three trailing areas it identified are largely closed — single-GPU
memory ceiling, time-to-first-step, WENO throughput (entries in
[`done.md`](done.md)). Still open:

- **Re-run the comparison suite** on post-fix dev (projected weno5
  edge ~1.8x), and fix its chunk metric to report compile separately
  (`_CHUNK_COMPILE_LOG`).
- **Cold-compile HLO volume.** Step-body compile scales ~O(ops^1.35);
  HLO-volume reduction is the only cold-start lever for weno5
  (~8.5–10 s honest compile) and for the mapped solve's 16–18 s cold
  compile (vs 2–3 s flat, 2026-07-13 — the same HLO-volume problem in
  the CG body).
- **Async two-tier chunk compile** (optional, interactive-UX). Measured
  (first advance −24..31%, steady state bitwise-unchanged), default-off
  patch preserved, unlanded.
  [`../research/time_to_first_step.md`](../research/time_to_first_step.md)
- **WENO selected-input follow-ups.** The pre-existing forced-4
  knife-edge divergence test now also tips `weno5` (a kernel-shape
  roundoff flip — see
  [`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md)).
  Negative results are recorded in
  [`../research/stencil_lowering.md`](../research/stencil_lowering.md)
  and — for upwind5: one-path spellings, XLA flags, Pallas — in
  [`../research/upwind5_revisit.md`](../research/upwind5_revisit.md)
  (the 2026-07-17 RTX 3060 re-baseline; entry in
  [`done.md`](done.md)) — do not revisit them without reading both.
- **Storage-halo width probe.** Biased order-5 pads storage to `n+8`
  per axis where the nominal reach needs `n+6` (centered: `n+4` vs
  `n+2`) — ~6% inflation on every upwind5 buffer, est. 2–3 ms/step
  @192³ on RTX-3060-class hardware. A core staggering-policy
  question, parity-sensitive, unprobed
  ([`../research/upwind5_revisit.md`](../research/upwind5_revisit.md)
  §6).

## Channel eigenmodes on multi-device — two upstream repros to file

The fridom-side work shipped (2026-07-17, T5): the channel projection now
fails loudly with a taught `NotImplementedError` on a grid that shards a
periodic axis, instead of dying in the HLO verifier — see
[`done.md`](done.md). Both underlying faults are **upstream**
(jax/jaxlib 0.10.2; the earlier "fridom-side c64/c128 dtype mix" reading
is refuted — the traced jaxpr carries zero complex64), and both now have
a minimal fridom-free repro + a drafted jax issue awaiting the owner's
go-ahead to file:

- **GPU (T5).** Not the FFT-norm constant (refuted: reproduces with
  `norm=None`). XLA:GPU/GSPMD lowers a **sharded-transform-axis** FFT
  through its distributed Cooley-Tukey decomposition whose
  **twiddle-factor** constants are `complex64` against `complex128`
  data; the HLO verifier rejects `multiply c64[] c128[]`. Not covered by
  `multi_output_fusion`. Repro + issue:
  [`../research/artifacts/channel_fftnorm_gpu/`](../research/artifacts/channel_fftnorm_gpu/).
- **CPU (T5b).** A batched-`eigh` heap corruption in jaxlib's CPU LAPACK
  on many-core hosts (not the `sort` lowering; that was aliasing).
  Repro + issue:
  [`../research/artifacts/channel_sort_segfault/`](../research/artifacts/channel_sort_segfault/).

Remaining open work:

- **File the two jax issues** (owner go-ahead required — the drafts are
  ready).
- **Optional real GPU fix** (make the projection *run* multi-device,
  not just skip): route the channel transforms through the slab /
  distributed-transform lowering the spectral solver already uses
  (`operators/distributed_solve.py`), so each transform axis is
  device-local when its FFT runs — the `with_sharding_constraint`
  "replicate the transform axis" workaround is proven bit-for-bit exact
  vs the single-device result. Bigger blast radius; deferred.

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

- **Stretched + terrain-following combined** — the correctness
  question is answered
  ([`../research/stretched_terrain_combined.md`](../research/stretched_terrain_combined.md),
  2026-07-17): **no double-count** — stretching (measure widths) and
  terrain (chart J) factor exactly, and the conservative FV
  advection is already correct on the combined grid (conservation
  machine zero, constancy aligned with the projection divergence,
  2nd order under 15:1 stretch). What remains is downstream: the
  mapped pressure solve dies on a stretched column (spectral
  preconditioner unbuildable — cryptic `DispatchError`, no gate;
  SPD lost in the corner cross hops — measure-adjoint down-hop
  recipe probed to machine zero, record §3; multigrid V-cycle is
  the preconditioner candidate), the hydrostatic model has **no
  terrain support at all** (runs silently with 27%-wrong `p_hyd`;
  four metric-free sites, record §4), and the cumint `jacobian=`
  seam is unwired for `maps=` grids (silent no-op / unknown-metric
  raise). All four owner calls **ruled 2026-07-17** (record §7
  addendum): N1+N2 with a plain-CG stopgap; hydrostatic core build
  H0–H2+H4 with the explicit/split depth fix (implicit H3 deferred
  behind a taught error); multigrid learns `grid.measure` widths
  (no interim Thomas route); the `jacobian=` seam wired properly
  (`sqrt_g` for `maps=`, name re-key, taught error — H1 consumes
  the seam, superseding route a). Implementation open.
- **Re-record the FV/nodal step baselines on 4 GPUs** — the gpu4 step
  baseline predates the nodal sibling cases and the walled step
  baselines predate the FV default flip; re-record both
  `benchmarks/baselines/step-gpu{1,4}.json` on the GPU campaign.
  (The distributed FV solve on average origins is now **validated on
  real 4 GPUs** — T1, 2026-07-17: 11/11 `test_distributed_projection.py`
  green, 1-vs-4 FV step smoke matches to ~8e-15; see the scoping record
  §10.6.)

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

## Diffusion/friction closures at walls and on terrain

*Staged; scoped and sized 2026-07-17
([`../research/diffusion_walls_terrain_scoping.md`](../research/diffusion_walls_terrain_scoping.md)).*
The explicit family rejects every bounded grid (which also catches
all terrain columns); `VerticalMixing` is Neumann-rows-only. Stages:
free-slip walls are nearly structural (flux retag to
`Inner[Dirichlet]`, the advection precedent; ~150-250 LOC src);
no-slip adds the one new stencil (MITgcm-style wall rows, `slip=`
API; ~150-300); implicit Dirichlet bottom/top rows (~80-160, high
value — stiff bottom-drag regime, and the merge key must learn BC
structure); mapped along-σ with honest tilt naming (~100-200);
full-metric/rotated tensor deferred. Five owner calls in the record
(default slip, biharmonic no-slip pair, terrain fidelity bar,
slip ownership, stage-0 scope). **Not deferred — stage 0**: taught
gates for a *live silent-wrongness* — `VerticalMixing` binds on
stretched and terrain columns and silently solves the wrong operator
(`second_difference_matrix` infers one uniform `dz` from the first
two nodes; the chart never enters `evaluation_nodes`); a fully
periodic mapped grid likewise binds the explicit family with no
cross terms. Gate both now (~30-60 LOC + raises tests).

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

The semicoarsened V-cycle preconditioner shipped 2026-07-17 (the
"multigrid with vertical line smoothing" lever above, taken; entry in
[`done.md`](done.md); record
[`../plans/active/multigrid_pathway_plan.md`](../plans/active/multigrid_pathway_plan.md)
§3). Open, none blocking:

- **GPU wall-clock lever (only if wanted)** — the GB-2 wall-clock
  leg was measured on the A100 and **fails** (5.5–13.4× *slower*
  than spectral at 128–256³; the vertical-line Thomas smoother is
  latency-bound at full n_z on every semicoarsened level — entry in
  [`done.md`](done.md), evidence
  [`../research/multigrid_gb2_wallclock.md`](../research/multigrid_gb2_wallclock.md)).
  Spectral stays the production default on GPU. A GPU wall-clock win
  would need a z-parallel smoother (Chebyshev / stronger point
  variants — recorded levers in the plan §2) or restructuring away
  from the sequential vertical solve; open only if the owner wants
  that win — the iteration-count robustness result stands regardless.

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
