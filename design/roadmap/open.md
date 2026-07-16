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
- **4-GPU memory signature.** 1024x1024x512 fits (~31 GiB/GPU steady),
  1024x1024x768 dies *in compile* (remat) — needs its own attribution
  (per-device arena + fragmentation vs genuine remat). The single-GPU
  ceiling is resolved
  ([`../research/gpu_memory_ceiling.md`](../research/gpu_memory_ceiling.md)).
- **Cold-compile HLO volume.** Step-body compile scales ~O(ops^1.35);
  HLO-volume reduction is the only cold-start lever for weno5
  (~8.5–10 s honest compile) and for the mapped solve's 16–18 s cold
  compile (vs 2–3 s flat, 2026-07-13 — the same HLO-volume problem in
  the CG body).
- **Async two-tier chunk compile** (optional, interactive-UX). Measured
  (first advance −24..31%, steady state bitwise-unchanged), default-off
  patch preserved, unlanded.
  [`../research/time_to_first_step.md`](../research/time_to_first_step.md)
- **WENO selected-input follow-ups.** Multi-host (`srun -n P`)
  confirmation of the walled selected path (single-controller forced-4
  exercised, real multi-process not); and the pre-existing forced-4
  knife-edge divergence test now also tips `weno5` (a kernel-shape
  roundoff flip — see
  [`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md)).
  Negative results are recorded in
  [`../research/stencil_lowering.md`](../research/stencil_lowering.md) —
  do not revisit them without reading it.

## Channel eigenmodes are broken on multi-device

Two independent pre-existing faults, both attributed **upstream**
(jax/jaxlib 0.10.2; the earlier "fridom-side c64/c128 dtype mix"
reading is refuted — the traced jaxpr carries zero complex64 on either
path): a `sort`-lowering **segfault** on forced-CPU meshes (blocks even
testing; re-verified exit 139 on dev), and an **XLA:GPU/GSPMD lowering
fault** that synthesizes a c64 FFT-norm constant against the c128 cuFFT
output inside the large sharded projection module, so the HLO verifier
kills the projection on real multi-GPU (not covered by the
`multi_output_fusion` workaround). The single-device path is fine.

Work: minimal upstream repros for both faults (ready to file with jax —
filing needs the owner's go-ahead), plus a fridom-side mitigation (e.g.
keep the FFT norm scaling outside the fused sharded kernel) or a taught
multi-device skip on the channel eigenbasis so the projection fails
loudly instead of in the HLO verifier. Evidence, provenance probes, and
corrections:
[`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md).

## Finite-volume nonhydro — walls, mapped, hygiene (F4–F6)

Stages F0–F3 shipped: the periodic nonhydro model is FV by default
(decision FV-D2 option A; entry in [`done.md`](done.md)). Walled and
mapped grids stay nodal by default — explicit `family="fv"` there is a
taught error until the stages below land. Open:

- **F4** walls (FV-D4) — plus the F2-found blocker: BC-tagged face →
  `CellAvg` reconstruction is unseeded, so a *walled stratified* mixed
  model does not assemble.
- **F5** mapped/chart FV.
- **F6** hygiene (G7 dealiasing on average origins, G8 quadrature
  `discretize`, G9 one-sided wall reconstruct).
- **Validate the FV default on 4 GPUs** — the distributed solve on
  average origins ran only 1-GPU so far, and the gpu4 step baseline
  predates the nodal sibling cases.

[`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md)

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

## Generalized adiabatic ramping

An `AdiabaticRamping` base transform that ramps declared parameters from a
start to an end value over a ramp period (continuous stage-time `Ramp`
evaluation; curves `"linear"` / `"cosine"` / `"exp"` or a callable), with
**`OptimalBalance` as a subclass** contributing only the balancing policy.
Also buys adiabatic spin-up and parameter continuation. Its dependency
(2.8) has shipped, and `Propagator(updates={param: Ramp(...)})` already
covers much of the mechanism — so this is largely an ergonomics/factoring
task, not new capability.
[`../plans/active/adiabatic_ramping.md`](../plans/active/adiabatic_ramping.md)

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
asserting those raises. Both models use only Dirichlet/Neumann walls, and
no bottom-drag or flux-BC module exists or is planned. Do it when a model
needs one — otherwise it ships untested-by-use.

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

- **Multigrid with vertical line smoothing** — the only preconditioner
  family that could beat the spectral solve (coefficient-robust *and*
  mesh-independent). A real project (staggered mapped-grid
  restriction/prolongation, smoothers, coarse solve, all under a static
  trace); justified only if steep bathymetry (4.5× depth ratio, 45
  iterations) becomes a real workload. Every cheaper alternative was
  measured and rejected — read
  [`perf_geometry_merge_plan.md`](../plans/active/perf_geometry_merge_plan.md)
  §4b lever 1 before re-proposing one.
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

---

# Long-term goals

| #   | Task | Notes |
|-----|------|-------|
| 3.1 | **Hydrostatic model** | Dropped from the cutover (2026-07-08, executed 2026-07-11): the old `hydrostatic` package is removed and this is a greenfield feature. **Implementation plan ready** (2026-07-16): [`../plans/active/hydrostatic_model_plan.md`](../plans/active/hydrostatic_model_plan.md) — stages H0–H6 (shared advection rehoming, cumulative integral, kinematics + explicit oracle, implicit free surface, mixing + eigenmodes, pyOM3/Veros/Oceananigans comparison protocol, split-explicit), decisions HY-D1..D7 awaiting owner sign-off. Implicit vertical mixing and the split-explicit free surface build on 2.5 as designed. |
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
