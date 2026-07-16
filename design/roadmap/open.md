---
status: active
date: 2026-07-13
---

# Roadmap — open work

The single open-work tracker for FRIDOM, ordered: **next steps** first,
then **long-term goals**. Shipped phases are recorded in
[`done.md`](done.md). Task numbers are historical and stable — other
records cite them ("ROADMAP 3.5") — so they do not run in order.

The four items that were unplaced were sized on 2026-07-13 (against real
diffstats of comparable landed work); each carries its estimate below.

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

## Performance optimization (merged 2026-07-13)

Cross-cutting: single- and multi-device, compile and runtime, memory,
cpu and gpu. The optimization line **has now landed** — it was developed
in a parallel checkout and merged here on 2026-07-13. Plan and status:
[`plans/active/perf_geometry_merge_plan.md`](../plans/active/perf_geometry_merge_plan.md).

Shipped: shard-local re-blocking (multi-device pad/unpad without
collectives), the distributed transform planner (the reshard/pencil
stages that work around the XLA SPMD FFT fault), a fused `rfftn` fast
path, storage-frame field arithmetic with ghost-claim propagation, scan
unroll by stepper ring period, past-only tendency rings, and general
non-divisible ghost sharding.

Still open, and the next work:

1. **There is no performance guard anywhere in the repo.** The
   optimization wave added zero benchmarks; every headline number lives
   only in a commit message, and `benchmarks/results/` is gitignored.
   The CI benchmark job states "No timing assertions." A regression that
   silently reverts an optimization leaves the whole suite green. **A
   reproducible A/B harness is the prerequisite for everything below.**
2. **Nothing asserts the fast paths are actually taken.** Every
   *fallback* branch is tested; the fast path is not. A change that
   pushes the production pressure solve off the distributed path would
   pass the suite and quietly cost ~2x at 512³.
3. The optimizations do not yet **reach inside** the mapped PCG solve —
   *resolved, see the update below.*

*Update (2026-07-16): item 1's stated prerequisite — a reproducible A/B
harness — now exists. `benchmarks/model/bench_step.py` records
**committed** baselines (`benchmarks/baselines/step-gpu{1,4}.json`, no
longer gitignored) and runs them `--fail-on-regression`; the
indivisible-shard campaign added the `nh_flat_prime` / `nh_flat_walled_x`
guard cases. What is still open is wiring it as a **gate**: the CI
benchmark job still only smoke-runs ("No timing assertions",
`.github/workflows/tests.yml`), so a silent regression on the untimed CI
path stays green. Item 2 is only partially met —
`tests/nonhydro2/test_distributed_projection.py` now asserts the
distributed fast path for all four solve geometries (periodic, walled-z,
walled-x, prime), but the other fast paths (storage-frame arithmetic
aside, which the halo-claim assertions guard) remain unasserted. Item 3
is **closed**: the geometry-merge stage 3 pass (2026-07-14/15) took the
optimizations inside the CG loop — `Grid.measure` memoized, the f32
preconditioner on mapped grids, and the mixed distributed transform
distributing the preconditioner inside the scan — and priced what
remains as the algorithm itself, not unreached optimization
([`plans/active/perf_geometry_merge_plan.md`](../plans/active/perf_geometry_merge_plan.md)
§4b; closure entry in [`done.md`](done.md)).*

## Gain targets from the Oceananigans reference comparison (2026-07-16)

A matched-protocol comparison against Oceananigans.jl 0.105.3 (nonhydro,
(P,P,walled-z), f-plane, float64, four advection schemes, 64³→max size,
1/2/4×A100-80GB; suite and full report live in the untracked
`benchmarks/comparison` — out-of-tree by design, see its README) confirms
the new stack wins where it is pressure-solve-bound (1.86× at 512³
linear; 4.5–14× on 2–4 GPUs, where Oceananigans' distributed transpose
solve does not scale at all) and identifies three places where fridom
measurably trails the reference:

1. **Transient-allocation memory ceiling.** *Single-GPU part RESOLVED
   2026-07-16* ([`../research/gpu_memory_ceiling.md`](../research/gpu_memory_ceiling.md)):
   the OOM was BFC *fragmentation*, not capacity — the chunk's
   transients are ONE contiguous 30.55 GiB XLA temp arena, and the
   non-donating `_canonicalize` full-carry copies (the `jit_copy` at
   ~62 GB resident) shredded the pool before the first step. Fixed on
   dev: `_canonicalize` donates its carry (setup peaks 44.9/59.2 →
   26.5/36.7 GiB) plus a one-time pre-chunk carry defragmentation
   (`FRIDOM_DISABLE_DEFRAG=1` opts out). 1024×512×512 advective now
   runs on one A100 at `MEM_FRACTION=0.92` (153 ms/step, unroll=3,
   bitwise-identical physics, per-step perf unchanged at all sizes);
   `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` is a validated env-only
   alternative (VMM defeats fragmentation; multi-GPU unvalidated).
   Still open: the 4-GPU signature — 1024×1024×512 fits (~31 GiB/GPU
   steady), 1024×1024×768 dies *in compile* (remat) — needs its own
   attribution (per-device arena + fragmentation vs genuine remat).
2. **Time-to-first-step.** Attributed 2026-07-16
   ([`../research/time_to_first_step.md`](../research/time_to_first_step.md)):
   the report's "11–71 s" conflated compile with executing the whole
   first chunk — honest compile is size-independent at ~2 s (centered)
   to ~8.5–10 s (weno5) plus ~3 s of throwaway eager compiles from the
   `dry_run` validation pass. Three fixes are prototyped and measured
   (patches preserved, nothing merged): `dry_run` under
   `jax.eval_shape` (construction compiles 111→12, build −85%,
   bitwise-identical steps — brings centered total compile to ~1.75 s,
   meeting the <2 s goal), a persistent compilation cache with
   `min_compile_time_secs=0` (warm TTFS −48%; jax's default threshold
   silently skips the 113 small compiles), and a default-off two-tier
   async chunk compile (first advance −24..31%, steady state
   bitwise-unchanged). **The first two LANDED on dev 2026-07-16**
   (merge `7842242b`; post-merge 64³ GPU: cold TTFS 7.5→5.05 s, warm
   2.83 s, per-step unchanged). Remaining: the async two-tier patch
   (optional, interactive UX); HLO-volume reduction in the step body
   (~O(ops^1.35) compile scaling) is the only cold-start lever for
   weno5 — and for the mapped solve's 16–18 s cold compile (vs 2–3 s
   flat, 2026-07-13), the same HLO-volume problem in the CG body; fix
   the comparison suite's metric to report compile separately
   (`_CHUNK_COMPILE_LOG`).
3. **Advection-kernel throughput.** The single-GPU edge collapses from
   1.86× (linear, solve-bound) to 1.05× (upwind5: 131 vs 137 ms/step)
   and 1.10× (weno5: 187 vs 205) — the biased-reconstruction kernels are
   only at parity with Oceananigans' KernelAbstractions kernels, unlike
   every other part of the step. Headroom likely in the
   reconstruct/select pipeline (face-velocity `Where` selects, WENO
   weight evaluation, fusion across the three flux axes). Profile
   against a roofline before optimizing.

## Multi-device follow-ups from the indivisible-shard campaign

The indivisible-extent sharding hole itself is **fixed** (2026-07-16,
all four phases; outcomes and merges in
[`../plans/done/indivisible_shard_plan.md`](../plans/done/indivisible_shard_plan.md);
entry in [`done.md`](done.md)). A same-day follow-up sweep closed
three of the four residual items: **multi-host validation** passed (a
real `srun -n 4` launch, one process per GPU, of the walled-x and
prime guard configs matches single-process runs to ≤ 3.3e-14 of the
state scale — resolution in the plan's "Open questions"), the
**forced-4 test sensitivities** were triaged (`test/forced4-triage`,
merge `0d139fc5`: `single_device` marks where the old-stack reference
is not device-count invariant, the backend-aware `invariant` pattern
on the bitwise asserts; the ninth case was never a forced-4
sensitivity and got its relative bound independently in `cbfc032a` —
with the eigenmode test deselected, the whole-dir `tests/nonhydro2`
forced-4 run is a green gate again), and the **surplus staggered leg**
moved to "Sized, deferred" below. What remains open:

- **Channel eigenmodes are broken on multi-device** — two independent
  pre-existing faults, both now attributed **upstream** (jax/jaxlib
  0.10.2; the earlier "fridom-side c64/c128 dtype mix" reading is
  refuted — the traced jaxpr carries zero complex64 on either path):
  a `sort`-lowering **segfault** on forced-CPU meshes (blocks even
  testing; re-verified exit 139 on dev), and an **XLA:GPU/GSPMD
  lowering fault** that synthesizes a c64 FFT-norm constant against
  the c128 cuFFT output inside the large sharded projection module,
  so the HLO verifier kills the projection on real multi-GPU (not
  covered by the `multi_output_fusion` workaround). The single-device
  path is fine. Work: minimal upstream repros for both faults (ready
  to file with jax — filing needs the owner's go-ahead), plus a
  fridom-side mitigation (e.g. keep the FFT norm scaling outside the
  fused sharded kernel) or a taught multi-device skip on the channel
  eigenbasis so the projection fails loudly instead of in the HLO
  verifier. Evidence, provenance probes, and corrections:
  [`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md).

## Grid follow-ups — the ergonomics half

*Landed 2026-07-15 (see [done.md](done.md)).* The convenience
constructors — `fr.spatial.cartesian.Grid(shape=, extent=, periodic=)`,
the `fr.spatial.spherical.Grid` fast assemble, and the
`fr.spatial.charts.lonlat_sphere` chart primitive — and the API shims —
`ImmutableStateError` on `.data` assignment (a raising setter guiding to
`with_data`), the transform classes re-exported at
`fr.spatial.operators.*` and `NodeSet` at `fr.spatial.*`, and
`Grid.dispatch` typed — all shipped. Getting Started no longer opens with
hand-built `IntervalMesh` factors. One item stays open, and it is
deferred:

**Deferred, not next:** coefficient-space product/power rows. That is not
a missing row but a semantics decision — elementwise multiplication of two
Fourier-coefficient fields is *not* the product of the represented
functions (it is a convolution), so registering it under the same
`("multiply", space)` kind invites silent nonsense. Needs an owner call
first; it blocks nothing.
[`../plans/active/phase2_grid_followups.md`](../plans/active/phase2_grid_followups.md)

## Finite-volume nonhydro

Move the nonhydro model to the average family (`CellAvg` scalars,
face-normal velocities — decision FV-D2 **option A**, owner 2026-07-12).
**No FV code is written yet**; all nine operator gaps are open. Staged:
the four FV symbol rows (the long pole — they block `SpectralSolve` and
hence the pressure solve), the missing conversion rows, an FV tracer slice
— which is where the payoff lands: **exact tracer-mass conservation and
the cut-cell path** — then the C-grid profile with **bitwise parity** as
the gate. Do **not** flip the default wholesale first: walls and mapped
grids work today and would regress.
[`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md)

## Docs & examples rebuild

The CI skeleton and one executed pilot example landed. What remains is the
bulk: **12 example ports** (only `shallowwater/barotropic_instability.py`
is on the new stack) and the **entire prose page tree** — `docs/source/`
still holds the old-stack pages. Also retires the pre-rendered-media
machinery (`@skip_on_doc_build`, the git-LFS videos).

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

Both were sized on 2026-07-13. Neither is hard to justify *technically*;
both fail the "who wants it" test today. Promote either the moment a
consumer appears.

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
difficulty. The plan was corrected: the C-grid biased advection tendency
is only 2nd order on **any** mesh once the advecting velocity varies, so
the mapped divisor does **not** buy asymptotic order back for
`UpwindAdvection` / `WENOAdvection`. The real payoff is ENO/dispersion
behaviour restored on stretched meshes (the reason those schemes exist),
honest design order for standalone `WenoReconstruction` and
`FiniteDifference` order > 2, and mapped one-sided FD. That is quality on
stretched meshes, not a new capability headline — worth doing, not urgent.

**The Jacobian spike ran 2026-07-16 and answered the plan's blocker**:
the divisor is the **same-row discrete Jacobian** (wide linear row on
the seam-unwrapped node coordinates) — it restores design order
(3/5/5, FD 4/6) *and* satisfies the discrete metric identity exactly,
where the analytic Jacobian restores the same order but misses the
identity at O(h^p); the widths are static, data-independent fields.
All that remains is the lift itself, on the recorded route
([`../plans/active/high_order_mapped_plan.md`](../plans/active/high_order_mapped_plan.md)
§3, numbers in
[`../research/mapped_jacobian_spike.md`](../research/mapped_jacobian_spike.md)).

## Mapped-solve residual levers — measured, none currently worth taking

*The "multi-device compile and execution cost" line (formerly 3.9)
closed 2026-07-16 (entry in [done.md](done.md)): on real hardware the
mapped solve scales, the walled/mapped solves distribute, and the CG
loop's remaining premium is priced as the algorithm itself. What
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
| 3.1 | **Hydrostatic model** | Dropped from the cutover (2026-07-08, executed 2026-07-11): the old `hydrostatic` package is removed and this is a greenfield future feature, not a cutover gate. Scope when picked up: linear tendency, hydrostatic pressure solver, advection wiring, eigenvectors. Implicit vertical mixing (and an optional split-explicit free surface) build on 2.5. |
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
