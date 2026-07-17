---
status: done
date: 2026-07-16
---

# Roadmap — shipped

The completed half of the FRIDOM roadmap, kept as a record. Open work
lives in [`open.md`](open.md). Task numbers are stable: other records
cite them ("ROADMAP 3.6"), so rows keep their original numbering even
after moving here.

**Hygiene rule (binding):** an item's record moves here the moment it
ships — in the same change that reports it shipped — and the `open.md`
entry is trimmed to what actually remains. `open.md` never accumulates
"shipped/landed/resolved" narrative; this file is where it lives.

## Phase 0 — Foundations

Benchmark infrastructure, domain-decomposition unification, repo
cleanup, and removal of the experimental spectral grid — all on the
old `framework`.

## Phase 1 — grid, operators, decomposition

The function-space core, decoupled from the model and testable
standalone. Built as `framework2`, shipped as **`fridom.spatial`**.
Design: [`../specs/grid/`](../specs/grid/00_overview.md), class designs
in [`../specs/grid/classes/`](../specs/grid/classes/README.md), operator
algebra in
[`../specs/operator_algebra/`](../specs/operator_algebra/00_overview.md).
Implementation record:
[`../plans/done/phase1_implementation_plan.md`](../plans/done/phase1_implementation_plan.md).

| #   | Task | Outcome |
|-----|------|---------|
| 1.1 | Meshes, spaces, products | The `Mesh` / `FunctionSpace` families (nodal, average, coefficient, Galerkin, `ConstantSpace`), `TensorProductSpace`, interning. |
| 1.2 | Field core + registry + FD / interpolate | `ScalarField` + `grid.create_field`, the `OperatorRegistry`, `FiniteDifference` / `LinearInterp`, the `Operator` hierarchy with bind-only axis naming and `@` composition. |
| 1.3 | Average family, FV, algebra | `CellAvg`/`FaceAvg`, the FV operators, `integrate`, the field dunders, and the operator algebra (`Composite`, `OperatorSum`, `ScaledOperator`, `Block`, `Dispatched`, the `grad`/`div`/`curl`/`laplacian` factories). |
| 1.4 | Transforms | `Fourier`, `Sine`/`Cosine`, `Chebyshev`, `refined()` padding as space-mapping operators; `Symbol` eigenvalues and spectral solves. |
| 1.5 | Domain decomposition | `negotiate`, `MeshDecompositionTraits`, `HaloSpec`/`HaloTracer` (halo accounting by tracing operator requirements), multi-device shard maps. |
| 1.6 | Immersed subset + export | `grid.immersed` (per-space masks derived on demand) and `f.xr` export to xarray. |
| 1.7 | Standalone validation | Hand-rolled PDEs driven by fields + operators + decomposition, single and multi device, plus the numerical checks in [`05_validation.md`](../specs/grid/05_validation.md). Findings: [`../research/phase1_findings.md`](../research/phase1_findings.md). |
| 1.8 | Sync-strategy redo (2026-07-07) | Consumption-side halo-validity tracking: fields carry a trace-time valid-halo depth; operators sync iff input depth < requirement; `store` stops syncing. Cut the composed step from one exchange per operator application to ~one per state component. Plan: [`../plans/done/sync_redo_plan.md`](../plans/done/sync_redo_plan.md). |

## Phase 2 — model, modules, time-stepping, IO

The model layer on the Phase-1 grid. Shipped as **`fridom.model`**.
Design: [`../specs/model/`](../specs/model/00_overview.md) (decisions
D1–D5) and [`../specs/model/classes/`](../specs/model/classes/README.md).
Implementation record:
[`../plans/done/phase2_implementation_plan.md`](../plans/done/phase2_implementation_plan.md)
(its Phase-2.9 wave 11 — symbolic `L` — was built and then reverted; see
[`../decisions/blocksymbol_l_assembly.md`](../decisions/blocksymbol_l_assembly.md)).

| #   | Task | Outcome |
|-----|------|---------|
| 2.1 | Design: model composition (2026-07-08) | Decisions D1–D5 in [`../specs/model/`](../specs/model/00_overview.md); grid follow-ups filed as [`../plans/active/phase2_grid_followups.md`](../plans/active/phase2_grid_followups.md). |
| 2.2 | Field registration + parameters in modules | `Module` declares `FieldMetadata`; parameters live in modules (`FPlaneCoriolis`/`BetaPlaneCoriolis`, `ConstantStratification`, shallow-water `csqr`, Rossby scaling). |
| 2.3 | Modules modify anything | Modules and grid in the traced state; direct `Model(...)` assembly over a single `modules=` tuple (D4). |
| 2.4 | Single `jax.jit` for the full run | Chunked `lax.scan` (`step_chunk`, AOT-compiled, donated carry); trace-friendly `Clock`; per-step NaN reduction + chunk-boundary abort; `fr.ops.Session` + `Model.run()`. |
| 2.5 | Staged / split time stepping | Ordered stages by `StageKind`; IMEX explicit/implicit partition; CNAB2/SBDF2; the RK family + `LowStorageRK3`; by-variable Gauss-Seidel `advance_stages`. |
| 2.6 | IO: TensorStore writer + diagnostics | `fr.io.Writer` (zarr-format store via tensorstore), `fr.io.TimeSeries` CSV, triggers, pickle-free snapshots, restart-under-scan. |
| 2.7 | Port nonhydro + shallowwater | `nonhydro2` / `shallowwater2`: tendencies, pressure solvers as function-space operators, model-side eigenmodes. Physics parity is closed ([`../research/parity_audit.md`](../research/parity_audit.md), 23/23); the examples/docs half runs on [`../plans/active/docs_examples_plan.md`](../plans/active/docs_examples_plan.md). |
| 2.8 | State transforms | `fr.StateTransform` + the algebra (`@`, arithmetic, `FixedPoint`, `Shift`), `model.variant(term_filter=...)`, `fr.closures.ClosureBase`, and the ported family: Vortical/Wave/Divergence projections, `Propagator`, `TimeAverage`, `OptimalBalance`. **NNMD shipped too** (2026-07-11), as `fr.transforms.BalanceExpansion` — record: [`../plans/done/nnmd_rewrite_plan.md`](../plans/done/nnmd_rewrite_plan.md). |

## Phase 3 — completed items

| #   | Task | Outcome |
|-----|------|---------|
| 3.4 | **Coordinate systems** (2026-07-12) | Mapped, spherical and boundary-fitted grids — the Phase-1 designed-for metric seams, filled in: measures as fields + `MappedIntervalMesh` (C0), `CoordinateMapping` / `grid.metric` / `physical_diff` with the CS-D1 chart embedding (C1), metric-aware vector calculus + spherical shallow water (C2), the CS-D2 preconditioned-CG mapped pressure solve + terrain-following nonhydro (C3), dynamic metrics + the optional CS-D4 ALE module (C4). Record: [`../plans/done/coordinate_systems_plan.md`](../plans/done/coordinate_systems_plan.md). |
| 3.6 | **CG compile cost: `lax.scan` the Krylov loop** (2026-07-13) | The mapped pressure CG was an unrolled fixed-iteration loop, so tracing and XLA compilation were O(iterations). Converted to `lax.scan` (not `fori_loop`: `jax.grad` must keep working) by carrying raw arrays plus a static space and rebuilding fields inside the body, sidestepping the `ScalarField.halo_valid` treedef-stability obstacle. HLO is now flat in the iteration count (468 lines at 12, 60 and 300 iterations, against 2102/5198/10358 unrolled). One loose end, carried to [`open.md`](open.md): the jitted forced-4 multi-device solve was never re-measured. Record: [`../plans/done/krylov_scan_plan.md`](../plans/done/krylov_scan_plan.md). |

| 3.1 | **Hydrostatic model** (2026-07-17) | Greenfield `fridom.hydrostatic` (plan decisions HY-D1..D7 signed off 2026-07-16): prognostic `u,v,b` + `ps = g·eta`, diagnosed `w`/`p_hyd` on DIAGNOSE stages via the new staggered `CumulativeIntegral` (exact discrete fundamental theorem / pyOM half-cell forms); three free-surface variants — explicit term (oracle), implicit CONSTRAINT-stage 2D Helmholtz with the pyOM `epsilon` knob (`epsilon=0` = rigid lid, mean-gauged), split-explicit ADVANCE subcycle with the SM2005 filter per spec §5.4 as frozen; shared advection rehomed to `fr.model.modules` (+ the exact `Outer -> Inner` restriction row); `fr.closures.VerticalMixing` (mergeable tridiagonal, CNAB2/SBDF2); exact numeric eigenbasis (the cumint pair is an exact transpose pair) with vortical/wave + barotropic/baroclinic projections; `hy.comparison_model` preset + physics suite — discrete Rossby-adjustment target (the γ² Coriolis-interpolation correction, matched to 3e-5), wave-packet group velocity, Eady growth vs the discrete operator to 2e-4 (`ThermalWindBackground`; the QG 0.31·fΛ/N gap is the physical Stone/Ri correction). Open remnants (external comparison legs, example review) stay in [`open.md`](open.md) 3.1. Record: [`../plans/active/hydrostatic_model_plan.md`](../plans/active/hydrostatic_model_plan.md) §8. |
| 3.8 | **Generalized adiabatic ramping** (2026-07-17) | Deform a model between reference and target operator configurations, `L(s) = (1-rho(s)) L_ref + rho(s) L_target`, with shared terms never computed twice (blend taxonomy: untouched / affine-parameter / term-weight; decisions AR-D1..D9, driving consumer the Rosenau et al. JFM draft). Shipped R1–R6: time-dependent scalar parameters + the declarative AR-D7 ETDRK4 taught error; `FieldBlend` (author-level affine field blends; ramped Coriolis `f0(t) + beta(t)·y`, static paths bit-identical); `fr.transforms.AdiabaticRamping` (four legs `.down`/`.backward`, `replace()`, window + composition protocol surfaces, AR-D6 irreversibility guard); `OptimalBalance` rebuilt *on* the legs bit-identically; phase-neutral `AdiabaticProjection` (backward–forward; forward–forward counter-example pinned) + `relative_imbalance`; docs page + double-ramp example. Post-landing audit verified the stretched-exponential leakage law to roundoff (`log eta = -2.52 sqrt(tau)`, R² 0.997; [`../research/adiabatic_leakage_scaling.md`](../research/adiabatic_leakage_scaling.md)) and pinned it as a regression shard. Example content review deferred at owner instruction — open in [`open.md`](open.md). Record: [`../plans/done/adiabatic_ramping.md`](../plans/done/adiabatic_ramping.md). |

## Landed since, outside the numbered tasks

- **Upwind5 advection — revisited, attributed, closed at the XLA
  ceiling** (2026-07-17) — the owner's charge to re-research the
  "cannot do better" verdict on new hardware (RTX 3060 Laptop, f64
  1/64-rate — the opposite regime from the A100). Independent matched
  re-baseline against Oceananigans 0.105.3: edges centered 1.26×,
  upwind5 1.01–1.06×, weno5 1.56× — same shape as the A100, and
  fridom's upwind5/centered ratio is 1.82 on BOTH machines despite the
  63× f64-FMA gap, proving the overhead is memory, not arithmetic.
  Byte-level attribution (buffer assignment + live ranges): the +82%
  over centered = ~⅓ extra HBM traffic (XLA materializes 6 wide
  face-reconstruction arrays + more live flux products; `n+8` halo
  pad) + ~⅔ wide-stencil fusion efficiency (141 vs 201 GB/s, no SMEM
  tiling in XLA's loop emitter) + ~0 arithmetic; the one-path
  spellings are byte-identical in the compiled step (the second
  reconstruction was never materialized), which is the mechanism
  behind every failed probe on both machines. Negatives, all
  measured: one-path re-replication (+2.4/+5.2% at 160/192³;
  small-n win reverses exactly at production sizes), a 9-flagset
  bracketed XLA sweep (null ±0.2%; `multi_output_fusion` disable is a
  memory-only knob, u5 scratch 790→266 MB), and the deferred
  Pallas/Triton hand kernel (+34–54% slower on sm_86, bitwise-correct
  — the pow2 constraint forces 2× reconstruction arithmetic in an
  arithmetic-bound regime; GSPMD-incompatible besides). Upwind5
  ties/beats Oceananigans absolutely on both machines; the missing
  relative edge is structural to the XLA lowering. Do-not-revisit
  list extended with mechanisms; residuals (storage-halo width n+8
  vs n+6, axis-0 layout, Pallas-for-WENO on A100) stay in
  [`open.md`](open.md). Side finding: the FV biased-advection
  assembly blocker (tracked in the FV entry of `open.md`). Record:
  [`../research/upwind5_revisit.md`](../research/upwind5_revisit.md).

- **Single-GPU transient-memory ceiling resolved** (2026-07-16) — gap 1
  of the Oceananigans reference comparison. The 1024×512×512 OOM was
  BFC *fragmentation*, not capacity: the chunk's transients are ONE
  contiguous 30.55 GiB XLA temp arena, and the non-donating
  `_canonicalize` full-carry copies (the `jit_copy` at ~62 GB resident)
  shredded the pool before the first step. Fixed on dev (`b6b24644`):
  `_canonicalize` donates its carry (setup peaks 44.9/59.2 →
  26.5/36.7 GiB) plus a one-time pre-chunk carry defragmentation
  (`FRIDOM_DISABLE_DEFRAG=1` opts out). 1024×512×512 advective now
  runs on one A100 at `MEM_FRACTION=0.92` (153 ms/step, unroll=3,
  bitwise-identical physics, per-step perf unchanged at all sizes);
  `XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async` is a validated env-only
  alternative (VMM defeats fragmentation; multi-GPU unvalidated). The
  4-GPU memory signature still needs its own attribution
  ([`open.md`](open.md)). Record:
  [`../research/gpu_memory_ceiling.md`](../research/gpu_memory_ceiling.md).

- **Time-to-first-step attributed; the two main fixes landed**
  (2026-07-16, merge `7842242b`) — gap 2 of the Oceananigans reference
  comparison. The report's "11–71 s" conflated compile with executing
  the whole first chunk — honest compile is size-independent at ~2 s
  (centered) to ~8.5–10 s (weno5), plus ~3 s of throwaway eager
  compiles from the `dry_run` validation pass. Landed: `dry_run` under
  `jax.eval_shape` (construction compiles 111→12, build −85%,
  bitwise-identical steps — centered total compile ~1.75 s, meeting
  the <2 s goal) and a persistent compilation cache with
  `min_compile_time_secs=0` (warm TTFS −48%; jax's default threshold
  silently skips the 113 small compiles). Post-merge 64³ GPU: cold
  TTFS 7.5→5.05 s, warm 2.83 s, per-step unchanged. Remainders (the
  default-off async two-tier chunk-compile patch, HLO-volume
  reduction, the comparison-suite metric fix) stay in
  [`open.md`](open.md). Record:
  [`../research/time_to_first_step.md`](../research/time_to_first_step.md).

- **WENO selected-input one-pass reconstruction** (2026-07-16, merge of
  `perf/weno-selected-input`) — gap 3 of the Oceananigans reference
  comparison. The stencil-lowering study attributed the advection
  collapse (1.86× linear → 1.05/1.10× upwind5/weno5): the slice-window
  kernels already lower optimally (one fused kernel, zero temps) and
  composition is free; WENO is divide/instruction-bound and paid its
  nonlinear weights TWICE (both biased reconstructions computed, then
  `Where`-selected — Oceananigans selects stencil *indices* and
  evaluates once). Shipped: a module-private
  `_SelectedFaceReconstruction` operator
  (`nonhydro2/modules/advection.py`) — tap `where`s on the order+1
  union window, ONE left `weno_reconstruct` of the taps — used by
  `WENOAdvection._face_value` in place of both-then-select (linear
  `UpwindAdvection` kept on both-then-select, byte-identical chunk
  HLO). Production A/B (A100, matched config, fresh process): **weno5
  −39.3% @256³ (25.65→15.57 ms/step), −45.9% @512³ (239.80→129.78)**;
  20-step branch-vs-parent parity ≤1.9e-13 (weno5) / bitwise (weno3).
  The FV C-grid merge extended the selected kernel to the average
  family (`family="fv"`: the `_FVBiasedReconstruction` frame), so a
  `CellAvg` tracer takes the same one-pass spelling and the FV/nodal
  bitwise tendency identity holds. Negative results (do not revisit):
  single-divide weights (real-step temp blowup, 512³ OOM), f32 weights
  (net loss stacked on selected-input), linear-upwind one-path
  spellings (micro win reverses to +4–6% real), and the
  conv/tap-loop/per-point-kernel rewrites. Follow-ups (comparison
  re-run, multi-host confirmation, the forced-4 knife-edge test) stay
  in [`open.md`](open.md). Records:
  [`../research/stencil_lowering.md`](../research/stencil_lowering.md),
  A/B in
  [`../research/stencil_lowering/microbench/phase3/IMPLEMENTATION_AB.md`](../research/stencil_lowering/microbench/phase3/IMPLEMENTATION_AB.md).

- **FV nonhydro stages F0–F3 — the periodic model is FV by default**
  (2026-07-16) — the average-family move (`CellAvg` scalars,
  face-normal velocities; decision FV-D2 **option A**, owner
  2026-07-12), first four stages: the four FV symbol rows, the
  conversion rows (a new `"deconvolve"` kind), `family=` declarations
  (FV-D1b) plus the FV tracer slice (exact tracer-mass conservation to
  machine zero, periodic *and* walled), and the C-grid profile. Gated
  on bitwise parity with nodal and a clean step-suite run (FV/FD step
  time 0.997–1.003 at every size, 1× A100). Walled/mapped grids stay
  nodal by default; explicit `family="fv"` there is a taught error.
  F4–F6 and the 4-GPU validation stay in [`open.md`](open.md). Record:
  [`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md).

- **FV nonhydro stage F4 — walls on FV** (2026-07-16) — decision
  **FV-D4**: BC tags extend to the average family as wall-value
  claims (no DOF change); Neumann `CellAvg` is the DCT-II origin of
  the walled FV pressure solve; `_neumann_sibling` covers both
  families; the claim-consuming `("average", Inner(DIRICHLET))` row
  closes the F2 stratified blocker (`w.to(b)`); explicit
  `family="fv"` served on walled unmapped grids (the auto default
  flipped to FV the same day — next entry). Gates:
  manufactured walled Poisson <1e-12; machine-zero walled FV
  projection (z, x, y cases); **eager-bitwise parity** with the
  walled nodal model (jitted 12-step ≤1.2e-14, an XLA fusion-ordering
  artifact); stratified buoyancy conservation ~1e-18; symbols exact
  against composed operators; forced-4 FV fast-path siblings. Design
  and implementation corrections:
  [`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md) §11.

- **FV nonhydro — walled auto-default flip** (owner ruling
  2026-07-16) — the auto family default is FV on every grid that can
  carry it: `_fv_capable` = unmapped ∧ unimmersed (periodic **and**
  walled), the exact complement of the explicit-fv taught error.
  Shipped with the walled test sweep (nodal coverage pinned
  `family="nodal"` where the nodal path is the test's purpose, the
  F3-flip precedent) and nodal siblings for the walled step-benchmark
  cases (baselines re-record next GPU campaign). The flip surfaced
  the walled-FV analytic-eigenmode taught gap — closed the next day
  (next entry).

- **FV nonhydro stage F5 — mapped/chart FV** (2026-07-17) — the last
  FV stage: terrain-following grids serve explicit `family="fv"`
  (`_require_fv_capable` now rejects immersed only; the mapped
  auto-default stays nodal, an owner call — [`open.md`](open.md)).
  Zero spatial-layer changes: the F0–F4 rows sufficed; the work was
  family-aware routing — `MappedPressureSolver`'s corner face→cell
  down-hops resolve the `"average"` kind on FV (`_to_cell`), and a
  conservative J-weighted mapped FV advection form for pure `CellAvg`
  tracers (per-axis telescoping: `∫J·τ = 0.0` exactly, vs O(1) drift
  on the nodal consistent form — the genuinely new FV property on
  terrain; velocities keep the consistent form, momentum is not
  FV-conserved per FV-D2). Findings: the mapped FV pressure operator
  is **bitwise** the nodal one (uniform computational columns);
  adjointness exactly 0.0 (the SPD/CG license carries over); the
  operator-level mapped FV DOF is the **chart-cell average**
  (reconciles G8; the "physical-volume" reading holds only for
  stretched meshes); `_reconstruct_walled_face` stays chart-uniform
  (measure-weighting would break the transpose pairing). Record:
  scoping §13.

- **FV biased advection assembles** (2026-07-17) — the blocker the
  upwind5 campaign filed (biased velocity self-advection on any
  FV-default grid died at a `Center` vs `CellAvg` bare-`retag` bridge
  in `_face_value`, live since the F3 flip) is closed. Fix: the
  family-aware bridge `_to_flux_space` — co-located `Center → CellAvg`
  crossings go through a real `.to` (the `"deconvolve"` kind, a
  one-point pass-through, **bitwise** at 2nd order), BC-only
  differences stay `retag` (the walled adopt-then-strip seam), pure
  nodal reduces to the old bridge. The diagnosis was also *extended*:
  the biased `_velocity_face` silently dropped FV operands to the
  2-point `"average"` row instead of the order-coupled `_interp` —
  fixed by deconvolving to the `Center` skeleton first, so the FV
  velocity face is order-coupled and bitwise nodal. Parity: periodic
  **exactly bitwise** for upwind3/5, weno3/5 (eager and 12-step
  jitted); walled bitwise eager, ≤1e-14 jitted (the known F4
  pressure-solve fusion artifact, not the schemes). Coverage gap
  closed: `nh.Model`-driven FV biased velocity self-advection tests +
  biased parity variants in `test_fv_default.py`. Coexists cleanly
  with the concurrent hydrostatic `Outer→Inner` restriction work
  (orthogonal branches of the shared `_velocity_face`). Diagnosis
  record: [`../research/upwind5_revisit.md`](../research/upwind5_revisit.md) §7.

- **Walled-FV analytic eigenmodes** (2026-07-17) — the gap the
  default flip surfaced, closed as an early F5 slice: the kit mints
  its BC-tagged `CellAvg` analysis origins itself
  (`_fv_tagged_vertical`, the pressure solver's `mesh.average(kind,
  bc=...)` seam — the declaration layer stays BC-free, C8 intact);
  the DST-II Dirichlet-`CellAvg` origin is seeded (updating the F4
  "no consumer" stance); `LinearReconstruction` carries the interp
  trig eigenvalues (`fv_trig_interp_codomain`, `cos(k dz/2)` —
  bitwise the nodal `LinearInterp` values). Gates: the **FV walled
  eigenbasis is bit-identical to the nodal one** (frequencies,
  eigenvector data, m=0/m=N edge blocks — max diff exactly 0.0);
  strong eigenrelation `L q = −iω q` through the composed FV field
  operators ~1e-16; round-trip completeness <1e-13 including a
  globally-uniform buoyancy (wave projection structurally 0.0); the
  walled battery parametrized over both families. Physics note
  pinned during gating: a z-constant `b` with *horizontal* structure
  legitimately excites waves (`∂w/∂t = b`) — the ω=0 claim holds at
  `kh=0` only. Record: scoping §11 second addendum.

- **FV nonhydro stage F6 — hygiene G7/G8/G9** (2026-07-16) —
  dealiased padded transforms on average origins (sinc-bracketed
  embed/trim; `FaceAvg` included; no model-level 2/3-rule consumer
  exists yet); per-cell Gauss–Legendre quadrature `discretize` behind
  `create_field(order=)` (default bitwise the midpoint rule = 1-point
  Gauss; exact to degree 2·order−1; chart-cell semantics on mapped
  meshes); one-sided `CellAvg → Outer` wall reconstruction
  (`LinearReconstruction(target=OUTER, boundary="one_sided")`,
  geometry-derived weights, per-instance opt-in on the nodal
  precedent, R1 kept for the closed default). Records:
  [`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md) §12.

- **The mapped-Jacobian spike** (2026-07-16) — the 1D throwaway
  experiment (no production edits) that was the stated blocker of the
  high-order-mapped plan. Answer: the divisor for option (i) is the
  **same-row discrete Jacobian** — it restores design order (upwind-3
  3.00, upwind-5 4.99, weno-5 masked 5.00, FD-4/6 3.99/5.97) *and*
  satisfies the discrete metric identity exactly, while the analytic
  Jacobian restores the same order but misses the identity at O(h^p)
  and the current measure divisor caps everything at 2 (reproducing
  the documented 5→2 trap). The widths are static per
  (space, order, bias) — a weno flux over a linear-row width keeps
  order 5. The full lift stays deferred on payoff (see
  [`open.md`](open.md)). Records:
  [`../research/mapped_jacobian_spike.md`](../research/mapped_jacobian_spike.md),
  [`../plans/active/high_order_mapped_plan.md`](../plans/active/high_order_mapped_plan.md) §3.

- **Multi-device execution cost — measured, optimized, closed**
  (2026-07-16; formerly roadmap 3.9) — the krylov-era scare figure
  ("mapped solve 694× slower on 4 forced-cpu devices") was a harness
  artifact: on real A100s the mapped solve *scales*. The line closed
  in three waves: eager field ops on a multi-device operand no longer
  replicate (the operator template runs its stencil kernel under one
  jit trace — `fix/eager-operator-sharding`, plus the
  `fix/traced-measure-cache` follow-up so traced measure queries stay
  uncached); the mixed distributed transform distributes walled (trig)
  and mapped solves (`40dd4ea5`: walled 256³ −52%/chunk, scaling
  0.96×→2.02×; mapped 256³ −48%, 1.21×→2.34× on 4×A100), retiring the
  "walled flat grids do not scale" finding; and indivisible-extent
  sharding was fixed (own entry below). What remains of the
  mapped-step premium is the CG loop itself, now priced and accepted:
  one CG iteration ≈ one flat timestep (6.15 ms marginal at 256³; the
  preconditioner is 58% of it and is the *right algorithm* — every
  cheaper alternative measured and rejected), and
  `pressure_iterations=30` stays (not a trace-era default: steep
  terrain needs it, and the budget is resolution-independent). The
  reach levers landed (`Grid.measure` memo, the f32 mapped
  preconditioner, halo gating); the storage-frame CG carry was
  reverted by measurement (re-attempt gate in the krylov docstring).
  The deferred residual levers live in [`open.md`](open.md)
  ("Mapped-solve residual levers"). Records:
  [`../plans/active/perf_geometry_merge_plan.md`](../plans/active/perf_geometry_merge_plan.md)
  §§4a–4b,
  [`../plans/active/distributed_transform_plan.md`](../plans/active/distributed_transform_plan.md).

- **Indivisible-extent sharding fixed** (2026-07-16) — the owner-flagged
  multi-device hole (4 GPUs 2–5× *slower* than 1 whenever the sharded
  axis carried a `P`-indivisible extent: walled staggered legs, prime
  domain sizes). Four phases in one campaign: the reblock gate admits
  the divisible staggered-deficit leg to the fast padded-even plan
  (x-walled 256³ collapses 15.6 → 3.5 ms/step onto the periodic
  scaling); the distributed solve accepts indivisible split axes via a
  padded balanced all-to-all with local pad/trim (prime 257³
  9.18 → 4.02 ms/step, 4-GPU now 2.14× faster than 1-GPU, no
  replicated cube); stepper scaling helpers spelled on the storage
  frame (bitwise-identical, **1-GPU nh_flat −11..−17% and
  shallowwater −9..−23%/step** as a bonus); shard-axis selection
  ranks periodic-divisible axes first
  (walled-x default now shards y: 3.56 → 3.36 ms/step). Divisible
  programs stayed byte-identical throughout; baselines re-recorded
  with new `nh_flat_prime` and `nh_flat_walled_x` guard cases.
  Multi-host validation closed 2026-07-16: a real `srun -n 4` launch
  (one process per GPU, `jax.distributed.initialize()`) of the
  walled-x and prime guard configs matches single-process runs to
  ≤ 3.3e-14 of the state scale, with no host fetch of a global array
  (resolution in the plan's "Open questions"). The forced-4 test
  sensitivities the campaign surfaced were triaged the same day
  (`test/forced4-triage`, merge `0d139fc5`; corrections and
  resolutions appended to the faults note); the channel-eigenmode
  multi-device faults remain open (re-attributed upstream — roadmap).
  Records:
  [`../plans/done/indivisible_shard_plan.md`](../plans/done/indivisible_shard_plan.md),
  probe evidence in
  [`../research/indivisible_shard_probes.md`](../research/indivisible_shard_probes.md),
  pre-existing faults surfaced along the way in
  [`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md).

- **Grid setup ergonomics — the fast assemble** (2026-07-15) — building
  a spherical grid was a 9-line copy-from-the-docstring incantation. Now
  `fr.spatial.spherical.Grid((nlon, nlat), radius=a,
  lat_extent=(-lat_max, lat_max))` is one call: it builds the
  periodic-lon x bounded-lat interval meshes and attaches the orthogonal
  lat-lon sphere chart (the extent-free `fr.spatial.charts.lonlat_sphere`
  primitive). `lat_extent` is required (the poles are metric-singular; a
  loud guard rejects `|lat| >= pi/2`) and asymmetric bands give a
  hemisphere; an optional `lon_extent` closes the zonal walls (a
  longitude sector — still orthogonal, so the diagonal index moves
  assemble across it). The sibling `fr.spatial.cartesian.Grid`
  (`shape=`/`extent=`/`periodic=`) landed alongside it, together with
  the API shims (`ImmutableStateError` on `.data` assignment — a
  raising setter guiding to `with_data` — the transform classes
  re-exported at `fr.spatial.operators.*`, `NodeSet` at
  `fr.spatial.*`, and `Grid.dispatch` typed), closing the Phase-2 grid
  follow-ups up to one deferred semantics item (coefficient-space
  product/power rows — [`open.md`](open.md)). Record:
  [`../plans/done/grid_ergonomics_plan.md`](../plans/done/grid_ergonomics_plan.md).
- **Chart / sphere setup ergonomics** (2026-07-15) — the E1–E5
  follow-ups to coordinate systems (3.4). The last, E2: a chart with a
  **bounded** axis no longer needs the undiscoverable
  `merge_overrides(RaiseIndex(..., diagonal=True), ...)` incantation —
  declare `CoordinateMapping(chart=..., orthogonal=True)` and the grid
  seeds the diagonal index moves; a taught error names the fix
  otherwise. Auto-detection was rejected (the induced metric is numeric
  autodiff, so an orthogonal chart reads ~1e-16, not exact zero — a
  tolerance would risk silently dropping real cross terms). Record:
  [`../plans/done/chart_ergonomics_plan.md`](../plans/done/chart_ergonomics_plan.md).
- **Decomposed / gather-free output** (2026-07-14) — the it-1 IO sink
  `decomposition.gather`ed the whole true field to rank 0 / host
  before writing, which cannot fit large grids (768³+). The Writer
  now binds from the values-free `export_layout` and writes each
  replica-0 jax shard's true-DOF tile straight into the zarr store
  via `decomposition.shard_writes` (tensorstore, async per firing),
  with `decomposition.chunk_hint` aligning the default chunk grid to
  the per-shard cell blocks — no gather, no global array, and the
  storage padding (halo ghosts, stagger reserve, cell padding) never
  reaches the file. Independent of divisibility. Record:
  [`../plans/done/gather_free_output_plan.md`](../plans/done/gather_free_output_plan.md).
- **The performance-optimization line merged** (2026-07-13) — the
  optimization wave developed in a parallel checkout, merged onto dev:
  shard-local re-blocking (multi-device pad/unpad without collectives),
  the distributed transform planner (the reshard/pencil stages that
  work around the XLA SPMD FFT fault), a fused `rfftn` fast path,
  storage-frame field arithmetic with ghost-claim propagation, scan
  unroll by stepper ring period, past-only tendency rings, and general
  non-divisible ghost sharding. The reproducible A/B harness it lacked
  followed on 2026-07-16: `benchmarks/model/bench_step.py` records
  **committed** baselines (`benchmarks/baselines/step-gpu{1,4}.json`,
  no longer gitignored) and runs them `--fail-on-regression`, with the
  `nh_flat_prime` / `nh_flat_walled_x` guard cases from the
  indivisible-shard campaign. Open remainders (wiring the harness as a
  CI gate; the unasserted fast paths) stay in [`open.md`](open.md).
  Record:
  [`../plans/active/perf_geometry_merge_plan.md`](../plans/active/perf_geometry_merge_plan.md).
- **Docs & examples rebuild — CI skeleton + pilot** — the
  executable-examples CI skeleton and one executed pilot port
  (`shallowwater/barotropic_instability.py`) landed; the 12 remaining
  example ports and the whole prose page tree stay in
  [`open.md`](open.md). Record:
  [`../plans/active/docs_examples_plan.md`](../plans/active/docs_examples_plan.md).
- **Boundary closures** (2026-07-11, merge `9a95202a`) — the R1 flip
  (exterior reads on a BC-free bounded axis raise; the extrapolation
  fill is gone), `BC.ROBIN` structure, and one-sided opt-in rows. The
  Robin *dynamic data path* remains open:
  [`../plans/active/boundary_plan.md`](../plans/active/boundary_plan.md).
- **Graded near-wall fallback operator** — `Fallback` + the shared
  graded ladder, on both the FV (WENO) and nodal routes, working on a
  sharded bounded axis. Record:
  [`../plans/done/fallback_operator_plan.md`](../plans/done/fallback_operator_plan.md).
- **Exactly-conserving Coriolis** (2026-07-13, merge `fc0b61c8`) — both
  routes in shallowwater2: the optional correction term (keeps `L` and
  the eigenmodes valid) and the full nonlinear module. Record:
  [`../plans/done/coriolis_energy_correction.md`](../plans/done/coriolis_energy_correction.md).
- **The symbol substrate** — scalar `Symbol` + `SpectralSolve` + the
  banded primitive; the block/eigen layer on top was built and
  withdrawn. Record:
  [`../plans/done/operator_symbols_plan.md`](../plans/done/operator_symbols_plan.md).
- **Projections & eigenmodes** — the energy-metric design (`EnergyMetric`,
  `p = M q`, `eigh(iML, M)`, the channel eigenbasis) and the transform
  family. Record:
  [`../plans/done/projection_eigenmode_plan.md`](../plans/done/projection_eigenmode_plan.md);
  the one remaining tier is phase I of
  [`../plans/active/projection_eigenmode_roadmap.md`](../plans/active/projection_eigenmode_roadmap.md).
- **The package split** (2026-07-11) — `framework2` became
  `fridom.spatial` + `fridom.model`. Record:
  [`../plans/done/spatial_model_split_plan.md`](../plans/done/spatial_model_split_plan.md).
