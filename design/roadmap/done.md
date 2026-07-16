---
status: done
date: 2026-07-13
---

# Roadmap — shipped

The completed half of the FRIDOM roadmap, kept as a record. Open work
lives in [`open.md`](open.md). Task numbers are stable: other records
cite them ("ROADMAP 3.6"), so rows keep their original numbering even
after moving here.

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

## Landed since, outside the numbered tasks

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
  (`shape=`/`extent=`/`periodic=`) landed alongside it, closing the last
  open item of the Phase-2 grid follow-ups. Record:
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
