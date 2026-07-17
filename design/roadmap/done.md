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

| 3.1 | **Hydrostatic model** (2026-07-17) | Greenfield `fridom.hydrostatic` (plan decisions HY-D1..D7 signed off 2026-07-16): prognostic `u,v,b` + `ps = g·eta`, diagnosed `w`/`p_hyd` on DIAGNOSE stages via the new staggered `CumulativeIntegral` (exact discrete fundamental theorem / pyOM half-cell forms); three free-surface variants — explicit term (oracle), implicit CONSTRAINT-stage 2D Helmholtz with the pyOM `epsilon` knob (`epsilon=0` = rigid lid, mean-gauged), split-explicit ADVANCE subcycle with the SM2005 filter per spec §5.4 as frozen; shared advection rehomed to `fr.model.modules` (+ the exact `Outer -> Inner` restriction row); `fr.closures.VerticalMixing` (mergeable tridiagonal, CNAB2/SBDF2); exact numeric eigenbasis (the cumint pair is an exact transpose pair) with vortical/wave + barotropic/baroclinic projections; `hy.comparison_model` preset + physics suite — discrete Rossby-adjustment target (the γ² Coriolis-interpolation correction, matched to 3e-5), wave-packet group velocity, Eady growth vs the discrete operator to 2e-4 (`ThermalWindBackground`; the QG 0.31·fΛ/N gap is the physical Stone/Ri correction). Oceananigans execution leg ran 2026-07-17 (out-of-tree harness, single A100): machine-precision linear parity; per-step ratios oc/fridom — implicit-linear ~2x fridom at every rung, split-explicit centered 1.04–1.29x, weno5 parity; it surfaced the implicit+advection surface-closure instability → root-caused and fixed same day (constancy-preserving surface advective flux, default on; plan §H7, merge `8bd91dfe`). Open remnants (Veros/pyOM3 legs, example review) stay in [`open.md`](open.md) 3.1. Record: [`../plans/active/hydrostatic_model_plan.md`](../plans/active/hydrostatic_model_plan.md) §8. |
| 3.8 | **Generalized adiabatic ramping** (2026-07-17) | Deform a model between reference and target operator configurations, `L(s) = (1-rho(s)) L_ref + rho(s) L_target`, with shared terms never computed twice (blend taxonomy: untouched / affine-parameter / term-weight; decisions AR-D1..D9, driving consumer the Rosenau et al. JFM draft). Shipped R1–R6: time-dependent scalar parameters + the declarative AR-D7 ETDRK4 taught error; `FieldBlend` (author-level affine field blends; ramped Coriolis `f0(t) + beta(t)·y`, static paths bit-identical); `fr.transforms.AdiabaticRamping` (four legs `.down`/`.backward`, `replace()`, window + composition protocol surfaces, AR-D6 irreversibility guard); `OptimalBalance` rebuilt *on* the legs bit-identically; phase-neutral `AdiabaticProjection` (backward–forward; forward–forward counter-example pinned) + `relative_imbalance`; docs page + double-ramp example. Post-landing audit verified the stretched-exponential leakage law to roundoff (`log eta = -2.52 sqrt(tau)`, R² 0.997; [`../research/adiabatic_leakage_scaling.md`](../research/adiabatic_leakage_scaling.md)) and pinned it as a regression shard. Example content review deferred at owner instruction — open in [`open.md`](open.md). Record: [`../plans/done/adiabatic_ramping.md`](../plans/done/adiabatic_ramping.md). |

## Landed since, outside the numbered tasks

- **Stretched + terrain-following combined — answered and shipped**
  (2026-07-17, merges `1c8614c0`, `0a887fa9`, `d338538c`, `4906b8db`,
  `7fba1bfe`) — the roadmap correctness question resolved
  ([`../research/stretched_terrain_combined.md`](../research/stretched_terrain_combined.md):
  **no double-count** — stretching lives in `grid.measure` widths,
  terrain in the chart `J`, exactly complementary factors of one
  Jacobian; the conservative FV advection was *already*
  correct/conservative/2nd-order on combined grids) and every
  downstream gap closed the same day on the record's §7-addendum
  rulings: **(1)** the `jacobian=` seam wired for analytic `maps=`
  grids — `sqrt_g` derived (∏ column Jacobians, single-base only),
  the gate re-keyed chart-coordinate → base-axis through the shared
  `jacobian_weight` helper, the silent no-op now a taught error;
  **(2)** the mapped pressure solve on stretched columns — the
  measure-adjoint base-axis corner down-hop (`down_b =
  diag(1/m_cell)·up_bᵀ·diag(m_inner)`) restores SPD under the
  physical inner product to machine zero, spectral-on-stretched is a
  taught error, `preconditioner="none"` added as the correctness
  stopgap; **(3)** the multigrid V-cycle consumes `grid.measure`
  widths (vertical bands, diagonal, coarse levels; the transfers
  were already measure-adjoint): **7 PCG iterations flat** across
  16²–64² and up to ~20:1 stretch vs 221→non-convergent
  unpreconditioned, uniform columns bitwise unchanged; **(4)** the
  pre-existing storage-frame measure-divide masked singularity
  (`staggering.py`) guarded with the double-`where` — every bounded
  stretched-mesh `diff`, and the end-to-end stretched+terrain solve,
  is now reverse-differentiable (grad = FD in-suite; the one
  flagged-but-safe sibling, `MetricScaled`, in `open.md`); **(5)**
  the hydrostatic model gained its **sigma-coordinate core**:
  J-weighted `p_hyd` (2nd order on uniform and stretched columns;
  previously silently 27% wrong on terrain), contravariant-`w`
  diagnosis (machine-exact flux-form FTC, exact bottom seed), the
  slope-corrected pressure gradient built as the exact discrete
  adjoint of continuity (rest-over-topography converges at 2nd
  order; baroclinic and barotropic energy legs machine-precision),
  physical-depth explicit free surface; implicit + split-explicit
  on charts are taught errors (the variable-csqr route, hydrostatic
  plan §7–§8). Residuals tracked in [`open.md`](open.md).

- **CG pressure solve — opt-in convergence tolerance**
  (2026-07-17) — a keyword-only `tolerance: float | None = None` on
  `ConjugateGradient`
  ([`krylov.py`](../../src/fridom/spatial/operators/krylov.py)) stops
  refining once the measure-weighted true relative residual clears
  `tolerance · sqrt(<b,b>)` (compared squared; zero RHS converges
  immediately). Mechanism: a **masked `lax.scan`** — the fixed scan
  keeps its static length, but each step wraps the real CG step in
  `lax.cond(converged, no-op, real_step)`, so past convergence every
  step is a runtime no-op (`lax.cond` → real `stablehlo.case`, ~3×
  forward on CPU at converge-at-5-of-30). Chosen over
  `while_loop` + `custom_linear_solve` because `scan` + `cond`
  differentiate the **actual truncated algorithm** — `jax.grad` stays
  exact to FD precision (the repo invariant, no `custom_vjp`, no
  transpose machinery) whereas the IFT gradient errs ∝ tolerance and
  hides a `symmetric=True` 70 %-wrong-grad trap on measure-weighted-
  self-adjoint operators. `tolerance=None` is the pre-existing fixed
  path **byte for byte** (a sub-floor tolerance that never fires is
  measured bitwise-identical). Threaded as `tolerance=` /
  `pressure_tolerance=` through the mapped
  ([`mapped_pressure.py`](../../src/fridom/nonhydro2/modules/mapped_pressure.py))
  and immersed
  ([`immersed_pressure.py`](../../src/fridom/nonhydro2/modules/immersed_pressure.py))
  pressure solvers, `DynamicalCore`, the `nh.Model` factory, and
  `hy.ImplicitFreeSurface` — all default `None`, pure passthrough; the
  flat spectral paths are untouched. Caveats documented: keep the
  tolerance above the ~1e-14 residual floor (below it the cond never
  fires and the fixed-iteration post-floor `tiny/tiny` NaNs the reverse
  gradient — a firing tolerance *removes* this pre-existing hazard) and
  do not `vmap` it (`cond` → compute-both `select`). Default-off on
  purpose (results shift at the tolerance level in tuned configs; a safe
  default is problem-dependent). The end-to-end tolerance autodiff
  regression rides the **immersed** consumer — the mapped model is
  pre-existing-non-reverse-differentiable in this geometry (a metric
  singularity, `tolerance=None` NaNs identically). GPU step-level
  re-measure + a default-on revisit: [`open.md`](open.md). Research:
  [`../research/cg_stopping_criterion.md`](../research/cg_stopping_criterion.md).
  **Default-on at `1e-8` since 2026-07-17** (owner decision, superseding
  the default-off above): `tolerance` / `pressure_tolerance` now default
  to `1e-8` (`sqrt(f64 eps)`, Oceananigans' PCG precedent — fires above
  the ~4.5e-14 floor so the T4 trap cannot engage in f64); `None` is the
  explicit fixed-iteration opt-out, and determinism-pinned tests pass it.
  **GPU-validated 2026-07-17** (A100, 256³ terrain nonhydro2, linear
  mapped step inside the real chunked scan): the early-exit survives
  XLA:GPU as a true `conditional`; ms/step 206.0 → 83.4 gentle (9 of 30
  iterations, −59.5 %) / 153.0 strong (19 iterations, −25.7 %) at the
  `1e-8` default, −65 %/−40 % at `1e-6`; 50-step state matches the fixed
  budget to ≤ 1.7e-9 relative. The win *exceeds* the CPU micro-timing —
  the standalone-vs-chunked reversal fear did not materialize.
  Addendum with the full table in the research record.
- **Multigrid pathway, phase A — the grid transfer layer**
  (2026-07-17, merge `fae44be4`) — grid-to-grid transfer on the new
  stack: `Mesh.coarsened` / `Grid.coarsened` (independent coarse
  siblings, no adoption links, attachments cloned and re-derived),
  `negotiate(allow_replicated=)` (a coarse level below the
  shardability floor replicates on the same device mesh, MG-D5),
  `check_level_shardability` (the gate on the silent-halo hazard,
  research F3), and `GridTransfer` — adjoint restriction /
  prolongation between coexisting grids (order-1/2 pairs,
  `R = M_H^-1 P^T M_h` via `jax.linear_transpose`, so adjointness
  holds by construction: measured <= 2.5e-15 across mapped /
  immersed / semicoarsened cases; conservation exact; forced-4-device
  parity; autodiff regression per the differentiability policy).
  Dual-use: the multigrid substrate (phase B, below) and
  the coupling regrid primitive (CS-15/§11.1). Record:
  [`../plans/active/multigrid_pathway_plan.md`](../plans/active/multigrid_pathway_plan.md)
  §2; research:
  [`../research/multigrid_pathway.md`](../research/multigrid_pathway.md).
- **Multigrid pathway, phase B — the V-cycle pressure preconditioner**
  (2026-07-17, merge `87aeabea`) — geometric semicoarsened multigrid
  as an opt-in preconditioner for the mapped and immersed PCG pressure
  solves (`pressure_preconditioner="multigrid"` + `multigrid_levels`
  on both solvers, `DynamicalCore`, and `nh.Model`; fingerprint-static,
  spectral stays the default; flat grids ignore the knob). Engine:
  `spatial.operators.multigrid` — fixed-count symmetric V(1,1)
  (MG-D7/D8: static level tuple, trace-time unrolled, SPD in the
  weighted product so plain CG stays valid), damped point-Jacobi and
  vertical-line block-Jacobi smoothers (batched Thomas kernel in
  `banded.py`, ω = 0.8), per-level projections (mean-free / the
  level's own wet-mean, MG-D6). Hierarchy:
  `nonhydro2.modules.multigrid_hierarchy.coarsen_levels` — per-level
  re-discretization of the same solver on `Grid.coarsened` levels
  (incl. re-merging the FV `diff` profile per coarse grid),
  semicoarsening ×2 horizontal with a 4-cell floor and graceful
  degradation to smoothing-only; `Grid.coarsened` memoized per
  (factors, devices) for retrace stability. Measured (steep mapped
  a = 0.8, iterations to 1e-10): **44 → 13, flat over 32/64/96³**
  (default depth pinned to 5 by the levels×sweeps campaign; depth 3
  missed the ≤ 15 gate at 18); immersed genuine partials **15 (16³) /
  18 (32³) against the 30 budget** where spectral needs ~80 — closes
  the "preconditioner quality on heavily-masked domains" residual.
  Gates: GB-4 compile-once, HLO flat in the CG iteration count; GB-5
  forced-4 parity incl. a replicated coarse level (MG-D5); autodiff
  regression through the immersed multigrid step (the smoothers'
  dry-cell double-`where` guards hold). GB-5 real multi-GPU validated
  2026-07-17 (DKRZ 4×A100): the eight multi-device parity tests pass on
  real 4 GPUs single-process, and a real `srun -n 4` multi-process
  steep-mapped multigrid model run matches its single-device reference
  to 5.6e-17 (plan §3, GB-5 outcome). The GB-2 wall-clock leg was
  measured on the A100 the same day and **fails**: 5.5–13.4× slower
  ms/step than spectral at 128/192/256³ (one V-cycle ≈ 66× a spectral
  CG iteration at 128³ — the sequential vertical-line Thomas smoother
  runs at full n_z on every semicoarsened level, latency-bound on
  GPU), so **spectral stays the production default on GPU**; the
  iteration-count win stands as a robustness/CPU result. Evidence:
  [`../research/multigrid_gb2_wallclock.md`](../research/multigrid_gb2_wallclock.md).
  Record:
  [`../plans/active/multigrid_pathway_plan.md`](../plans/active/multigrid_pathway_plan.md)
  §3 (B0 spike numbers + the three recorded corrections, not yet
  owner-reviewed).
- **Mapped projection reverse-NaN fixed** (2026-07-17, merge
  `ee350bda`) — `jax.grad` through a mapped (terrain-following) run
  returned an all-NaN gradient while the primal stayed finite: the
  `velocity_correction` metric quotient `F_i / J` divided by the
  column Jacobian `J = dm/db`, strictly positive on valid cells but
  zero-filled in never-valid storage padding, so the sealed-`inf`
  primal there carried a singular divide VJP (`0 * inf -> NaN` —
  the masked-singularity class the differentiability policy names).
  First noted as "a metric singularity" at the CG-tolerance landing,
  localized by the multigrid B5 pass, fixed with the double-
  `jnp.where` guard (`MappedPressureSolver._divide_by_jacobian`):
  bitwise-identical primal on every valid cell (parity + treedef
  tests unchanged), finite reverse pass. Ships the mapped autodiff
  regression the path never had
  (`tests/nonhydro2/test_mapped_model_autodiff.py`: grad finite and
  FD-matched to rel-err ~6e-12 against gate 1e-4, nodal + FV). The
  new-stack step path is now reverse-differentiable on **all** grid
  types — flat, walled, mapped, immersed — with no known exception.
- **Immersed partial cells — all dimensions, all three models**
  (2026-07-17, merges `ee257bc0` I0+I1, `b447b8e5` I2, `a5aec29d` I4,
  `3858d977` I3, plus the autodiff regression gates) — the immersed
  grid works in nonhydro2, shallowwater2 and hydrostatic with genuine
  partial cells in every dimension (a sloping `B(y, z)` gives
  x-partials). Grid layer: quadrature volume fractions
  (`ImmersedDomain(order=, min_fraction=)`, hFacMin floor, min-rule
  face transfer, boolean default bitwise iteration-1). Models:
  masked-Poisson PCG projections (spectral inverse masked onto the
  wet cells as preconditioner, V-orthogonal wet-mean gauge),
  fraction-weighted flux-form advection/continuity, `MaskState`
  hygiene stage, wet-column free surface (implicit PCG +
  split-explicit), taught gates on everything that cannot honestly
  run masked. Keystone gates: staircase ≡ walled model at ~1e-16
  (nonhydro2 FV) / ~3e-17 (sw2 Sadourny); masked divergence and
  θ-mass conservation at machine zero in all three models; hydrostatic
  column equivalence ≤ 2e-15; 2nd-order masked-Poisson convergence on
  genuine x/z-partials; `jax.grad` through every immersed step path
  FD-matched at ≤ 1e-8. **4-GPU validated** (gpu4 campaign T2,
  2026-07-17): the masked cut-cell PCG step is device-count invariant on
  real 4× A100 (`test_[partial_]immersed_step_is_device_count_invariant`,
  face-aligned box + a new genuine-partial obstacle smoke, 1-vs-4 ≤
  1.8e-15; the fusion workaround is not even needed at 16³). Residuals in
  [`open.md`](open.md). Record +
  per-stage corrections:
  [`../plans/active/immersed_partial_cells_plan.md`](../plans/active/immersed_partial_cells_plan.md).
- **Variable boundary forcing — wind stress, surface buoyancy flux**
  (2026-07-17, merge `24ee6fd0`) — prescribed wall-face fluxes as
  tendency contributions in the wall-adjacent cell
  (`fr.model.modules.BoundaryFlux(field, coord, side, flux=, scale=)`:
  sign(side)·scale(t)·F(x_tang)·W, W the index-built `1/Δn`
  wall-weight profile over the grid measure, F a tangential profile,
  scale a published side-qualified dynamic-leaf parameter) — the
  surveyed Oceananigans/MITgcm/Veros mechanism riding the structural
  NEUMANN zero-gradient halos; **no ghost-fill machinery consumed**
  (boundary-closure 2e stays consumer-less, its entry corrected).
  Two generic `TimeDependent` curves: `fr.model.TimeFunction`
  (static law, dynamic params — zero-recompile sweeps) and
  `fr.model.TimeSeries` (tabulated `jnp.interp`, the EXF/Veros
  two-record blend). nonhydro2 wrappers `WindStress` /
  `SurfaceBuoyancyFlux` own the oceanographic signs
  (`wind_stress.<coord>_<side>.scale`, opposite-wall instances
  coexist — the Rayleigh–Bénard idealization). Gates: analytic
  oracles (`q·t/Δz` wall row, global budget `= q·A` to rounding,
  hand-stepped Ramp/TimeFunction), forced-4 bitwise, 100% branch
  coverage on both new modules, nonhydro2 suite green. Deferred
  designed-fors (volumetric `Forcing`/SW wind, bottom drag,
  tabulated 2-D+time maps, value BCs): plan §BF-D5. Record:
  [`../plans/done/boundary_forcing_plan.md`](../plans/done/boundary_forcing_plan.md).

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
  alternative (VMM defeats fragmentation; multi-GPU validated in the
  4-GPU entry below). Record:
  [`../research/gpu_memory_ceiling.md`](../research/gpu_memory_ceiling.md).

- **4-GPU memory signature attributed — same fix, not remat**
  (2026-07-17) — second bullet of the Oceananigans reference comparison
  gap. The comparison probe's read that 1024x1024x768 "dies in compile
  (remat)" on 4 A100s is **wrong**: on current dev it FITS at ~44
  GiB/GPU steady. The original death (recorded 2026-07-16 10:01 UTC) was
  per-device BFC arena *fragmentation* — the 18.36 GiB contiguous chunk
  temp arena could not be placed in a pool churned by the non-donating
  `_canonicalize` — one rung larger than the single-GPU ceiling, and
  closed by the SAME donation+defrag fix (`b6b24644`), which merged 5.5 h
  *after* the observation. The `hlo_rematerialization.cc` line that named
  it is a non-fatal warning whose peak estimate (~62 GiB) is ~1.4x
  pessimistic vs the real 44 GiB. Reproduced directly (revert the fix
  -> BFC OOM; pre-fix + cuda_async -> fit, which also validates
  cuda_async multi-GPU). Lever for 768: none, it fits at the default BFC
  0.75. Next rung 1024x1024x1024 (~51 GiB/GPU) is a harder wall BFC
  clears at neither 0.75 nor 0.92 (GPU0 cannot place the 24.71 GiB
  arena) — `cuda_async`'s job. Record:
  [`../research/gpu_memory_ceiling.md`](../research/gpu_memory_ceiling.md)
  §7.

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
  conv/tap-loop/per-point-kernel rewrites. Multi-host validation closed
  2026-07-17: a real `srun -n 4 --gpu-bind=none` launch (walled-**and**-
  sharded x, weno5, 30 steps, fusion workaround set) matched the
  single-device serial reference to machine precision (max abs 2.3e-15,
  ≤5.2e-15 of field scale — sharded-vs-serial reduction roundoff), with
  the selected-input walled path asserted active on the sharded axis.
  Remaining follow-ups (comparison re-run, the forced-4 knife-edge test)
  stay in [`open.md`](open.md). Records:
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
  cases (baselines re-recorded 2026-07-17, gpu4 campaign T7 — see the
  step-baseline re-record entry). The flip surfaced
  the walled-FV analytic-eigenmode taught gap — closed the next day
  (next entry).

- **FV nonhydro — mapped auto-default flip** (owner ruling
  2026-07-17: "auto = FV wherever capable, no surprising family
  changes by grid type") — static terrain-following grids now
  auto-default to FV; the C3 validation battery holds on the FV
  default, and a frozen-`MovingGeometry`-without-ALE FV run is
  bitwise the static FV run. **Dynamic geometry is carved out**:
  ALE (`MeshVelocityCorrection`) is nodal-only, and since
  time-dependence is a model property the `nh.Model` factory supplies
  `dynamic_geometry` to `resolve_model_family` — moving-geometry
  models stay nodal-auto (default path never breaks), explicit
  `family="fv"` + ALE is a taught error at `bind`. Benchmark
  `nh_mapped` now measures the FV default (`nh_mapped_nodal` sibling
  added; baselines re-recorded 2026-07-17, gpu4 campaign T7). The auto
  rule is
  now: **FV iff unimmersed and statically-mapped-or-flat.** Records:
  scoping §13 addendum; the ALE-FV gap stays in [`open.md`](open.md).

- **ALE on FV — the last family-awareness gap closed** (owner ruling
  2026-07-17: option C ratified, the auto default flips) — moving
  geometry now runs on the FV family. `MeshVelocityCorrection` is
  **family-aware**, routed per field at `bind` on its column factor:
  the conservative **flux form** on the `CellAvg` columns of
  `b`/`u`/`v` (`corr = (1/J)[D(f_face·w) − f̄·D(w)]`, the
  Reynolds-transport identity, `flux_diff` the exact face→cell
  telescoping — mirroring `_mapped_fv_divergence`), the **advective
  form** on `w`'s point-valued wall-normal column. The moving-wall
  mesh flux (`w·n ≠ 0`, where an advective flux vanishes) is carried by
  the one-sided `CellAvg → Outer` reconstruction + `Outer` `flux_diff`,
  not the `Inner` zero pad. This is the F5 pattern: module-level
  routing, **zero** spatial-layer changes. The `dynamic_geometry`
  carve-out is retired (`_fv_capable` now unconditionally True; the
  flag + factory plumbing removed) — the auto rule is now simply **FV
  wherever capable**. New invariant: the semi-discrete tracer budget
  closes to machine precision at every resolution (telescoping
  identity, the F5-advection analogue), so the morph tracer drift
  collapses from the nodal 6.9e-3 spatial-truncation drift to a pure
  time residual (1.35e-4, halves with `dt`). Gates: frozen-motion
  bitwise, constancy machine-zero, `H(t)` 2nd order (order 1.88 on the
  resolved pair), morph (exact volume, machine-zero divergence),
  autodiff FD-matched (isolated from a **pre-existing** mapped-pressure
  reverse-mode NaN — flagged, roadmap follow-up). Records:
  [`../research/ale_on_fv.md`](../research/ale_on_fv.md) + scoping §13
  addendum 2.

- **FV/nodal step baselines re-recorded on the campaign's final dev**
  (2026-07-17, gpu4 campaign T7) — both `benchmarks/baselines/step-gpu{1,4}.json`
  re-recorded on clean merged dev `5af2e370` (metadata `dirty=false`).
  `step-gpu4.json` re-recorded **in full** on a single-process 4×A100
  node (fusion workaround `--xla_disable_hlo_passes=multi_output_fusion`),
  gaining all nodal sibling cases (`nh_flat_periodic_nodal`,
  `nh_flat_advective_nodal`, `nh_flat_walled_nodal`,
  `nh_flat_walled_x_nodal`, `nh_mapped_nodal`); only the walled rows of
  `step-gpu1.json` were re-recorded (single device; other rows carried
  forward byte-identical). Regression check vs the old baselines: every
  large delta attributed to a known landing, no unexplained regression.
  The **walled** cases (`nh_flat_walled{,_x}`) now run the **FV** C-grid
  by default (walled auto flip), +0.8..+9.7% on 4 GPUs / +0.6..+5.0% on
  1 GPU vs the old FD numbers — confirmed as the FV-vs-FD cost by the new
  nodal siblings matching the old FD baseline to within noise. The
  **mapped** cases (`nh_mapped`, FV default) are **much faster**
  (4-GPU: −53..−58% at 30 iters; 1-GPU `nh_mapped[256,30]` 10.19 s →
  4.17 s, −59.1%) from the CG-tolerance default flip to 1e-8 (the bench
  cases pass no `pressure_tolerance`, so they run the 1e-8 convergence
  break with `pressure_iterations` as the max budget; the "pin fv morph
  to fixed-iteration" commit pinned a *test*, not these cases). FV and
  nodal siblings step at parity (small residual gaps are compiler/fusion
  artifacts, not physics). `sw_flat[1024]` +6.1% (4 GPU) attributed to
  the reverse-mode-safe Sadourny PV-division double-`where` guard
  (`4f6e86b6`, differentiability policy). Self-check
  `--fail-on-regression` vs the new baselines: green on both.

- **Coefficient profiles follow the family** (owner ruling
  2026-07-17) — the `MeridionalStratification` `n2` Profile question
  closed as: physical-coefficient fields declared through the
  family-agnostic patterns follow the grid family **by design** (on
  an FV model `n2(y)` is a `CellAvg(y)` profile — midpoint-initialized,
  so the numbers are unchanged at 2nd order, and every product stays
  co-located). No pins; one uniform rule. The distinction becomes
  real only under `create_field(order>=2)` quadrature init, where the
  cell-average reading is the correct FV semantics anyway.

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
- **Channel eigenmode projection — taught multi-device skip**
  (2026-07-17, T5) — the channel projection engine
  ([`_eigenbasis.py`](../../src/fridom/model/_eigenbasis.py),
  `_reject_sharded_projection` guarding `_contract_planes`) now raises a
  taught `NotImplementedError` when the grid shards a **periodic
  (Fourier) axis** across devices, instead of dying deep in the HLO
  verifier. Root cause (re-attributed on real 4× A100, refuting the
  earlier "c64 **FFT-norm** constant" reading): XLA:GPU/GSPMD lowers a
  sharded-transform-axis FFT through its distributed Cooley-Tukey
  decomposition (`fft_collective_permute_body`) whose **twiddle-factor**
  constants are synthesized at `complex64` against the `complex128`
  data — the fault reproduces with `norm=None`, so it is **not** the
  jax FFT normalization and **not** covered by `multi_output_fusion`.
  The gate keys off `default_layout.is_local(name)`, so it never fires
  on a single-device grid or a `device_ids=(0,)` grid on a multi-device
  host, and never on a grid too small to shard (the collapsed
  many-device case). GPU-scoped mirrored test
  (`test_channel_projection_rejects_a_sharded_periodic_axis`, skipped on
  the CPU backend so the batch-144 eigenbasis eigh does not trip the T5b
  heap-corruption crash). Minimal fridom-free repro + drafted (unfiled)
  jax issue:
  [`../research/artifacts/channel_fftnorm_gpu/`](../research/artifacts/channel_fftnorm_gpu/).
  Record:
  [`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md).
