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

## Guiding target (end state) — delivered

The end-state vision that steered the rewrite. Phases 1 and 2 delivered
the first four points (the grid is `fridom.spatial`, the model layer is
`fridom.model`); the coupled-models remainder of the fifth is tracked
as tasks 3.2/3.3 in [`open.md`](open.md).

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
| 2.1 | Design: model composition (2026-07-08) | Decisions D1–D5 in [`../specs/model/`](../specs/model/00_overview.md); grid follow-ups filed as [`../plans/done/phase2_grid_followups.md`](../plans/done/phase2_grid_followups.md). |
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

- **Time-dependent-field follow-ups: state-sourced energy metric +
  `FieldBlend` unification** (2026-07-19, rulings TDF-D10/D11, merges
  `2d222425` + `27790fdf`). TDF-D10: a time-dependent field weight in
  `EnergyMetric` becomes a state-sourced descriptor, so the metric is
  measured at the analysed state's own time instead of a frozen `t=0`
  bake (the eigen/channel tools keep the explicit `snapshot=True`
  frozen path, TDF-D6). TDF-D11: `FieldBlend` (the affine ramped
  Coriolis blend) now rewrites its carried field per substage through a
  SELF_UPDATE stage, so the term reads the fresh carry and the I/O
  staleness wart (`f_coriolis` I/O showing the `t=0` snapshot) is gone;
  the sw2 energy-correction refusal on a ramped `f` is lifted (the
  `conserving − linear` total telescopes exactly on the stage-time
  carry). AR-D2's mechanism half is superseded, its generality half
  preserved. Record:
  [`../plans/done/td_fields_followups.md`](../plans/done/td_fields_followups.md).
  **TDF-D7 nonhydro `n2(y, t)` law landed 2026-07-19** (merge
  `01628a30`): `MeridionalStratification` now accepts a
  `fr.model.ProfileFunction` law alongside its static callable, marking
  `n2` `time_dependent` and rewriting it each substage via a SELF_UPDATE
  stage — so the previously-unreachable TDF-D10 `n2` reciprocal energy
  weight is now live through a real model. This **closes** the `open.md`
  "Time-dependent fields — remaining follow-ups" entry entirely; its only
  other item, a re-diagonalization contract for the analysis tools, stays
  **declined** per TDF-D6 (a recorded decision, not open work — a
  time-dependent `L` has no fixed eigenbasis).
- **`EnergyMetric` `ps` depth weight + eigen-channel physical
  measure** (2026-07-18, `59b6047a`/`72427813` — the metric-side
  remainder of the stretched+terrain item, after the physical-integral
  default fixed the `u`/`v`/`b` legs). The hydrostatic `ps` weight is
  now `H/c²`: the vertical physical extent on flat/stretched-only
  grids (fixing the silent depth != 1 error — flat depth-2 bilinear
  skew 1.0 → 7.5e-15), the column-Jacobian integral on `maps=`
  columns (scalar on stretched-z, an `H(x, y)` field on terrain gated
  behind `allow_field_weights`; terrain barotropic bilinear skew
  −1.8e-15 through the public metric). The depth axis is read off
  `ps`'s own `ConstantSpace` factor (walled-horizontal channels keep
  assembling). The dense channel engine holds the depth in the same
  weight: `_bounded_measure` gives the constant `ps` leg **unit**
  measure (no double-count) and J-weights the nodal per-node measure,
  so stretched-z channel eigenpairs are Hermitian/M-orthonormal under
  the *physical* product (hermiticity 1.8e-16, orthonormality
  3.1e-15); a genuinely terrain-following column is refused by name in
  `channel_eigenpairs` (per-column eigenproblems — replacing the
  misleading Hermiticity-residual message).
- **Differentiability closure campaign — the public differentiable run
  surface plus the last VJP seals** (2026-07-18, executing
  [`../plans/active/differentiability_plan.md`](../plans/active/differentiability_plan.md))
  — the masked-singularity VJP class is now **closed across the step
  path** and `jax.grad` through a run has a public spelling. Five
  merges. Record hygiene (`45629c35`): three stale records corrected
  (jax_grad_run addendum for the mapped-pressure seal `ee350bda`;
  fv_nonhydro_scoping + hydrostatic_model_plan dated in-place notes).
  The three coriolis `metric_weight` divides sealed via new
  `_safe_metric_divide` (`93049651`: H1 `linear_rotation` `/ w.to(v)`,
  H2/H3 `chart_rotation` `/ w_1`, `/ w_2`; double-`where` + halo-trace
  escape hatch, 3 autodiff regressions — immersed weighted-rotation
  IC-grad FD 1.2e-13, chart-rotation term-level FD 1.35e-11). The
  `MetricScaled` reciprocal/coefficient divides sealed via
  `_sealed_metric_divide` (`7fdbc900`; see D4 note below). The Sadourny
  chart kinetic-energy `ekin/sqg_p` and conserving-Coriolis
  `f.to(corner)/h.to(corner)` PV divides sealed (`39fe604c`: a
  `_sealed_metric_divide` helper and new `_safe_pv_divide`, covering
  `CoriolisEnergyCorrection` and every
  `Nonlinear{FPlane,BetaPlane,Rotation}Coriolis` route). And the public
  surface `Model.propagator(*, wrt=(), steps, remat=None)` returning a
  pure `(theta, state=None) -> ModelState` (`dfd8ce0a`): calls
  `_chunk_body` directly (no donation, no host panic sync), splices
  `wrt` bound-parameter / `TIME_STEP` / PROGNOSTIC leaves by identity,
  defaults to a fresh stepper state (the warm-up ramp is in the
  gradient), teaches four refusals (unknown name, identity-defaulted
  constant, frozen-L under `freezes_linear_operator`, materialized
  owner); `remat` is a keyword-only `jax.checkpoint` hook forcing
  `unroll=1` (grad bitwise-equal to plain), forward parity with
  `advance` bitwise (maxdiff 0.0), 17 tests in
  `tests/model/test_model_propagator.py`. A package-wide sweep of the
  sw2/nonhydro2/hydrostatic tendency modules then verdict-tabled every
  other metric/thickness divide as already-sealed or cannot-fire — the
  masked-singularity class is CLOSED across the step path. AGENTS.md's
  differentiability policy now names `Model.propagator` the canonical
  pattern (the private `_chunk_body` shards stay valid). **Owner-review
  notes (items 1–4 owner-ratified 2026-07-19):**
  (1) D4 was executed as a *seal*, not the plan's
  approved comment-only watch-item — the premise was DISPROVEN: the
  `MetricScaled` divides fire live (walled/sphere IC-grad through
  `_chunk_body` NaNs, isolated by bisection, fires with `coriolis=None`
  too); grounds were the live NaN + the roadmap's own "guard when a
  composition exposes them" trigger + a 0.000%-added-FLOP cost proof
  (static-geometry mask constant-folds, singular quotient DCE'd; the
  ALE metric tangent handled by construction). (2) the propagator
  checks frozen-L *before* materialized (more specific message; every
  reachable L-param today is also materialized). (3) the frozen-L set
  is `linear_params`-only — params feeding L via `linear_fields` AUX
  fields are caught by the materialized refusal, so refusal
  completeness is identical today, but when TDF wave 2 makes
  linear-consumed fields recomputed-in-trace the frozen-L set must
  learn `linear_fields` (verified against the shipped TDF merges
  `bb2fb96f`/`ceb9db75`/`3d2d1e1a` 2026-07-19: law params are not
  `wrt`-bindable and a recomputed-in-trace linear-consumed field fails
  assembly via the `time_dependent`-marker guard, so no live hole —
  the deferral is forward-looking hardening for `n2(z,t)`/TDF-D7).
  (4) the materialized refusal over-refuses a differentiable param that
  merely shares an owner with a materialized field (e.g. sw
  `scaling.rossby`); in-trace rematerialization is the recorded
  follow-on. The candidate `nonhydro.dsqr` frozen-L hole — a
  stratification-free nonhydro2 ETDRK4 model slipping
  `wrt=("nonhydro.dsqr",)` past both refusals (`dsqr` enters L through
  the pressure-projection CONSTRAINT, not a `linear=True` term) — was
  investigated 2026-07-19 and is **unreachable**: ETDRK4's mandatory
  eigenbasis needs `stratification.n2` for the energy metric
  (`energy.py:340-355`), so no strat-free nonhydro2 ETDRK4 model is
  constructible, and every constructible one's stratification module
  class-declares `DSQR` in a `buoyancy_force` `linear_params`, keeping
  `nonhydro.dsqr` in the refusal set (empirically refused under ETDRK4,
  accepted under `AdamBashforth`; no code change). Phase 3 (D5
  `TangentPropagator`, `jax.jvp` of
  `model.tendency`) is **deferred** — no consumer (NNMD descoped); the
  shared name-resolution piece already shipped, so it stays a small
  lift ([`open.md`](open.md) sized-deferred). Record:
  [`../plans/active/differentiability_plan.md`](../plans/active/differentiability_plan.md)
  §10.

- **Stretched+terrain GPU validation — complete; semicoarsening
  multi-device break root-caused and closed** (2026-07-18) — the
  multi-GPU leg validated the core stretched+terrain paths (N2
  measure-adjoint hop, plain-CG, differentiability, terrain
  hydrostatic files, realistic 3D model device-invariant to the CG
  tolerance floor under single-process GSPMD AND real `srun -n 4`;
  addendum in
  [`stretched_terrain_combined.md`](../research/stretched_terrain_combined.md)),
  and its two remainders turned out to be one bug and are **closed**:
  the semicoarsening V-cycle multi-device parity break was bisected
  to `4ca61a96` (interval halo accounting halved registry widths →
  previously-replicated coarse levels silently flipped to
  **sigma-sharded**), which exposed the latent bounded-axis
  `(0,0)`-exterior-reach sync hole — the corrupt op was the
  coarse-level mapped operator apply at sigma shard seams, the line
  solve was always clean; the lone sharded bounded 1-D
  `MappedIntervalMesh` failures were the same hole. Cured by the
  invariants campaign's `b57e3e78` (per-shard `footprint_reach`);
  verified: forced-CPU-4 all victims green (23/23 stretched +
  lone-1D), **real 4x A100 three-file battery 47 passed / 1 skipped**
  (9 failed the previous day). Coarse levels remain sigma-sharded by
  negotiation — now machine-precision-correct (`1.6e-15..5.2e-14`
  vcycle parity incl. stretched 16/shard) — so the residue is
  perf/hardening only (open.md, multigrid section). Record:
  [`semicoarsen_multidevice_regression.md`](../research/semicoarsen_multidevice_regression.md).

- **Coefficient-space product/power rows — ruled closed by design**
  (owner-ratified 2026-07-18) — the open-roadmap semantics question
  ("should coefficient-space fields get `("multiply"|"divide"|"power"|
  "abs", space)` rows?") is settled: coefficient-space `ScalarField`s
  are a **vector space, not an algebra**. Only transform-commuting
  operations are field arithmetic (add/sub of same-space fields,
  scalar multiply/divide — already exact); an elementwise product of
  two coefficient fields is a **convolution** of the represented
  functions, not their product, so those elementwise rows are
  **permanently absent by design**, not an "iteration 1" deferral.
  Per-mode (diagonal) coefficient algebra lives on `Symbol`; the
  pointwise function product lives in nodal space; `Convolution` and
  the zero-mode `ConstantBroadcast` stay reserved distinct kinds,
  unbuilt until a consumer exists (the census found **zero**
  coefficient×coefficient product consumers). Shipped: taught-error
  rewording of the coefficient-space dunders/guards
  (`spatial/fields/scalar_field.py`) + pinned tests, spec note, and
  the `products.py` docstring line. **This closed the last open item
  of the Phase-2 grid follow-ups.** Record:
  [`coefficient_space_arithmetic_semantics.md`](../research/coefficient_space_arithmetic_semantics.md).

- **Mapped + advection + chunked scan non-finite — root-caused,
  already fixed** (2026-07-18 investigation; the fix itself landed
  2026-07-17 in `44b5cb8d`) — the open-roadmap fault (recorded from
  the CG GPU measurement at `b77f8582`: mapped advective runs
  non-finite at `chunk_size ≥ 2` while the same steps run finite one
  at a time) was the unguarded mapped velocity-correction `flux / J`
  planting `inf` in the never-valid storage padding: the per-chunk
  `_scrub_ghost_storage` cleansed it at chunk=1, while inside a
  chunk≥2 scan the carry seal refills only negotiated halo lanes, so
  the next step's masked wall arithmetic hit `0·inf = NaN` and the CG
  dot products globalized it (u/v/w/p 100 % non-finite at it=2, b one
  step behind). **Never a GPU or compiler fault**: CPU reproduces the
  signature bit-identically at n=64 (the "on GPU" title was an
  observation artifact — the CPU leg was never run; the
  `multi_output_fusion` non-fix is thereby explained). Fixed
  *accidentally* by the velocity-correction **VJP** guard `44b5cb8d`
  ("the forward projection is untouched" — it was the forward fault
  too); adjacent-commit bisect (`30f4a624` broken → `44b5cb8d`
  fixed), and HEAD runs chunk=2 bit-identical to chunk=1 through
  it=22 at 256³. Bonus finding: the bench config itself (unclosed
  inviscid centered advection) blows up physically at it≈24
  (t≈0.12), cadence/backend-independent and bit-identical across the
  254 intervening commits — the original chunk=1 "control" looked
  finite only because it stopped earlier. The **chunk-cadence parity
  regression shipped with this merge**
  (`test_mapped_advection_chunk_cadence_parity` in
  `tests/model/test_step_chunk.py`): a terrain-following advective
  model stepped four times at `chunk_size` 1 vs 2 must stay finite and
  agree to `rtol 1e-12`. Red-checked — reverting the
  `_divide_by_jacobian` guard reproduces `PanicError` at it=2 already
  at n=8 (a smaller floor than the n=64 recorded above). Not bitwise:
  CPU scan-length grouping reassociates FP at ~3e-15, while the GPU
  256³ measurement above was bitwise. The last hardening residual —
  the held `MetricScaled` pad-inf seal, owner decision D4 — closed
  2026-07-19: the differentiability campaign's own seal (`7fdbc900`,
  live reverse-NaN + 0.000%-FLOPs cost proof) was owner-ratified and
  the redundant held branch deleted; nothing remains open. Record:
  [`mapped_chunk_nonfinite_rootcause.md`](../research/mapped_chunk_nonfinite_rootcause.md).

- **Storage-halo width recovered — two-sided (interval) halo
  accounting** (2026-07-18, probe merge `a8a9aefb`, implementation
  merge `40a24df8`) — the "n+8 vs nominal n+6" roadmap question
  resolved: the extra layer was the halo trace **summing symmetrized
  scalar reaches** along the sync-free advection chain (biased
  reconstruct 3 + flux-diff 1 → width 4), losing the biased window's
  asymmetry; the true Minkowski-composed footprint of
  `flux_diff ∘ reconstruct` is 3/side (not staggering, not
  even-rounding — upwind3's odd width 3 refutes that reading).
  `HaloSpec`/`OperatorRequirements`/trace/`halo_valid` now carry
  per-name two-sided reaches (symmetric max presented to storage;
  per-op reaches derived from implemented kernel geometry, not
  hardcoded). upwind5/weno5 negotiate width 3 → storage `n+6`
  (−5.7% step bytes @96³, −3.0% @192³; compiled `memory_analysis`
  tracks `(n+6)³/(n+8)³` to 4 s.f.); per-step sync counts identical
  to dev (scalar-validity control re-syncs 28 vs 16 — the two-sided
  runtime validity is load-bearing); narrow-vs-wide same-code parity
  **bitwise** (0.0, upwind5 and centered, 32³×10). The empirical
  probe first proved the width-3 floor (width 2 fails the taught
  dry-run guard) and that forcing a narrow store *without* interval
  validity trades bytes for mid-chain re-syncs. Bonus fixes: immersed
  fraction/mask and coordinate-measure caches re-keyed on the
  negotiated halo (latent stale-array bug under wider
  re-negotiation); bounded shrinking stencils claim reach 0; one FV
  mapped-pressure "bitwise" assertion honestly relaxed to 1e-14
  (width-coincidental pairwise-reduction tie on dev). Gates:
  new-stack suites 6329 passed / 0 failed post dev-merge (FV
  fusion-guard ratchet included), forced-4 decomposition 334 passed,
  3 autodiff files green, reblock HLO golden regenerated (pure shape
  shift), ruff clean. Centered stays width 2 at the assembled model
  (`DynamicalCore.extra_halo = 2` floor) — remainder tracked in
  [`open.md`](open.md). Record:
  [`storage_halo_width.md`](../research/storage_halo_width.md)
  (probe scripts + RESULTS under `storage_halo_width/probe/`).

- **Storage-halo GPU A/B executed — shape-luck verdict** (2026-07-18,
  owner-requested, single A100-80GB): compiled bytes track
  `(n+6)³/(n+8)³` to 4 s.f. at every size, but wall-clock is
  per-(scheme, size) XLA:GPU kernel-selection luck, ±10-20% in *both*
  directions — upwind5 +12-18% @192³ yet −3-14% @512³, weno5 +5-17%
  @256³/512³, 128³/256³-upwind5 neutral, centered A/A noise ~0.3%.
  The 192³ width scan (w3-w6 uniform + per-axis) refutes both the
  bytes-monotone and the alignment reading: only the uniform 200³
  (width-4) shape is fast, so there is no padding rule to chase.
  Cross-checked against real pre-merge dev (detached worktree at
  `4c287c12`): forced-wide new code compiles byte-identical to old
  dev (persistent-cache-served). Narrowing kept (memory −3%, CPU
  faster, GPU mixed); follow-ups (biased `bench_step` cases, optional
  width-floor knob) tracked in [`open.md`](open.md). Record:
  [`storage_halo_gpu_ab.md`](../research/storage_halo_gpu_ab.md)
  (harness + raw results under `storage_halo_gpu_ab/`).

- **Pressure-solver halo demand researched — "silently wrong"
  disproven, true demand 1, derivation design** (2026-07-18): every
  default pressure/constraint solver path needs **1**/side, not the
  declared 2 — the nonhydro2 projection's div and grad are 2-point
  legs on opposite sides of a global transform that acts as a runtime
  validity barrier (legs merge by max, not Minkowski sum);
  empirically forcing the declaration to 1 is bitwise on the periodic
  and walled spectral paths, ~1 ulp on mapped CG, and the hydrostatic
  surface solve already declares 1. The owner's stale-declaration
  fear does not hold: under-provisioning fails **loudly** (registry
  consumption guards; the physics floor keeps width ≥ 1 in any
  runnable model), and the spectral symbol derives from the same
  registry rows the stage applies, so an operator swap cannot
  silently desync. Also over-declared: hydrostatic core terrain 2→1,
  sw2 orthogonal-chart gravity 2→1 (Sadourny/Coriolis corner-chain 2s
  are genuine). Design: derived `extra_halo` ("structure declared,
  numbers derived") is feasible — the declaration is read after
  `bind` and the dispatch merge, so it can resolve the bound operator
  rows and compose their two-sided `reach`. Implementation (and the
  centered `n+4 → n+2` it unlocks, GPU-gated) tracked in
  [`open.md`](open.md). Record:
  [`pressure_solver_halo.md`](../research/pressure_solver_halo.md)
  (probes under `pressure_solver_halo/`).

- **Storage-shape "luck" root-caused — loop-emitter remainder + DRAM
  stride, no free mitigation** (2026-07-18, merge `70f88db0`): the
  192³ upwind5 n+6-slower-than-n+8 regression is *not* kernel
  mis-selection — the step is GPU-bound (the contrary nsys read was a
  CUDA-graph tracing artifact; profile with `--cuda-graph-trace=node`)
  and the whole gap lives in the advection loop fusions on padded
  storage: (A) a bounds-checked remainder tail when the padded flat
  count is not divisible by 512 (200³ is, 198³ is not) on the
  register-capped flux fusions, and (B) the larger effect, a
  DRAM-partition stride penalty on the elementwise fusions (39% vs
  54% achieved DRAM at identical access/occupancy). No exploitable
  rule (208³ ÷512 yet slow; centered's sweet spot is 196³, not 200³)
  and no free flag (`multi_output_fusion` off equalizes by slowing
  the fast shape 22%); the only lever is measure-and-pin per flagship
  config. The reported instability was concurrent-tenant
  contamination — on a dedicated GPU slow shapes are deterministically
  slow. The centered `n+4 → n+2` gate executed the same day (§8b):
  uniform +2.3-4.3% at 128³-512³, worst at 192³ as predicted, while
  args bytes drop 1.2-4.5%; a linear pair on the *identical* 194³
  storage is −1.2% *faster* (the sign is the scheme's fusion
  population). Record:
  [`upwind5_shape_regression.md`](../research/upwind5_shape_regression.md)
  (probes + raw + gate harness under `upwind5_shape_regression/`).

- **Derived `extra_halo` shipped — pressure-solver halo from bound
  operators** (2026-07-18, merged in `b931452c` — the merge landed
  under a parallel session's design-commit message in the shared
  checkout; parents `70b012d8` + `feat/derived-extra-halo`
  `8a92c8b8`): the three over-declaring cores now declare their
  stage's dataflow *structure* and derive the numbers from the merged
  registry rows the stage applies (`model/halo_demand.py`: within-leg
  Minkowski sum, per-side max across transform barriers, symmetric
  collapse; `nonhydro2.DynamicalCore` gained the `bind` this needs).
  Negotiated widths: nonhydro2 flat spectral + mapped CG 2→**1**
  (centered/linear storage `n+4 → n+2`), hydrostatic terrain 2→**1**
  and immersed 2→**(1,1,0)** (no vertical stencil), sw2 chart gravity
  2→**1** — including non-orthogonal charts: the research table's "2
  (tight)" was symmetrize-then-sum over-counting; the cross-interp
  telescopes against the gradient bias (`[0,+1] ⊕ [-1,0] = [-1,+1]`),
  bitwise-verified on three sheared charts. Parity: bitwise on all
  spectral/hydro paths, 8.3e-17 mapped CG, 8.7e-19 sw sphere. The CG
  diagonal builders (the one consumption-guard bypass) now assert
  solved-axis width ≥ 1 at build. Gates: mirrored + model suite
  (2133) + forced-4 decomposition green, ruff clean; GPU gate run
  *before* landing (entry above) — ships on memory/tightness/CPU
  grounds with the centered +2.3-4.3% priced in.

- **Storage-width follow-ups closed — biased `bench_step` cases added,
  width-floor knob REFUSED** (2026-07-18, owner rulings in chat):
  `nh_flat_advective_upwind5` / `_weno5` cases added to
  `benchmarks/model/bench_step.py` (append-only; order pinned at 5;
  default family — biased FV assembly works now, the A/B record's
  "blocked" note is stale) on a dedicated size grid `[32, 192, 256,
  512]` that includes the measured 192³ knife-edge size, so the
  biased/WENO fusion families' storage-shape sensitivity is
  guard-visible from now on. Verified end-to-end on the A100 at all
  sizes (per-step numbers reproduce the A/B: upwind5 6.26 @192³ /
  123.6 ms @512³, weno5 6.94 / 138.8); **baseline recorded at the
  owner's next batched guard run**, not before. The storage-width
  floor knob is **refused** (owner, 2026-07-18): no public knob;
  measure-and-pin stays a benchmark-internal technique (the forcing
  monkeypatch in the research harnesses), and the shipped widths
  stand as measured. Records:
  [`upwind5_shape_regression.md`](../research/upwind5_shape_regression.md),
  [`storage_halo_gpu_ab.md`](../research/storage_halo_gpu_ab.md) §4.

- **Upstream jax issues filed for the two T5 faults** (2026-07-18,
  owner-filed) —
  [jax-ml/jax#39291](https://github.com/jax-ml/jax/issues/39291)
  (T5: SPMD-partitioned FFT emits `complex64` twiddles for a
  `complex128` transform, HLO verifier rejects the mixed multiply) and
  [jax-ml/jax#39292](https://github.com/jax-ml/jax/issues/39292)
  (T5b: batched-`eigh` heap corruption from nested Eigen×OpenBLAS
  thread oversubscription). Pre-filing validation on the day-of-release
  jax 0.11.0: **both still reproduce** (changelog touches neither
  path). New findings folded into the reports: the FFT fault
  reproduces **without GPUs** (`JAX_PLATFORMS=cpu` +
  `--xla_force_host_platform_device_count=4`, identical verifier
  error), so it sits in the backend-agnostic SPMD partitioner and the
  filed repro needs no hardware; the eigh threshold scales with host
  thread count (256-thread node: batch 144; 128-core login node: batch
  144 passes, 1024 crashes — 0.10.2 SIGABRT, 0.11.0 SIGSEGV). Prior-art
  sweep found no existing upstream report of either root cause
  (closest context: jax#15680 sharded-FFT all-gather for T5;
  OpenBLAS #2839/#4216/#5639 for T5b's mechanism, all closed as
  config-expected — cited as evidence the nesting is jax-side).
  Final drafts (owner voice, trimmed to repro + evidence) live with
  the repro scripts:
  [`channel_fftnorm_gpu/`](../research/artifacts/channel_fftnorm_gpu/),
  [`channel_sort_segfault/`](../research/artifacts/channel_sort_segfault/).

- **Half-axis-sharded 3-D channel served — layout-aware half-axis
  re-designation** (2026-07-18, merge `feade7fa`; coverage follow-up
  merge `964a9117`) — the last remainder case with a fast path: when
  the default layout shards the engine's half (`rfft`) axis (reachable
  only when an earlier periodic axis is indivisible by P — see the
  exposure survey), `channel_eigenpairs` now designates a **local**
  periodic axis as the half axis (`_designate_half_axis`), so the
  shipped fused contraction serves the layout with roles swapped —
  zero new collective code, and the basis is *built* directly in the
  chosen frame (no runtime re-layout). Byte-identical whenever the
  last periodic axis is local (single device and all
  previously-served layouts). Touched: the pick + Fourier-axis
  ordering (`model/eigen_channel.py`), `fourier_ops`
  (`model/_eigenbasis.py`), the nh Leray labeler
  `_constrained_column` generalization
  (`nonhydro2/channel_eigenmodes.py`), stale-remainder docstring trim,
  a GPU-scoped end-to-end regression test + a CPU-safe pick unit test
  (forced-4 CI leg). Gates: frame freedom proven single-device
  (both frames agree ≤ 2.5e-14); 4×A100 many-vs-one 8.5e-15,
  idempotency 1.4e-14, HLO all-to-all present / all-gather absent;
  per-test outcomes across the 8 mirrored/adjacent eigen test files
  **byte-identical to base dev** on the 4-GPU node (334 outcomes, the
  only delta the new passing test; the pre-existing multi-device
  setup faults unchanged); ruff clean. Non-perf-sensitive (build-time
  pick + projection utilities; step path untouched) — no guard run.
  Record: [`eigen_remainder_investigation.md`](../research/eigen_remainder_investigation.md)
  (the validated patches it archived landed as this merge).

- **Channel eigenmodes run multi-GPU — fused distributed contraction**
  (2026-07-18, merge `e60259de`) — the projection/`f(L)` application
  (`_eigenbasis._contract_planes`) no longer rejects a grid that
  shards a periodic axis: a new fused lowering
  ([`distributed_contract.py`](../../src/fridom/spatial/operators/distributed_contract.py),
  the sibling of the fused spectral solve) runs forward / per-plane
  `Q diag(w) Qᴴ M` contraction / backward in **one `jax.shard_map`
  region** so every FFT axis is device-local when its transform runs
  — the upstream XLA:GPU distributed-FFT c64-twiddle fault
  ([`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md))
  is never reached. Geometry is engine-constrained (the half axis is
  fixed to the engine's `rfftn` frame, never relocated; the transpose
  partner is the half **coefficient** axis, padded on its `n//2+1`
  extent with empty trailing pad shards allowed — the lanes are
  transient, and zero-padded `q`/`w` make their output exactly zero),
  which is why the generic `_distributed_geometry` (fully-complex
  `h=None` on a two-stage transform) could not serve it. `q`/weights
  enter as `in_specs`-sliced arguments (per-device basis memory ÷
  device count; plan memoized per grid, 0 warm recompiles). Measured
  (4× A100, n=16 nonhydro channel): many-vs-`device_ids=(0,)` max abs
  diff **1.08e-14**, idempotency 4.9e-15, HLO all-to-all only (no
  all-gather/all-reduce), reverse-mode gradients finite. Covers
  nonhydro2, shallowwater2, and hydrostatic through the shared base;
  the taught `NotImplementedError` narrows to the unsupported
  remainder (2-D channel's single periodic axis, the half axis itself
  sharded, non-1-D mesh). Single-device and bounded-axis-sharded
  paths byte-for-byte unchanged; the pressure-solve path untouched
  (perf merge gate: non-perf-sensitive — the new module is imported
  only by `_eigenbasis.py`, nothing on the step path). Tests on the
  forced-4 CI leg (`test_distributed_contract.py`,
  `test_eigenbasis_distributed.py`); the sharded-periodic rejection
  test flipped to a runs-and-matches gate. Two **pre-existing**
  multi-device eigenbasis faults surfaced (setup `GridFrozenError`,
  `mode()` synthesis crash) stay open in [`open.md`](open.md).

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
  plan §7–§8). Same-day follow-up: the shared mapped advection's
  `Z/J` divide sealed (`_safe_ratio` double-`where`), making
  nonlinear terrain advection reverse-safe (grad = FD to 9e-10; the
  hydrostatic terrain autodiff gate now runs `advection=True`).
  Residuals tracked in [`open.md`](open.md).

- **Terrain buoyancy slope-advection term**
  (`fix/terrain-buoyancy-slope-term`, 2026-07-18) — the terrain
  hydrostatic buoyancy equation now couples `b` to the **physical**
  vertical velocity `w_true = Jω + u·Zₓ + v·Z_y`
  (`stratification.restoring`'s terrain branch adds the slope-advection
  half `−N²(u·Zₓ + v·Z_y)`, absent since the sigma core landed). This
  fixes the O(slope)-wrong terrain internal-wave physics and restores an
  energy-consistent KE↔PE exchange under the physical (J-weighted)
  metric — the root of the apparent baroclinic/barotropic asymmetry
  ([`../research/energy_metric_asymmetry.md`](../research/energy_metric_asymmetry.md)).
  It also **corrects the over-claim** in the stretched+terrain entry
  above ("baroclinic energy legs machine-precision"): that gate passed
  by *state-selection accident* (single-mode states sit in the leak's
  null set); the pre-fix pair actually leaks O(slope), resolution-
  independent. **Analytic spelling shipped, not the exact discrete
  adjoint.** The adjoint was proven to reach machine-zero skew (a
  probe confirmed the plain-z-gradient↔`Jω` pair is *already* exactly
  skew on terrain, so `corr`'s exact metric-adjoint closes the pair for
  arbitrary states) — but it bakes the grid quadrature weights into the
  buoyancy tendency and requires transposing the C-grid interpolation
  chain (`jax.linear_transpose` of the pressure-gradient machinery every
  step, or measure-ratio adjoint rows that only reduce cleanly on
  uniform-periodic/uniform-z columns), so it fights the staggering and
  is against the differentiability policy's spirit. The local physical
  `w_true` is the shipped form: manifestly the physics, cheap, trivially
  reverse-differentiable, O(h²)-consistent (the full correctness fix —
  right continuum limit). Gate: the accidental smooth single-mode
  `test_baroclinic_energy_conversion` replaced by a **bilinear
  random-state physical-skew gate** (independent broadband X, Y, all
  components; hand-built physical metric, `ps` leg lifted for `H/c²`);
  pre-fix ~0.8 flat in n, with the term it collapses at ~2nd order
  (1.8e-1 / 2.6e-2 / 5.7e-3 at n = 16/32/64, orders 2.7 / 2.2). Rest
  state preserved (term vanishes at u = v = 0), flat path byte-identical
  (`self._column is None`), and a propagator autodiff gate (grad wrt
  initial `u`, which the new term feeds into `db/dt`) is finite +
  FD-matched. Follow-up in [`open.md`](open.md): the diagnosed-`w`
  output labeling (`Jω` vs physical `w`).

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
- **Multigrid generalization: terrain implicit surface + coarsening
  freedom + warm starts** (2026-07-18, plan
  [`../plans/active/multigrid_generalization_plan.md`](../plans/active/multigrid_generalization_plan.md),
  all five phases owner-ratified and shipped same day; merges
  `51db9ba6` A, `70b012d8` E, `a0eb7027` D, `6e32b4b4` B,
  `5e7a0eff` C) — closes the **H3 implicit** taught error:
  `hy.ImplicitFreeSurface` now runs on sigma charts via the new
  `BarotropicPressureSolver` (GM-D1 volume-exact variable-csqr
  operator: volume drift ≤ 1e-12, correction cancellation 6e-16,
  flat-limit exact, autodiff green), preconditioned by the flat
  spectral inverse or the new 2-D point-Jacobi multigrid (iterations
  11–13 h- **and** steepness-flat vs spectral's 27 at a=0.8;
  forced-4 replicated-coarse-level parity green). All pressure/surface
  CG solves warm-start from the previous step (GB-2 128³ step
  −19.5 % multigrid / −16.8 % spectral, physics-neutral ≤ 3.5e-9),
  and full 3-D coarsening is the mapped-solver multigrid default
  (GM-D9: −4.3 % on top, parity 3.1e-10, automatic semicoarsening
  fallback for Chebyshev / indivisible n_z / stretched-base columns).
  Residuals stay in `open.md`: the split-explicit chart variant and
  the real multi-process 4-GPU leg. The hydrostatic
  walled-horizontal gap found in phase B was closed the same day
  (flat + immersed; see its own entry below).
- **Multigrid size-scaling root cause: the depth cap, not the
  algorithm** (2026-07-18, measurement-only; record
  [`../research/multigrid_depth_scaling.md`](../research/multigrid_depth_scaling.md))
  — the post-swap "deficit vs spectral widens with n" verdict was an
  artifact of the fixed `multigrid_levels=5` default: h-independence
  breaks once the coarsest level outgrows its 8 sweeps (iterations
  10 → 15 → 27 at 128/256/512³ while spectral stays flat 36;
  per-iteration cost is healthy — 42× per 64× more cells vs
  spectral's 54×, the cost ratio *improving* 3.14× → 2.45×). At
  floor-scaled depth (coarsest 8×8×n_z; L=6 at 256³, L=7 at 512³):
  flat **10** iterations at every size, per-cycle cost unchanged,
  and the in-model GB-2 step **beats spectral 1.23× at 256³
  (237.8 vs 291.6 ms) and 1.22× at 512³ (1873.3 vs 2277.9 ms)**
  (physics equivalence 2–5e-11). GB-2 (≥1.5×) still unmet at every
  measured size. **Floor-limited depth shipped as the default the
  same day** (owner-ratified; merge `b8b165f1`):
  `multigrid_levels: int | None = None` on `nh.Model` /
  `DynamicalCore` / both pressure solvers / `coarsen_levels` — None
  (default) coarsens to the 4-cell horizontal floor, an int stays an
  explicit cap; default-path 512³ in-model validation 2000.1 ms/step
  (1.14× vs spectral; realized depth 8 — one borderline CG iteration
  above the hand-capped L=7 row, physics 1.3e-10).
- **Multigrid V-cycle kernel swap** (2026-07-18, merge `0ece46b1`) —
  `banded.tridiagonal_solve_along_axis` grew a host-static
  `method` knob with three interchangeable kernels: `"scan"` (the
  old reference Thomas, kept verbatim), `"pcr"` (pure-jax parallel
  cyclic reduction, portable, arbitrary n) and `"cusparse"` (batched
  `lax.linalg.tridiagonal_solve`, gtsv2StridedBatch); `"auto"` (the
  default) resolves host-side to cuSPARSE on a GPU backend and PCR
  elsewhere, and an explicit `"cusparse"` off-GPU raises a taught
  ValueError. Threaded as `multigrid_tridiagonal_method` along the
  `multigrid_levels` route (`nh.Model` → `DynamicalCore` → both
  pressure solvers → `VerticalLineJacobi`), name-validated at
  construction. All kernels agree to ~1e-18 (convergence-neutral)
  and are natively reverse-differentiable; autodiff + garbage-ends +
  non-power-of-two + dispatch tests shipped in the mirrored files.
  Microbench (A100, n_z = 128, batch 128²): scan 2.80 →
  pcr 0.37 → cusparse 0.20 ms/solve. In-model steep mapped
  (GB-2 protocol): 128³ step 542.7 → **42.0** ms (12.9×; spectral
  41.0 — parity, 0.975×); 512³ scan 7394 → cusparse **3402** ms
  (2.2×; spectral 2278 — 0.67×, so the GB-2 1.5× bar stays unmet
  and spectral stays the mapped GPU default; the study's "likelier
  at larger n" projection is refuted at 512³). Physics equivalence
  spectral↔cusparse ~5e-11 at both sizes. 512³ memory: spectral
  28.5 / multigrid-cusparse 43.8 GiB peak (fits one A100-80GB);
  the pcr variant OOMs at 512³ (XLA live set ≥ 76 GiB) — on GPU
  the cuSPARSE default is also the memory-viable kernel. Two
  corrections to the study record: the projected 128³ post-swap
  1.17× measured as 0.975×, and the "free IMEX side benefit" was
  wrong (`model/implicit.py` uses the dense `solve_along_axis`,
  not this kernel). *Corrected same day: the "deficit widens with
  n / spectral stays default at every size" conclusion was the
  `multigrid_levels=5` depth cap — see the size-scaling entry
  above.* Open residue (residual mapped-GPU levers): [`open.md`](open.md);
  the cuSPARSE-under-GSPMD leg is now closed (entry below). Evidence:
  [`../research/multigrid_kernel_study.md`](../research/multigrid_kernel_study.md)
  §§Addendum, Addendum 2.
- **cuSPARSE-under-GSPMD HLO/perf leg + immersed post-swap standing —
  measured** (2026-07-18) — closed the two residues the kernel-swap
  entry above left open, on real 4× A100 (jax 0.10.2, dev `0c950a33`).
  **cuSPARSE under GSPMD**: XLA partitions the batched custom call
  cleanly along the sharded batch axes — per-shard operands
  (`f64[(128/4)·128, 128, 1]` down to `f64[4, 4, 1]`) at every one of
  the six full-3-D-coarsening levels, with no feeding collective (the
  module's all-gathers are the projection global-mean and a `take`
  index gather, neither a cuSPARSE operand), both in a minimal
  standalone jit and in the in-model `jit__chunk_body`. Parity 1-vs-4
  and cuSPARSE-vs-pcr ~1e-14, CG iterations flat 10; 4-GPU timings
  mg-cuSPARSE 1.11× at 512³ but 0.37× at 128³ (per-level collective
  latency), and **pcr fits 512³ multi-device** (12.1 GiB/dev — the
  one-GPU ≥ 76 GiB wall is sharded away). The `banded.py` multi-device
  caveat is rewritten to record the validated partitioning (observed
  XLA lowering, not a contract; pcr stays the portable kernel).
  **Immersed post-swap standing**: mg-cuSPARSE 1.09×/1.12× at 128³/256³
  at the production budget=100 — the study's projected 1.3–2.0× was a
  budget=30 artifact (at budget=100 spectral converges at 71–73 iters);
  mg is the only converged option below budget ≈70. GB-2 (≥ 1.5×) stays
  unmet at every size/device count. Data + scripts + HLO excerpts:
  [`../research/artifacts/multigrid_gspmd_validation/`](../research/artifacts/multigrid_gspmd_validation/);
  narrative:
  [`../research/multigrid_kernel_study.md`](../research/multigrid_kernel_study.md)
  Addendum 2.
- **Coarse-level agglomeration — replicate the deep multigrid levels**
  (2026-07-19, merge `9e08493f`) — the census-driven lever (kernel-study
  Addendum 3): coarse V-cycle levels below a per-shard-extent threshold
  are built with a replicated layout (MG-D5 `Layout({})` end to end, no
  new comm pattern), so their smoother/operator/projection issue zero
  collectives and the cross-boundary reshard rides
  `jax.device_put` on restrict / a local slice on prolong. Knob
  `multigrid_agglomerate: int | None = None` (`τ`, default OFF) on
  `nh.Model` → `DynamicalCore` → both mapped/immersed solvers; the switch
  fires at the first level whose shortest would-be per-shard extent `< τ`
  **and** whose replicated per-device footprint `≤ 4 MiB`. Builder in
  `spatial/operators/multigrid_hierarchy.py`
  (`negotiate(force_replicated=)`, `Grid.coarsened(replicated=)`,
  `coarsen_levels(agglomerate=τ)`). Gates green at forced 4/16 host
  devices (CPU): parity — identical CG iterations ON vs OFF vs 1-device
  (mapped 10, immersed 17); **not** bitwise multi-device (the coarse
  mean projection reduces over a replicated array, so the last bits
  reassociate) but below the `1e-8` solve tolerance and **ON no worse
  than OFF** vs the 1-device truth (mapped model drift 3.4e-16; immersed
  ON-vs-1dev 1.7e-11 < OFF-vs-1dev 3.5e-11); one-device knob is a bitwise
  no-op. Capability — ON replicates a below-`τ` sharded coarse level at
  the same floor depth. Autodiff — `jax.grad` through a short ON immersed
  run finite + central-FD matched (reshard is a pure relayout). CPU
  forced-4 HLO census confirms the targeted collectives vanish: the
  coarse sub-KB z-halo permutes 24→0 and the vertical-line-smoother
  column-transpose all-to-alls 6→0 per V-cycle. **Two corrections to the
  plan's motivation** (record §4): (1) the *capability* claim — "a
  P-device sharded axis cannot coarsen below P cells; depth capped" —
  did **not** reproduce: at forced 16/32 devices floor depth is already
  reached, because layout negotiation (MG-D5 `allow_replicated`) already
  replicates below the shardability floor with no crash, no empty shards,
  no depth cap (the earlier h-independence break was the fixed
  `multigrid_levels=5` cap, not device count). (2) The reproduced driver
  is **latency only**: in semicoarsen/immersed hierarchies the sharded
  axis flips x→z as horizontals coarsen, and the coarse levels stay
  z-sharded at 2 planes/shard — the census's sub-KB regime. **Phase 3
  4-GPU wall-clock ran** 2026-07-19 (job `26355284`, artifacts
  [`../research/artifacts/multigrid_agglomeration_phase3/`](../research/artifacts/multigrid_agglomeration_phase3/)),
  closing the GPU-leg, `τ`-sweep and folding follow-ups: **no `τ` is a
  wall-clock win** (the ~9–14 ms/128³ projection did not reproduce — best
  τ8 recovers 5.8 ms, τ4 0.6 ms; immersed τ4 regresses off −0.9/−4.3%;
  GPU does not fold the replicated reductions, collectives net −6%), so
  **default OFF stands — owner-ratified 2026-07-19**; item fully closed,
  the knob remains the large-P capability escape hatch. Record:
  [`../plans/done/multigrid_agglomeration_plan.md`](../plans/done/multigrid_agglomeration_plan.md)
  §4; driver:
  [`../research/multigrid_kernel_study.md`](../research/multigrid_kernel_study.md)
  Addendum 3.
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
- **Mapped + immersed composition** (2026-07-18/19, merges
  `5bc27631` M0+M1, `48ac9052` M2–M4, `dc164825` M5) — the second
  immersed residual closed: chart/terrain + immersed grids now run
  in nonhydro2 and hydrostatic. Jacobian-weighted chart fractions
  (MI-D1, tensor GL × column J, separable path bitwise-untouched);
  the composed cut-cell metric pressure projection
  (`ComposedPressureSolver`: face-α direct legs + shared corner-α
  inside the cross hops — the symmetry-gate-selected spelling;
  symmetry 5.3e-16, all-wet ≡ mapped **bitwise**, identity-chart +
  mask ≡ flat immersed ≤ 1e-12); fraction-weighted composed
  multigrid (~15 iters where masked spectral needs ~200;
  `pressure_preconditioner` now `None`=auto, composed→multigrid);
  the M4 cross-flux conservation fix (θJ-mass 4.4e-16, was 1.17e-3);
  stretch-aware immersed bands (first assembling stretched-z
  immersed models); hydrostatic wet-column terrain barotropic solve
  + masked contravariant continuity (column equivalence 7e-16,
  all-wet byte-identical, θ-mass drift exactly 0.0). `order=None`
  chart masks and split-explicit/multigrid-on-terrain+immersed are
  taught errors. Follow-ups in [`open.md`](open.md). Plan +
  decisions + per-stage records:
  [`../plans/active/mapped_immersed_composition_plan.md`](../plans/active/mapped_immersed_composition_plan.md).
- **Biased/upwind/WENO advection on immersed grids** (2026-07-18,
  merge `02663933`) — the first immersed residual closed: the
  IP-D8 taught error replaced by the **mask-keyed graded ladder**
  (`graded.apply_graded_mask`), generalizing the wall closure's rung
  ladder from index-distance keying to per-face distance-to-dry
  selectors (static union-window products of the boolean masks;
  pre-masked operand; full-array rungs + trace-time-constant
  `jnp.where` select — the PALM precompute-and-select precedent,
  subsuming NEMO/MITgcm difference-zeroing). Covers both families
  (FV `_FVBiasedReconstruction`, nodal `_BiasedFaceReconstruction`
  both shifts, centered velocity interp, WENO both-then-select).
  Keystone gate: **staircase advection tendency ≡ walled graded FV
  at machine zero** (up3/up5 exactly 0.0, weno5 1.4e-17); all-wet ≡
  unimmersed bitwise on both families; θ-mass ≤ 1e-12 on genuine
  partials; autodiff FD-matched; forced-4 green. Plan + decisions +
  corrections:
  [`../plans/active/immersed_graded_advection_plan.md`](../plans/active/immersed_graded_advection_plan.md).
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
  TTFS 7.5→5.05 s, warm 2.83 s, per-step unchanged. Of the
  remainders, only the default-off async two-tier chunk-compile
  patch stays in [`open.md`](open.md); HLO-volume reduction and the
  comparison-suite metric fix are closed (entries below). Record:
  [`../research/time_to_first_step.md`](../research/time_to_first_step.md).

- **Async two-tier chunk compile** (2026-07-18) — the time-to-first-
  step §3c follow-up, now landed behind the default-off knob
  `Model(async_chunk_compile=True)`. On a chunk cache miss whose
  natural unroll > 1, `step_chunk` lowers the full-unroll chunk on the
  calling thread (a `Lowered` holds HLO, not the donated carry's
  buffers), synchronously compiles a cheap `force_unroll=1` variant of
  the same length, serves it while the full executable compiles in a
  daemon thread (`.compile()` releases the GIL), and swaps the cache
  entry to the full executable at a later chunk boundary — both tiers
  share the `out_shardings` pin, so the swap is reshard-free. Measured
  (single GPU, from the prototype): first advance −24..31% (256³/64³),
  steady-state per-step bitwise-unchanged after the swap; a background
  compile failure re-raises on the next `step_chunk` call. The shipped
  version drops the prototype's `eager1` mode (measured strictly worse)
  and serves the length-C unroll-1 tier. Forced-4 CPU keeps the
  bitwise-equality invariant (no `single_device` mark needed). Record:
  [`../research/time_to_first_step.md`](../research/time_to_first_step.md)
  §3c.

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
  Both follow-ups closed: the 2026-07-17 single-GPU suite recheck
  confirmed the win in the suite itself (512³ weno5 186.8→130.5
  ms/step, oc edge 1.10→1.57×; `results/recheck-2026-07-17/` in the
  bench repo — the full-table refresh and chunk-metric fix remain a
  separate [`open.md`](open.md) item), and the forced-4 knife-edge
  divergence-gate flip for `weno5` was covered by the same
  residual-vs-tendency bound (`cbfc032a`) that fixed the pre-existing
  `upwind5` case — verified 2026-07-18, 12/12 green under forced-4
  CPU on dev. Further negative results for upwind5 (one-path
  spellings, XLA flags, Pallas) are in
  [`../research/upwind5_revisit.md`](../research/upwind5_revisit.md);
  do not revisit any of them without reading the records. Records:
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
- **Diffusion/friction closures: walls (free-slip / no-slip), mapped
  columns, implicit no-slip rows** (2026-07-17, branch
  `feat/diffusion-walls-terrain`) — stages 0–4 of
  [`../research/diffusion_walls_terrain_scoping.md`](../research/diffusion_walls_terrain_scoping.md).
  The explicit `_DiffusionClosure` family runs on walled grids
  (nodal family): no-flux tracers and free-slip tangential close
  structurally (interior flux retagged onto its `Inner[Dirichlet]`
  sibling — the advection flux-space precedent; wall-normal targets
  close on their own tag); `slip="no"` on the friction closures adds
  the `-2 ν q/Δn²` wall rows (grid-measure widths, positive
  true-shape weights — singularity-free reverse mode), realizing the
  same odd-mirror `-3` corner as the implicit band; biharmonic
  applies the same wall treatment on both passes (G&H 2000).
  `VerticalMixing(bottom=/top=)` gains no-slip Dirichlet rows
  (`second_difference_matrix(bc=)`, merge key carries the BC rows so
  unlike-BC legs never kappa-sum) and taught gates against
  stretched/terrain solve columns (the uniform-dz band was silently
  wrong there). Mapped/terrain: along-coordinate semantics validated
  and documented (order 1.97 stretched convergence, measure-weighted
  conservation 2.2e-15, terrain legs H-independent bitwise, `nu_v`
  along the named column coordinate — the ROMS `MIX_S_UV`
  convention); source change docstring-only. The spatial-layer
  prerequisite — the `divide_by_codomain_measure` VJP seal (any
  `jax.grad` through a bounded stretched-mesh diff was NaN on
  boundary-adjacent cells) — was found here independently and landed
  the same day via `fix/stretched-diff-vjp` (reconciled; the
  closure-level stretched autodiff regressions ship on this branch).
  Gates: discrete cosine/sine mode decay at
  exact rates, conservation to machine zero, periodic path bitwise
  unchanged, autodiff regressions (free/no-slip × walled/stretched/
  terrain) vs central FD, 1697-test sweep green, ruff clean. FV
  (`CellAvg`) walled targets were a taught rejection at this landing
  (lifted next day, entry below) — remaining residuals in
  [`open.md`](open.md).
- **FV walled diffusion/friction closures** (2026-07-18, branch
  `feat/fv-walled-diffusion`) — the finite-volume residual of the
  entry above: walled BC-free `CellAvg` targets (the nonhydro2
  default family) now take the **same** flux-retag wall closure as
  the nodal family, because under the FV C-grid `diff` profile the
  interior flux stagger-lands on the same nodal `Inner` face and the
  F4 `Inner[Dirichlet] → CellAvg` row closes it with the structural
  zero wall flux — free-slip/no-flux verbatim, no-slip reusing the
  wall-adjacent `-2 ν u₁/Δn²` correction unchanged (`CellAvg` ghosts
  bit-identical to `Center`; `grid.measure` gives true per-cell
  widths). New code is classification + a bind-time face-exposing
  probe only: a raw grid (collocated `FVDerivative` profile) is
  taught-rejected instead of running the wrong stencil; a tagged FV
  cell wall cannot even be declared (space-layer C8 gate). Deliberate
  deviation from the open.md lean: the `Outer`-flux-slot spelling
  (§3.3-b) was **not** built (needs a new `CellAvg → Outer` row + a
  wall-slot constructor; stays the open-boundaries Tier-2
  unification) — record §9 addendum. Mapped/stretched FV columns
  validated on the along-σ semantics rather than gated
  (measure-weighted conservation machine-zero, terrain bitwise
  H-independent, autodiff FD-exact walled + stretched); FV-vs-nodal
  walled parity 1e-12; periodic FV chain got first numeric coverage
  (bitwise vs nodal). Tests:
  `tests/model/closures/test_diffusion_fv.py` (24 tests); gates:
  closures suite 150 green, nonhydro2 596 green, ruff clean.

- **FV-vs-nodal step-time gap — closed** (2026-07-18, branch
  `perf/fv-walled-storage-frame`; record
  [`../research/fv_nodal_step_gap.md`](../research/fv_nodal_step_gap.md)).
  Re-measure first corrected the folklore: the gap was **4-GPU-only**
  (1-GPU FV/nodal parity everywhere; the T7 "+1…+10% gpu1 walled" was
  FV-vs-old-FD-baseline) — walled n=256 +3.9%, mapped n=128 +8..9%,
  n=256 +16..18%, all in the CG-iteration-independent part.
  HLO-attributed and causally confirmed (monkeypatch A/B): the two FV
  walled special branches' true-frame excursion
  (`f.data` → `jnp.pad` → `store`) made the SPMD partitioner
  materialize a transposed `{2,1,0}` layout and reroute the
  periodic-axis halo collective-permutes through it (58 vs 26
  transposed collectives; claim-loss/refill hypothesis refuted —
  collective counts equal). Fix: storage-frame windowed spelling
  (wall-zero ghost writes + the ordinary `apply_fv_staggered` window,
  sealed measure divide), gated `wall_slots_addressable`, true-frame
  kept as the distributed-axis fallback. **4-GPU FV/nodal after:
  walled 1.003/0.995, mapped 1.002–1.008** (from 1.04–1.18); flat +
  1-GPU unchanged; physics bitwise both device counts; step-guard
  green; ratchet baseline re-recorded (counts up, wall-clock down).
  Residuals in the record §4: 1-GPU mapped-256 +1.6% (sealed-divide
  cost; `custom_jvp` is the lever if ever needed), distributed walled
  axis keeps the slow spelling, stale gpu1 mapped baseline replaced
  in the follow-up re-record.

- **Cold-compile HLO volume — closed as a measured negative**
  (2026-07-18) — the HLO-volume remainder of the 2026-07-16
  time-to-first-step entry above. Four-way campaign (census refresh,
  frame plumbing, advection batching, mapped/CG body): the motivating
  numbers were stale — weno5 chunk compile is 4.37 s, not 8.5–10 s
  (the selected-input landing already delivered −38%), and the mapped
  "16–18 s vs 2–3 s" was the first-advance-wall metric artifact
  (today: ~3.8 s vs 1.3 s GPU, size-independent) — and every
  remaining reduction buys a measured runtime regression: the seal
  DUS spelling is the runtime-optimal one (+2.8 ms/step
  alternatives), pads fold at jax lowering (69 jaxpr → 9 HLO),
  call-dedup of the 12 flux kernels is erased by XLA's CallInliner
  (−37% unopt, ±0 compile), true batching needs a stacked state
  (temp 0→571 MB, 2.6–8× kernel time, shapes diverge on
  walled/mapped), and the multigrid V-cycle is structurally linear
  in levels with the cuSPARSE auto-default already smallest+fastest.
  Compile tracks *optimized* HLO (unopt is unroll-invariant) —
  corrected in the record. The one honest cold-start lever left is
  the async two-tier chunk compile, tracked in
  [`open.md`](open.md). Record (incl. do-not-revisit list):
  [`../research/hlo_volume.md`](../research/hlo_volume.md).

- **Performance guard — deterministic CI gates + hardened compare +
  manual A100 guard** (2026-07-18, plan + owner rulings:
  [`../plans/active/perf_guard_plan.md`](../plans/active/perf_guard_plan.md))
  — the buildable surface of the "wire the benchmark harness as a CI
  gate" item, after the research verdict that a wall-clock gate in
  GitHub CI is malpractice (shared-runner noise ~2.7% CoV; no
  surveyed project PR-gates on timing) and the rulings: PR CI gates
  *structure*, the A100 node gates *time*, manual-trigger only.
  Shipped: **G1** six fast-path guards — multigrid line-smoother
  per-level isinstance + steep/sloped convergence budgets (a locally
  swapped point smoother stalls at rel ~1 / 1.7e-4 vs 3.2e-9 /
  1e-15, so the budgets bite), tridiagonal auto→pcr/cusparse
  end-to-end wiring + pcr HLO while-absence (scan positive control),
  WENO selected-input jaxpr div-halving (3 vs 6, 4 vs 8 — both
  sides computed in-test, never hardcoded), carry-donation
  `is_deleted()` guard (the compile-pin was already covered),
  periodic FV=nodal per-op HLO equality (measured **byte-identical**
  compiled HLO), walled 1.0153 / mapped 1.0116 FV/nodal op-count
  **ratchet** (+10% band, regen via `FRIDOM_REGEN_FV_RATCHET=1`,
  committed CPU/1-device baseline, failure message cites the
  compiler-artifact caveat), and the uniform-mesh scalar-dx fold
  (no field-shaped divisor in the jaxpr; mapped positive control).
  **G2** `compare` hardened: environment guard on
  backend/device_count/device_kind/jax_version (missing field =
  mismatch, exit 2, `--allow-env-mismatch` downgrade),
  min-estimator statistic (noise is one-sided), per-case
  `max(threshold, 3·CoV_base)` tolerance with the winning rule
  shown per case. **G3** `benchmarks/ci/step_guard.sbatch` + README
  (mirrors the T7 baseline-record invocation exactly; results are
  retained, never deleted) + the AGENTS.md **perf merge gate**
  line. Combined gates on merged dev: 382 passed / 14 skipped
  (gpu-only + multi-device + ratchet self-skips), ruff clean. Open
  remainder tracked in [`open.md`](open.md): the first green,
  manually submitted guard run on the A100 node and the gpu-marked
  cusparse legs.

- **Comparison-suite chunk-metric fix — honest `compile_s`**
  (2026-07-18, out-of-tree bench repo only; no fridom src change) —
  the metric-fix remainder of the 2026-07-16 time-to-first-step
  entry above. The fridom harnesses (`fridom/bench_compare.py`,
  `fridom_hydro/bench_hydro.py`,
  `fridom_multi/bench_{multi,maxfit}.py`) now report `compile_s` —
  the AOT chunk-compile seconds fridom records in
  `_CHUNK_COMPILE_LOG`, read as a before/after delta around the
  first advance (the log is process-global; a sweep touches it many
  times) — beside the unchanged, now explicitly-flagged-as-conflated
  `first_advance_s`. `analyze.py`/`analyze_hydro.py` render three
  distinct columns (fridom compile / fridom 1st-adv / oc 1st-step —
  the old table put oc's first `time_step!` under a column titled
  "oc compile s"), keep pre-fix JSONs rendering (`—†` + footnote),
  and both reports regenerate cleanly against the existing results.
  Smoke-verified: 64³ linear compile 1.70 s vs first-advance 2.46 s
  (single A100) and 1.19 s vs 19.0 s (CPU); hydro 256²×32 CPU 0.82 s
  vs 84.7 s (the artifact vividly). The multi-GPU scripts were edited
  by careful reading only (no 4-GPU allocation) and get exercised at
  the next full sweep — tracked with the suite re-run in
  [`open.md`](open.md). Record:
  [`../research/time_to_first_step.md`](../research/time_to_first_step.md)
  §1 + the bench repo README (Metrics).

- **Performance guard — CLOSED: first checkpoint green** (2026-07-18,
  completing the entry above; full first-day log in
  [`../plans/active/perf_guard_plan.md`](../plans/active/perf_guard_plan.md)
  §7) — the remaining criterion (one green, manually submitted
  `step_guard.sbatch` run) is met by run 26346802→26347156's
  measurements judged green: gpu1 exit 0 (39 ok, 1 faster), gpu4
  exit 0 (40 ok). The day's four runs told the whole story: run 1
  RED = **true positive** (tiny-nodal shift from the FV
  storage-frame spelling; attributed + re-baselined by the owning
  session), runs 3–4 RED = **false alarms from a per-process slow
  mode** (~0.8 ms/chunk, every sample in the affected subprocess
  uniformly high, a different random tiny case each run; proven
  benign by a two-fresh-process probe reading baseline level and by
  zero `src/` delta between runs 3 and 4). Fix (owner-ratified):
  `compare` verdicts now also require an **absolute 1.2 ms/chunk
  floor** (symmetric for faster/slower; suppressions annotated
  `(floor)` in reports) — the floor is the harness's declared
  resolution limit, large cases unaffected. The re-adjudicated run
  even exposed one baseline entry recorded from a slow-mode process
  (advective_nodal[32] −9.6% `(floor)`), motivating the optional
  future upgrade noted in the plan: min-across-2–3-processes for
  sub-16 ms cases. Guard cadence stands per ruling §5.5:
  owner-batched checkpoints, never per-merge, agents never submit.

- **Hydrostatic walled-horizontal gap — closed (flat + immersed)**
  (2026-07-18, found that morning as a multigrid-generalization
  phase-B residual; root cause + three fixes same day; merges
  `d26d3d9d` fix 1, `48e841ec` fix 3, `48a15c00` fix 2). The
  recorded "staggering never wires wall BCs" diagnosis was wrong:
  the Velocity-role bind derivation tags the wall-normal velocity
  correctly per axis. The real seam was `ScalarField.to` (and its
  mirror `HaloTracer.to`): no arm for a *tag-only* factor
  difference (same node set, BC-siblings), so every BC-free
  gradient output mis-classified as a node-set conversion and
  resolved a deliberately-absent bare-face row. Fix 1 adds the
  two-line sibling arm to both `.to`s (adopt via `retag`; fires
  only where the old code guaranteed an error) — the whole gap for
  `ExplicitFreeSurface`: walls x/y/x+y, advection on/off, immersed
  included, mirror-symmetry vs a doubled periodic domain 5.6e-17,
  volume drift 1.7e-18, autodiff FD-matched 3.7e-12
  (`tests/hydrostatic/test_free_surface_walled.py`). Fix 2 ports
  the nh2 F4 wall closure into `ImplicitFreeSurface._flat_spectral`
  (solve on the `_neumann_sibling` space, `_dirichlet_mid` mid
  legs, retag seam around `SpectralSolve`; immersed CG operator
  retags its flux legs): periodic path bitwise-identical (sha256
  state hash), mirror gate 3.2e-15, rigid-lid gauge 1.3e-17,
  immersed+walled divergence 3.2e-9, autodiff 5.1e-12. Fix 3
  wall-tags the split-explicit barotropic transports at
  declaration (`wall_bc={staggered: DIRICHLET}` on the transport
  `SpacePattern` — the same channel the Velocity role uses), after
  which the subcycle needed no further seam: periodic path
  sha256-identical, mirror gate exact 0.0, volume drift 3.5e-18,
  autodiff 3.1e-12. Every free-surface variant now assembles and
  runs on walled horizontal grids on flat and immersed geometry;
  the one remaining layer (terrain chart + walled horizontal — a
  genuine missing interpolate in the mapped slope gradient, not a
  tag issue) is tracked in [`open.md`](open.md).
- **Multi-device eigenbasis setup + synthesis faults — FIXED**
  (2026-07-18). The two "pre-existing faults surfaced by the
  projection validation": (a) the setup `GridFrozenError` was a
  negotiate/verify **cap asymmetry** (freeze sealed the sharding-capped
  halo, verify compared the raw demand — every `model.variant`
  faulted on cap-engaged grids, n ∈ {8,11,14,17}@4dev for the nh
  channel); fixed by capping the verify side identically (merge
  `8a787452`, restores the `variant` ⊆ lemma). (b) `mode()` /
  `channel_random_state` on a sharded periodic axis now route through
  the fused backward-only synthesis (`ContractPlan.synthesize`, merge
  `e316987d`; bit-identical parity, finite VJP; 2-D channel remains
  the ratification item in `open.md`). Record:
  [`../research/halo_sharding_invariants.md`](../research/halo_sharding_invariants.md).

- **Naive GSPMD transform path illegal (Tier 1) — SHIPPED**
  (2026-07-18, merge `83fbc56c` + CI `867537e1` + fixture pins
  `a6bc4b39`). `Transform.forward/backward` raise a taught error when
  the operand's layout shards a transform axis (the path silently
  all-gathered on CPU and crashed XLA:GPU's distributed-FFT lowering,
  jax#39291); safe lowerings (`SlabPlan`, `ContractPlan`) bypass the
  seam by construction. The `numeric_eigenpairs` probe gathers first.
  Suite converted (device-pinned fixtures; taught-error counterparts).
  Study + campaign record:
  [`../research/gspmd_naive_transform_illegality.md`](../research/gspmd_naive_transform_illegality.md);
  phases 2+ in
  [`../plans/active/gspmd_transform_illegality_plan.md`](../plans/active/gspmd_transform_illegality_plan.md).

- **Interval-accounting sharding regressions — FIXED (one loud, one
  silent)** (2026-07-18). Surfaced by the campaign's forced-4 residual
  sweep; both from the two-sided-accounting landing `40a24df8`.
  **Loud:** the sharding-cap floor was blind to trace-only wide
  stencils, so small axes sharded with a halo below one application's
  reach (97 advection forced-4 failures); fixed by flooring the cap
  with the traced per-application reach in negotiate + frozen verify,
  plus an assembly pre-validation negotiate so construction-time
  sharding collapses before `dry_run` (in dev via `94786a7c`).
  **Silent wrong physics:** bounded staggered kernels published a
  wall-cancelled `exterior_reach` of (0,0), eliding the inter-shard
  halo sync on sharded walled axes — diffusion/friction tendencies
  wrong by O(1)–O(10) at real multi-GPU scale, growing with N; fixed
  by publishing the per-shard `footprint_reach` (merge `b57e3e78`;
  periodic bit-identical, the n+8→n+6 storage win survives; also
  cured the sadourny doubly-walled sharded failures and closed a
  lone-bounded-op width-0 negotiation hole). Record:
  [`../research/halo_sharding_invariants.md`](../research/halo_sharding_invariants.md).

- **Time-dependent fields — general non-affine mechanism SHIPPED**
  (2026-07-18, three waves). Closes the general half of the
  "Time-dependent parameters and time-dependent fields" roadmap ask: a
  *profile* that itself evolves — `f(y, t)`, `csqr(y, t)` — with
  **non-affine** time dependence, beyond the affine-blend subset that
  shipped 2026-07-17 (`adiabatic_ramping.md` R1/R2). Also closed: the
  `dsqr` AR-D7 cross-module report (owned by `DynamicalCore`, consumed
  by `ConstantStratification.buoyancy_force` — a live silent-wrongness
  hole under `ETDRK4`), and the mandatory `ETDRK4` answer. Waves:
  - **Wave 1 — structural frozen-`L` guard** (merge `bb2fb96f`,
    `refactor/linear-term-guard`; 11 files, +636/−68). `@fr.term`
    gains `linear_params` / `linear_fields`; the frozen-`L` sweep moves
    into the **base-class default** of
    `Module.time_dependent_linear_parameters` (the hand-written Coriolis
    / stratification overrides deleted), keeping the assembly guard
    call-site, offender attribution and error surface byte-compatible.
    Resolution is module-local; cross-module deps like `dsqr` are
    reported by the **owning** module (`DynamicalCore`).
  - **Wave 2 — mechanism + consumers** (merge `ceb9db75`,
    `feat/time-dependent-fields`; 14 files, +1424/−78). The
    `(coords, t)` recompute core and `ProfileFunction` extracted into
    `model/scheduled_field.py` (`MovingGeometry` re-expressed on it
    bitwise); `FieldDeclaration.time_dependent` marker + the composer
    `SELF_UPDATE` coverage lint (+ `Stage(writes=)`); the law-valued
    `BetaPlaneCoriolis(f=ProfileFunction(...))` and shallow-water
    `DynamicalCore` `csqr` paths; and the `linear_fields` wiring so
    `ETDRK4` refuses a scheduled `f`.
  - **Wave 3 — honesty sweep + records** (`chore/td-fields-honesty`,
    this merge). `PolarizedWaveMaker` refuses a `ProfileFunction`-valued
    parameter or a `time_dependent`-marked dependency field at bind
    (TDF-D6); the eigen/analysis surfaces (`eigenbasis` /
    `ChannelEigenmodes` / `Eigenmodes.from_model`, `EnergyMetric`)
    document their fixed-`at_time` snapshot semantics — the
    `EnergyMetric` check confirmed the weights are **baked once at
    `from_model`**, never re-read at evaluation time.

  Rulings: TDF-D5 (`ETDRK4` refuses a time-dependent `L`, never
  auto-splits), TDF-D6 (setup-baked consumers get the same honest
  refusal; the analysis tools stay time-frozen, no re-diagonalization
  contract), TDF-D7 (`f` / `csqr` in scope; nonhydro `n2(z, t)` a named
  follow-up), TDF-D9 (`FieldBlend` NOT unified onto the rewrite path —
  owner decision). Open remainders in `open.md`: the `n2(z, t)` path,
  the `FieldBlend`-unification question, and the declined
  re-diagonalization contract. Plan:
  [`../plans/done/time_dependent_fields.md`](../plans/done/time_dependent_fields.md).

- **Split-explicit barotropic-IC gap — ruled + fixed** (2026-07-19,
  merge `f3d96306`, branch `fix/split-explicit-barotropic-ic`). The
  2026-07-18 srun-validation finding — a z-independent `set_fields`
  velocity vanished from the whole carry in one step (max|u| 0.98 →
  2.6e-4) — was ruled an **IC gap** by the owner (not rest-start
  semantics). Mechanism: `hy.SplitExplicitFreeSurface` declares the
  barotropic transports `U, V` PROGNOSTIC (zero-initialized at build)
  and nothing projected the IC's depth mean into them, so the first
  CONSTRAINT stage replaced the depth mean of `u, v` with
  `U/H = 0`. Fix: a minimal host-side IC hook —
  `Module.derive_initial_fields(state, provided)` (base no-op),
  called by `Model.set_fields` after the user's fields are applied
  and re-homed identically; the split-explicit override seeds
  `U = ubar/(1/H)` (immersed: `ubar_wet * H_col`, land columns 0)
  exactly as the subcycle commit computes, iff the velocity was set
  without its transport (an explicit `U`/`V` is respected bitwise; a
  ps-only call derives nothing). Entirely outside the jit step path
  (differentiability policy exempt). Tests: seed-equality to 1e-13 +
  3-step survival, explicit-override, ps-only, immersed
  transport-depth consistency; forced-4 green. Scoping verified:
  the explicit/implicit variants carry no slaved barotropic
  prognostic — this was the only module with the gap. Caveat for the
  comparison ladder: the pre-fix se rungs ran near-zero-velocity
  flows while Oceananigans got the full IC (wall-time ratios are
  data-independent, physics trajectories were not comparable).

- **Z-sharded hydrostatic seam divergence — FIXED** (2026-07-19, merge
  `ef1a4d08`, branch `fix/hydro-z-shard-seam`). Found while
  root-causing the owner's "why can't weno5 run z-sharded" question:
  the crash itself was the shardability-cap negotiation bug already
  fixed by `a802fbea` (registry-sourced cap floor blind to trace-only
  weno reconstructions), but z-sharded runs that DID run were silently
  wrong — buoyancy diverged rel ~5e-2 over 20 steps exactly at the
  shard-seam z-levels (deterministic, halo-width-independent;
  x-sharded bit-exact, which is why the 2026-07-18 srun validation at
  64x64x16 never saw it). Mechanism (third hypothesis; the
  investigation's `w.to(b)` localization was disproven by
  instrumenting `_ensure_valid`): `Restriction` (`Outer -> Inner`,
  the vertical advective flux's relocation of the diagnosed `w` onto
  interior flux faces) reads `Outer[m+1]` — one slot above each
  output — but declared no `requirements`, inheriting halo 0, so the
  operand's seam ghost was never synced and the last interior face of
  every shard read the reshard's zero fill. u/v only looked bit-exact
  because their horizontal interpolation's sync filled z as a
  side-effect. Fix: the `(0, 1)` footprint declaration on the
  operator (same `reach_or` pattern as every sibling; Restriction was
  the unique offender — sweep confirmed all other `SeparableOperator`s
  declare theirs, and nonhydro2 never applies Restriction).
  Single-device and x-sharded results bitwise-unchanged (HLO gains a
  local ghost-fill; outputs identical); centered z-shard parity now
  1e-11-tight all fields, weno5 b 2.4e-4 → ~1e-7. Regression:
  `tests/hydrostatic/test_z_shard_parity.py` (forced-4, centered +
  weno5, demonstrated red pre-fix) + footprint/seam/autodiff tests in
  `test_restrict.py`. Residual in `open.md`: the weno5 momentum
  ~1e-5 z-seam (`WenoReconstruction` vertical footprint vs the
  negotiated z-halo of 2 — halo-cap machinery, owner-governed).
