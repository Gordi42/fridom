---
status: normative
date: 2026-07-13
---

# FRIDOM design records

Internal design records for FRIDOM, organized by lifecycle. **Specs**
(`specs/`) are the living, normative truth: when code and spec
disagree, one of them is wrong and must be fixed. **Decisions**
(`decisions/`) are immutable records of settled design rulings — they
are never edited, only superseded. **Plans** (`plans/active/`) are work
in flight; when a plan ships it moves to `plans/done/` (its record is
kept, not deleted). **Research** (`research/`) holds frozen inputs —
option analyses, precedent surveys, audits — that fed the decisions;
where research disagrees with a spec, the spec wins. **Archive**
(`archive/`) holds proposals that were put on hold or superseded.

[`roadmap/open.md`](roadmap/open.md) is the single open-work tracker;
[`roadmap/done.md`](roadmap/done.md) records the shipped phases. Both
link into the design records behind each task.

**Naming map.** The specs predate two renames and keep the historical
spellings:

1. The 2026-07-11 package split
   ([`plans/done/spatial_model_split_plan.md`](plans/done/spatial_model_split_plan.md)):
   read `framework2.grid` as **`fridom.spatial`**, and `framework2.model`
   / `framework2.{transforms,io,ops,modules}` as **`fridom.model`**.
2. The flat `fr.*` namespace the specs assume (`fr.Grid`, `fr.meshes`,
   `fr.ScalarField`, `fr.Model`) **did not ship** — the root package
   re-exports subpackages only. Read a bare `fr.X` as `fr.spatial.X` or
   `fr.model.X`. Whether to re-add the flat aliases is an open
   ergonomics question, not a design one.

## Conventions

1. Every note gets YAML front-matter with at least `status:` and
   `date:` (plus `supersedes:` / `superseded_by:` where applicable).
2. A plan is never edited into "done" in place — its status flips to
   `done` and the file **moves** to `plans/done/`.
3. `plans/active/` holds only work that is still ahead. Finished work
   does not live there: it moves to `plans/done/`, and what remains in
   an active plan is trimmed to the open items.
4. Decision records are immutable. When landed code overtakes one, it
   gets a `superseded_by:` pointer and a header note — the ruling itself
   is never rewritten.
5. The roadmap follows the same hygiene: `roadmap/open.md` holds only
   open work. When an item ships, its record moves to
   `roadmap/done.md` — in the same change that reports it shipped —
   and the open entry is trimmed to what actually remains. Status
   narrative ("shipped", "landed", "resolved") never accumulates in
   `open.md`.

## `specs/` — living, normative design specs

### `specs/grid/` — the grid abstraction (implemented as `fridom.spatial`)

| File | Status | Description |
|------|--------|-------------|
| [`00_overview.md`](specs/grid/00_overview.md) | normative | Grid abstraction redesign — document map, motivation, decision index. |
| [`01_concepts.md`](specs/grid/01_concepts.md) | normative | Core concepts: meshes, function spaces, operators, fields, the Grid assembly. |
| [`02_rules.md`](specs/grid/02_rules.md) | normative | Rules: space algebra, coefficient spaces, boundaries, FV semantics, dispatch. |
| [`03_api_sketches.md`](specs/grid/03_api_sketches.md) | normative | API sketches for the grid layer. |
| [`04_decomposition.md`](specs/grid/04_decomposition.md) | normative | Domain decomposition: negotiation, halos, shard maps. |
| [`05_validation.md`](specs/grid/05_validation.md) | normative | Paper validation — the numerical checks gating the grid layer. |
| [`06_open_threads.md`](specs/grid/06_open_threads.md) | normative | Grid threads: resolved map + the residuals that remain. |
| [`07_iteration1_api.md`](specs/grid/07_iteration1_api.md) | normative | Iteration-1 public API surface. |

### `specs/grid/classes/` — grid-layer class designs

| File | Status | Description |
|------|--------|-------------|
| [`README.md`](specs/grid/classes/README.md) | normative | Class-design phase overview, document map, staging. |
| [`meshes.md`](specs/grid/classes/meshes.md) | normative | Meshes and function spaces. |
| [`spaces.md`](specs/grid/classes/spaces.md) | normative | Function spaces (nodal, average, coefficient, Galerkin, constant). |
| [`product_spaces.md`](specs/grid/classes/product_spaces.md) | normative | TensorProductSpace and the Field cluster. |
| [`fields.md`](specs/grid/classes/fields.md) | normative | Fields: ScalarField, vector/tensor fields, metadata. |
| [`grid.md`](specs/grid/classes/grid.md) | normative | Grid assembly, decomposition entry points, the mapping/metric seams. |
| [`decomposition.md`](specs/grid/classes/decomposition.md) | normative | Domain decomposition: traits, halo accounting, the consumption-side sync contract. |
| [`operators_base.md`](specs/grid/classes/operators_base.md) | normative | Operator base hierarchy, binding, Symbol. |
| [`operators_composed.md`](specs/grid/classes/operators_composed.md) | normative | Composed operators and the dispatch registry. |
| [`operators_products.md`](specs/grid/classes/operators_products.md) | normative | Pointwise, product and reduction operators. |
| [`operators_stencils.md`](specs/grid/classes/operators_stencils.md) | normative | Stencil, FV and spectral operators. |
| [`operators_transforms.md`](specs/grid/classes/operators_transforms.md) | normative | Transform operators (Fourier, trig, Chebyshev, padding). |
| [`operator_algebra_merge.md`](specs/grid/classes/operator_algebra_merge.md) | normative | Decisions merging the operator algebra into the class designs. |

### `specs/operator_algebra/` — the operator algebra

| File | Status | Description |
|------|--------|-------------|
| [`00_overview.md`](specs/operator_algebra/00_overview.md) | normative | Operator algebra design — overview and document map. |
| [`01_taxonomy_and_binding.md`](specs/operator_algebra/01_taxonomy_and_binding.md) | normative | Taxonomy and axis binding. |
| [`02_algebra.md`](specs/operator_algebra/02_algebra.md) | normative | Algebra rules: composition, sums, scaling, blocks. |
| [`03_api_sketches.md`](specs/operator_algebra/03_api_sketches.md) | normative | API sketches. |
| [`04_open_threads.md`](specs/operator_algebra/04_open_threads.md) | normative | Resolved map + the two residuals (index-aware tensor blocks; `VectorField.map` vs `Block`). |

### `specs/docs/` — the documentation rebuild

| File | Status | Description |
|------|--------|-------------|
| [`structure.md`](specs/docs/structure.md) | normative | The docs page tree: per-page scope, authoring format, v1 cut. |
| [`style_guide.md`](specs/docs/style_guide.md) | normative | Docs style guide: voice, banned patterns, page anatomy, figures, citations, enforcement. |

### `specs/model/` — the model layer (implemented as `fridom.model`)

| File | Status | Description |
|------|--------|-------------|
| [`00_overview.md`](specs/model/00_overview.md) | normative | Model layer redesign — modules, model, time stepping; document map. |
| [`01_concepts.md`](specs/model/01_concepts.md) | normative | Core concepts and load-bearing decisions (D1–D5). |
| [`02_rules.md`](specs/model/02_rules.md) | normative | Rules: purity, precision, bitwise reproducibility, clock. |
| [`03_time_stepping.md`](specs/model/03_time_stepping.md) | normative | Staged / split time stepping. |
| [`04_run_loop_io.md`](specs/model/04_run_loop_io.md) | normative | Composition, the run loop, and IO. |
| [`05_api_sketches.md`](specs/model/05_api_sketches.md) | normative | API sketches. |
| [`06_validation.md`](specs/model/06_validation.md) | normative | Paper validation for the model layer. |
| [`07_open_threads.md`](specs/model/07_open_threads.md) | normative | Model threads: resolved map + the residuals that remain. |
| [`08_state_transforms.md`](specs/model/08_state_transforms.md) | normative | The state-transform algebra. |
| [`09_coupling_designfor.md`](specs/model/09_coupling_designfor.md) | normative | Coupled models — design-for constraints (CS-1..18); the pre-design for roadmap 3.2. |

### `specs/model/classes/` — model-layer class designs

| File | Status | Description |
|------|--------|-------------|
| [`README.md`](specs/model/classes/README.md) | normative | Class-specification phase overview and document map. |
| [`declarations.md`](specs/model/classes/declarations.md) | normative | Declarations: fields, parameters, terms, roles. |
| [`module.md`](specs/model/classes/module.md) | normative | The Module base and closures. |
| [`model.md`](specs/model/classes/model.md) | normative | Model: assembly pipeline, run loop, results. |
| [`time_steppers.md`](specs/model/classes/time_steppers.md) | normative | Time steppers: Clock, schedule, RK/AB/IMEX families. |
| [`transforms.md`](specs/model/classes/transforms.md) | normative | State transforms: base surface, algebra, projection family. |
| [`io_ops.md`](specs/model/classes/io_ops.md) | normative | IO & ops: Writer, triggers, snapshots, Session. |

### `specs/nnmd/` — nonlinear normal-mode decomposition

| File | Status | Description |
|------|--------|-------------|
| [`nnmd_design_note.md`](specs/nnmd/nnmd_design_note.md) | normative | The slaving recursion; implemented as `fr.transforms.BalanceExpansion`. |

## `decisions/` — immutable decision records

| File | Status | Description |
|------|--------|-------------|
| [`symbol_stack_design.md`](decisions/symbol_stack_design.md) | accepted, partially superseded | The symbol stack — dynamic symbols, mixed transforms, banded axes. `SpectralSolve`, layout-faithful `eigenvalues` and the banded primitive landed; the `BlockSymbol` projector layer did not. |
| [`blocksymbol_l_assembly.md`](decisions/blocksymbol_l_assembly.md) | superseded | Assembling `L` as a `BlockSymbol` from the operator algebra — built (`026f4c62`), then removed (`d3309640`) for want of a consumer; `L` is now derived by numeric probe or hand-written eigenmodes. |

## `plans/active/` — work in flight

| File | Status | Description |
|------|--------|-------------|
| [`cutover_parity_plan.md`](plans/active/cutover_parity_plan.md) | active | Cutover, parity half: physics parity is closed; awaits owner sign-off of the intentional-deltas table. |
| [`cutover_checklist.md`](plans/active/cutover_checklist.md) | active | Cutover, mechanical half: the executable swap list — consumer map (45 source + 16 test imports, 136 modules, 107 test files), order of operations, gates. |
| [`docs_examples_plan.md`](plans/active/docs_examples_plan.md) | active | Docs & examples rebuild: the CI skeleton and the pilot example landed; 12 example ports and the full prose tree remain. |
| [`boundary_plan.md`](plans/active/boundary_plan.md) | active | Boundary closures: R1, the one-sided rows and `BC.ROBIN` structure landed 2026-07-11; open is stage 2e (the Robin dynamic `(α, g)` ghost-fill path) and the optional 2f halo-claim refinement. |
| [`projection_eigenmode_roadmap.md`](plans/active/projection_eigenmode_roadmap.md) | active | Projection / eigenmode build order: phases A–H landed; tracks only the unscheduled phase-I `Banded` / mixed-representation tier. |
| [`phase2_grid_followups.md`](plans/active/phase2_grid_followups.md) | active | Grid follow-ups from the Phase-2 reconciliation: every model-layer blocker landed; the `cartesian.Grid` constructor (R15) and the API shims (`.data` `ImmutableStateError`, transform/`NodeSet` re-exports, `Grid.dispatch` typing) landed 2026-07-15, leaving one deferred item (coefficient-space product/power rows — a semantics decision). |
| [`fv_nonhydro_scoping.md`](plans/active/fv_nonhydro_scoping.md) | active | Finite-volume nonhydro (roadmap 3.5): decisions FV-D1..D4 (D2 = option A), the staged F0–F6 plan. F0–F3 shipped 2026-07-16 (the periodic model is FV by default); open are F4 walls, F5 mapped, F6 hygiene, and the 4-GPU validation. |
| [`high_order_mapped_plan.md`](plans/active/high_order_mapped_plan.md) | draft | High-order stencils on mapped grids: the four standing mapped refusals, the metric-identity obstacle, the option table. Walled prerequisite paid; Jacobian spike answered 2026-07-16 (same-row discrete divisor, [`research/mapped_jacobian_spike.md`](research/mapped_jacobian_spike.md)); the full lift stays deferred on payoff. |
| [`adiabatic_ramping.md`](plans/active/adiabatic_ramping.md) | idea | Generalized adiabatic ramping (roadmap 3.8) — an `AdiabaticRamping` base transform with `OptimalBalance` as a subclass. Planned, not scheduled; its dependency (2.8) has shipped. |
| [`perf_geometry_merge_plan.md`](plans/active/perf_geometry_merge_plan.md) | active | Reconciling the performance line with the geometry line: the merge, the geometry stages and the mapped-PCG interior pass (§§4a–4b) landed; the A/B step harness exists with committed baselines — open is wiring it as a CI gate plus the unasserted fast paths (roadmap). |
| [`distributed_transform_reconciliation.md`](plans/active/distributed_transform_reconciliation.md) | active | Reconcile the distributed spectral solve with the §5.1 layout-in-space design. Stages 1–4 landed (the transform planner); closes the push gate. |
| [`distributed_transform_plan.md`](plans/active/distributed_transform_plan.md) | active | The distributed transform planner: layout-annotated stages, the fused `shard_map` lowering, and the A100 gate results. |
| [`multigrid_pathway_plan.md`](plans/active/multigrid_pathway_plan.md) | active | Multigrid pathway: phase A (the grid transfer layer — `Mesh.coarsened` / `Grid.coarsened` / `GridTransfer`, dual-use with coupling CS-15) landed 2026-07-17; open are the B0 two-level spike and the workload-gated V-cycle preconditioner (phase B). |

## `plans/done/` — shipped plans (kept as records)

| File | Status | Description |
|------|--------|-------------|
| [`multihost_writer_plan.md`](plans/done/multihost_writer_plan.md) | done | Multi-host writer — `fr.io.Writer` correct under a real `srun -n N` multi-process run: rank-0-owns-metadata + per-rank disjoint shard writes + conditional `process_allgather` of the coordinate labels. Shipped 2026-07-15 (`8b6642bf`); async falls back to blocking under multi-process (v1). |
| [`grid_ergonomics_plan.md`](plans/done/grid_ergonomics_plan.md) | done | Grid setup ergonomics — the fast assemble: `fr.spatial.spherical.Grid` (required `lat_extent`, optional `lon_extent` → closed zonal walls) over the `charts.lonlat_sphere` primitive, plus the sibling `cartesian.Grid`. Landed 2026-07-15. |
| [`chart_ergonomics_plan.md`](plans/done/chart_ergonomics_plan.md) | done | Chart / sphere setup ergonomics (E1–E5) — closed 2026-07-15 by `orthogonal=True` on `CoordinateMapping` (seeds the diagonal index moves) plus a taught-error safety net; auto-detect rejected (numeric metric → tolerance → silent-wrong-physics risk). |
| [`krylov_scan_plan.md`](plans/done/krylov_scan_plan.md) | done | Krylov scan (roadmap 3.6) — O(1) trace for the mapped pressure CG; HLO is now flat in the iteration count. Shipped 2026-07-13. |
| [`coriolis_energy_correction.md`](plans/done/coriolis_energy_correction.md) | done | Exactly-conserving shallow-water Coriolis — the optional correction term (keeps `L`) and the full nonlinear module. Shipped 2026-07-13 (`fc0b61c8`). |
| [`fallback_operator_plan.md`](plans/done/fallback_operator_plan.md) | done | Graded-order near-wall fallback operator — `Fallback` + the shared graded ladder, FV and nodal routes, sharded bounded axes. |
| [`nnmd_rewrite_plan.md`](plans/done/nnmd_rewrite_plan.md) | done | NNMD rewrite — shipped 2026-07-11 as `fr.transforms.BalanceExpansion` (P0–P5, benchmarks included). |
| [`operator_symbols_plan.md`](plans/done/operator_symbols_plan.md) | done | Operator symbols — the spectral-solve substrate; `Symbol` + `SpectralSolve` + banded shipped, the block layer withdrawn. |
| [`projection_eigenmode_plan.md`](plans/done/projection_eigenmode_plan.md) | done | Projections & eigenmodes — the energy-metric design (`EnergyMetric`, `p = M q`, `eigh(iML, M)`, the channel eigenbasis). |
| [`composition_refactor_plan.md`](plans/done/composition_refactor_plan.md) | done | Composition refactor — the realized-map category and one composition core. |
| [`coordinate_systems_plan.md`](plans/done/coordinate_systems_plan.md) | done | Coordinate systems (roadmap 3.4) — mapped, spherical and boundary-fitted grids (CS-D1..D4); C0–C4 landed 2026-07-12. |
| [`spatial_model_split_plan.md`](plans/done/spatial_model_split_plan.md) | done | Split framework2 into `fridom.spatial` + `fridom.model` — merged 2026-07-11. |
| [`bc_free_boundaries.md`](plans/done/bc_free_boundaries.md) | done | BC-free bounded spaces — resolved by the R1 landing: exterior reads raise; `one_sided` is the explicit closure. |
| [`phase1_implementation_plan.md`](plans/done/phase1_implementation_plan.md) | done | Phase-1 implementation (the grid layer, today's `fridom.spatial`). |
| [`phase2_implementation_plan.md`](plans/done/phase2_implementation_plan.md) | done | Phase-2 implementation (the model layer, today's `fridom.model`); its Phase-2.9 wave 11 (symbolic `L`) was built and reverted. |
| [`sync_redo_plan.md`](plans/done/sync_redo_plan.md) | done | Task 1.8 — consumption-side sync. |

## `research/` — frozen inputs to the decisions

Superseded where they disagree with the specs. The `c*`/`d*` reports
are the per-decision research behind `specs/model/` (see
[`research/README.md`](research/README.md) for their map).

| File | Status | Description |
|------|--------|-------------|
| [`README.md`](research/README.md) | frozen | Map of the design research reports. |
| [`c1_coupling_precedents.md`](research/c1_coupling_precedents.md) | frozen | C1 — Coupling precedent survey. |
| [`c2_coupled_walk.md`](research/c2_coupled_walk.md) | frozen | C2 — Concrete atmosphere–ocean walk (adversarial). |
| [`c3_coupling_architecture.md`](research/c3_coupling_architecture.md) | frozen | C3 — Coupling architecture pre-design. |
| [`d1_1_declaration_and_ics.md`](research/d1_1_declaration_and_ics.md) | frozen | D1.1 — FieldDeclaration surface and the home of initial conditions. |
| [`d1_2_space_binding.md`](research/d1_2_space_binding.md) | frozen | D1.2 — How module field declarations get their function spaces. |
| [`d1_3_core_ownership.md`](research/d1_3_core_ownership.md) | frozen | D1.3 — Who declares the core prognostic variables. |
| [`d1_4_roles.md`](research/d1_4_roles.md) | frozen | D1.4 — The role system. |
| [`d1_5_access_and_collisions.md`](research/d1_5_access_and_collisions.md) | frozen | D1.5 — Access surface, collisions, namespacing, functional updates. |
| [`d2_1_resolution_mechanism.md`](research/d2_1_resolution_mechanism.md) | frozen | D2.1 — Cross-module parameter resolution: mechanism design. |
| [`d2_2_representation.md`](research/d2_2_representation.md) | frozen | D2.2 — Parameter representation and time dependence. |
| [`d2_3_diagnostics.md`](research/d2_3_diagnostics.md) | frozen | D2.3 — Where physics diagnostics live, and their API. |
| [`d2_4_host_consumers.md`](research/d2_4_host_consumers.md) | frozen | D2.4 — Host-side parameter access and the eigenmode seam. |
| [`d3_1_term_surface.md`](research/d3_1_term_surface.md) | frozen | D3.1 — The TendencyTerm surface and treatment declarations. |
| [`d3_2_stepper_core.md`](research/d3_2_stepper_core.md) | frozen | D3.2 — The stepper as a pure scan body: stepper state, dt, Clock. |
| [`d3_3_stage_schedule.md`](research/d3_3_stage_schedule.md) | frozen | D3.3 — The stage schedule: kinds, ordering, traced signatures. |
| [`d3_4_imex_splitting.md`](research/d3_4_imex_splitting.md) | frozen | D3.4 — Split integrators: IMEX by term, Gauss-Seidel by variable. |
| [`d4_1_assembly_model.md`](research/d4_1_assembly_model.md) | frozen | D4.1 — Model composition and assembly pipeline. |
| [`d4_2_run_loop.md`](research/d4_2_run_loop.md) | frozen | D4.2 — The run loop. |
| [`d4_3_io_seams.md`](research/d4_3_io_seams.md) | frozen | D4.3 — The IO seams. |
| [`d4_4_lifecycle_coupling.md`](research/d4_4_lifecycle_coupling.md) | frozen | D4.4 — Post-assembly lifecycle; sweep/multi-device/coupling proofing. |
| [`d5_1_algebra.md`](research/d5_1_algebra.md) | frozen | D5.1 — StateTransform: base surface, algebra semantics, laws. |
| [`d5_2_variants.md`](research/d5_2_variants.md) | frozen | D5.2 — model.variant() and the term-predicate vocabulary. |
| [`d5_3_family_ports.md`](research/d5_3_family_ports.md) | frozen | D5.3 — The ported projection family, walked end-to-end. |
| [`phase1_findings.md`](research/phase1_findings.md) | frozen | Phase-1 validation findings (the roadmap-1.7 gate; input to Phase 2). |
| [`parity_audit.md`](research/parity_audit.md) | frozen | The cutover-parity audit (23/23 rows covered). |
| [`nnmd_literature.md`](research/nnmd_literature.md) | frozen | NNMD literature sweep. |
| [`boundary_design_explainer.md`](research/boundary_design_explainer.md) | frozen | Extrapolation-fill vs the boundaries R1 flip — the side-by-side that grounded the R1 decision (made 2026-07-11). |
| [`xla_spmd_fft_fault.md`](research/xla_spmd_fft_fault.md) | frozen | XLA SPMD-FFT fault: a jitted FFT on a sharded axis RET_CHECKs; reproducer, condition matrix, the fridom-side mitigation. |
| [`multigrid_pathway.md`](research/multigrid_pathway.md) | frozen | Multigrid pathway research (pinned `731089fc`): framework substrate, solver seam, decomposition constraints, the forced-4-device sharding probe, external prior art; input to the pathway plan. |
| [`diffusion_walls_terrain_scoping.md`](research/diffusion_walls_terrain_scoping.md) | frozen | Diffusion/friction closures at walls (free-slip/no-slip) and on mapped terrain: machinery inventory, external practice, per-case design, staged sizing; flags the live `VerticalMixing` stretched/terrain silent-wrongness. |

## `archive/` — held / superseded

| File | Status | Description |
|------|--------|-------------|
| [`linear_term_blocks_plan.md`](archive/linear_term_blocks_plan.md) | superseded | Linear-term block signatures (the H1 pivot) — built in full, then deleted for want of a consumer (`d06441d1`, `d3309640`). |
| [`linear_dsl_proposal.md`](archive/linear_dsl_proposal.md) | superseded | Readable linear-term DSL — superseded on its premise: the `LinearBlock` IR it lowers to was deleted the same day the proposal was held. |
