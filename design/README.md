---
status: normative
date: 2026-07-11
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

[`ROADMAP.md`](../ROADMAP.md) at the repo root remains the single
open-work tracker; it links into `design/` for the normative designs
and plans behind each task.

## Conventions

1. Every new note gets YAML front-matter with at least `status:` and
   `date:` (plus `supersedes:` / `superseded_by:` where applicable).
2. A plan is never edited into "done" in place — its status flips to
   `done` and the file **moves** to `plans/done/`.

## `specs/` — living, normative design specs

### `specs/grid/` — the grid abstraction (Phase 1, implemented)

| File | Status | Description |
|------|--------|-------------|
| [`00_overview.md`](specs/grid/00_overview.md) | normative | Grid abstraction redesign — document map, motivation, decision index. |
| [`01_concepts.md`](specs/grid/01_concepts.md) | normative | Core concepts: meshes, function spaces, operators, fields, the Grid assembly. |
| [`02_rules.md`](specs/grid/02_rules.md) | normative | Rules: space algebra, coefficient spaces, boundaries, FV semantics, dispatch. |
| [`03_api_sketches.md`](specs/grid/03_api_sketches.md) | normative | API sketches for the grid layer. |
| [`04_decomposition.md`](specs/grid/04_decomposition.md) | normative | Domain decomposition: negotiation, halos, shard maps. |
| [`05_validation.md`](specs/grid/05_validation.md) | normative | Paper validation — the numerical checks gating the grid layer. |
| [`06_open_threads.md`](specs/grid/06_open_threads.md) | normative | Open threads from the grid design phase. |
| [`07_iteration1_api.md`](specs/grid/07_iteration1_api.md) | normative | Iteration-1 public API surface. |

### `specs/grid/classes/` — grid-layer class designs

| File | Status | Description |
|------|--------|-------------|
| [`README.md`](specs/grid/classes/README.md) | normative | Class-design phase overview, document map, staging. |
| [`meshes.md`](specs/grid/classes/meshes.md) | normative | Meshes and function spaces. |
| [`spaces.md`](specs/grid/classes/spaces.md) | normative | Function spaces (nodal, average, coefficient, Galerkin, constant). |
| [`product_spaces.md`](specs/grid/classes/product_spaces.md) | normative | TensorProductSpace and the Field cluster. |
| [`fields.md`](specs/grid/classes/fields.md) | normative | Fields: ScalarField, vector/tensor fields, metadata. |
| [`grid.md`](specs/grid/classes/grid.md) | normative | Grid assembly and decomposition entry points. |
| [`decomposition.md`](specs/grid/classes/decomposition.md) | normative | Domain decomposition: traits, halo accounting, sync strategy. |
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
| [`04_open_threads.md`](specs/operator_algebra/04_open_threads.md) | normative | Open threads. |

### `specs/model/` — the model layer (Phase 2)

| File | Status | Description |
|------|--------|-------------|
| [`00_overview.md`](specs/model/00_overview.md) | normative | Model layer redesign — modules, model, time stepping; document map. |
| [`01_concepts.md`](specs/model/01_concepts.md) | normative | Core concepts and load-bearing decisions (D1–D5). |
| [`02_rules.md`](specs/model/02_rules.md) | normative | Rules: purity, precision, bitwise reproducibility, clock. |
| [`03_time_stepping.md`](specs/model/03_time_stepping.md) | normative | Staged / split time stepping. |
| [`04_run_loop_io.md`](specs/model/04_run_loop_io.md) | normative | Composition, the run loop, and IO. |
| [`05_api_sketches.md`](specs/model/05_api_sketches.md) | normative | API sketches. |
| [`06_validation.md`](specs/model/06_validation.md) | normative | Paper validation for the model layer. |
| [`07_open_threads.md`](specs/model/07_open_threads.md) | normative | Open threads. |
| [`08_state_transforms.md`](specs/model/08_state_transforms.md) | normative | The state-transform algebra. |
| [`09_coupling_designfor.md`](specs/model/09_coupling_designfor.md) | normative | Coupled models — design-for constraints (CS-1..18). |

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
| [`nnmd_design_note.md`](specs/nnmd/nnmd_design_note.md) | normative | NNMD design note (P0 of the NNMD rewrite plan): the slaving recursion. |

## `decisions/` — immutable decision records

| File | Status | Description |
|------|--------|-------------|
| [`blocksymbol_l_assembly.md`](decisions/blocksymbol_l_assembly.md) | accepted | Assembling `L` as a `BlockSymbol` from the operator algebra (resolves eigenmode-roadmap decision 4). |
| [`symbol_stack_design.md`](decisions/symbol_stack_design.md) | accepted | The symbol stack — dynamic symbols, mixed transforms, banded axes; refines the operator-symbols plan. |

## `plans/active/` — work in flight

| File | Status | Description |
|------|--------|-------------|
| [`spatial_model_split_plan.md`](plans/active/spatial_model_split_plan.md) | active | Split framework2 into fridom.spatial + fridom.model (owner-approved names & mapping). |
| [`cutover_parity_plan.md`](plans/active/cutover_parity_plan.md) | active | Cutover-parity work plan: drop the old framework/nonhydro/shallowwater stack. |
| [`docs_examples_plan.md`](plans/active/docs_examples_plan.md) | active | Docs & examples rebuild plan. |
| [`fallback_operator_plan.md`](plans/active/fallback_operator_plan.md) | active | Graded-order boundary fallback operator — implementation plan. |
| [`nnmd_rewrite_plan.md`](plans/active/nnmd_rewrite_plan.md) | active | NNMD rewrite plan for framework2. |
| [`operator_symbols_plan.md`](plans/active/operator_symbols_plan.md) | active | Operator symbols — the spectral-solve substrate (superseded in part by the symbol-stack decision). |
| [`composition_refactor_plan.md`](plans/active/composition_refactor_plan.md) | active | Composition refactor — realized-map category & one composition core. |
| [`boundary_plan.md`](plans/active/boundary_plan.md) | blocked | Boundary-closure plan (BC-free spaces, Robin/mixed BCs) — blocked on an owner decision between conflicting designs. |
| [`bc_free_boundaries.md`](plans/active/bc_free_boundaries.md) | open | BC-free bounded spaces: exterior values are untouchable — owner-flagged open question. |
| [`projection_eigenmode_plan.md`](plans/active/projection_eigenmode_plan.md) | active | Projections & eigenmodes — the energy-metric design (research + plan). |
| [`projection_eigenmode_roadmap.md`](plans/active/projection_eigenmode_roadmap.md) | active | Projection / eigenmode build roadmap (dependency-ordered). |
| [`linear_term_blocks_plan.md`](plans/active/linear_term_blocks_plan.md) | active | Linear-term block signatures (the H1 pivot) — sub-plan. |
| [`phase2_grid_followups.md`](plans/active/phase2_grid_followups.md) | active | Phase-2 reconciliation — grid-layer follow-up work items. |

## `plans/done/` — shipped plans (kept as records)

| File | Status | Description |
|------|--------|-------------|
| [`phase1_implementation_plan.md`](plans/done/phase1_implementation_plan.md) | done | Framework2 Phase-1 implementation plan (grid layer). |
| [`phase2_implementation_plan.md`](plans/done/phase2_implementation_plan.md) | done | Framework2 Phase-2 implementation plan (model layer, wave execution). |
| [`phase1_findings.md`](plans/done/phase1_findings.md) | done | Phase-1 validation findings (input to Phase 2). |
| [`sync_redo_plan.md`](plans/done/sync_redo_plan.md) | done | Task 1.8 implementation plan — consumption-side sync. |

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
| [`nnmd_literature.md`](research/nnmd_literature.md) | frozen | NNMD literature sweep (R1 of the NNMD rewrite plan). |
| [`parity_audit.md`](research/parity_audit.md) | frozen | framework2 §8.8 cutover-parity audit. |

## `archive/` — held / superseded

| File | Status | Description |
|------|--------|-------------|
| [`linear_dsl_proposal.md`](archive/linear_dsl_proposal.md) | held | Readable linear-term DSL — proposal held by the owner (2026-07-09); the `LinearBlock` form stays. |
