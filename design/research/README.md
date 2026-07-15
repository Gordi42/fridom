---
status: frozen
date: 2026-07-12
---

# Design research reports

The `c*` / `d*` table below is the per-decision research behind the
model-layer design. The bucket also holds standalone frozen
inputs (audits and investigations), listed in the second table.

Per-decision research reports produced during the model-layer design
(deep option analysis + precedent research, AI-assisted). These are
**inputs to the decisions**, not normative text: the resolved state
of each decision lives in [`../specs/model/01_concepts.md`](../specs/model/01_concepts.md);
disagreements between a report and the concepts file were reconciled
there (see the "Reconciliations" subsection of D1). Sketches in
these reports also predate two later refinements: the naming rule
(the state vector is `state`, never `z`) and the tendency
read/write rule (terms receive the full state vector; contribution
keys are gated to PROGNOSTIC) — the concepts file is authoritative.

| Report | Decision |
|--------|----------|
| [`d1_1_declaration_and_ics.md`](d1_1_declaration_and_ics.md) | D1.1 — FieldDeclaration surface, initial conditions, defaults, restart |
| [`d1_2_space_binding.md`](d1_2_space_binding.md) | D1.2 — how declarations get function spaces (SpacePattern) |
| [`d1_3_core_ownership.md`](d1_3_core_ownership.md) | D1.3 — who declares the core prognostic variables |
| [`d1_4_roles.md`](d1_4_roles.md) | D1.4 — lifecycle axis + role system |
| [`d1_5_access_and_collisions.md`](d1_5_access_and_collisions.md) | D1.5 — access surface, collisions, functional-update idiom |
| [`d2_1_resolution_mechanism.md`](d2_1_resolution_mechanism.md) | D2.1 — provides/requires resolution, jax aliasing semantics, namespace |
| [`d2_2_representation.md`](d2_2_representation.md) | D2.2 — scalar-vs-field placement rule, static/dynamic discipline, `fr.Ramp` |
| [`d2_3_diagnostics.md`](d2_3_diagnostics.md) | D2.3 — diagnostics as functions, bound namespace, writer seam |
| [`d2_4_host_consumers.md`](d2_4_host_consumers.md) | D2.4 — `model.parameters`, eigenmode seam, projections, mutation rules |
| [`d3_1_term_surface.md`](d3_1_term_surface.md) | D3.1 — TendencyTerm, treatments, the implicit-operator surface |
| [`d3_2_stepper_core.md`](d3_2_stepper_core.md) | D3.2 — stepper as scan body, warm-up, dt, Clock, backward runs |
| [`d3_3_stage_schedule.md`](d3_3_stage_schedule.md) | D3.3 — stage kinds/ordering, project-the-state, StepContext |
| [`d3_4_imex_splitting.md`](d3_4_imex_splitting.md) | D3.4 — IMEX families, buffers, split-explicit free surface |
| [`d4_1_assembly_model.md`](d4_1_assembly_model.md) | D4.1 — constructor, assembly pipeline, dispatch-merge call site, Model status |
| [`d4_2_run_loop.md`](d4_2_run_loop.md) | D4.2 — run()/advance, chunking, NaN mechanism, debug/profiling tiers |
| [`d4_3_io_seams.md`](d4_3_io_seams.md) | D4.3 — chunk-boundary IO, triggers, Writer/snapshot seams, dill successor |
| [`d4_4_lifecycle_coupling.md`](d4_4_lifecycle_coupling.md) | D4.4 — mutation surface, update_parameters/reset, sweeps, coupling proofing |
| [`d5_1_algebra.md`](d5_1_algebra.md) | D5.1 — StateTransform base, algebra semantics, the three laws, FixedPoint/norm |
| [`d5_2_variants.md`](d5_2_variants.md) | D5.2 — model.variant(), term predicates, ClosureBase, snapshot semantics |
| [`d5_3_family_ports.md`](d5_3_family_ports.md) | D5.3 — Vortical/Wave/Divergence, TimeAverage, OptimalBalance, NNMD ports |
| [`c1_coupling_precedents.md`](c1_coupling_precedents.md) | Coupling — production coupler survey (OASIS/ESMF/CESM/FMS/YAC/IFS-NEMO), irreducible problems, what evaporates under jax |
| [`c2_coupled_walk.md`](c2_coupled_walk.md) | Coupling — adversarial concrete A–O walk (windowed accumulation gap, re-materialization contradiction, advance surface) |
| [`c3_coupling_architecture.md`](c3_coupling_architecture.md) | Coupling — Phase-3 architecture pre-design (facade, dispatch-then-sync, mediator-as-model, product-state Schwarz, hooks H1–H11) |

## Standalone frozen inputs

| Report | Feeds |
|--------|-------|
| [`parity_audit.md`](parity_audit.md) | Cutover — the §8.8 cutover-parity claim → test map ([`../plans/active/cutover_parity_plan.md`](../plans/active/cutover_parity_plan.md)) |
| [`boundary_design_explainer.md`](boundary_design_explainer.md) | Boundaries — dev extrapolation-fill vs the R1 flip, side by side (decision made 2026-07-11: R1 landed) |
| [`nnmd_literature.md`](nnmd_literature.md) | NNMD — literature sweep, R1 of [`../plans/active/nnmd_rewrite_plan.md`](../plans/done/nnmd_rewrite_plan.md) |
| [`xla_spmd_fft_fault.md`](xla_spmd_fft_fault.md) | Coordinate systems — the jitted-FFT-on-a-sharded-axis XLA fault; reproducer + fridom-side mitigation |
| [`indivisible_shard_probes.md`](indivisible_shard_probes.md) | Indivisible-extent sharding — HLO forensics (per-op collective attribution, the `tensor.py:525` gate A/B), padded all-to-all transpose validation, pad-cost measurements, prior art ([`../plans/active/indivisible_shard_plan.md`](../plans/active/indivisible_shard_plan.md)) |
