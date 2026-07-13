---
status: normative
date: 2026-07-13
---

# Model layer redesign — Class designs

Part of the model redesign notes; see
[`../00_overview.md`](../00_overview.md) for the document map.

Status: **implemented** (drafted 2026-07-08; all six cluster files
placed and cross-reconciled, then built out as `fridom.model`,
ROADMAP 2.2–2.8). The class surfaces below are the normative
reference; where the implementation deviated, the deviation is called
out in the owning file, and the two surfaces the specs promise but the
code never grew (`model.blank_state`/`state_space`,
`fr.modules.WindowAccumulator`) are listed in
[`../07_open_threads.md`](../07_open_threads.md) §9.1.
Reconciliation notes from the drafting round:

- **Package layout**: where the clusters' layout sketches disagree,
  the *owning* cluster's placement wins — `io/` and `ops/` sit
  directly under `framework2/` (io_ops.md), `transforms/` directly
  under `framework2/` (transforms.md), `time_steppers/` under
  `model/` (time_steppers.md); declarations.md's whole-tree sketch
  is indicative only, and model.md is canonical for `model/`
  itself.
- `Ramp.reversed()` is pinned in declarations.md (window
  reflection); transforms.md's open question 2 now points there.
- Each file carries an explicit deviations/spec-completions list
  (flagged inline per the rule below) and an Open-questions list
  restricted to 07_open_threads residuals — the consolidated set
  feeds the 2.2–2.8 implementation briefs.

This directory is the
second design phase for the model layer: it turns the resolved
decisions D1–D5 (concept notes `../01_concepts.md`, rules
`../02_rules.md`, full designs `../03_time_stepping.md`,
`../04_run_loop_io.md`, `../08_state_transforms.md`, the coupling
design-for `../09_coupling_designfor.md`, and the validation record
`../06_validation.md`) into concrete class specifications — for
every class the module placement, constructor, and full public
surface (methods, properties, dunders) with Python signatures. It
bridges the design to ROADMAP Phase 2 implementation (tasks
2.2–2.8), exactly as [`../../grid/classes/`](../../grid/classes/README.md)
did for the grid (Phase 1).

**Nothing here re-decides.** All decisions are signed; a spec that
needs to deviate from the notes must call the deviation out
explicitly and record it in `../07_open_threads.md`. The API
sketches (`../05_api_sketches.md`) are the acceptance surface —
every spelling there must remain expressible. The coupling
constraint list **CS-1..18**
(`../09_coupling_designfor.md` §11.3) is binding on these specs.

## Document map

| Cluster | File | Classes owned |
|---------|------|---------------|
| Declarations | [`declarations.md`](declarations.md) | `FieldDeclaration` (+ templates), `SpacePattern` family (`Collocated`/`Staggered`/`Profile`/`SpaceRule`, `Dof`), `Lifecycle`, `Role` + `fr.roles` (incl. `Velocity` and its validation-amended rules), `FieldReference`, `ParameterDeclaration`/`ParameterReference`/`fr.params` registry (+ `USE_PROVIDED`/`fr.Param`), `fr.Ramp`/`TimeDependent`/`resolve_at`, `TendencyTerm`/`@fr.term`, `ImplicitOperator` + `fr.implicit` families, `fr.terms` predicates. |
| Module | [`module.md`](module.md) | `Module` base (the capability menu: declarations, references, dispatch, terms, stages, `self_update`, `bind`, `extra_halo`, `state_type`), `fr.closures.ClosureBase`, `Stage`/`StageKind`, `StepContext`. |
| Model | [`model.md`](model.md) | `fr.Model` (constructor, the nine-step assembly, lifecycle methods incl. `set_aux`/`variant`/`tendency`/`blank_state`/`advance`/`run`/snapshots), `FieldTable`/`VelocitySelector`, `TendencyComposer`, the binding/re-materialization tables, `AssemblyReport`, the fingerprint, `State` vocabulary classes, `AdvanceResult`/`RunResult`/`PanicError`, preset factories, the error-type registry. |
| Time stepping | [`time_steppers.md`](time_steppers.md) | `TimeStepper` base, `StepperState`, `AdamBashforth` (eps order-2-only), the RK family, `IMEXMultistep` (CNAB2/SBDF2), `Clock`, schedule composition internals. |
| Transforms | [`transforms.md`](transforms.md) | `fr.StateTransform` + the algebra nodes, `StateSignature`/`TransformInfo`, `Identity`/`Shift`/`FixedPoint`, `Propagator`, `TimeAverage`, `OptimalBalance`, the projection wrappers, `model.variant` mechanics, `fr.linearize`. |
| IO & ops | [`io_ops.md`](io_ops.md) | Trigger objects + the plan-time lowering function, `OutputStream`/`Writer`/`TimeSeries`, `Snapshots` + the snapshot store/manifest, **`fr.ops.Session`** + the normative ops protocols (walltime predictor, progress renderer), `fr.io.resubmit`/`fr.slurm`. |

## Shared template (per class)

- a one-line role plus a table of *kind* (ABC/concrete/final),
  *pytree status* (static/dynamic per the hardened-jaxify
  discipline; or "host object, not a pytree"), *task* (the ROADMAP
  2.x row it lands in, or `designed-for`), and *design refs* (the
  normative sections implemented);
- a Python skeleton (constructor and every public member as a
  signature with a one-line docstring, no bodies);
- prose notes: semantics, invariants, extension contracts, and
  **pointers** to rejected alternatives (the research archive —
  never re-argued);
- a closing `## Open questions` — genuinely unresolved points only
  (the parked residuals in `../07_open_threads.md` relevant to the
  cluster); decided questions are not reopened.

## Cross-cluster seam anchors

The state vector is `state`, never `z`; parameters are
dotted-by-concept, fields dot-free; every in-trace hook is
`(self, state, ctx) -> dict` applied per the kind's write gate;
`fn`/`default=` references to module behavior are **unbound**
(the aliasing rule); Model is a host driver, the carry is
`(state, modules, stepper_state, clock, panicked)`; `advance` is
the primitive and everything else is layered sugar with bitwise
equivalence tests (`run ≡ Session loop`, `preset ≡ explicit
assembly`); declarations are transient assembly data except
retained AUX-default closures; consent flags (`host_writable`) are
lifecycle-polymorphic (AUX ∪ DIAG); no coupling/run-loop state
outside the carry.
