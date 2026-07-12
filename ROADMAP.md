# FRIDOM Roadmap

Planned architectural evolution of FRIDOM, in phases. Each task is small
enough to develop, test, and merge on its own.

## Guiding target (end state)

- **No ModelSettings**: a `Model` is assembled from a grid and modules;
  every physical parameter (`f0`, `n2`, `csqr`, ...) lives in a module.
- **Everything is one pytree**: modules can modify anything during
  `update`, and the whole run is a single `jax.jit` call (no Python time
  loop).
- **Grid = function spaces**: fields live on function spaces, operators
  map between spaces, mesh arrays are lazy, and new grid types
  (stretched, spherical) slot into the same abstraction.
- **Fields are ergonomic**: init from callables (`f(x, y, z) -> value`),
  dimension reduction (`g = f.sel(x=a)`).
- **Models**: nonhydro, shallowwater, and coupled multi-model runs
  (multi-device, later multi-host); a hydrostatic model is a future
  greenfield feature (see 3.1).

## Approach

The new architecture is built in a parallel package `fridom.framework2`,
reusing `framework.utils`; at the end it is renamed to `framework`. Build
order is **grid-first**: the function-space grid, its operators, and the
domain decomposition (Phase 1) are validated standalone; the model and
time-stepping layer is then built on top of the finished grid (Phase 2);
existing models are ported last. This designs the model once, on the
final grid — the model's field registration, halo negotiation, and pytree
shape all depend on grid concepts.

The one cost — no end-to-end model run until Phase 2 — is covered by the
grid layer's standalone testability and an early hand-rolled-PDE smoke
test (1.7).

---

## Phase 0 — Foundations (done)

Benchmark infrastructure, domain-decomposition unification, repo
cleanup, and removal of the experimental spectral grid — all on the
existing `framework`.

## Phase 1 — `framework2`: grid, operators, decomposition

The function-space core, decoupled from the model and testable
standalone. Direct implementation of the class designs in
[`design/specs/grid/classes/`](design/specs/grid/classes/README.md)
(concepts in [`design/specs/grid/`](design/specs/grid/), operator
algebra in [`design/specs/operator_algebra/`](design/specs/operator_algebra/00_overview.md));
the `classes/README` staging section is the basis for the breakdown.

| #   | Task | Notes |
|-----|------|-------|
| 1.1 | **Meshes, spaces, products** | The `Mesh` and `FunctionSpace` families (nodal, average, coefficient, Galerkin, `ConstantSpace`), static markers, `TensorProductSpace`, interning. Pure static structure. |
| 1.2 | **Field core + registry + FD / interpolate** | `ScalarField` + `grid.create_field` (nodal, single device), the `OperatorRegistry`, `FiniteDifference` / `LinearInterp`, the base `Operator` hierarchy with bind-only axis naming (`op["x"]`) and `@` composition. |
| 1.3 | **Average family, FV, algebra** | The `CellAvg`/`FaceAvg` family, FV operators (`FVDerivative = flux_diff @ Dispatched("reconstruct")`), `integrate`, the field dunders, and the operator algebra (`Composite`/`SeparableComposite`, `OperatorSum`, `ScaledOperator`, `Block`, `Dispatched`, and the `grad`/`div`/`curl`/`laplacian` factories). |
| 1.4 | **Transforms** | `Fourier`, `Sine`/`Cosine`, `Chebyshev`, and `refined()` padding as space-mapping operators; `Symbol` eigenvalues and spectral solves. Spectral differentiation returns here as a function-space operator. |
| 1.5 | **Domain decomposition** | `negotiate` + `MeshDecompositionTraits` + `HaloSpec`/`HaloTracer` (halo accounting by tracing operator requirements) + multi-device shard maps (class doc 04). The grid is a static pytree aux with per-coordinate halos. |
| 1.6 | **Immersed subset + export** | `grid.immersed` (per-space boolean masks derived on demand) and `f.xr` export to xarray. |
| 1.7 | **Standalone validation** | A hand-rolled PDE (advection / diffusion) driven by fields + operators + decomposition under a plain loop, single and multi device, plus the numerical checks in [`05_validation.md`](design/specs/grid/05_validation.md). The correctness gate before the model layer exists. |
| 1.8 | **Sync-strategy redo: consumption-side halo-validity tracking** — *done (2026-07-07)* | *Decided 2026-07-08; implemented on `framework2-sync-redo`.* Replace the iteration-1 sync-after-every-operator placement: fields carry a trace-time valid-halo depth (static attribute, zero runtime cost); operators sync iff input depth < requirement; `store` stops syncing. Cuts the composed model step from one exchange per operator application *and per field `+`/`-`* to ~one per state component per step; results-neutral by construction (syncs only rewrite ghost cells). Decision record: [decomposition open questions](design/specs/grid/classes/decomposition.md#open-questions); work item 8 in [`phase2_grid_followups.md`](design/plans/active/phase2_grid_followups.md). Not blocking 2.2–2.3; land before performance-sensitive multi-device work (2.7 benchmarks, 3.3). Implementation plan: [`design/plans/done/sync_redo_plan.md`](design/plans/done/sync_redo_plan.md). |

Grid extensions specified as `designed-for` (may defer): stretched /
coordinate-map grids, terrain-following coordinates, spherical grids with
`RaiseIndex`/`LowerIndex`, and immersed fractions.

## Phase 2 — `framework2`: model, modules, time-stepping, IO

The model layer, built on the Phase 1 grid. **The design is complete
(2026-07-08)**: five resolved decisions (field registration,
parameter ownership, staged/split stepping, composition/run-loop/IO,
and the state-transform algebra) in
[`design/specs/model/`](design/specs/model/00_overview.md) —
task 2.1's design doc, grown into the full note set covering
2.1–2.6 plus the transforms. The rows below are now implementation
tasks against that design.

| #   | Task | Notes |
|-----|------|-------|
| 2.1 | **Design doc: model composition** — *done (2026-07-08)* | Resolved as decisions D1–D5 in `design/specs/model/` (concepts, rules, full designs for stepping/run-loop/transforms, API sketches, research archive, class specs in `design/specs/model/classes/`). Reconciled against the landed Phase-1 code (2026-07-08): the per-step sync-amplification question's model half is discharged (the signed term surface is sync-policy-neutral; the grid-side strategy redo is decided as task 1.8, decision record in the [decomposition open questions](design/specs/grid/classes/decomposition.md#open-questions)), the Phase-1 validation findings are consumed (metadata ruling in `design/specs/grid/classes/fields.md`, bitwise umbrella in `design/specs/model/02_rules.md`, precision ruling in `design/specs/model/classes/declarations.md`), and the grid follow-up work items are filed in [design/plans/active/phase2_grid_followups.md](design/plans/active/phase2_grid_followups.md). |
| 2.2 | **Field registration + parameters in modules** — *implemented (2026-07-08)* | `Module` API to declare `FieldMetadata` for the state; parameters move into modules (`FPlaneCoriolis`/`BetaPlaneCoriolis`, `ConstantStratification`, shallowwater `csqr`, Rossby scaling); stratification modules register `b`. Declaration vocabulary + assembly tables landed in Phase-2 waves 2–3; the concrete physics modules land with 2.7. |
| 2.3 | **Modules modify anything** — *implemented (2026-07-08)* | Modules and grid in the traced state; `Model(grid=..., tendencies=..., diagnostics=..., time_stepper=...)` direct assembly. `Module` base + `fr.Model` + the nine-step assembly (waves 3–4). |
| 2.4 | **Single `jax.jit` for the full run** — *implemented (2026-07-08)* | Chunked `lax.scan` (`step_chunk`, AOT-compiled, donated carry, lengths {C,1}); trace-friendly `Clock`; scan-body time steppers; per-step S5 NaN reduction + chunk-boundary abort. `fr.ops.Session` + `Model.run()` (wave 5). |
| 2.5 | **Staged / split time stepping** — *implemented (2026-07-08)* | Ordered stages by `StageKind`; IMEX explicit/implicit partition with `VerticalDiffusion` `(1 − dt·γ·L)^-1`, CNAB2/SBDF2; the RK family + `LowStorageRK3`; by-variable Gauss-Seidel `advance_stages`. Pressure projection (a CONSTRAINT stage) lands with 2.7; IMEX-RK stays designed-for. |
| 2.6 | **IO: TensorStore writer + diagnostics** — *implemented (2026-07-08)* | `fr.io.Writer` writes a zarr-format store **via tensorstore** (no zarr-python import; xarray/xgcm-openable), `fr.io.TimeSeries` CSV; triggers, the pickle-free snapshot store, restart-under-scan, walltime/progress/NaN under the chunked scan. Follow-up: partial / decomposed-slice output (designed-for behind the sink seam). |
| 2.7 | **Port nonhydro + shallowwater** | Tendencies, pressure solvers as function-space operators, model-side eigenmode objects (`Eigenmodes` / `from_model`). Update examples, docs, tests. |
| 2.8 | **State transforms** (design: `design/specs/model/08_state_transforms.md`) | `fr.StateTransform` + the algebra (`@`, arithmetic, `FixedPoint`, `Shift`), `model.variant(term_filter=...)` + term predicates + `fr.closures.ClosureBase`, and the ported family: Vortical/Wave/Divergence projections, `Propagator`, `TimeAverage`, `OptimalBalance`. NNMD deferred to its own future rewrite (no model propagator). |

## Phase 3 — Models & coupling

| #   | Task | Notes |
|-----|------|-------|
| 3.1 | **Hydrostatic model** | **Dropped from the cutover (2026-07-08, executed 2026-07-11): the old `hydrostatic` package is removed and this becomes a greenfield future feature, not a cutover gate.** Scope when picked up: linear tendency, hydrostatic pressure solver, advection wiring, eigenvectors. Implicit vertical mixing (and optional split-explicit free surface) build on 2.5. |
| 3.2 | **Coupled models — design** | `jax.distributed`, field exchange between models on different meshes/devices/processes, a `Coupler` module plus regridding operators, synchronization schedule. **Pre-designed** in [`design/specs/model/09_coupling_designfor.md`](design/specs/model/09_coupling_designfor.md) (precedent survey + adversarial A–O walk + architecture; the class specs carry its CS-1..18 constraints so 3.2 stays a pure addition). |
| 3.3 | **Coupled models — implementation** | Same-process multi-device, then multi-host. |
| 3.4 | **Coordinate systems** — *done (2026-07-12)* | Mapped, spherical, and boundary-fitted grids — the Phase-1 designed-for metric seams, filled in: measures as fields + `MappedIntervalMesh` (C0), `CoordinateMapping`/`grid.metric`/`physical_diff` with the CS-D1 chart embedding (C1), metric-aware vector calculus + spherical shallow water (C2), the CS-D2 preconditioned-CG mapped pressure solve + terrain-following/boundary-fitted nonhydro (C3), dynamic metrics + the optional CS-D4 ALE module (C4). Record, with per-stage outcomes and follow-ups: [`design/plans/done/coordinate_systems_plan.md`](design/plans/done/coordinate_systems_plan.md). |

## Cutover

Once the models reach parity on `framework2`, rename it to `framework`
and retire the old package in one swap; update imports, examples, docs.

---

## Dependency sketch

```
Phase 1 (standalone):
  1.1 ► 1.2 ► 1.3 ► 1.4 ► 1.5 ► 1.6 ► 1.7 (PDE validation, no model)
  1.5 ► 1.8 sync-strategy redo — done (2026-07-07)
Phase 2 (on the grid):
  1.x ► 2.1 ► 2.2 ► 2.3 ► 2.4 ► 2.5 ► 2.6 ► 2.7 port models
  2.2 declarations ► 2.5 staged stepping
  2.4 + 2.5 (+ eigenmodes from 2.7) ► 2.8 state transforms
Phase 3: 2.x ► 3.1;  2.4 ► 3.2/3.3
Cutover: 2.7 (+3.1) ► rename framework2 → framework
```

## Cross-cutting rules

- Mirrored tests (95% branch coverage gate), ruff-clean.
- `framework2` reuses `framework.utils`; it does not import the old
  model/grid stack.
- Benchmarked with the 0.1 infrastructure (runtime, compile, memory).
- The old `framework` stays runnable until the cutover; new work does not
  go into it.

## Open points

- NaN checking / early exit under a fully jitted run — part of 2.4.
- Designing the model-facing seams (`State`, module registration,
  eigenmode objects) in Phase 1 without a model consumer; bounded by the
  grid/model separation in the design notes.
- Unstructured grids stay out of scope; the designed-for grid extensions
  must not be precluded by the iteration-1 core.
