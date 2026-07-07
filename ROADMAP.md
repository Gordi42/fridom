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
- **Models**: nonhydro, shallowwater, hydrostatic, and coupled
  multi-model runs (multi-device, later multi-host).

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
[`notes/framework2/classes/`](notes/framework2/classes/README.md)
(concepts in [`notes/framework2/`](notes/framework2/), operator
algebra in [`notes/framework2/operator_algebra/`](notes/framework2/operator_algebra/00_overview.md));
the `classes/README` staging section is the basis for the breakdown.

| #   | Task | Notes |
|-----|------|-------|
| 1.1 | **Meshes, spaces, products** | The `Mesh` and `FunctionSpace` families (nodal, average, coefficient, Galerkin, `ConstantSpace`), static markers, `TensorProductSpace`, interning. Pure static structure. |
| 1.2 | **Field core + registry + FD / interpolate** | `ScalarField` + `grid.create_field` (nodal, single device), the `OperatorRegistry`, `FiniteDifference` / `LinearInterp`, the base `Operator` hierarchy with bind-only axis naming (`op["x"]`) and `@` composition. |
| 1.3 | **Average family, FV, algebra** | The `CellAvg`/`FaceAvg` family, FV operators (`FVDerivative = flux_diff @ Dispatched("reconstruct")`), `integrate`, the field dunders, and the operator algebra (`Composite`/`SeparableComposite`, `OperatorSum`, `ScaledOperator`, `Block`, `Dispatched`, and the `grad`/`div`/`curl`/`laplacian` factories). |
| 1.4 | **Transforms** | `Fourier`, `Sine`/`Cosine`, `Chebyshev`, and `refined()` padding as space-mapping operators; `Symbol` eigenvalues and spectral solves. Spectral differentiation returns here as a function-space operator. |
| 1.5 | **Domain decomposition** | `negotiate` + `MeshDecompositionTraits` + `HaloSpec`/`HaloTracer` (halo accounting by tracing operator requirements) + multi-device shard maps (class doc 04). The grid is a static pytree aux with per-coordinate halos. |
| 1.6 | **Immersed subset + export** | `grid.immersed` (per-space boolean masks derived on demand) and `f.xr` export to xarray. |
| 1.7 | **Standalone validation** | A hand-rolled PDE (advection / diffusion) driven by fields + operators + decomposition under a plain loop, single and multi device, plus the numerical checks in [`05_validation.md`](notes/framework2/05_validation.md). The correctness gate before the model layer exists. |

Grid extensions specified as `designed-for` (may defer): stretched /
coordinate-map grids, terrain-following coordinates, spherical grids with
`RaiseIndex`/`LowerIndex`, and immersed fractions.

## Phase 2 — `framework2`: model, modules, time-stepping, IO

The model layer, built on the Phase 1 grid.

| #   | Task | Notes |
|-----|------|-------|
| 2.1 | **Design doc: model composition** | `Model` as the composition root; setup order and halo negotiation via the doc-04 machinery; the full model pytree; the `update(mz)` signature; how tendency terms declare their integration treatment and which fields they advance (so 2.5 is not precluded); resolves the per-step sync-amplification question ([decomposition open questions](notes/framework2/classes/decomposition.md#open-questions)) — n tendency modules must not mean n syncs — and consumes the [Phase-1 validation findings](notes/framework2/phase1_findings.md) (jit ulp-invariance contract, metadata-in-treedef scan issue, API-gap backlog). |
| 2.2 | **Field registration + parameters in modules** | `Module` API to declare `FieldMetadata` for the state; parameters move into modules (`FPlaneCoriolis`/`BetaPlaneCoriolis`, `ConstantStratification`, shallowwater `csqr`, Rossby scaling); stratification modules register `b`. |
| 2.3 | **Modules modify anything** | Modules and grid in the traced state; `Model(grid=..., tendencies=..., diagnostics=..., time_stepper=...)` direct assembly. |
| 2.4 | **Single `jax.jit` for the full run** | Choose between a full-run `lax.scan`/`while_loop` with `io_callback` and a chunked scan; trace-friendly `Clock`; scan-body time steppers; a NaN-check / early-exit strategy under scan. |
| 2.5 | **Staged / split time stepping** | Generalize the stepper into ordered stages. Split by term (IMEX): explicit/implicit partition, implicit modules exposing `(1 - dt·γ·L)^-1 rhs` (tridiagonal / spectral solves), CNAB/SBDF and IMEX-RK. Split by variable (Gauss-Seidel): advance fields in order, each reading updated earlier fields. The two compose; pressure projection fits the same abstraction. |
| 2.6 | **IO: TensorStore writer + diagnostics** | A `TensorStoreWriter` (zarr store via tensorstore, xarray-openable) fitting the `io_callback` model; progress bar, NaN checker, restart under scan. Follow-up: partial / decomposed-slice output. |
| 2.7 | **Port nonhydro + shallowwater** | Tendencies, pressure solvers, projections/eigenvectors as function-space operators, model-side eigenmode objects (`omega`/`vec_q`/`vec_p`). Update examples, docs, tests. |

## Phase 3 — Models & coupling

| #   | Task | Notes |
|-----|------|-------|
| 3.1 | **Hydrostatic model** | Linear tendency, hydrostatic pressure solver, advection wiring, eigenvectors. Implicit vertical mixing (and optional split-explicit free surface) build on 2.5. |
| 3.2 | **Coupled models — design** | `jax.distributed`, field exchange between models on different meshes/devices/processes, a `Coupler` module plus regridding operators, synchronization schedule. |
| 3.3 | **Coupled models — implementation** | Same-process multi-device, then multi-host. |

## Cutover

Once the models reach parity on `framework2`, rename it to `framework`
and retire the old package in one swap; update imports, examples, docs.

---

## Dependency sketch

```
Phase 1 (standalone):
  1.1 ► 1.2 ► 1.3 ► 1.4 ► 1.5 ► 1.6 ► 1.7 (PDE validation, no model)
Phase 2 (on the grid):
  1.x ► 2.1 ► 2.2 ► 2.3 ► 2.4 ► 2.5 ► 2.6 ► 2.7 port models
  2.2 declarations ► 2.5 staged stepping
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
