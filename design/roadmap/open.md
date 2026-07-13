---
status: active
date: 2026-07-13
---

# Roadmap — open work

The single open-work tracker for FRIDOM. Shipped phases are recorded in
[`done.md`](done.md); each task below links to the design record that
governs it. Task numbers are historical and stable — other records cite
them ("ROADMAP 3.5").

## Guiding target (end state)

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

Phases 1 and 2 delivered the first four; the grid is `fridom.spatial`
and the model layer is `fridom.model`. What is left is the cutover, the
model-level features below, and coupling.

## Cutover — retire the old stack

The old `framework` / `nonhydro` / `shallowwater` packages are still on
disk (136 modules, 107 test files) and still exported from
`src/fridom/__init__.py`. **Physics parity is closed**; what remains is
mechanical.

- Rehome the two survivors (`framework/utils/`, `framework/logger.py`) —
  everything else depends on this decision (likely a top-level
  `fridom.utils`); then rewrite the 45 source + 16 test imports that
  reach into them.
- Delete the old packages and their tests; rename `nonhydro2` /
  `shallowwater2` to the canonical names; fix the root exports,
  `tests/conftest.py`, the CI multi-device path, coverage config,
  benchmarks.
- Gated on: owner sign-off of the intentional-deltas table, and the
  docs/examples rebuild being far enough along not to break the build.

Note the old "rename `framework2` → `framework`" plan is obsolete: the
split already landed the new stack as `spatial` + `model`, so only the
two model packages still carry a `2`.

Records: [`../plans/active/cutover_parity_plan.md`](../plans/active/cutover_parity_plan.md)
(parity, sign-off) and
[`../plans/active/cutover_checklist.md`](../plans/active/cutover_checklist.md)
(the executable swap list).

## Phase 3 — models & coupling

| #   | Task | Notes |
|-----|------|-------|
| 3.1 | **Hydrostatic model** | Dropped from the cutover (2026-07-08, executed 2026-07-11): the old `hydrostatic` package is removed and this is now a greenfield future feature, not a cutover gate. Scope when picked up: linear tendency, hydrostatic pressure solver, advection wiring, eigenvectors. Implicit vertical mixing (and an optional split-explicit free surface) build on 2.5. |
| 3.2 | **Coupled models — design** | `jax.distributed`, field exchange between models on different meshes/devices/processes, a `Coupler` module plus regridding operators, a synchronization schedule. **Pre-designed** in [`../specs/model/09_coupling_designfor.md`](../specs/model/09_coupling_designfor.md) (precedent survey + adversarial walk + architecture; the class specs carry its CS-1..18 constraints, so 3.2 stays a pure addition). |
| 3.3 | **Coupled models — implementation** | Same-process multi-device, then multi-host. Depends on 3.2. |
| 3.5 | **Finite-volume nonhydro** | Move the nonhydro model to the average family (`CellAvg` scalars, face-normal velocities — decision FV-D2 **option A**, owner 2026-07-12). **No FV code is written yet**; all nine operator gaps are open. Staged: the four FV symbol rows (the long pole — they block `SpectralSolve` and hence the pressure solve), the missing conversion rows, an FV tracer slice — which is where the payoff lands: **exact tracer-mass conservation and the cut-cell path** — then the C-grid profile with **bitwise parity** as the gate. Do **not** flip the default wholesale first: walls and mapped grids work today and would regress. Scoping record: [`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md). |
| 3.7 | **Spherical nonhydro** — *long-term* | The 3D spherical chart (`X(lon, lat, h)`, so the metric comes out diagonal and `w = dh/dt` is already physical) needs the C2 chart metrics and the C3 elliptic machinery to meet: the pressure operator becomes the Laplace–Beltrami on the chart — still SPD under the sqrt(g)-weighted product, so the PCG structure carries over, but the operator assembly must be written. Not the first 3D-spherical consumer: a hydrostatic model needs no pressure solve and is the likelier first use (3.1). |
| 3.8 | **Generalized adiabatic ramping** — *planned, not scheduled* | An `AdiabaticRamping` base transform that ramps declared parameters from a start to an end value over a ramp period (continuous stage-time `Ramp` evaluation; curves `"linear"` / `"cosine"` / `"exp"` or a callable), with **`OptimalBalance` as a subclass** contributing only the balancing policy. Also buys adiabatic spin-up and parameter continuation. Its dependency (2.8) has shipped, and `Propagator(updates={param: Ramp(...)})` already covers much of the mechanism — so this is largely an ergonomics/factoring task. Record: [`../plans/active/adiabatic_ramping.md`](../plans/active/adiabatic_ramping.md). |

## Cross-cutting work in flight

Not numbered roadmap tasks, but live plans:

- **Docs & examples rebuild** — the CI skeleton and a pilot example
  landed; 12 example ports and the whole prose page tree remain.
  [`../plans/active/docs_examples_plan.md`](../plans/active/docs_examples_plan.md)
- **Boundary closures, stage 2e** — the Robin dynamic `(α, g)` ghost-fill
  data path. Unblocked.
  [`../plans/active/boundary_plan.md`](../plans/active/boundary_plan.md)
- **High-order stencils on mapped grids** — four mapped refusals still
  stand; the walled prerequisite is paid, so the next step is the
  Jacobian spike (which divisor preserves free-stream).
  [`../plans/active/high_order_mapped_plan.md`](../plans/active/high_order_mapped_plan.md)
- **Chart ergonomics, E2** — auto-seed `diagonal=True` index moves on
  orthogonal charts; today a bounded chart does not assemble without an
  undiscoverable `merge_overrides` incantation.
  [`../plans/active/chart_ergonomics_plan.md`](../plans/active/chart_ergonomics_plan.md)
- **Eigenmode phase I** — promote `Banded` to a first-class realized map;
  the mixed `Fourier ⊗ Chebyshev` per-mode solve.
  [`../plans/active/projection_eigenmode_roadmap.md`](../plans/active/projection_eigenmode_roadmap.md)
- **Grid follow-ups** — two ergonomics items left (the `cartesian.Grid`
  convenience constructor is still a stub; assorted API shims).
  [`../plans/active/phase2_grid_followups.md`](../plans/active/phase2_grid_followups.md)

## Known gaps in the shipped surface

Surfaced by the 2026-07-13 spec audit; small, but they are promises the
specs make that the code does not keep:

- `model.blank_state()` / `model.state_space(name)` — specified, never
  built. Either add the two one-liners or strike them from the spec.
- `fr.modules.WindowAccumulator` — named as shipping in four places;
  does not exist.
- `add_prognostic` — every stepper family's combine step wants it.
- The multi-device mapped-pressure gates still solve **eagerly**, at a
  measured **85x** per-call penalty (16.4 s vs 194 ms on forced-4). Since
  3.6 the jitted forced-4 solve compiles (9.1 s, was: never), so
  `test_mapped_projection_is_device_count_invariant` can be jitted —
  numbers in
  [`../plans/done/krylov_scan_plan.md`](../plans/done/krylov_scan_plan.md).

## Cross-cutting rules

- Mirrored tests (95% branch coverage gate), ruff-clean.
- Benchmarked with the 0.1 infrastructure (runtime, compile, memory).
- The old `framework` stays runnable until the cutover; new work does not
  go into it.
- Unstructured grids stay out of scope; the designed-for grid extensions
  must not be precluded by the iteration-1 core.
