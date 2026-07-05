# FRIDOM Roadmap

This document outlines the planned architectural evolution of FRIDOM.
It is organized into phases; each phase is split into tasks that are
small enough to be developed, tested, and merged individually.

## Guiding target (end state)

- **No ModelSettings**: a `Model` is assembled directly from a grid and
  modules; every physical parameter (`f0`, `n2`, `csqr`, ...) lives in
  a module.
- **Everything is one pytree**: modules can modify anything during
  `update` (their own parameters, other modules, grid parameters), and
  the whole model run is a single `jax.jit` call (no Python time loop).
- **Grid = function spaces**: fields live on function spaces (e.g.
  `TensorProductSpace(CellSpace(), FaceSpace())`), operators map fields
  between spaces, mesh arrays (`x`, `k`, ...) are lazy and cached, the
  water mask is a wrapper grid class, and new grid types (stretched,
  spherical) slot into the same abstraction.
- **Fields are ergonomic**: initialization from callables
  (`f(x, y, z) -> value`), dimension reduction (`g = f.sel(x=a)`).
- **Models**: nonhydro (buoyancy optional, provided by stratification
  modules), shallowwater, hydrostatic (real dynamics), and coupled
  multi-model runs (multi-device, later multi-host).

---

## Phase 0 — Foundations

Low-risk work that unblocks and de-risks everything else.

| #   | Task | Notes |
|-----|------|-------|
| 0.1 | **Benchmark infrastructure** **(done)** | New `benchmarks/` package: wall time, compile time, peak device memory (`jax.profiler` / device memory profiles), and comparison reports between two commits/branches. Replaces the stale `benchmark/` directory (obsolete API). Comes first so that every later refactor is measured. |
| 0.2 | **Unify domain decomposition** **(done)** | Delete `SingleDecomposition`; make `JaxDecomposition` the only implementation and make it work for `jax.device_count() == 1`. Fill its gaps: lift the "at least 2 dims" restriction. (Spectral padding EXTEND/TRIM is removed in 0.4, not implemented.) Collapse the base-class abstraction if only one implementation remains. Outcome: one concrete `DomainDecomposition`; a single device is the degenerate case where all shardings are replicated and shard maps reduce to plain calls. Known rough edges (the first axis must divide the device count on several devices, per-call transform-wrapper rebuilds in `grid.fft`) are deferred to the Phase 4 grid rewrite, which revisits the decomposition. |
| 0.3 | **Repo cleanup** **(done)** | Remove `src.bak/` and the stale `benchmark/` scripts. (Both were already removed together with the 0.1 benchmark work.) |
| 0.4 | **Remove experimental spectral grid support (temporary)** **(done)** | Delete the spectral grid classes (`grid/spectral/` in framework, nonhydro, shallowwater), the `SpectralAdvection` modules, `SpectralDiff`, the `spectral_grid` flag, and the pseudo-spectral `FFTPadding.TRIM/EXTEND` machinery — dropping the `padding=` argument from `fft`/`ifft` and the `pad_trim`/`pad_extend`/`unpad_extend` domain-decomposition methods entirely. Keep all Fourier-space-on-cartesian machinery (`is_spectral`, `fft`/`ifft`, `discrete_spectral_operators`, the `*Spectral` projections, RFFT/Spectral cartesian pressure solvers, spectral-energy initial conditions); for `SpectralPressureSolver` keep the cartesian branch, dropping only its spectral-grid case. The current implementation is experimental and complicates 0.2 and Phase 4; spectral methods are reintroduced as function-space operators in Phase 4. |

## Phase 1 — Field ergonomics

Quick wins, largely independent of the architectural work.

| #   | Task | Notes |
|-----|------|-------|
| 1.1 | **Set fields via functions** | `field.set(lambda x, y: ...)` (or a constructor argument): evaluate the callable on the mesh at the field's position/space, handling staggering automatically. |
| 1.2 | **Dimension reduction / selection** | `g = f.sel(x=a)` returns a field with reduced `topo` (e.g. to extract boundary values). Requires making partial-domain fields first-class: resolve the `TODO(Silvano): make this work for non full domain fields` cluster in `scalar_field.py` (fft, sync, diff, interpolate, ...) and replace the `__getitem__` / `__setitem__` `NotImplementedError`. |
| 1.3 | **Lazy grid arrays** | `x_mesh`, `k_mesh`, `x_global`, ... become cached properties computed on demand instead of eagerly in `Grid.setup()`. Small, self-contained stepping stone for the Phase 4 grid rewrite. |
| 1.4 | **TensorStore output writer** | Remove the NetCDF (`netcdf_writer.py`) and Zarr (`zarr_writer.py`) writers and drop the `netcdf4`/`zarr` deps; replace with a single `TensorStoreWriter` that writes a zarr-format store via [tensorstore](https://google.github.io/tensorstore/) with xarray-openable metadata (dimension names/coords, consolidated metadata). TensorStore's async, chunk-wise writes also fit the Phase 3 `io_callback` model. Follow-up: partial-array output — write only a sub-region / decomposed slice of a field (connects to 0.2 decomposition and 1.2 selection). |

## Phase 2 — Composition refactor

The big architectural break: remove ModelSettings, move all parameters
into modules, let modules contribute state fields and modify anything.
No backward-compatibility shims; examples/docs/tests are updated in the
same phase.

| #   | Task | Notes |
|-----|------|-------|
| 2.1 | **Design doc: model composition** | How `Model` replaces `ModelSettingsBase` as the composition root; setup order and halo negotiation without mset; the shape of the "full model pytree"; module lifecycle; the `update` signature (likely `update(mz)` where `mz` reaches modules and grid). |
| 2.2 | **Module field registration** | `Module` API to declare `FieldMetadata` it contributes to the state vector (replaces `mset.custom_state_fields`). State vectors are built from grid defaults plus module registrations. |
| 2.3 | **Modules can modify anything** | Put modules and grid into the traced state so `update` can change module parameters, grid parameters, and other modules. Also removes the forced-re-setup property spaghetti (`halo`/`tendencies` setters triggering global re-setup). |
| 2.4 | **Parameters move into modules** | `FPlaneCoriolis` / `BetaPlaneCoriolis` (own `f0`, `beta`), `ConstantStratification` (owns `n2`), shallowwater `csqr` module, Rossby-number scaling module. Migrate nonhydro and shallowwater. |
| 2.5 | **Buoyancy de-defaulting** | Remove `b` from nonhydro's default state; stratification modules register it: `NoStratification`, `ConstantStratification`, later `TemperatureSalinity` (buoyancy implicit via an equation of state). |
| 2.6 | **Delete ModelSettings** | Remove `ModelSettingsBase` and all per-model `ModelSettings`; direct assembly via `Model(grid=..., tendencies=..., diagnostics=..., time_stepper=...)`. Update all examples, docs, and tests. |

## Phase 3 — Single `jax.jit` for the full run

Depends on Phase 2 (module purity + full model pytree).

> The 0.1 GPU baseline (A100) quantifies the prize: per-call dispatch
> overhead pins every jitted call to a ~100 us floor, and model steps
> to ~1.3 ms — nonhydro steps at 32-64^3 are overhead-dominated, and
> only ~256^3 becomes compute-bound. A scan-based single-jit run
> should recover roughly an order of magnitude at small and medium
> resolutions.

| #   | Task | Notes |
|-----|------|-------|
| 3.1 | **Design/prototype (open question)** | Prototype both strategies: (a) one `lax.scan`/`while_loop` over the full run with `io_callback` for IO/diagnostics; (b) chunked scan between IO events with a thin Python driver. Benchmark with the 0.1 infrastructure (runtime, compile time, memory) and decide. Includes a strategy for NaN checking / early exit under scan (checkify, panicked flag + `while_loop`, ...). |
| 3.2 | **Trace-friendly clock & scheduling** | Rework `Clock` / `ClockTrigger` / schedules so "every N steps / every T seconds" works under scan without Python branching on traced values. |
| 3.3 | **Rework diagnostics/IO modules** | TensorStore writer (from 1.4), progress bar, NaN checker, restart module — adapted to the chosen strategy from 3.1. |
| 3.4 | **Scan-based main loop** | Replace the Python loops in `model.py` (`_main_loop_steps` / `_main_loop_time`); remove the per-step jitted helpers in `adam_bashforth.py` / `runge_kutta.py`; time steppers become pure scan-body components. |
| 3.5 | **Simplify jit machinery** | With a single jit entry point, the structural-equality / memoization layer in `utils/jax_utils.py` (a known complexity hotspot) can likely shrink substantially. |

## Phase 4 — Grid abstraction rewrite (function spaces)

Highest-risk workstream. The design task (4.1) starts early, in
parallel with Phases 2–3; implementation lands after Phase 3.

> **Design notes.** The detailed design lives in
> [`notes/grid_redesign/`](notes/grid_redesign/) (start at
> [`00_overview.md`](notes/grid_redesign/00_overview.md)). The notes
> map to the tasks below as:
>
> | Task | Design notes |
> |------|--------------|
> | 4.1 | Whole set — core concepts ([`01_concepts.md`](notes/grid_redesign/01_concepts.md)), rules ([`02_rules.md`](notes/grid_redesign/02_rules.md)), API sketches ([`03_api_sketches.md`](notes/grid_redesign/03_api_sketches.md)); pytree/decomposition treatment in [`04_decomposition.md`](notes/grid_redesign/04_decomposition.md). |
> | 4.2 | `Mesh`/`FunctionSpace` (concepts 2.1–2.2), coefficient spaces & dispatch & shapes & FV & discretization (rules 3.2, 3.4, 3.5, 3.9, 3.10), decomposition. |
> | 4.3 | Strict space algebra and constant broadcast (rules 3.1, 3.3), `Field`/`VectorField` (concepts 2.4), sketches 4.1/4.3/4.8. |
> | 4.4 | Immersed/masked domains (rule 3.7) and open thread 1 ([`06_open_threads.md`](notes/grid_redesign/06_open_threads.md)). |
> | 4.5 | Terrain-following coordinates (rule 3.8) and open thread 10. |
> | 4.6 | Sphere / curvilinear validation (section 6.3, [`05_validation.md`](notes/grid_redesign/05_validation.md)) and open thread 4. |
>
> Model physics leaving the grid (`omega`/`vec_q`/`vec_p` -> model-side
> eigenmode objects, concepts 2.6, open thread 8) interacts with
> Phase 2; the transform-API richness that lets solvers stop bypassing
> the decomposition ([`04_decomposition.md`](notes/grid_redesign/04_decomposition.md))
> connects back to 0.2/0.4.

| #   | Task | Notes |
|-----|------|-------|
| 4.1 | **Design doc: function spaces** | Captured in [`notes/grid_redesign/`](notes/grid_redesign/): `FunctionSpace`, `TensorProductSpace`, bases/transforms (Fourier, DCT; Chebyshev-ready), operators `A: F_i -> F_j`, error on mixed-space arithmetic, how fields carry their space, pytree/equality treatment, interaction with the domain decomposition. Must not preclude unstructured grids. |
| 4.2 | **Cartesian function spaces** | Implement Cell/Face spaces and tensor products; port FFT/DCT, finite differences, and interpolations as space-mapping operators. Replaces `Position` / `AxisPosition` staggering. Reintroduce spectral (Fourier) differentiation — removed in 0.4 — as a space-mapping operator here. |
| 4.3 | **Port fields** | `ScalarField.function_space`; `f + g` across different spaces raises; `diff`/interp return fields on the mapped space. Port the eigenvector/projection machinery (`nonhydro/grid/cartesian/eigenvectors.py` is the biggest item). Revisit 1.2: a slice at `x = a` naturally lives on a reduced tensor-product space. |
| 4.4 | **Water mask as wrapper grid** | `MaskedGrid(inner_grid)` grid class; remove the default `WaterMask` from `GridBase`; masks derived per function space; masked operators wrap the inner operators. |
| 4.5 | **Stretched coordinates** | Coordinate-map grid (metric terms / Jacobians) on top of the new abstraction. |
| 4.6 | **Spherical coordinates** | Spherical grid class; lat-lon with metric terms first. Unstructured grids remain out of scope for this roadmap. |

## Phase 5 — Models & coupling

| #   | Task | Notes |
|-----|------|-------|
| 5.1 | **Hydrostatic model** | Currently a stub (empty `MainTendency`, `NotImplementedError` eigenvectors). Implement linear tendency, hydrostatic pressure solver, advection wiring, and eigenvectors — built once, directly on the Phase 2 architecture. |
| 5.2 | **Coupled models — design** | `jax.distributed`, exchanging fields between models on different meshes/devices/processes, a `Coupler` module plus regridding operators, synchronization schedule. Interacts with Phase 3 (exchange points inside/between scans). |
| 5.3 | **Coupled models — implementation** | Milestone 1: same-process, multi-device coupling. Milestone 2: multi-host. |

---

## Dependency sketch

```
0.1 benchmarks ──────────────► 3.1 (measure), all phases
0.2 unify decomposition ─────► 3.x, 4.x, 5.3
0.4 remove spectral ─────────► simplifies 0.2; revisited by 4.2
1.x field ergonomics ────────► (independent; 1.2 revisited by 4.3)
1.4 tensorstore writer ──────► 3.3 (adapt IO to scan)
2.1 design ► 2.2 ► 2.3 ► 2.4 ► 2.5 ► 2.6 ─► 3.x, 5.1
4.1 design (parallel) ► 4.2 ► 4.3 ► 4.4 ► {4.5, 4.6}
3.x single jit ──────────────► 5.2 / 5.3 coupling
```

## Cross-cutting rules (every task)

- Ships with mirrored tests (95% branch coverage gate) and stays
  ruff-clean.
- Benchmarked against the previous phase with the 0.1 infrastructure
  (runtime, compile time, memory).
- Examples and docs updated at each phase boundary; breaking changes
  are fine, but examples must run at every merge to main.

## Tradeoffs and open points

1. **Phase 3 before Phase 4**: single-jit first gives immediate
   performance wins and informs how heavy tracing of the new grid may
   be; it accepts some rework in 4.3 because the function-space rewrite
   churns field internals that the scan traces. Swapping the phases is
   defensible if the 4.1 design lands fast.
2. **1.2 is partially reworked in 4.3** (slices become reduced
   tensor-product spaces); done early anyway because of its immediate
   research usefulness.
3. **NaN checking / early exit under a fully jitted run** needs a
   dedicated strategy; part of 3.1.
4. Presentation items explicitly deferred: unstructured grids,
   Chebyshev basis (4.1 must not preclude them).
5. **Spectral grid removed temporarily (0.4)**: the current
   pseudo-spectral grid is experimental and complicates the
   decomposition (0.2) and grid rewrite (Phase 4); spectral methods
   return as function-space operators once the abstraction (4.1/4.2)
   can host them cleanly.
