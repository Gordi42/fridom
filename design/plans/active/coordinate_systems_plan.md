---
status: active
date: 2026-07-12
---

# Coordinate-systems plan — mapped, spherical, boundary-fitted grids

Staged implementation plan for non-Cartesian coordinate systems on
the new stack (`fridom.spatial` / `fridom.model`). The design side is
already normative — terrain-following in
[`../../specs/grid/02_rules.md`](../../specs/grid/02_rules.md) §3.8,
sphere/curvilinear in
[`../../specs/grid/05_validation.md`](../../specs/grid/05_validation.md)
§6.3/§6.5, the `CoordinateMapping` class in
[`../../specs/grid/classes/grid.md`](../../specs/grid/classes/grid.md),
`MappedIntervalMesh`/`SphereMesh` in
[`../../specs/grid/classes/meshes.md`](../../specs/grid/classes/meshes.md),
the metric-coefficient operator sketch in
[`../../specs/operator_algebra/03_api_sketches.md`](../../specs/operator_algebra/03_api_sketches.md)
§4.4 — so this plan is a fill-in of designed-for seams, not a
redesign.

## 1. Targets (owner, 2026-07-12)

- **(a) Spherical shallow water** (nonhydro later if it translates).
- **(b) Terrain-following nonhydro**: water depth `H(x, y)`.
- **(c) Boundary-fitted horizontal boundaries**: northern boundary
  position `y_N(x, z, t)`, **time-dependent** — one planned
  experiment morphs the northern boundary from sloped to flat during
  a run.

## 2. Decisions (owner sign-off in chat, 2026-07-12)

- **CS-D1 — chart route for curved manifolds.** Curved 2D geometries
  are built from **1D tensor factors with grid-level metric
  coupling**, not a monolithic 2D mesh factor: the factors carry
  topology only (DOF layout, staggering, halos, transforms); all
  geometry enters as metric coefficient fields. This adopts the
  owner-favored "manifold abstraction layer" direction recorded in
  the `SphereMesh` notes
  ([`meshes.md`](../../specs/grid/classes/meshes.md) §SphereMesh);
  the monolithic `SphereMesh` stub is not pursued. The
  `CoordinateMapping` gains an **embedding form** (a chart
  `X: (u, v) -> R^3`) from which the induced metric
  `g_ij = dX/du_i · dX/du_j`, its inverse, and `sqrt(g)` are derived
  by **autodiff of the chart callable** at the requested space's
  evaluation nodes. Reachable topologies with one chart:
  interval x interval (patch), circle x interval (lat-lon sphere
  minus polar caps), circle x circle (torus — a closed manifold with
  no singularities). Multi-chart atlases (full sphere via cubed
  sphere, genus >= 2) are out of scope but must not be precluded:
  metric derivation is per-chart from day one.
- **CS-D2 — pressure solve: preconditioned CG.** Non-separable
  elliptic operators (mapped coordinates couple the axes) are solved
  by matrix-free CG **preconditioned by the separable spectral
  inverse** (the flat-geometry `Symbol` inverse `SpectralSolve`
  already builds — its documented metric seam). Fixed iteration
  count (static trace, reverse-mode differentiable); mean/null-space
  projection; `custom_vjp` via the implicit function theorem when
  adjoints through the solve are needed (deferred). Geometric
  multigrid is deferred, not rejected — if PCG iteration counts blow
  up on steep geometry it slots in as a preconditioner inside the
  same CG loop.
- **CS-D3 — boundary-fitted before immersed for curved walls.**
  Target (c) uses the boundary-fitted mapping route (§3.8's own
  framing), which is structurally identical to (b) on a horizontal
  coordinate and keeps BCs conforming. The immersed-mask route
  (`ImmersedDomain`, already implemented, boolean) remains available
  as a cross-check reference, not the primary path.
- **CS-D4 — mesh-velocity (ALE) terms are a module, and optional.**
  A time-dependent mapping adds grid-velocity corrections
  (`du/dt|_phys = du/dt|_comp − x_grid_dot · grad u`) to every
  prognostic equation while the geometry moves. These live in a
  dedicated tendency module, **not** in the grid layer, and the
  owner explicitly requires the option to run without them
  (2026-07-12: one project deliberately trades physical correctness
  during the morph). Omitting the module is the off switch; its
  docs must state the correctness caveat plainly.

## 3. Implementation seams (survey, 2026-07-12)

Current-state anchors; line numbers as of dev @ `c6c7a949`:

- `grid.measure(space, name)` exists but its materializers
  hard-branch on `isinstance(mesh, IntervalMesh)` and raise
  otherwise (`src/fridom/spatial/grid.py:1344`, `:1408`).
- Stencil operators read a **scalar** spacing via
  `uniform_spacing()`
  (`src/fridom/spatial/operators/staggering.py:411`), documented as
  the iteration-1 stand-in for `grid.measure`.
- `MappedIntervalMesh`, `SphereMesh`, `CoordinateMapping`,
  `UnstructuredMesh` are ~10-line stubs; `Variance` defaults to
  `None`; `RaiseIndex`/`LowerIndex` are reserved in
  `operators/composed.py` but unimplemented.
- `Laplacian(metric=...)`/`Diag` already thread constant per-axis
  diagonal weights (`operators/composed.py`); `SpectralSolve`'s
  `Symbol` form is the documented metric injection point
  (`operators/spectral_solve.py`).
- `ImmersedDomain` is fully implemented (boolean masks).
- Deeply Cartesian model surfaces: `shallowwater2/modules/sadourny.py`
  (fluxes, vorticity, KE), the eigenmode/transform stack
  (Fourier-periodic assumptions).

## 4. Stages

Ordering: C0 → C1 → {C2, C3 in either order / parallel} → C4.
C2 and C3 both consume C1 and are independent of each other.

### C0 — measures become fields (the keystone)

- Generalize the grid's node/measure materializers off the
  `IntervalMesh`-only branches: the mesh supplies computational
  nodes and an optional self-coordinate mapping; the grid composes
  and shards.
- Route `FiniteDifference` / `flux_diff` / interpolation /
  `integrate` from scalar `uniform_spacing` to field-valued
  `grid.measure` coefficients. A uniform mesh yields a constant
  field XLA folds — the swap is results-neutral there.
- Implement `MappedIntervalMesh`
  ([`meshes.md`](../../specs/grid/classes/meshes.md) spec): strictly
  monotone self-coordinate mapping; the two staggered `dx` measures
  become genuinely different fields on their spaces.
- Deliverable: stretched vertical grids in nonhydro2 (standalone
  win; validates the whole metric-field mechanism).

### C1 — `CoordinateMapping` + `grid.metric` + metric derivative kinds

- Implement `CoordinateMapping` per its class spec (analytic maps
  and supplied-metrics forms), **plus the CS-D1 embedding form**
  (chart into ambient coordinates; metrics via autodiff). The
  embedding form is an amendment to record in
  [`classes/grid.md`](../../specs/grid/classes/grid.md).
- `grid.metric(space, name, params=...)`: per-staggered-space
  derivation, one owner for staggered consistency (H at u-, v-,
  w-points).
- Register the metric-coefficient derivative kinds (constant-z vs
  constant-sigma) as dispatch composites — sketch 4.4 made real:
  `ddx_z = fd["x"] - c * fd["sigma"]` with `c` a dynamic leaf.
- The `params=` overload accepts traced fields from day one (CS-D4
  needs dynamic metrics; the spec already forbids operators from
  caching them).

### C2 — chart-embedded manifolds; spherical shallow water

- Embedding-form metrics (`g_ij`, `g^ij`, `sqrt_g`) as
  `grid.metric` entries; off-diagonal terms supported (sums of
  separable operators with field coefficients).
- Metric-aware `grad`/`div`/`curl`/`laplacian` dispatch entries
  (`div u = (1/sqrt_g) d_i(sqrt_g u^i)`, Laplace–Beltrami);
  `Variance` tags + `RaiseIndex`/`LowerIndex`
  ([`operators_composed.md`](../../specs/grid/classes/operators_composed.md)).
- **Torus first** as the curved-manifold validation case (closed,
  regular metric, FFT along both axes, no poles), then the sphere
  as circle x bounded-lat with polar-cap boundaries.
- Spherical shallow water: Coriolis `2 Omega sin(lat)` as a
  `Profile("lat")` field; metric-aware rework of the Sadourny
  module (every flux/vorticity/KE expression); pole-cap BC choice.
  No elliptic solve exists in SW, so C2 does not depend on the
  CS-D2 solver.
- Eigenmode/transform machinery is **not** generalized here
  (Fourier-periodic Cartesian assumptions); spherical runs
  initialize without eigenmode-based ICs for now.

### C3 — boundary-fitted / terrain-following nonhydro (static)

- Static mappings: `z = sigma * H(x, y)` (target b) and
  `y = Y_N(x, z)` (target c) — the same machinery applied to a
  vertical and a horizontal coordinate.
- Tendency terms through the constant-z derivative kinds.
- **Pressure solve (CS-D2)**: matrix-free PCG; operator = the
  mapped `Div @ metric @ Grad` composite; preconditioner = the
  separable spectral symbol inverse; fixed iterations; mean
  projection for the null space. Mimetic check: flux-form
  discretization with Jacobian weights so div and grad are
  negative adjoints under the sqrt(g)-weighted inner product
  (SPD-ness is what licenses CG).
- Validation: **mapped-flat identity** (constant H / straight
  boundary through the mapping reproduces the unmapped run to
  rounding); sloped cases cross-checked against an
  `ImmersedDomain` staircase reference; convergence order.

### C4 — dynamic metrics + optional ALE (CS-D4)

- Time-dependent mapping parameters as module-owned state through
  `grid.metric(..., params=...)`; recomputed every step, traced.
- The ALE tendency module (grid-velocity corrections), assembled
  explicitly like any module — **omitting it is the off switch**;
  docstring/docs carry the correctness caveat.
- Validation: the sloped-to-flat northern-boundary morph; frozen
  motion (`x_grid_dot = 0`) reduces to the static C3 result
  bitwise; with ALE enabled, tracer/volume consistency during the
  morph.

## 5. Gates

- Mirrored tests + `ruff` per AGENTS.md at every stage; forced-4
  multi-device suite for new operator rows.
- Identity gates: uniform meshes results-neutral through the C0
  swap; mapped-flat == unmapped (C3); frozen-motion == static (C4).
- jit stability: sweeping mapping parameters (H fields, morph time)
  compiles **once** (compile-counter test); `jax.grad` through
  metric params runs.
- PCG: iteration count on reference sloped cases recorded and
  budgeted; preconditioner effectiveness (iterations roughly
  resolution-independent) asserted in tests.
- Torus/sphere numerics: Laplace–Beltrami eigenfunctions, solid-body
  rotation divergence-free, SW conservation diagnostics.

## 6. Out of scope (designed-for, not precluded)

- Multi-chart atlases (full sphere via cubed sphere / Yin-Yang,
  genus >= 2 surfaces): chart-boundary exchange is structurally a
  halo exchange with a transition map; keep metric derivation
  per-chart so the extension stays additive.
- Geometric multigrid (CS-D2 fallback preconditioner).
- Spherical nonhydro: needs C2 metrics + C3 solver; natural
  follow-up once both land.
- Eigenmode/state-transform machinery on mapped/curved grids.
- Unstructured meshes (unchanged ROADMAP stance).
