# Grid abstraction redesign — Core concepts

Part of the grid redesign notes; see [`00_overview.md`](00_overview.md)
for the document map and motivation.

---

## 2. Core concepts

The design decomposes the grid into six concepts, layered bottom-up:

```
Mesh                atomic factor of the domain (geometry + topology)
FunctionSpace       discrete representation of fields on one mesh
TensorProductSpace  product of per-mesh spaces, named coordinates
Field               (function_space, array) + minimal metadata
Operator            typed map between function spaces
Grid                assembly: meshes + decomposition + defaults
```

> **Terminology note.** The FRIDOM presentation slides use "mesh" for
> the staggered node collections (Omega_1, Omega_2, ...). In this
> design those are *node sets owned by function spaces*; `Mesh`
> denotes the geometric factor (geometry + topology, no
> discretization). Consequently, the current `grid.get_mesh()` (which
> returns coordinate *arrays*) is **removed** — not to be confused with
> the new `grid.factors` property ([section 2.6](#26-grid--the-assembly-object)),
> which returns the `Mesh` *factor objects*: fields are constructed
> through the grid field factory — by discretizing functions of
> physical coordinates, by assigning coefficients, or from raw data
> ([section 3.10](02_rules.md#310-discretizing-continuous-functions));
> coordinate access exists only as intrinsic space
> properties (evaluation nodes, wavenumbers).

### 2.1 Mesh — atomic factor of the domain

A *mesh* is an atomic factor of the domain geometry. It owns only the
**static geometry descriptors** of one factor — extent, periodicity,
cell count, cell/vertex structure (hence `dx`), topology, and (later)
metric structure. It does *not* store per-space evaluation nodes or
materialized coordinate meshgrids; those are derived, dynamic, and
grid-owned ([section 2.7](#27-where-coordinate-data-lives)).

- Meshes may have **any dimension >= 1**, but geometry is factored down
  to 1D meshes whenever possible. The canonical uniform 3D grid is a
  product of three 1D meshes. An unstructured horizontal mesh or a
  sphere is a single 2D mesh.
- The mesh owns cell count and geometry only; **DOF shapes belong to
  function spaces**
  ([section 3.5](02_rules.md#35-shape-is-a-property-of-the-space)):
  different spaces on the same mesh
  generally have different shapes.
- Planned mesh types (only the first is implemented in iteration 1):
  - `IntervalMesh(shape, extent, periodic)` — uniform 1D
  - `MappedIntervalMesh` — stretched 1D (vertical coordinates)
  - `ChebyshevMesh(shape, extent)` — Gauss-Lobatto nodes
  - `SphereMesh` — 2D, metric terms
  - `UnstructuredMesh` — 2D triangular
- The mesh is the **factory and registry of its function spaces**
  (section 2.2): spaces are created through the mesh and interned, so
  two requests for "the center space of mesh m" return the same
  object.
- The mesh declares its **decomposition traits**: whether it can be
  sharded across devices, with what halo strategy (ghost cells,
  transpose-based transforms, graph partition), see
  [section 5](04_decomposition.md#5-domain-decomposition).
- Every mesh exposes its **conforming boundary** as a mesh of
  dimension d - 1 (`mesh.boundary`: a 2-point 0D mesh for an interval,
  a closed curve for a 2D mesh). Boundary data therefore needs no new
  data type — it consists of ordinary fields on trace product spaces
  ([section 3.6](02_rules.md#36-boundaries-i-conforming-bc-structure-vs-boundary-data)).
- A mesh mapping (e.g. `MappedIntervalMesh`) may only depend on the
  mesh's own coordinates; cross-factor mappings (terrain-following,
  [section 3.8](02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted))
  are grid-level metric data, not mesh structure.

### 2.2 FunctionSpace — where discretization strategy lives

A `FunctionSpace` describes *how* a continuous field is represented on
one mesh. The discretization strategy is a property of the space, not
of the grid:

| Strategy          | Space kind on a mesh                            |
|-------------------|--------------------------------------------------|
| finite difference | nodal values at a node set (`Center`, `Right`, `Outer`, ...) |
| finite volume     | cell/dual-cell averages (`CellAvg`, `FaceAvg`) — distinct from nodal |
| spectral Galerkin | modal coefficients (`Fourier`, `Sine`, `Cheb`)   |
| FEM / DG          | element-local polynomials (out of scope, but not precluded) |

Notes:

- **Node sets follow the xgcm vocabulary**: `Center` (n points),
  `Left`/`Right` (cell edges on periodic meshes, n points), `Outer`
  (all faces, n + 1), `Inner` (interior faces, n - 1). The names
  `Cell`/`Face` are deliberately not used; the current FRIDOM `FACE`
  corresponds to `Right`. This maps one-to-one to xarray/xgcm
  staggered-coordinate export.
- Nodal and cell-average spaces are **distinct** even though they store
  one number per cell: high-order reconstruction (Shu) differs between
  point values and cell means. Today this distinction is a flag inside
  `PolynomialInterpolation`; it becomes a type.
- **Averages have no position.** A `CellAvg` DOF is a functional over
  the cell (`(1/dx) int_cell u dx`), not a value at a point;
  identifying it with the center value is a second-order
  approximation. Coordinate labels (centers for `CellAvg`, faces for
  `FaceAvg`) are export metadata, not mathematical positions. The
  average family mirrors the nodal family: `CellAvg` lives on primal
  cells, `FaceAvg` on dual cells, exactly as `Center` and `Right`
  live on primal and dual nodes
  ([section 3.9](02_rules.md#39-finite-volume-semantics-the-average-family-and-the-fv-derivative)).
- **Staggering is a choice of space per variable.** A C-grid velocity
  u lives on `Right(x) ⊗ Center(y) ⊗ Center(z)`; B-grid velocities
  live on `Right(x) ⊗ Right(y)`; a triangular C-grid velocity lives on
  the "edge-normal" space of the unstructured mesh. The per-axis
  `Position`/`AxisPosition` enums disappear from field metadata.
- **The homogeneous BC structure is baked into the space** (for
  Galerkin bases: Shen-type bases with built-in BCs; for nodal spaces:
  a BC attribute); it determines the free DOFs, the shape, and the
  admissible coefficient bases. Boundary *data* (inflow profiles,
  prescribed fluxes) is dynamic and per-field
  ([section 3.6](02_rules.md#36-boundaries-i-conforming-bc-structure-vs-boundary-data)).
  `bc_types` disappears from field metadata.
- `ConstantSpace` is a one-DOF space replacing `topo=False` axes
  ([section 3.3](02_rules.md#33-constantspace-replaces-topo-with-automatic-broadcast)).
- **A space's *defining* attributes are static descriptors**:
  `space.shape`
  ([section 3.5](02_rules.md#35-shape-is-a-property-of-the-space)),
  the node set or basis, the BC structure, the **scalars** it is
  defined over (`space.scalars`, `fr.Real` or `fr.Complex`, default
  `fr.Real` — the field of scalars / *Körper*,
  [section 3.1](02_rules.md#31-strict-space-algebra)), and — for
  coefficient spaces — their origin
  ([section 3.2](02_rules.md#32-coefficient-representations-are-separate-spaces)).
  These are the hashable jit/dispatch key, so real and complex variants
  of a space are distinct (and dispatch to distinct default transforms,
  rfft vs fft). Its *derived* coordinate
  quantities — evaluation nodes, wavenumbers, metric measures — are
  **not** stored on the space; they are dynamic, device-sharded
  `ScalarField`s (`.data` for the raw array) the grid materializes on
  demand through grid-mediated accessors
  (`grid.evaluation_nodes(space)`, `grid.wavenumbers(space)`; the space
  itself holds no grid reference)
  ([section 2.7](#27-where-coordinate-data-lives),
  [section 3.10](02_rules.md#310-discretizing-continuous-functions)).
  Spaces have *no relational properties*: all relations between spaces are
  **operator signatures**, and all default choices are **dispatch
  entries**
  ([section 3.4](02_rules.md#34-generic-operator-dispatch)). Two designs
  were considered and rejected:
  - `space.shift()` (a staggering-partner involution): ambiguous on
    bounded meshes (`Center` -> `Right`, `Outer`, or `Inner`?) and
    undefined for vertex/edge/cell node sets on unstructured meshes.
    Operators know their codomain (`diff: Center(x) -> Right(x)`);
    modules name target spaces explicitly.
  - `space.spectral` (a canonical coefficient dual): a space admits
    infinitely many coefficient bases (sine, cosine, Chebyshev, ...);
    "spectral" is only meaningful relative to an operator whose
    eigenbasis it is. The default transform is a dispatch entry
    `("transform", space)`, not a space property.
- Spaces are **static, hashable, and interned**. They serve as jit
  cache keys and dispatch keys; equality checks (`f + g` legality) are
  identity comparisons. Field arrays are the dynamic pytree leaves; the
  *structure* — spaces, operators, grids — is static
  ([section 2.7](#27-where-coordinate-data-lives)). The grid's
  materialized coordinate/metric arrays are *transient trace-time
  values* (recomputed on demand, never stored), and time-dependent
  geometry data lives in module-owned state fields — the grid itself
  carries no dynamic pytree leaves (class-design decision; see
  [`classes/04_grid_and_decomposition.md`](classes/04_grid_and_decomposition.md)).

An explicit FEEC-style "discrete de Rham complex" object is *not*
introduced; the staggering relations remain implicit in the space
family of a mesh. Naming should stay FEEC-compatible so this can be
revisited.

### 2.3 TensorProductSpace and named coordinates

A field on the full domain lives on a `TensorProductSpace`, one factor
space per mesh. Coordinates are addressed **by name** (`"x"`, `"y"`,
`"z"`; a 2D mesh contributes several names, e.g. `"lon"`, `"lat"`).

**The product is associative and flat: it normalizes to a single tuple
of per-mesh factor spaces, and no factor is itself a
`TensorProductSpace`.** Nesting is collapsed on construction, so if
`uniform(x, y)` is `interval(x) ⊗ interval(y)`, then
`uniform(x, y) ⊗ chebyshev(z)` *is*
`interval(x) ⊗ interval(y) ⊗ chebyshev(z)`, never
`(interval(x) ⊗ interval(y)) ⊗ chebyshev(z)`. This keeps the
coordinate-name namespace flat (names must be unique across factors,
so the product rejects duplicates), makes spaces interned so equality
is identity comparison
([section 2.2](#22-functionspace--where-discretization-strategy-lives)),
and lets operator dispatch
([section 3.4](02_rules.md#34-generic-operator-dispatch)) and
decomposition ([section 5](04_decomposition.md#5-domain-decomposition))
stay strictly per mesh. Flatness is at the **mesh** level, not the axis
level: a factor lives on an atomic mesh, which is 1D whenever geometry
factors ([section 2.1](#21-mesh--atomic-factor-of-the-domain)) but
stays a single, non-decomposable 2D factor for a sphere or unstructured
mesh — such a factor contributes several coordinate names yet remains
one entry in the flat tuple.

The payoff of the product structure: separable operators factor.
`d/dx` on a product space is `(mesh-x operator) ⊗ identity`, so a
mixed grid `uniform(x, y) ⊗ chebyshev(z)` needs no special code — each
axis brings its own operator family.

Two qualifications:

- The tensor product is **topological, not necessarily geometric**:
  DOF layout, shapes, staggering, and sharding always factor, but the
  geometry may be coupled across factors through **metric fields**
  (terrain-following coordinates,
  [section 3.8](02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted);
  curvilinear meshes,
  [section 6.3](05_validation.md#63-sphere--curvilinear)). Metric/Jacobian
  terms are ordinary fields on product
  spaces, and operators may carry field coefficients.
- **Coordinate mappings — and hence metric fields — may be
  time-dependent** (free-surface-following z*, moving meshes). Metric
  fields are then dynamic arrays like any other field data; no
  operator implementation may assume static metrics.

### 2.4 Field

- `ScalarField = (function_space, array) + metadata`. Metadata shrinks
  to name/units/nc-attrs. `position`, `bc_types`, `topo`, and
  `is_spectral` are all subsumed by the function space.
- **The array dtype is derived, not stored as a flag.** It follows the
  space's `scalars` and basis (`fr.utils.dtype_real()` /
  `dtype_comp()`): `is_spectral`'s old double duty — marking both the
  representation *and* the dtype — splits into the coefficient space
  ([section 3.2](02_rules.md#32-coefficient-representations-are-separate-spaces))
  and the scalars/Körper
  ([section 3.1](02_rules.md#31-strict-space-algebra)). `f.as_complex()`
  promotes a real field; `f.real` / `f.imag` extract the two real parts
  of a complex one.
- `VectorField` / `TensorField` are collections of scalars living on
  *different but related* spaces — the type system embraces that
  C-grid components have different spaces rather than fighting it.
- **`VectorField` is thin: it carries no metric.** The metric lives
  where the design already puts it — in operators (metric-aware
  `grad`/`div`/`curl` registered per mesh, sections 3.4,
  [6.3](05_validation.md#63-sphere--curvilinear)) and in metric fields
  on product spaces (section 2.3). **Component variance (covariant vs
  contravariant) is a property of the component's *space*** — distinct
  spaces, exactly like distinct staggered spaces — so the strict
  algebra (section 3.1) catches mixing variances, and **raising/lowering
  indices is an explicit metric-consuming operator** reading the
  grid-owned metric, not a method on the vector. One metric owner (the
  grid), and the sphere needs no metric-aware vector type.
- **`VectorField` supports `VectorField.map(fn)`**: it applies `fn` to
  each component field on its own space and returns a new collection.
  This is the functional-map surface consumed by spectra-based initial
  conditions (sketch
  [4.9](03_api_sketches.md#49-random-spectra-initial-condition-spectral-space-construction))
  and by the model-side eigenmode objects
  ([section 2.5](#25-operator--typed-maps-between-spaces)).
- **`State` *is* a `VectorField`** (a subclass): it is the state vector
  of a model, so it inherits component iteration and `map` but adds
  model-level semantics and user-defined diagnostics (energy, potential
  vorticity, ...). Where this design says `State.map`, the surface is
  the inherited `VectorField.map`; `State` is the physics-carrying
  specialization.

### 2.5 Operator — typed maps between spaces

Operators are **free-standing, parameterized objects** with an explicit
signature, not grid methods and not module-owned slots:

- signature: `(domain_space, codomain_space)` (resolved per axis for
  separable operators),
- per-axis requirements: halo width, shardability constraints,
- examples: `FiniteDifference(order=2)`, `LinearInterp()`,
  `WenoReconstruction(order=5)`, `Fourier()`,
  `PhaseShift()`, `SpectralDerivative()`.

Operators form an **algebra** — composition `C = A @ B`, sums with
scalar/field coefficients, axis binding `op["x"]`, and tuple
(direct-sum) signatures for vector/tensor-valued maps with `@` as
block-matrix multiplication — designed in the sibling note set
[`operator_algebra/`](operator_algebra/00_overview.md)
([section 3](operator_algebra/02_algebra.md) there); composites are
ordinary operators (registrable, halo-accountable, symbol-bearing).

Operators are **callable**: applying one to its operands is
`op(field, axis=...)` (unary) or `op(f, g)` (binary). The `axis`
keyword is **optional**: a separable 1D kernel applied to a multi-axis
field needs `op(field, axis="x")` to name the factor it acts on, but it
is omittable when the operator's domain is already a full product space
(a composed `grad`, a binary product) — there the axis is fixed by the
signature. Most operators are unary, but the signature generalizes to
**binary operators** `(domain_a, domain_b) -> codomain` for products
([section 3.11](02_rules.md#311-field-operations-linear-ops-and-the-product-problem)):
`CollocationProduct`, `Convolution`, and `Hadamard` are binary
operators, and the infix `f * g` is sugar for the dispatched
([section 3.4](02_rules.md#34-generic-operator-dispatch)) default
physical product. `*` is overloaded, disambiguated by operand type:

| Left `*` Right      | Meaning                                        |
|---------------------|------------------------------------------------|
| space `*` space     | tensor product (`mx.center * mz.galerkin`), section 2.3 |
| field `*` field     | physical (pointwise) product, dispatched per space (section 3.11) |
| `Symbol` `*` `Symbol` | diagonal (elementwise) composition of diagonal operators (section 3.11) |

The same table governs `**`: `k ** 2` on a `Symbol` is diagonal, `f ** 2`
on a field is the physical product.

Transforms are the one deliberate exception to "free-standing": they
bind the grid at construction (`fr.operators.Fourier(grid, axes=...)`)
and are applied through `.forward`/`.backward` rather than
`op(field, axis=...)`, because they need the domain decomposition and
FFT plan up front (sketch
[4.3](03_api_sketches.md#43-transform-round-trip-with-per-origin-coefficient-spaces)).
Stencil operators stay grid-free and take the grid only at
`.eigenvalues(grid, space)` time.

Two design points:

1. **Transforms are ordinary operators.** `fft` is nothing special:
   it maps a nodal space to a coefficient space
   ([section 3.2](02_rules.md#32-coefficient-representations-are-separate-spaces)).
   `FFTPadding`-style dealiasing is a property of the transform — a
   degree-p pad factor selecting a padded transform into a finer nodal
   space — not of every call site
   ([section 3.12](02_rules.md#312-dealiasing)).
2. **Operators may expose their eigenvalues — relative to a
   diagonalizing basis** (the operator's *symbol*, in the
   pseudo-differential sense). For a separable, translation-invariant
   operator the coefficient basis functions are eigenfunctions, so the
   operator acts as multiplication by an eigenvalue per mode. Three
   points fix the API:
   - **Queried per coefficient factor space**, not per axis:
     `op.eigenvalues(grid, coeff_space)` (grid-mediated, since the
     eigenvalues derive from the grid-materialized wavenumbers,
     [section 2.7](#27-where-coordinate-data-lives)). The operator is
     *axis-agnostic* — a `FiniteDifference` is a separable 1D kernel
     that supplies only its stencil (from `order`); the factor space
     supplies the axis, its spacing `dx` (via the mesh), and the
     wavenumbers (`grid.wavenumbers(space)`,
     [section 3.10](02_rules.md#310-discretizing-continuous-functions)).
     There is no `(kx, ky, kz)` tuple: a multi-axis operator's
     eigenvalues are **composed** from the per-factor ones (the
     discrete Laplacian is the broadcast sum over factors, sketch
     [4.6](03_api_sketches.md#46-operator-eigenvalues-for-exact-spectral-solvers)).
   - **Returns a `Symbol`** — a diagonal operator on the coefficient
     space, *not* a `ScalarField`. On a `Symbol` the operations `*`,
     `**`, `+`, and `1 / .` are the **diagonal (elementwise) algebra**
     (composition/inverse of diagonal operators), so `k ** 2` and
     `1 / lap` are well-defined and never the physical
     product/convolution that a `ScalarField`'s `*` would denote; a
     `Symbol` is *applied* to a field as a `Hadamard` multiply (it is
     callable, section 3.11). Per-factor symbols broadcast across the
     product via `ConstantSpace` (section 3.3), so the composed
     eigenvalues and the spectral solve stay inside the strict algebra.
     The same `Symbol` type carries other diagonal operators — spectral
     filters, the 2/3 truncation mask (section 3.12), inter-origin
     phase shifts, and the `sinc(k dx / 2)` cell-averaging factor
     (section 3.2). It is distinct from the coordinate accessors
     `grid.wavenumbers(space)` / `grid.evaluation_nodes(space)`, which
     stay `ScalarField`s of *sampled coordinate values* (section 2.7):
     a `Symbol` is built from their `.data`.
   - **Defined only relative to a diagonalizing basis.**
     `fd.eigenvalues(grid, fourier_x)` exists (Fourier diagonalizes a
     constant-coefficient FD on a periodic mesh);
     `fd.eigenvalues(grid, chebyshev_z)` does not. Eigenvalue-based
     exact
     solves therefore require *every* factor to diagonalize the
     operator (fully periodic grids); mixed grids fall back to banded
     per-column solves.

   This gives the homeless `discrete_spectral_operators` module a home:
   spectral pressure solvers and eigenmode machinery build *exact
   inverses of the discrete operators* from the eigenvalues instead of
   hardcoding `k_hat` variants.

**Eigenmode objects are model-side and out of scope for the grid
redesign.** The `omega`/`vec_q`/`vec_p` successor is a model-side object
(sketch
[4.9](03_api_sketches.md#49-random-spectra-initial-condition-spectral-space-construction))
that will be rewritten independently; the grid classes do not have to
worry about it. The grid supplies only the *primitives* it consumes —
operator eigenvalues as `Symbol`s (above), per-origin coefficient spaces
([section 3.2](02_rules.md#32-coefficient-representations-are-separate-spaces)),
`grid.wavenumbers`, `grid.random`, and `State.map`
([section 2.4](#24-field)). The eigenvector/projector **reuse `State`**
(component fields on their per-variable coefficient spaces, with the
inter-origin phase shifts baked into the discrete eigenvectors); `omega`
is a `Symbol`; per-mode normalization (energy 1) lives inside the
eigenmode object where the parameters do. A `Symbol` is the per-operator
eigenvalue primitive and does **not** subsume the eigenmode object,
which is a State-valued, physics-parameterized assembly built *from*
Symbols.

### 2.6 Grid — the assembly object

What remains of `Grid` is ergonomics and wiring:

- the tuple of meshes and coordinate names, exposed through the
  `grid.factors` property (the `Mesh` factor objects, in `names` order —
  the successor of the removed coordinate-array `get_mesh()`; the name is
  deliberately distinct from `get_mesh()` to avoid the near-collision,
  [section 2.1](#21-mesh--atomic-factor-of-the-domain)),
- **two constructors, not one overloaded signature**: `fr.Grid` is the
  model-agnostic assembly root, taking `meshes=` (pre-built meshes of
  any type, incl. sphere / unstructured; coordinate names are mandatory
  mesh-constructor arguments, so the grid only collects and validates
  them — class-design decision);
  `fr.grid.cartesian.Grid` is a convenience **subclass** taking
  `shape=`/`extent=`/`periodic=` and building the uniform `IntervalMesh`
  factors internally (no `N`/`L`). Splitting the two forms across a base
  class and a subclass avoids a single constructor that silently accepts
  two disjoint kwarg sets,
- the domain decomposition (negotiated per mesh,
  [section 5](04_decomposition.md#5-domain-decomposition)),
- the **operator dispatch registry**
  ([section 3.4](02_rules.md#34-generic-operator-dispatch)),
- the immersed domain — `grid.immersed`, an `ImmersedDomain` (a static
  descriptor) materializing the wet volume fraction and the per-space
  masks/fractions on demand; time-dependent geometry data is
  module-owned state
  ([section 3.7](02_rules.md#37-boundaries-ii-immersed-masked-domains)),
- the field factory (`grid.create_field(space, init=...)`,
  [section 3.10](02_rules.md#310-discretizing-continuous-functions)).

All mathematics lives in spaces and operators. In particular, **model
physics leaves the grid**: `omega`, `vec_q`, `vec_p` become model-side
eigenmode objects that consume `(grid, parameters)`. Models no longer
subclass grids.

### 2.7 Where coordinate data lives

Coordinate data (evaluation nodes, wavenumbers, quadrature weights,
metric terms) splits along a **structure vs data** line: the *structure*
that describes geometry is static and lives on the mesh and space; the
*arrays* are dynamic, device-sharded, and owned by the grid. This
replaces today's monolithic grid, which stores materialized, sharded
`x_mesh`/`k_mesh` meshgrids and rebuilds staggered variants on demand.

| Lives on | What | Status |
|----------|------|--------|
| **Mesh** ([2.1](#21-mesh--atomic-factor-of-the-domain)) | extent, periodicity, cell count, cell/vertex structure (source of `dx`), topology; 2D-mesh geometry + metric structure | static descriptor |
| **Space** ([2.2](#22-functionspace--where-discretization-strategy-lives)) | mesh-ref, node set / basis, shape, BC, origin | static descriptor (jit/dispatch key) |
| **Grid** ([2.6](#26-grid--the-assembly-object)) | `evaluation_nodes`, `wavenumbers`, metric measures (`dx`, cell widths, Jacobians) | dynamic, device-sharded `ScalarField`s (`.data` = raw array) |

Rules:

- **Materialization is grid-mediated; spaces hold no grid reference.**
  A space is a grid-free static key, so it cannot shard by itself; the
  **grid is the entry point** for every derived array —
  `grid.evaluation_nodes(space)`, `grid.wavenumbers(space)`,
  `grid.random.normal(space, seed)`, and operator eigenvalues
  `op.eigenvalues(grid, space)`
  ([section 2.5](#25-operator--typed-maps-between-spaces)). The grid
  owns the decomposition; the space is only the key. A **field**, being
  a grid-created instance, carries its grid (it needs it for `f.diff`
  dispatch anyway, [section 3.4](02_rules.md#34-generic-operator-dispatch)),
  so these calls are reachable from any field.
- **The derived quantities are `ScalarField`s, never stored
  statically.** `grid.evaluation_nodes(space)` and
  `grid.wavenumbers(space)` are *derived* per space (different spaces on
  one mesh have different node sets/shapes, so they cannot be a single
  mesh array) and returned as **`ScalarField`s tagged with that space**
  — one per named axis, queried per factor and broadcast into the
  product via `ConstantSpace` (section 3.3), consistent with operator
  eigenvalues. `.data` exposes the raw array. They are recomputed on
  demand from the static `(extent, n, basis)` descriptors via sharded
  `linspace`/`fftfreq`, to the **local shard matching field layout**
  ([section 5](04_decomposition.md#5-domain-decomposition)).
  Recompute-on-demand keeps the grid free of persistent array state, so
  a renegotiated decomposition can never leave a stale shard. (A
  higher-order rule's *within-cell* quadrature points/weights have more
  entries than DOFs, so those stay static reference-rule data, not
  fields; only the aggregate cell measure below is a field.)
  Recompute-on-demand is the **semantics**; the baseline cost story is
  XLA's CSE and loop-invariant code motion (adequate for `iota`-based
  uniform nodes). For large stretched/mapped node arrays the **opt-in
  performance knob** is to materialize once *outside* the scanned
  region and close over the array (a real store, at the price of
  resident memory) — `jax.checkpoint` is the opposite trade
  (rematerialization) and does not apply. The knob is invisible to the
  abstraction, bound by the same re-decomposition-invalidation rule —
  never a semantic change — and mapped-mesh materialization carries a
  compile-time benchmark item (constant-folding large traced mappings
  is a known slow-constant-folding trigger).
- **Operators derive spacing-dependent stencil coefficients
  dynamically, never as hardcoded constants.** An operator's
  *structure* (order, stencil width, neighbor pattern) is static and
  part of its dispatch identity; its *coefficient values* on a
  stretched/mapped mesh are dynamic arrays derived from the grid's
  metric fields (the `dx` measures below) at trace time and traced
  through jit — the "operators carry field coefficients" rule
  ([section 2.3](#23-tensorproductspace-and-named-coordinates)) applied
  to the base separable case. A uniform mesh is the special case where
  that field is constant and XLA constant-folds it; no operator bakes
  `dx`/node positions into Python constants. (Whether repeated
  materializations warrant memoization is the benchmark question of the
  materialize-outside-scan knob above — the goal is to avoid
  unnecessary recomputation without reintroducing stale state.)
- **The N-D meshgrid is never stored.** It is materialized transiently
  by broadcasting the 1D-per-factor `evaluation_nodes` inside the field
  factory / `discretize`
  ([section 3.10](02_rules.md#310-discretizing-continuous-functions));
  separability makes this O(n) per factor, not O(n^d).
- **`dx` is a staggered metric field, not a scalar.** Grid spacing is
  a metric *measure*, and — like staggering
  ([section 2.2](#22-functionspace--where-discretization-strategy-lives))
  — the staggered space it lives on is set by *which* measure it is:
  the primal **cell width** `x_{i+1/2} - x_{i-1/2}` is a field on
  `Center`/`CellAvg` (the FV integration weight; the denominator of
  `flux_diff: Outer -> CellAvg`,
  [section 3.9](02_rules.md#39-finite-volume-semantics-the-average-family-and-the-fv-derivative)),
  while the dual **center-to-center spacing** `x_{i+1} - x_i` is a field
  on `Right`/`Outer`/`FaceAvg` (the denominator of
  `diff: Center -> Right`). On a uniform mesh these collapse to one
  constant (hence today's single `dx` scalar per axis); on a
  stretched/mapped mesh they genuinely differ and live on different
  spaces. This is the metric-field machinery
  ([section 3.8](02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted))
  applied to the base separable case, not only to terrain-following
  coordinates.
- **Coupled or time-dependent geometry is a field, not a descriptor.**
  Metric fields, Jacobians, terrain `H`, and `z*`
  ([section 2.3](#23-tensorproductspace-and-named-coordinates),
  [section 3.8](02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted))
  are ordinary dynamic fields on product spaces — the same
  representational class as the separable coordinate arrays, just
  coupled across factors and possibly prognostic. As fields they are
  **owned by modules / the model state, never stored on the grid**:
  the grid holds only static descriptors and materializes derived
  arrays on demand, while metric-consuming accessors take the dynamic
  data explicitly (class-design decision; see
  [`classes/04_grid_and_decomposition.md`](classes/04_grid_and_decomposition.md)).
