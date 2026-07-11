---
status: normative
date: 2026-07-06
---

# Grid abstraction redesign — Paper validation

Part of the grid redesign notes; see [`00_overview.md`](00_overview.md)
for the document map. Concepts are in
[`01_concepts.md`](01_concepts.md), rules in
[`02_rules.md`](02_rules.md).

---

## 6. Paper validation against future grid types

Iteration 1 implements the uniform FD/FV tensor grid at feature parity
with the current cartesian grid, plus the full coefficient-space layer
(Fourier, DST/DCT, Chebyshev collocation incl. `ChebyshevMesh`). The Shen/Galerkin *BC bases* of 6.2 and
everything in 6.3–6.5 remain design-for-only. The abstraction is
validated on paper against five future grid types; each names the API
pressure points it creates.

### 6.1 Fourier x Fourier (pure spectral, collocated Galerkin)

- Expression: both factors are `Fourier` coefficient spaces; the model
  state lives in coefficient space permanently. This is a spectral
  Galerkin grid, and it is **collocated (un-staggered)**: all fields
  share a single Fourier space (one origin), since staggering is an
  FD/FV device — avoiding pressure-velocity decoupling — that spectral
  methods do not need.
- `diff` dispatches to `SpectralDerivative` (multiply by `i k`). There
  is **no interpolation and no phase shift** on this grid: a single
  un-staggered Fourier space has no staggered positions to convert
  between, so the current `DummyInterpolation` simply disappears (it
  exists today only because the FD-oriented framework routes every field
  through position/interpolation machinery even when that is the
  identity). The per-origin `PhaseShift`
  ([section 3.2](02_rules.md#32-coefficient-representations-are-separate-spaces),
  sketch [4.3](03_api_sketches.md#43-transform-round-trip-with-per-origin-coefficient-spaces))
  is the machinery for *mixed* grids that deliberately stagger, not for
  this one.
- Initial conditions and forcing enter via `discretize` (sample at
  the origin's nodes + transform,
  [section 3.10](02_rules.md#310-discretizing-continuous-functions)) — standard
  pseudo-spectral practice. Spectral-native constructions (random
  spectra, prescribed-phase ICs) use `init_coeff=`/`data=` and
  `grid.random` instead
  ([section 3.10](02_rules.md#310-discretizing-continuous-functions),
  sketch [4.9](03_api_sketches.md#49-random-spectra-initial-condition-spectral-space-construction)).
- Pressure points:
  - **dealiasing policy** for products (transform to nodal, pad/trim,
    multiply, transform back) must be expressible as a product policy
    of the space rather than per-call `FFTPadding` flags;
  - nonlinear modules need a clean idiom for "temporarily nodal"
    fields;
  - masked domains have no spectral route except **volume
    penalization**
    ([section 3.7](02_rules.md#37-boundaries-ii-immersed-masked-domains)):
    coastline + Fourier means a
    penalization forcing module, not mask-aware operators.

### 6.2 Uniform FV x Chebyshev-Galerkin

- Expression: `IntervalMesh ⊗ ChebyshevMesh`; vertical fields on
  Shen-type bases with BCs baked in; `f.diff("z")` is a coefficient
  recurrence, `f.diff("x")` an FD stencil.
- Decomposition: shard x (halo), keep z local — exercised by the
  per-mesh negotiation.
- Pressure points:
  - **no vertical staggering exists in Galerkin**: modules written for
    C-grids (e.g. `LinearTendency` interpolating w in z) must express
    "interpolate to space S" and receive an identity/exact operator
    when S is already right — module code must be written against
    spaces, never against CENTER/FACE assumptions;
   - implicit vertical solves (Helmholtz per column) need
     operator-eigenvalue/assembly access along one axis;
  - quadrature-based integrals differ per factor (uniform weights vs
    Clenshaw-Curtis) — `integrate` must compose per-mesh weights;
  - Shen bases with two boundary conditions have n - 1 free modes
    (n cells -> n + 1 Lobatto modes -> 2 constraints) — a
    shape example
    ([section 3.5](02_rules.md#35-shape-is-a-property-of-the-space)) that
    vertical solves and transforms
    must carry.

### 6.3 Sphere / curvilinear

- Expression: a single 2D `SphereMesh` contributing two coordinate
  names (`"lon"`, `"lat"`) and owning metric terms.
- Pressure points:
  - **operators become non-separable** on the mesh: `grad`/`div` are
    metric-aware operators registered for sphere-mesh spaces, not
    compositions of 1D `diff` — the generic dispatch registry must
    allow mesh-level (not just axis-level) operator kinds;
  - **vector fields need the metric**, but `VectorField` stays thin
    (section [2.4](01_concepts.md#24-field)): component variance
    (covariant/contravariant) is a space attribute, and index
    raising/lowering plus metric-aware `grad`/`div`/`curl` are explicit
    metric-consuming operators reading the grid-owned metric — the
    vector carries no metric;
  - a 2D mesh contributes several coordinate *names*: `init=`
    functions receive `lon` and `lat` keywords from the one mesh
    ([section 3.10](02_rules.md#310-discretizing-continuous-functions)).

### 6.4 Unstructured horizontal x structured vertical

- Expression: 2D `UnstructuredMesh` (node sets: vertices, edges,
  cells; edge-normal velocity space for triangular C-grids) ⊗ 1D
  vertical mesh — the prism structure of ICON/FESOM.
- Dispatch and strict algebra carry over unchanged: spaces are just
  richer (vertex/edge/cell instead of center/right/outer/inner).
  Vertex/edge/cell node sets also show why spaces have no relational
  properties
  ([section 2.2](01_concepts.md#22-functionspace--where-discretization-strategy-lives)):
  there is no shift involution on an
  unstructured mesh — staggering transitions exist only as operator
  signatures (`div: edge-normal -> cell`).
- Pressure points:
  - decomposition requires **graph partitioning** and indirect-neighbor
    halos — a second decomposition backend besides jaxDecomp;
  - no tensor-product transform along the unstructured factor; solvers
    need iterative/matrix paths;
  - plotting/xarray export need a non-boxed data model (per-DOF
    coordinate arrays via `grid.evaluation_nodes(space)`).

### 6.5 Terrain-following vertical coordinate

- Expression: `uniform(x, y) ⊗ mapped(z)` with grid-level metric
  fields
  ([section 3.8](02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted));
  topological product intact, geometry coupled.
- Pressure points:
  - **metric fields as operator coefficients**: the operator model
    must support field-coefficient compositions, resolved by dispatch
    (constant-z vs constant-sigma derivative kinds);
  - metric fields must exist at staggered spaces (H at u-, v-,
    w-points) — derived consistently by the grid, not per module;
  - **dynamic metrics** (z*): metric fields are pytree-dynamic; jit
    must not bake them in, and operators may not cache them
    statically;
  - vertical integrals need Jacobian weights (dz = H dsigma), meeting
    the quadrature machinery of sections
    [2.2](01_concepts.md#22-functionspace--where-discretization-strategy-lives) and
    [3.5](02_rules.md#35-shape-is-a-property-of-the-space).

Design-for, do not implement: nothing in iteration 1 may assume all
meshes are 1D, that any space has a canonical transform, that
operators are separable, or that metric fields are static.
