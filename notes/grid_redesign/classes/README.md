# Grid abstraction redesign — Class designs

Part of the grid redesign notes; see
[`../00_overview.md`](../00_overview.md) for the document map and
motivation.

Status: **draft class design, no implementation.** This directory is
the second design phase: it turns the concept notes (sections 2–3, the
normative reference) into concrete class specifications — for every
class the module placement, constructor, and full public surface
(methods, properties, dunders) with Python signatures. It bridges
ROADMAP task 4.1 (design) to 4.2+ (implementation in the transitional
`framework.grid2` package).

All skeletons are the *intended* public API; names and signatures may
still shift during implementation, but any deviation from the concept
notes is called out explicitly. Coverage is the **full designed-for
surface**: every class/member carries an iteration tag (`1` = the
day-one uniform FD/FV tensor grid of
[`../07_iteration1_api.md`](../07_iteration1_api.md), `designed-for` =
specified so iteration 1 does not preclude it).

---

## Document map

| File | Cluster (classes owned) |
|------|-------------------------|
| [`01_meshes_and_spaces.md`](01_meshes_and_spaces.md) | `Mesh` family (`IntervalMesh`, `MappedIntervalMesh`, `ChebyshevMesh`, `SphereMesh`, `UnstructuredMesh`, `PointMesh`), the `FunctionSpace` families (nodal, average, coefficient, Galerkin, `ConstantSpace`), static markers (`Scalars`/`fr.Real`/`fr.Complex`, `BC`/`BCStructure`, `NodeSet`, `Variance`), space interning. |
| [`02_product_spaces_and_fields.md`](02_product_spaces_and_fields.md) | `TensorProductSpace`, `SpaceMismatchError`, `FieldMetadata`, `ScalarField`, `VectorField`, `TensorField`, `State`; the strict-algebra arithmetic surface, lifts/joins, field sugar (`diff`, `to`, `integrate`, `sel`, ...). |
| [`03_operators.md`](03_operators.md) | `Operator` hierarchy (unary/binary/separable), stencil kernels (`FiniteDifference`, interpolation/reconstruction, FV `FluxDifference`), transforms (`Fourier`, DST/DCT, `Chebyshev`), `Symbol` diagonal algebra, pointwise/product operators, reductions, composed vector calculus, and the `OperatorRegistry` dispatch class. |
| [`04_grid_and_decomposition.md`](04_grid_and_decomposition.md) | `Grid` + `grid2.cartesian.Grid`, `RandomFieldFactory`, `ImmersedDomain`, `CoordinateMapping`, decomposition (`HaloStrategy`, `MeshDecompositionTraits`, `HaloSpec`, `HaloTracer`, `ArrayLayout`, `Decomposition`/`TensorDecomposition`), and the canonical `grid2` package tree. |

## Shared template

Each doc specifies, per class:

- a one-line role plus a table of *kind* (ABC/concrete/final),
  *pytree status* (static interned key vs dynamic leaves, jaxify
  treatment), *iteration* (1 / designed-for), and *concept refs*
  (the normative sections implemented);
- a Python skeleton (constructor and every public member as a
  signature with a one-line docstring, no bodies);
- prose notes: semantics, invariants, space signatures
  (`domain -> codomain`), extension contracts, and rejected
  alternatives;
- a closing `## Open questions` section (only genuinely unresolved
  points; decided concept-note questions are not reopened).

## Cross-cluster seams

The four docs were cross-reviewed for consistency; the fixed seam
anchors are:

- spaces are produced by mesh factories (`mx.center`,
  `mx.fourier(origin=...)`) and combined with `*` into a flat
  `TensorProductSpace`; lone factors implement the product protocol;
- fields carry their grid; spaces and meshes hold no grid reference
  and no arrays; derived coordinate data is grid-mediated
  (`grid.evaluation_nodes(space)`, `grid.wavenumbers(space)`,
  `grid.measure(space, name=...)`);
- the single field factory is
  `grid.create_field(space, init=... | init_coeff=... | data=...)`;
  seeded randomness is `grid.random`;
- dispatch is grid-owned: `grid.dispatch` holds doc 03's
  `OperatorRegistry`; field sugar (`f.diff("x")`, `f * g`) resolves
  `(kind, space)` entries there;
- transforms bind the grid at construction
  (`fr.operators.Fourier(grid, axes=...)`, `.forward`/`.backward`);
  all other operators are grid-free until application;
- eigenvalues are per coefficient factor space:
  `op.eigenvalues(grid, coeff_space) -> Symbol`; a `Symbol` is a
  diagonal operator, never a `ScalarField`;
- decomposition traits are declared per space by meshes
  (`mesh.decomposition_traits(space) -> MeshDecompositionTraits`) and
  consumed by doc 04's negotiation; halos are per coordinate name,
  accumulated by tracing (`HaloTracer`), never a global integer.

Two normative-note amendments were made together with these docs:
the binary-product spelling in
[`../02_rules.md`](../02_rules.md#311-field-operations-linear-ops-and-the-product-problem)
is instance-call (`Hadamard()(f, a)`), and
[`../07_iteration1_api.md`](../07_iteration1_api.md#106-deferred-to-later-iterations-not-typed-by-a-day-one-user)
clarifies the iteration-1 immersed-domain subset.
