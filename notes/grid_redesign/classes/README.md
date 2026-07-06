# Grid abstraction redesign — Class designs

Part of the grid redesign notes; see
[`../00_overview.md`](../00_overview.md) for the document map and
motivation.

Status: **draft class design, no implementation.** This directory is
the second design phase: it turns the concept notes (sections 2–3, the
normative reference) into concrete class specifications — for every
class the module placement, constructor, and full public surface
(methods, properties, dunders) with Python signatures. It bridges
the design (these notes) to ROADMAP Phase 1 implementation
(tasks 1.1–1.7) in the `framework2.grid` package.

All skeletons are the *intended* public API; names and signatures may
still shift during implementation, but any deviation from the concept
notes is called out explicitly. Coverage is the **full designed-for
surface**: every class/member carries an iteration tag (`1` = the
day-one surface of
[`../07_iteration1_api.md`](../07_iteration1_api.md) — the uniform
FD/FV tensor grid at parity with the current cartesian grid plus the
full coefficient-space layer —, `designed-for` = specified so
iteration 1 does not preclude it).

---

## Document map

| File | Cluster (classes owned) |
|------|-------------------------|
| [`01_meshes_and_spaces.md`](01_meshes_and_spaces.md) | `Mesh` family (`IntervalMesh`, `MappedIntervalMesh`, `ChebyshevMesh`, `SphereMesh`, `UnstructuredMesh`, `PointMesh`), the `FunctionSpace` families (nodal, average, coefficient, Galerkin, `ConstantSpace`), static markers (`Scalars`/`fr.Real`/`fr.Complex`, `BC`/`BCStructure`, `NodeSet`, `Variance`), space interning. |
| [`02_product_spaces_and_fields.md`](02_product_spaces_and_fields.md) | `TensorProductSpace`, `SpaceMismatchError`/`GridMismatchError`, `FieldMetadata`, `ScalarField`, `VectorField`, `TensorField`, `State`; the strict-algebra arithmetic surface, lifts/joins, field sugar (`diff`, `to`, `integrate`, `sel`, ...). |
| [`03_operators.md`](03_operators.md) | `Operator` hierarchy (unary/binary/separable), stencil kernels (`FiniteDifference`, interpolation/reconstruction, FV `FluxDifference`/`DualFluxDifference`/`FaceDifference`), transforms (`Fourier`, `Sine`/`Cosine`, `Chebyshev`, padding), `Symbol` diagonal algebra, pointwise/product operators (`Where`, `ConstantBroadcast`), reductions, composed vector calculus, and the `OperatorRegistry` dispatch class. |
| [`04_grid_and_decomposition.md`](04_grid_and_decomposition.md) | `Grid` + `framework2.grid.cartesian.Grid` (lifecycle: seed -> negotiate -> freeze), `Discretizer`, `RandomFieldFactory`, `ImmersedDomain`, `CoordinateMapping`, `Slip`, decomposition (`HaloStrategy`, `MeshDecompositionTraits`, `HaloSpec`, `HaloTracer`/`trace_halo`, `ArrayLayout`, `Decomposition`/`TensorDecomposition`/`GraphDecomposition`, `negotiate`), the halo/storage contract, export (`f.xr`), and the canonical `framework2.grid` package tree. |

**Decision log (not a cluster spec):**

| File | Contents |
|------|----------|
| [`operator_algebra_merge.md`](operator_algebra_merge.md) | Decisions for merging the operator algebra ([`../../operator_design/`](../../operator_design/00_overview.md), from `dev`) into [`03_operators.md`](03_operators.md): algebra-derived composed operators, bind-only axis naming, `Dispatched` as the user verb, `SeparableComposite` typing, interning, and the iteration split, applied to `03_operators.md` (D8). |

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

The fixed seam anchors across the four docs are:

- coordinate names are mandatory mesh-constructor arguments
  (`IntervalMesh(..., name="x")`); `fr.Grid(meshes=...)` only collects
  and validates them;
- spaces are produced by mesh factories (`mx.center`,
  `mx.fourier(origin=...)` — `origin` always explicit) and combined
  with `*` into a flat `TensorProductSpace`; lone factors implement
  the product protocol;
- **everything structural is static and identity-hashed** (explicit
  `__eq__`/`__hash__` returning `self is other` — required by
  fridom's structural-equality machinery): meshes, spaces, operators,
  and the **grid itself, which carries no dynamic pytree leaves**;
  field arrays are the only dynamic leaves
  (`ScalarField` is `jaxify, dynamic=("_data",)`); time-dependent
  geometry is module-owned state consumed via explicit-data accessors;
- fields carry their grid (static aux); derived coordinate data is
  grid-mediated and materialized on demand at trace time
  (`grid.evaluation_nodes(space)`, `grid.wavenumbers(space)`,
  `grid.measure(space, name=...)`);
- the single field factory is
  `grid.create_field(space, init=... | init_coeff=... | data=...)`;
  seeded randomness is `grid.random`;
- binary field arithmetic requires identical spaces (up to the
  `ConstantSpace` / Real->Complex lifts, `SpaceMismatchError`) and the
  same grid object (`GridMismatchError`);
- dispatch is grid-owned: `grid.dispatch` holds doc 03's
  `OperatorRegistry`; field sugar (`f.diff("x")`, `f * g`) resolves
  `(kind, space)` entries there; transform rows are seeded lazily
  (doc 04's grid lifecycle);
- transforms bind the grid at construction
  (`fr.operators.Fourier(grid, axes=...)`, `.forward`/`.backward`);
  all other operators are grid-free until application;
- halo/storage contract: `_data` is storage-shaped, `.data` the
  true-shape view; iteration 1 syncs after every operator application
  (kernels run per-shard under a decomposition-supplied `shard_map`);
  sync-elision along traced chains is the designed-for optimization;
- eigenvalues are per coefficient factor space:
  `op.eigenvalues(grid, coeff_space) -> Symbol`; a `Symbol` is a
  diagonal operator, never a `ScalarField`; its product broadcast is
  the diagonal extension (`Identity ⊗ D`), not the field lift;
- decomposition traits are declared per space by meshes
  (`mesh.decomposition_traits(space) -> MeshDecompositionTraits`) and
  consumed by doc 04's negotiation; halos are per coordinate name,
  accumulated by tracing (`HaloTracer`), never a global integer.

## Suggested implementation staging (ROADMAP Phase 1)

The iteration-1 surface is larger than the ~4.9 kloc it replaces and
is not shippable as one unit; the natural internal ordering (each
stage testable in isolation) is:

1. static markers, meshes, spaces, products (pure static structure);
2. minimal `ScalarField` + `create_field(data=/init=)` on nodal
   spaces + registry + FD/interpolate, single device;
3. average family, FV operators, `integrate`, full strict-algebra
   dunders;
4. transforms (`Fourier`, DST/DCT, Chebyshev) + `refined()` padding;
5. negotiation + `HaloTracer` + multi-device;
6. immersed boolean subset + export.

Staging 2 lands a minimal field core; staging 3–6 then cover algebra
completion, the eigen machinery, and model-facing sugar.

## Normative-note amendments

Amendments made to the concept notes together with these docs: the
binary-product spelling in
[`../02_rules.md`](../02_rules.md#311-field-operations-linear-ops-and-the-product-problem)
is instance-call (`Hadamard()(f, a)`); the phase-shift exactness and
Hermitian-constraint caveats in
[`../02_rules.md`](../02_rules.md#32-coefficient-representations-are-separate-spaces)
(Nyquist mode, self-conjugate value constraint); the Shen DOF count
(n - 1) in section 3.5 / 6.2; `fr.Grid(meshes=...)` with names on
meshes (sections 2.6, 8, sketches 4.1/4.4); the grid carries no
dynamic pytree leaves and time-dependent geometry is module-owned
(sections 2.2, 2.6, 2.7); the materialize-outside-scan performance
knob replaces the `jax.checkpoint` phrasing (section 2.7); sketch
4.6/4.10 spellings; and the
[`../07_iteration1_api.md`](../07_iteration1_api.md) cheat sheet
(coefficient layer iteration-1, `cell_avg`/`face_avg`, required
`origin`, immersed subset).
