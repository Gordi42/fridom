# Grid abstraction redesign — Class designs: meshes and function spaces

Part of the grid redesign notes; see
[`../00_overview.md`](../00_overview.md) for the document map. Status:
draft class design, no implementation. Signatures are the intended
public API for `framework2.grid`; the numbered concept sections remain
the normative reference.

This document owns the **Mesh and FunctionSpace cluster**: the mesh
factor classes, the function-space classes they intern, and the small
static markers both depend on (scalars/Körper, BC structure).
Decomposition-trait *types* (`HaloStrategy`,
`MeshDecompositionTraits`) are owned by doc 04; meshes here only
declare traits through them. Sibling class docs own the rest of the
surface:

| Doc | Owns |
|-----|------|
| `02_product_spaces_and_fields.md` | `TensorProductSpace`, `ScalarField`, `VectorField`, `State` |
| `03_operators.md` | operators, transforms, `Symbol`, dispatch |
| `04_grid_and_decomposition.md` | `Grid`, decomposition, coordinate materialization, `ImmersedDomain`, metrics |

Where this cluster touches those seams it uses only the fixed anchor
signatures: `grid.evaluation_nodes(space)`, `grid.wavenumbers(space)`,
`grid.create_field(space, init=...)`,
`op.eigenvalues(grid, coeff_space) -> Symbol`, and
`space_a * space_b -> TensorProductSpace`.

---

## Module layout

The code lives in `fridom.framework2.grid` (part of the new parallel
`fridom.framework2` package), renamed to `fridom.framework.grid` at
cutover. Proposed internal layout for this cluster:

```
src/fridom/framework2/grid/
    scalars.py            # Scalars, Real, Complex, Variance
    bc.py                 # BC, BCStructure
    meshes/               # re-exported as fr.meshes
        mesh.py           # Mesh (ABC)
        structured_1d.py  # StructuredMesh1D (ABC)
        interval.py       # IntervalMesh
        mapped_interval.py# MappedIntervalMesh
        chebyshev.py      # ChebyshevMesh
        point.py          # PointMesh
        sphere.py         # SphereMesh
        unstructured.py   # UnstructuredMesh
    spaces/
        function_space.py # FunctionSpace (ABC)
        nodal.py          # NodeSet, NodalSpace (ABC),
                          # Center, Left, Right, Outer, Inner,
                          # PointValues
        average.py        # AverageSpace (ABC), CellAvg, FaceAvg
        coefficient.py    # CoefficientSpace (ABC), FourierSpace,
                          # SineSpace, CosineSpace, ChebyshevSpace
        galerkin.py       # GalerkinSpace, ExtendedGalerkinSpace
        constant.py       # ConstantSpace
        tensor_product.py # TensorProductSpace (owned by doc 02)
    decomposition/
        traits.py         # HaloStrategy, MeshDecompositionTraits
                          # (owned by doc 04)
```

Only the entries without an ownership annotation are specified in
this document; `framework2/grid/decomposition/traits.py` and
`framework2/grid/spaces/tensor_product.py` appear so the shared tree is
consistent across the four cluster docs. Doc 04 holds the canonical
whole-subpackage tree. Import direction: classes here import
`HaloStrategy` / `MeshDecompositionTraits` *from*
`framework2.grid.decomposition.traits`.

Top-level re-exports (lazypimp, per repo convention):

- `fr.meshes` = `fridom.framework2.grid.meshes` (the mesh factors,
  section 8 of the overview);
- `fr.Real`, `fr.Complex` from `framework2.grid.scalars` (section 3.1);
- `fr.BC` from `framework2.grid.bc` (sketch 4.4 spells `fr.BC.DIRICHLET`).

Function spaces get **no top-level namespace**: they are produced only
by the mesh factory attributes (`mx.center`, `mz.galerkin(...)`), per
section 8 of the overview. The `spaces/` modules are importable for
`isinstance` checks in operator/dispatch code, not for construction.

---

## Cluster-wide rules

These apply to every class below and are not repeated per class:

- **Static, hashable, interned.** Meshes and spaces hold no jax
  arrays and no grid reference (sections 2.1, 2.2, 2.7). They enter
  field pytrees as *static aux data* — never as leaves — and serve as
  jit-cache and dispatch keys. None of these classes is decorated with
  `@fr.utils.jaxify`.
- **Identity semantics — explicit, not defaulted.** `Mesh` and
  `FunctionSpace` define an *explicit*
  `def __eq__(self, other): return self is other` plus the matching
  identity `__hash__`. Relying on the object defaults is a trap:
  fridom's structural-equality machinery
  (`framework/utils/jax_utils.py`, `_values_equal`) routes any fridom
  object whose `__eq__` *is* the object default into a deep
  structural walk, so two distinct-but-equal meshes — and their
  interned space families — would compare equal in the jit cache,
  exactly what this cluster forbids. An explicit identity `__eq__`
  makes `_values_equal` fall through to plain `a == b`. Interning
  turns value equality into identity: two requests for "the center
  space of mesh `m`" return the same object, so the strict-algebra
  check (section 3.1) is `a is b`.
- **Meshes are *not* interned by value.** Each mesh construction is a
  new, distinct domain factor: a square domain needs two
  `IntervalMesh(shape=256, extent=(0, 1))` instances, and they must
  not collapse into one. Interning happens *per mesh*, one level down,
  in the mesh-owned space registry.
- **Space constructors are guarded.** The only construction path is
  the owning mesh's factory attributes: `__init__` takes a *private
  factory token* keyword and raises when invoked without it (i.e.
  outside the mesh's interning factory), because direct construction
  would silently void the identity-equality guarantee of the strict
  algebra. Signatures are given below for the record, prefixed with
  the mesh they bind to.
- **No large static arrays — at all costs.** Static descriptors are
  numbers, tuples, enums, callables. Bulk array data must never be
  baked into interned static objects: interned objects are immortal
  (memory blow-up) and static key content leaks into traced programs
  as baked constants (compile-time blow-up). Anything bulk is
  dynamic, grid-materialized-on-demand data. This rule is what makes
  the unstructured-mesh geometry an unresolved problem (see
  `UnstructuredMesh` and Open questions).
- **Naming.** Classes `PascalCase`, members `snake_case`, per
  `AGENTS.md`. Fixed factory spellings (section 10.2): `mx.center`,
  `mx.right`, `mx.outer`, `mx.inner`, `mx.cell_avg`, `mx.face_avg`,
  `mx.constant`, `mx.fourier(origin=...)`, `mx.galerkin(bc=...)`,
  `space.as_complex()`, `space_a * space_b`. The squashed spellings
  `cellavg`/`faceavg` were rejected for snake_case consistency; the
  class names `CellAvg`/`FaceAvg` are unchanged.

---

## Static markers

### Scalars — the Körper marker (`fr.Real` / `fr.Complex`)

One-line role: the field of scalars a function space is defined over
(section 3.1).

- Kind: final `enum.Enum`
- Static or dynamic: static (enum members are singletons)
- Iteration: 1
- Concept refs: sections 3.1, 2.2, sketch 4.11

```python
class Scalars(Enum):
    """The field of scalars (Körper) a function space is over."""

    REAL = auto()
    COMPLEX = auto()


# module-level aliases, re-exported at top level as fr.Real / fr.Complex
Real: Scalars = Scalars.REAL
Complex: Scalars = Scalars.COMPLEX
```

Notes:

- **Concrete form: a two-member `Enum`, not sentinel classes.** Enum
  members are hashable singletons, pickle/repr-friendly, usable in
  `match` statements, and iterable for parametrized tests. The
  rejected alternative — two empty marker classes `Real` / `Complex`
  used as class objects — hashes and compares fine but has no common
  type for annotations (`scalars: Scalars` reads better than
  `scalars: type`) and invites accidental instantiation.
- The marker denotes the Körper of the *represented function*, not
  the storage dtype (section 3.1): a `fr.Real` Fourier space stores a
  complex Hermitian half-spectrum. Dtype derivation from
  `(scalars, basis)` is a field/space concern documented per space
  class below; the enum itself carries no dtype.
- `space.scalars` participates in the interned identity of every
  space, so real and complex variants are distinct dispatch keys
  (rfft vs fft, section 3.1).

### BC and BCStructure — homogeneous BC structure

One-line role: the static boundary-condition *structure* baked into a
space (data stays per-field, section 3.6).

- Kind: `BC` final enum; `BCStructure` final value class
- Static or dynamic: static; `BCStructure` is hashed *by value*
- Iteration: 1 — `bc.py` ships day one; `NONE`, `DIRICHLET`, and
  `NEUMANN` are all exercised in iteration 1 (carried by the
  Sine/Cosine coefficient spaces and their structured origins);
  Galerkin/Shen spaces remain designed-for
- Concept refs: sections 2.2, 3.2, 3.5, 3.6

```python
class BC(Enum):
    """Homogeneous boundary-condition kind at one boundary component."""

    NONE = auto()       # BC-free: boundary DOFs stay in the space
    DIRICHLET = auto()
    NEUMANN = auto()


class BCStructure:
    """Normalized per-boundary-component tuple of BC kinds."""

    def __init__(self, components: tuple[BC, ...]) -> None:
        """Store the (left, right) tuple for 1D meshes."""
        ...

    @classmethod
    def normalize(cls, spec: BC | BCStructure | tuple[BC, ...],
                  n_components: int) -> BCStructure:
        """Expand a single BC to all components; validate length."""
        ...

    @property
    def components(self) -> tuple[BC, ...]:
        """BC kind per boundary component (left, right for 1D)."""
        ...

    @property
    def n_constraints(self) -> int:
        """Number of constrained DOFs (non-NONE components)."""
        ...

    @property
    def is_free(self) -> bool:
        """True if every component is BC.NONE."""
        ...

    def __eq__(self, other: object) -> bool:
        """Value equality (unlike spaces, which use identity)."""
        ...

    def __hash__(self) -> int:
        """Value hash; used inside the mesh interning keys."""
        ...
```

Notes:

- `BCStructure` is a small *value* type used inside interning keys; it
  is deliberately value-hashable, while spaces themselves remain
  identity-hashable. Factories accept the sugar `bc=fr.BC.DIRICHLET`
  (meaning: at every boundary component) and normalize.
- **Periodicity is not a BC member.** It is mesh topology
  (`mesh.periodic`), fixed at mesh construction; putting it in `BC`
  would duplicate the descriptor and allow inconsistent combinations.
- The structure decides free-DOF membership and hence `space.shape`
  (section 3.5): Dirichlet `Outer` drops its two boundary DOFs;
  Dirichlet `Center` keeps n DOFs (no boundary node in the set) but
  changes the admissible coefficient bases (DST-II, section 3.2).
- Robin/mixed conditions need a real parameter inside a static key;
  deferred (see Open questions).

---

## Meshes

### Mesh (ABC)

One-line role: atomic factor of the domain — geometry + topology, no
discretization, no arrays (section 2.1).

- Kind: ABC
- Static or dynamic: static (identity-hashed; pytree aux data)
- Iteration: 1
- Concept refs: sections 2.1, 2.7, 5; section 3.6 (`boundary`)

```python
class Mesh(ABC):
    """Atomic factor of the domain geometry (geometry + topology)."""

    def __init__(self, names: tuple[str, ...]) -> None:
        """Create the factor; coordinate names are mandatory."""
        ...

    # ------------------------------------------------------------
    #  Identity (explicit, see cluster rules)
    # ------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        """Identity: return self is other (explicit so fridom's
        structural-equality walk falls through to plain ==)."""
        ...

    def __hash__(self) -> int:
        """Identity hash, matching __eq__."""
        ...

    # ------------------------------------------------------------
    #  Geometry / topology
    # ------------------------------------------------------------
    @property
    @abstractmethod
    def dim(self) -> int:
        """Intrinsic dimension of the factor (>= 0)."""
        ...

    @property
    @abstractmethod
    def boundary(self) -> Mesh:
        """The conforming boundary as a mesh of dimension dim - 1."""
        ...

    # ------------------------------------------------------------
    #  Coordinate names
    # ------------------------------------------------------------
    @property
    def names(self) -> tuple[str, ...]:
        """Coordinate names, fixed at construction."""
        ...

    # ------------------------------------------------------------
    #  Decomposition traits (seam: types owned by doc 04)
    # ------------------------------------------------------------
    @abstractmethod
    def decomposition_traits(self, space: FunctionSpace
                             ) -> MeshDecompositionTraits:
        """Sharding/halo traits for one of this mesh's spaces
        (doc 04's frozen record; preference-ordered strategies)."""
        ...

    # ------------------------------------------------------------
    #  Space factory / interning registry
    # ------------------------------------------------------------
    @property
    def constant(self) -> ConstantSpace:
        """The one-DOF broadcast space on this mesh (section 3.3)."""
        ...

    def _intern(self, key: tuple,
                factory: Callable[[], FunctionSpace]) -> FunctionSpace:
        """Return the registry entry for key, building it on first
        request (the interning mechanism behind every factory)."""
        ...

    def __repr__(self) -> str:
        """E.g. 'IntervalMesh(x: n=256, extent=(0, 1), periodic)'."""
        ...
```

Notes:

- **Interning mechanism.** Each mesh owns a private registry
  `dict[tuple, FunctionSpace]`. Every factory (property or method)
  normalizes its arguments into a value-hashable key — e.g.
  `(Center, BCStructure(...), Scalars.REAL)` or
  `(FourierSpace, id-of-origin, Scalars.REAL)` — and calls
  `_intern`. Guarantees: (a) value-equal requests return the identical
  object, so space equality is identity; (b) the registry is mesh-
  owned, so spaces of two distinct-but-equal meshes are never
  conflated; (c) `space.as_complex()` routes back through the same
  registry, so scalar variants are interned too. Zero-argument
  factories are additionally exposed as `functools.cached_property`
  for attribute-access speed; the cache and the registry hold the
  same object.
- **Names are mandatory at construction** (owner decision). 1D
  meshes take `name: str`
  (`IntervalMesh(n, extent, periodic, name="x")`); multi-coordinate
  meshes take `names: tuple[str, ...]`
  (`SphereMesh(..., names=("lon", "lat"))`). There is no
  `bind_names`, no pre-binding state, and `Mesh.names` is always
  well-defined. Consequently the assembly root is
  `fr.Grid(meshes=...)` — **no `names=`** — which only *validates*
  that names are duplicate-free across meshes (doc 04 side); the
  cartesian convenience subclass keeps `names=` because it constructs
  the meshes itself. Mesh reuse across grids is naturally under fixed
  names. `len(names)` equals the number of coordinate names the
  factor contributes (1 for 1D factors, 2 for a sphere — independent
  of storage-axis count); reprs of spaces use the name (`Center(x)`).
- **`boundary` is stable.** Repeated access returns the identical
  boundary-mesh object (`cached_property`), so spaces interned on the
  boundary mesh obey the same identity guarantees as bulk spaces.
  Trace product spaces (section 3.6) are ordinary products with the
  boundary mesh's spaces as factors — no new data type.
- The base class deliberately declares **no nodal/average/coefficient
  factories**: those exist only where the node-set vocabulary is
  meaningful (structured 1D meshes below; vertex/edge/cell factories
  on `UnstructuredMesh`). Only `constant` is universal — "constant
  along this factor" makes sense for every mesh (section 3.3).
- **Decomposition traits are a per-space query** (section 5): nodal
  and coefficient spaces of one mesh differ (ghost halos vs
  transpose-based transforms), so the single seam method
  `decomposition_traits(space)` is a per-space query. The returned
  `MeshDecompositionTraits` (doc 04, `framework2.grid.decomposition.traits`) is
  a frozen record with a preference-ordered
  `strategies: tuple[HaloStrategy, ...]` (`GHOST` / `TRANSPOSE` /
  `LOCAL` / `GRAPH`) and `min_local_size: int = 1`. A separate
  `shardable` boolean is derivable (`strategies != (LOCAL,)`) and is
  therefore dropped. The negotiation consuming these records is
  doc-04 territory.
- Extension contract for subclasses: implement `dim`, `boundary`,
  `decomposition_traits`, and whatever factory surface the mesh
  family supports; keep every constructor argument a static
  descriptor (numbers, tuples, enums, callables — no jax arrays).

### StructuredMesh1D (ABC)

One-line role: shared 1D cell structure and the full space-factory
surface of interval-like meshes.

- Kind: ABC (subclasses: `IntervalMesh`, `MappedIntervalMesh`,
  `ChebyshevMesh`)
- Static or dynamic: static
- Iteration: 1
- Concept refs: sections 2.1, 2.2, 3.2, 3.5, 10.2

```python
class StructuredMesh1D(Mesh):
    """1D mesh with n cells: node-set, average, and coefficient
    space factories."""

    def __init__(self, shape: int, extent: tuple[float, float],
                 periodic: bool, *, name: str) -> None:
        """Store cell count, extent, periodicity, and the mandatory
        coordinate name."""
        ...

    # ------------------------------------------------------------
    #  Geometry descriptors
    # ------------------------------------------------------------
    @property
    def n_cells(self) -> int:
        """Number of primal cells n (the only count the mesh knows;
        DOF counts live on spaces, section 3.5)."""
        ...

    @property
    def extent(self) -> tuple[float, float]:
        """Physical interval (x_min, x_max)."""
        ...

    @property
    def periodic(self) -> bool:
        """Whether the interval is periodic (mesh topology, not a BC)."""
        ...

    # ------------------------------------------------------------
    #  Mesh interface
    # ------------------------------------------------------------
    @property
    def dim(self) -> int:
        """Always 1."""
        ...

    @property
    def boundary(self) -> PointMesh:
        """Two located endpoints; the empty PointMesh if periodic."""
        ...

    def decomposition_traits(self, space: FunctionSpace
                             ) -> MeshDecompositionTraits:
        """GHOST-first for nodal/average spaces; (LOCAL, TRANSPOSE)
        for coefficient spaces (local preferred); (LOCAL,) for
        ConstantSpace. Chebyshev overrides (transpose-first)."""
        ...

    def refined(self, factor: Fraction) -> Self:
        """A new, distinct mesh of the same type with the cell count
        scaled by factor (padded-transform target, section 3.12);
        interned per (mesh, factor)."""
        ...

    @property
    def refined_from(self) -> Self | None:
        """The parent this mesh was refined from (None on unrefined
        meshes); doc 03's padded transforms derive the coarse trim
        target from it."""
        ...

    # ------------------------------------------------------------
    #  Nodal factories (xgcm vocabulary, section 2.2)
    # ------------------------------------------------------------
    @property
    def center(self) -> Center:
        """Nodal cell centers (n DOFs); BC-free, fr.Real."""
        ...

    @property
    def left(self) -> Left:
        """Nodal left cell edges (n DOFs)."""
        ...

    @property
    def right(self) -> Right:
        """Nodal right cell edges (n DOFs)."""
        ...

    @property
    def outer(self) -> Outer:
        """All faces (n + 1 DOFs); bounded meshes only."""
        ...

    @property
    def inner(self) -> Inner:
        """Interior faces (n - 1 DOFs); bounded meshes only."""
        ...

    def nodal(self, node_set: NodeSet, *,
              bc: BC | BCStructure = BC.NONE) -> NodalSpace:
        """General nodal factory with BC structure; the properties
        above are sugar for bc=BC.NONE."""
        ...

    # ------------------------------------------------------------
    #  Average factories (section 3.9)
    # ------------------------------------------------------------
    @property
    def cell_avg(self) -> CellAvg:
        """Primal-cell averages (n DOFs)."""
        ...

    @property
    def face_avg(self) -> FaceAvg:
        """Dual-cell averages (n periodic / n - 1 bounded DOFs)."""
        ...

    # ------------------------------------------------------------
    #  Coefficient factories (section 3.2)
    # ------------------------------------------------------------
    def fourier(self, origin: FunctionSpace) -> FourierSpace:
        """Fourier coefficient space of the given origin (explicit,
        no default); periodic meshes only."""
        ...

    def sine(self, origin: FunctionSpace) -> SineSpace:
        """DST coefficient space of a Dirichlet-structured origin;
        bounded meshes only."""
        ...

    def cosine(self, origin: FunctionSpace) -> CosineSpace:
        """DCT coefficient space of a Neumann-structured origin;
        bounded meshes only."""
        ...

    def galerkin(self, *, bc: BC | BCStructure,
                 extended: bool = False) -> GalerkinSpace:
        """Modal Galerkin space with baked-in BCs (designed-for);
        implemented per concrete mesh (Shen on ChebyshevMesh)."""
        ...
```

Notes:

- The constructor argument is named `shape` for continuity with the
  sketches (`IntervalMesh(shape=256, ...)`) but it is the **cell
  count**, exposed as `n_cells`; per section 3.5 the mesh has no DOF
  shape.
- **Node-set names are topological; the mesh fixes physical
  placement.** `Center`/`Right` on a uniform interval are equispaced;
  on a `ChebyshevMesh` the same vocabulary lands on Gauss–Lobatto
  geometry (see below). This is what lets one nodal class family
  serve all structured 1D meshes.
- Factory validity is checked at call time: `outer`/`inner` raise on
  periodic meshes (no boundary faces; the wrap face would be
  double-counted), `fourier` raises on bounded meshes, `sine`/
  `cosine` raise on periodic ones. `left`/`right` exist on both
  topologies with n DOFs (xgcm semantics: right edges of the n cells;
  on a bounded mesh `Right` excludes the left boundary face).
- `nodal(node_set, bc=...)` is the BC-structure hook for nodal spaces
  required by section 3.6, and it is **iteration 1**: the Sine/Cosine
  coefficient spaces (promoted to iteration 1) need
  Dirichlet-/Neumann-structured nodal origins (Dirichlet `Center` ->
  DST-II, Dirichlet `Inner` -> DST-I, Neumann `Center` -> DCT-II,
  section 3.2). The `NodeSet` enum (`CENTER`, `LEFT`, `RIGHT`,
  `OUTER`, `INNER`, `POINTS`) lives in `spaces/nodal.py` and is the
  interning key component; there is deliberately no string-keyed
  variant.
- **Coefficient factories take an explicit `origin` — no default**
  (owner decision): an implicit
  `origin=center` default invites exactly the origin-mixup bugs the
  per-origin coefficient-space design (section 3.2) exists to catch.
- No `dx` on the ABC: uniform spacing is an `IntervalMesh` extra;
  measures in general are grid-materialized metric fields
  (section 2.7).
- **`refined(factor)` is the finer-mesh factory** behind doc 03's
  padded transforms (this answers doc 03's open question 2): the
  degree-p pad factor `(p + 1)/2` selects a padded transform into a
  finer nodal space (section 3.12), and that space needs a mesh to
  live on. The factor is a `fractions.Fraction` (the natural type for
  3/2- and 2/3-style ratios: exact, hashable, no float fuzz);
  `n_cells * factor` must be integral, else `ValueError`. The result
  is a **first-class mesh** — its spaces are interned on it and obey
  every rule above. Although meshes are not interned by value,
  `refined` results *are* memoized per (parent mesh, factor), so
  repeated requests return the identical finer mesh and its spaces
  stay identity-comparable. Grid-wise, refined meshes are **"adopted children"**
  of the grid that owns their parent (rule G10; doc 04 owns that
  paragraph). Iteration 1 on `IntervalMesh`; designed-for on the
  other 1D meshes.

### IntervalMesh

One-line role: uniform 1D interval — the iteration-1 workhorse
(section 2.1).

- Kind: concrete, final
- Static or dynamic: static
- Iteration: 1
- Concept refs: sections 2.1, 3.5, 10.1–10.2, sketch 4.4

```python
class IntervalMesh(StructuredMesh1D):
    """Uniform 1D interval with n equal cells."""

    def __init__(self, shape: int, extent: tuple[float, float],
                 periodic: bool = True, *, name: str) -> None:
        """Uniform interval; shape is the cell count; the coordinate
        name is mandatory."""
        ...

    @property
    def dx(self) -> float:
        """Uniform cell width (extent length / n_cells), a static
        descriptor."""
        ...
```

Notes:

- `dx` is a *descriptor convenience* (the mesh owns the cell
  structure, section 2.1). Operators must still obtain measures
  through the grid's metric fields (section 2.7: no operator bakes
  `dx` into Python constants); on this mesh those fields are constant
  and XLA folds them.
- Decomposition traits: the `StructuredMesh1D` default — GHOST-first
  for nodal/average spaces, `(LOCAL, TRANSPOSE)` for coefficient
  spaces, `(LOCAL,)` for `ConstantSpace`.
- `refined(factor)` is **iteration 1 here** (the padded-transform
  target of section 3.12 is needed for dealiased products at day-one
  parity); designed-for on the other 1D meshes.

### PointMesh

One-line role: 0D mesh — a finite set of located points; the boundary
of an interval (section 3.6).

- Kind: concrete, final
- Static or dynamic: static
- Iteration: 1 (minimal: exists so `mesh.boundary` is total; the
  trace-field enforcement machinery around it is designed-for)
- Concept refs: sections 2.1, 3.6

```python
class PointMesh(Mesh):
    """0D mesh: a finite tuple of located points."""

    def __init__(self, positions: tuple[tuple[float, ...], ...],
                 *, name: str) -> None:
        """Points given by their ambient coordinates; may be empty."""
        ...

    @property
    def n_points(self) -> int:
        """Number of points (0 for the boundary of a periodic mesh)."""
        ...

    @property
    def positions(self) -> tuple[tuple[float, ...], ...]:
        """Ambient coordinates of the points (static descriptor)."""
        ...

    @property
    def dim(self) -> int:
        """Always 0."""
        ...

    @property
    def boundary(self) -> PointMesh:
        """The empty PointMesh (a 0D mesh has no boundary)."""
        ...

    def decomposition_traits(self, space: FunctionSpace
                             ) -> MeshDecompositionTraits:
        """Always (LOCAL,): point meshes are replicated."""
        ...

    @property
    def points(self) -> PointValues:
        """One nodal DOF per point (the trace space factor)."""
        ...
```

Notes:

- `IntervalMesh.boundary` returns a two-point `PointMesh` at the
  extent endpoints, or the zero-point `PointMesh` when periodic —
  one type covers both, and "empty boundary" needs no special case.
  A boundary `PointMesh` inherits its name from the parent factor, so
  trace spaces print as `boundary(x)` (section 3.6).
- An inflow profile u(x=0, y, z, t) is a field on
  `boundary(x).points ⊗ Center(y) ⊗ Center(z)` — an ordinary product
  space (doc 02); this class adds no data type for boundary data.
- The factory is named `points`, not `nodal`: `StructuredMesh1D` has
  a *method* `nodal(node_set, bc=...)`, and reusing the name for a
  zero-argument property on a sibling would give the same spelling
  two arities across the mesh family.
- Point positions are a static tuple-of-tuples of floats: hashable by
  value at construction time (folded into no registry — mesh identity
  still rules), tiny, and host-side.

### MappedIntervalMesh

One-line role: stretched 1D interval via a self-coordinate-only
mapping (vertical coordinates), section 2.1.

- Kind: concrete, final
- Static or dynamic: static (the mapping callable is a static
  descriptor, hashed by identity)
- Iteration: designed-for
- Concept refs: sections 2.1, 2.7, 3.8 boundary note

```python
class MappedIntervalMesh(StructuredMesh1D):
    """1D interval with a monotone coordinate mapping."""

    def __init__(self, shape: int, extent: tuple[float, float],
                 mapping: Callable, periodic: bool = False,
                 *, name: str) -> None:
        """mapping: computational s in [0, 1] -> physical x; must be
        a pure, jnp-traceable, strictly monotone function."""
        ...

    @property
    def mapping(self) -> Callable:
        """The coordinate mapping (static descriptor)."""
        ...
```

Notes:

- The mapping may depend **only on this mesh's own coordinate**
  (section 2.1); cross-factor mappings (terrain-following) are
  grid-level metric data (section 3.8), not mesh structure.
- The mesh stores the *function*, never materialized node arrays: the
  grid composes `mapping` with uniform computational nodes on demand
  (section 2.7), and the two staggered `dx` measures become genuinely
  different metric fields on their respective spaces.
- No scalar `dx` property exists here — asking for one is the bug the
  metric-field design prevents.

### ChebyshevMesh

One-line role: bounded 1D mesh on Gauss–Lobatto geometry, host of the
Chebyshev/Shen space family (section 2.1, validation 6.2).

- Kind: concrete, final
- Static or dynamic: static
- Iteration: 1 (promoted with the rest of the coefficient machinery;
  doc 03 promotes the Chebyshev transform in parallel). `galerkin`
  stays designed-for.
- Concept refs: sections 2.1, 3.2, 3.5; validation 6.2

```python
class ChebyshevMesh(StructuredMesh1D):
    """1D interval with Gauss-Lobatto node geometry."""

    def __init__(self, shape: int, extent: tuple[float, float],
                 *, name: str) -> None:
        """Bounded by construction (periodic is always False)."""
        ...

    @property
    def lobatto(self) -> Outer:
        """Alias for self.outer: the Gauss-Lobatto collocation space
        (n + 1 points including the endpoints)."""
        ...

    def chebyshev(self, origin: FunctionSpace) -> ChebyshevSpace:
        """Chebyshev coefficient space of the given origin
        (explicit, no default)."""
        ...

    def galerkin(self, *, bc: BC | BCStructure,
                 extended: bool = False) -> GalerkinSpace:
        """Shen basis with baked-in BCs (designed-for;
        extended=True: the inhomogeneous variant with boundary
        modes, section 3.6)."""
        ...

    def decomposition_traits(self, space: FunctionSpace
                             ) -> MeshDecompositionTraits:
        """TRANSPOSE-first: shardable, with contiguous-axis needs
        (transform, banded solves) met via transpose layouts."""
        ...
```

Notes:

- `lobatto` **is** `outer` — same interned object. The xgcm node-set
  vocabulary is topological (all faces = n + 1 points including
  endpoints); this mesh places them at Gauss–Lobatto locations. The
  alias exists because spectral users think "Lobatto points", not
  "outer faces".
- **The space family is restricted** (owner decision):
  `outer`/`lobatto`, the `chebyshev`/`galerkin`
  coefficient spaces, and `constant` — no cell family. The nodal
  `center`/`left`/`right`/`inner` and average `cell_avg`/`face_avg`
  factories raise on this mesh until an FV-on-Chebyshev consumer
  exists.
- `fourier` raises (bounded); `sine`/`cosine` are admissible but the
  natural bases here are Chebyshev/Shen.
- **Decomposition traits are TRANSPOSE-capable, not LOCAL-preferring**
  (owner directive): a tensor product of Chebyshev meshes would
  otherwise be undecomposable in every direction. The mesh *is*
  shardable; operations that need a contiguous dimension (the
  Chebyshev transform, banded per-column solves) declare
  transpose-based layouts — `strategies=(TRANSPOSE, ...)` — and
  doc 04's negotiation consumes this. "Shard x/y, keep z on-device"
  (section 5) remains one negotiated *outcome* on mixed grids, not a
  trait forced by this mesh.

### SphereMesh

One-line role: single 2D factor for spherical geometry, contributing
two coordinate names and owning metric structure (validation 6.3).

- Kind: concrete (final)
- Static or dynamic: static
- Iteration: designed-for
- Concept refs: sections 2.1, 2.3; validation 6.3

```python
class SphereMesh(Mesh):
    """2D spherical factor (lon x lat), metric-aware."""

    def __init__(self, shape: tuple[int, int], radius: float,
                 names: tuple[str, str] = ("lon", "lat")) -> None:
        """Cell counts per coordinate; radius is a static descriptor."""
        ...

    @property
    def dim(self) -> int:
        """Always 2."""
        ...

    @property
    def boundary(self) -> Mesh:
        """The chart boundary: for a lat-lon parameterization the
        polar-cap latitude circles, not the empty mesh (see notes)."""
        ...

    @property
    def radius(self) -> float:
        """Sphere radius (metric structure descriptor)."""
        ...

    @property
    def center(self) -> NodalSpace:
        """Nodal cell centers of the lon-lat cells."""
        ...

    def decomposition_traits(self, space: FunctionSpace
                             ) -> MeshDecompositionTraits:
        """GHOST-first for nodal/average spaces (2D block sharding);
        (LOCAL,) otherwise."""
        ...
```

Notes:

- This factor contributes **two coordinate names** but stays *one*
  entry in the flat product tuple (section 2.3); `init=` functions
  receive `lon` and `lat` keywords from this one mesh (6.3).
- **The sphere is in practice not a closed manifold** (owner
  directive): a lat-lon grid is a chart with boundaries toward the
  poles, so `boundary` is *not* empty — it is the polar-cap latitude
  circles (plus periodic identification in lon).
- **Owner's favored direction — a manifold abstraction layer.**
  Rather than a monolithic 2D factor, the eventual design should
  explore local parameterization: charts `R^2 -> S^2`, under which
  the two angular coordinates could even be **two `IntervalMesh`
  factors with metric coupling at grid level** (the section 2.3
  "topological product, geometrically coupled through metric fields"
  rule applied to the sphere). This class is therefore a sketch of
  the *interface* a spherical factor must satisfy (interning,
  boundary, traits), not a committed shape.
- The staggered space family beyond `center` (and universal
  `constant`) is deliberately unspecified here: on a sphere the
  useful sets (C-grid faces, poles handling, cubed-sphere variants)
  are implementation questions for the iteration that ships it.
  `grad`/`div` are mesh-level operator registrations (6.3), not
  space features.
- Metric terms (cos-lat factors, area elements) are grid-materialized
  fields (section 2.7); the mesh stores only `radius` and counts.

### UnstructuredMesh

One-line role: 2D triangular factor with vertex/edge/cell DOF sets
(ICON/FESOM-style prisms when producted with a vertical mesh,
validation 6.4).

- Kind: concrete (final)
- Static or dynamic: static (connectivity is static topology, see
  notes)
- Iteration: designed-for
- Concept refs: sections 2.1, 2.2; validation 6.4; section 5 (GRAPH)

```python
class UnstructuredMesh(Mesh):
    """2D triangular mesh factor."""

    def __init__(self, vertices, triangles,
                 names: tuple[str, str] = ("lon", "lat")) -> None:
        """vertices: (n_vertices, 2) coordinates; triangles:
        (n_triangles, 3) vertex indices — bulk data whose storage
        home is unresolved (see notes: no-static-arrays rule)."""
        ...

    @property
    def dim(self) -> int:
        """Always 2."""
        ...

    @property
    def boundary(self) -> Mesh:
        """The boundary polyline as a 1D mesh (empty if closed);
        concrete 1D boundary-mesh type to be fixed when implemented."""
        ...

    @property
    def vertex(self) -> NodalSpace:
        """One DOF per vertex."""
        ...

    @property
    def edge(self) -> NodalSpace:
        """One DOF per edge."""
        ...

    @property
    def cell(self) -> NodalSpace:
        """One DOF per triangle."""
        ...

    @property
    def edge_normal(self) -> NodalSpace:
        """Edge-normal velocity component space (triangular C-grid)."""
        ...

    def decomposition_traits(self, space: FunctionSpace
                             ) -> MeshDecompositionTraits:
        """GRAPH-first for vertex/edge/cell spaces (graph
        partitioning, a later backend); (LOCAL,) otherwise."""
        ...
```

Notes:

- **Unresolved collision with the no-static-arrays rule** (owner
  directive): the connectivity and vertex coordinates of an
  unstructured mesh are **bulk array data**, and baking them into an
  interned static mesh violates the cluster rule that large static
  arrays are to be avoided at all costs (immortal interned memory;
  baked constants in traced programs). The earlier idea — host-side
  numpy frozen at construction and content-hashed once — mitigates
  hashing cost but not the memory/compile-time problem. The owner
  flags this as a **real problem requiring a careful rethink before
  any unstructured work starts** (see Open questions); candidate
  directions include keeping only a topology *fingerprint* static
  and materializing connectivity as dynamic grid-owned data. Derived
  per-DOF coordinates are unaffected: they were always materialized
  on demand through `grid.evaluation_nodes(space)` (6.4) by the
  fully static grid (G1, doc 04).
- The vertex/edge/cell family is why nodal spaces have no `shift()`
  involution (section 2.2): staggering transitions here exist only as
  operator signatures (`div: edge_normal -> cell`).
- Spaces reuse the `NodalSpace` ABC with mesh-specific node-set tags;
  whether they warrant dedicated classes (`Vertex`, `Edge`, `Cell`)
  is deferred to the implementing iteration — the interning and
  identity contract does not change either way.

---

## Function spaces

### FunctionSpace (ABC)

One-line role: static, hashable, interned descriptor of one discrete
representation on one mesh (section 2.2).

- Kind: ABC
- Static or dynamic: static (identity-hashed, interned; pytree aux
  data / jit and dispatch key)
- Iteration: 1
- Concept refs: sections 2.2, 2.7, 3.1, 3.5, 3.6

```python
class FunctionSpace(ABC):
    """How a continuous field is represented on one mesh."""

    def __init__(self, mesh: Mesh, scalars: Scalars,
                 bc: BCStructure, *, _token: object) -> None:
        """Raises unless _token is the owning mesh's private factory
        token: construction only through mesh factories (guarded,
        see cluster rules)."""
        ...

    # ------------------------------------------------------------
    #  Identity (explicit, see cluster rules)
    # ------------------------------------------------------------
    def __eq__(self, other: object) -> bool:
        """Identity: return self is other (explicit so fridom's
        structural-equality walk falls through to plain ==)."""
        ...

    def __hash__(self) -> int:
        """Identity hash, matching __eq__."""
        ...

    @property
    def mesh(self) -> Mesh:
        """The owning mesh factor (a static back-reference; spaces
        hold no *grid* reference, section 2.7)."""
        ...

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        """True DOF count per storage axis (section 3.5); one entry
        for 1D factors, more for 2D structured factors."""
        ...

    @property
    def scalars(self) -> Scalars:
        """The Körper the space is defined over (default fr.Real)."""
        ...

    @property
    def bc(self) -> BCStructure:
        """Homogeneous BC structure baked into the space; the free
        default for BC-free spaces (section 3.6)."""
        ...

    @property
    def variance(self) -> Variance | None:
        """Component variance (covariant/contravariant) for vector
        components on metric meshes; None for scalar/no-variance
        (designed-for, section 2.4)."""
        ...

    # ------------------------------------------------------------
    #  Product protocol (shared with TensorProductSpace, doc 02)
    # ------------------------------------------------------------
    @property
    def factors(self) -> tuple[FunctionSpace, ...]:
        """(self,): a lone factor is its own flat factor tuple."""
        ...

    @property
    def names(self) -> tuple[str, ...]:
        """The owning mesh's coordinate names (always defined; names
        are fixed at mesh construction)."""
        ...

    def factor(self, name: str) -> FunctionSpace:
        """Self if name is one of the mesh's names; KeyError else."""
        ...

    def as_complex(self) -> Self:
        """The interned fr.Complex variant of this space."""
        ...

    def as_real(self) -> Self:
        """The interned fr.Real variant (the codomain of f.real)."""
        ...

    def __mul__(self, other: FunctionSpace) -> TensorProductSpace:
        """Tensor product (flat, associative; doc 02 owns the type)."""
        ...

    def __repr__(self) -> str:
        """E.g. 'Center(x)', 'Fourier(x, origin=Right)'."""
        ...
```

Notes:

- **No relational properties, by decision** (section 2.2). There is
  no `space.shift()` — the staggering partner is ambiguous on bounded
  meshes (`Center` -> `Right`, `Outer`, or `Inner`?) and undefined on
  unstructured node sets; operators know their codomain instead
  (`diff: Center -> Right`). There is no `space.spectral` — a space
  admits infinitely many coefficient bases and none is canonical; the
  default transform is the dispatch entry `("transform", space)`.
  Anything relating two spaces is an operator signature or a dispatch
  entry, never a space property.
- **Defining attributes = interning key**: `(type, mesh identity,
  node set / basis, bc, scalars, origin)` — exactly the static
  descriptors of section 2.2. Derived coordinate quantities
  (evaluation nodes, wavenumbers, measures) are `ScalarField`s the
  grid materializes on demand via `grid.evaluation_nodes(space)` /
  `grid.wavenumbers(space)`; nothing array-like lives here, and the
  grid itself is fully static too (G1, doc 04) — time-dependent
  geometry lives in module-owned state fields.
- `as_complex` / `as_real` are *interning lookups*, not relational
  properties in the rejected sense: the scalar variant is part of the
  space's own defining data (its Körper), and both directions return
  interned spaces. `as_complex` on an `fr.Complex` space returns
  `self` (idempotent), likewise `as_real` on `fr.Real`.
- `__mul__` normalizes to a flat `TensorProductSpace` (doc 02);
  duplicate meshes among the factors are rejected there
  (`mx.center * mx.right` is an error — one factor per mesh,
  section 2.3).
- **Single-factor spaces implement doc 02's product protocol**
  (`factors`, `names`, `factor(name)`) with the trivial defaults
  above, so per-factor code (dispatch, decomposition, coordinate
  accessors) treats a lone factor space and a `TensorProductSpace`
  uniformly — no special case for the 1D-grid degenerate product.
  See doc 02 for the protocol's normative home.
- **`variance` is a designed-for defining attribute** (section 2.4,
  validation 6.3): covariant and contravariant components of a
  vector on a metric mesh live on *distinct spaces*, so the strict
  algebra catches variance mixing, and raising/lowering indices is an
  explicit metric-consuming operator (doc 03). `Variance` is a small
  two-member enum (`COVARIANT`, `CONTRAVARIANT`) placed beside
  `Scalars`; the default `None` means scalar/no-variance and keeps
  the attribute out of the interning key — it enters the key only
  when set, so iteration-1 keys are unchanged.
- Dispatch code classifies spaces by the three intermediate ABCs
  below (`NodalSpace`, `AverageSpace`, `CoefficientSpace`) plus
  `ConstantSpace`; no string "kind" attribute exists.

### NodalSpace (ABC) and Center, Left, Right, Outer, Inner

One-line role: point values at a node set — the finite-difference
representation (xgcm vocabulary, section 2.2).

- Kind: `NodalSpace` ABC; `Center`, `Left`, `Right`, `Outer`, `Inner`
  final concrete classes
- Static or dynamic: static (interned)
- Iteration: 1 (all five; `Left` is not on the day-one cheat sheet
  but is trivial and completes the xgcm set). BC-structured variants
  (`bc != NONE`): iteration 1 as well — they are the origins of the
  iteration-1 Sine/Cosine spaces.
- Concept refs: sections 2.2, 3.5, 3.6, 10.2

```python
class NodeSet(Enum):
    """Topological node-set tag of a nodal space."""

    CENTER = auto()
    LEFT = auto()
    RIGHT = auto()
    OUTER = auto()
    INNER = auto()
    POINTS = auto()      # PointMesh trace nodes


class NodalSpace(FunctionSpace):
    """Point values at a node set of the mesh."""

    @property
    def node_set(self) -> NodeSet:
        """The topological node-set tag."""
        ...


class Center(NodalSpace):
    """Nodal values at the n cell centers."""


class Left(NodalSpace):
    """Nodal values at the n left cell edges."""


class Right(NodalSpace):
    """Nodal values at the n right cell edges."""


class Outer(NodalSpace):
    """Nodal values at all n + 1 faces (bounded meshes)."""


class Inner(NodalSpace):
    """Nodal values at the n - 1 interior faces (bounded meshes)."""
```

Shapes on a mesh with n cells (section 3.5), before BC constraints:

| Space | Periodic | Bounded |
|-------|----------|---------|
| `Center` | (n,) | (n,) |
| `Left` / `Right` | (n,) | (n,) |
| `Outer` | — (factory error) | (n + 1,) |
| `Inner` | — (factory error) | (n - 1,) |

Notes:

- The names `Cell`/`Face` are deliberately not used; the current
  FRIDOM `FACE` corresponds to `Right`. This maps one-to-one to
  xarray/xgcm staggered-coordinate export. The old
  `Position`/`AxisPosition` enums (`grid/position.py`) disappear:
  staggering is the choice of space per variable.
- **BC structure reduces the shape** by `bc.n_constraints` *when the
  constrained boundary DOF is in the node set*: Dirichlet `Outer` has
  shape (n - 1,) (the same DOF set as BC-free `Inner`, yet a distinct
  interned space — types differ, and that is fine); Dirichlet
  `Center` keeps (n,) — no boundary node — but selects DST-II as its
  compatible coefficient basis (section 3.2). BC-free `Outer` keeps
  its boundary DOFs as the slots for prescribed boundary fluxes
  (section 3.6, FV row).
- Concrete classes are empty subclasses of `NodalSpace`: the class
  *is* the node-set tag for dispatch (`("diff", Center-of-mesh-x)`),
  with `node_set` as the enum mirror used in interning keys and
  generic code.

### PointValues

One-line role: nodal space on a `PointMesh` — the factor of trace
product spaces (section 3.6).

- Kind: concrete, final
- Static or dynamic: static (interned)
- Iteration: 1 (minimal, with `PointMesh`)
- Concept refs: sections 2.1, 3.6

```python
class PointValues(NodalSpace):
    """One nodal DOF per point of a PointMesh."""

    # shape == (mesh.n_points,); node_set == NodeSet.POINTS
```

Notes:

- Obtained as `mesh.boundary.points`. Not a `ConstantSpace`: it is a
  *located* boundary space (coordinates, generally > 1 DOF, possibly
  0 DOFs on periodic factors) restricted to the boundary manifold; it
  must not broadcast into the interior (section 3.6).

### AverageSpace (ABC) and CellAvg, FaceAvg

One-line role: cell-mean functionals — the finite-volume
representation, distinct from nodal ("averages have no position",
sections 2.2, 3.9).

- Kind: `AverageSpace` ABC; `CellAvg`, `FaceAvg` final
- Static or dynamic: static (interned)
- Iteration: 1
- Concept refs: sections 2.2, 3.9, 3.5; sketch 4.7

```python
class AverageSpace(FunctionSpace):
    """Averages over a cell family (primal or dual)."""


class CellAvg(AverageSpace):
    """Averages over the n primal cells."""


class FaceAvg(AverageSpace):
    """Averages over the dual cells around faces
    (n periodic / n - 1 bounded)."""
```

Notes:

- A `CellAvg` DOF is the functional `(1/dx) ∫_cell u dx`, not a value
  at any point; identifying it with the center value is a
  second-order approximation, made explicit as a `reconstruct` /
  evaluate-to-average operator pair (sketch 4.7). The average family
  mirrors the nodal family: `CellAvg` ↔ `Center` on primal cells,
  `FaceAvg` ↔ `Right`/`Outer` on dual cells.
- Coordinate labels for export (centers for `CellAvg`, faces for
  `FaceAvg`) are metadata the export layer attaches (doc 04
  territory); they are deliberately **not** a property of these
  classes — that would smuggle "position" back in.
- The FV derivative types against this family:
  `reconstruct: CellAvg(n) -> Outer(n+1)` (approximate),
  `flux_diff: Outer(n+1) -> CellAvg(n)` (exact) — operator classes in
  doc 03; the spaces just make the signatures expressible.
- Average spaces have their own coefficient spaces: the interning key
  of a `FourierSpace` with `origin=CellAvg` differs from
  `origin=Center` (the `sinc(k dx / 2)` factor, section 3.2).

### CoefficientSpace (ABC) and FourierSpace, SineSpace, CosineSpace, ChebyshevSpace

One-line role: modal coefficients relative to a basis, defined by
(basis, origin) — the spectral representation (section 3.2).

- Kind: `CoefficientSpace` ABC; concrete classes final
- Static or dynamic: static (interned)
- Iteration: 1 — all four (`FourierSpace`, `SineSpace`,
  `CosineSpace`, `ChebyshevSpace`; owner decision). Doc 03 promotes
  the DST/DCT/Chebyshev transforms in parallel, restoring
  bounded-axis spectral parity in iteration 1.
- Concept refs: sections 3.2, 3.1, 3.5; sketches 4.3, 4.9

```python
class CoefficientSpace(FunctionSpace):
    """Modal coefficients of a basis, tied to an origin space."""

    @property
    def origin(self) -> FunctionSpace:
        """The origin space: constitutive of this space, fixing the
        inverse transform and the shape (section 3.2)."""
        ...


class FourierSpace(CoefficientSpace):
    """Fourier coefficients of a periodic origin."""

    # shape: (n // 2 + 1,) if scalars is fr.Real (Hermitian
    # half-spectrum, rfft layout); (n,) if fr.Complex


class SineSpace(CoefficientSpace):
    """DST coefficients of a Dirichlet-structured bounded origin."""

    # DST-I of Dirichlet Inner: (n - 1,); DST-II of Dirichlet
    # Center: (n,)


class CosineSpace(CoefficientSpace):
    """DCT coefficients of a Neumann-structured bounded origin."""


class ChebyshevSpace(CoefficientSpace):
    """Chebyshev coefficients of a Gauss-Lobatto origin."""

    # shape: (n + 1,) — one mode per Lobatto point
```

Notes:

- **The origin is constitutive** (section 3.2): `mx.fourier(origin=
  mx.center)` and `mx.fourier(origin=mx.right)` are distinct interned
  spaces, related only by the exact `PhaseShift` operator (doc 03).
  Adding fields across origins is a caught `SpaceMismatchError`, the
  coefficient-space payoff of strict algebra (sketch 4.3).
- **Scalars follow the origin.** `space.scalars` is the Körper of the
  represented function, inherited from the origin: a real-origin
  Fourier space has `scalars = fr.Real` while *storing* complex
  numbers, and its half-spectrum shape makes the Hermitian constraint
  *largely* structural (section 3.2). Precisely: the shape removes
  only the conjugate half of the spectrum; the **realness of the
  k = 0 and Nyquist entries is a value constraint invisible to the
  shape**. That value-level invariant is owned at the seams — the
  field factory projects assigned coefficients at the self-conjugate
  modes (doc 02), and the random draw handles those modes as special
  indices (doc 04). `as_complex()` on a coefficient space returns
  the coefficient space of the complexified origin (full spectrum),
  i.e. it changes the shape too — it is never a dtype flag flip.
- The class names carry a `Space` suffix (`FourierSpace`, not
  `Fourier`) to avoid colliding with the transform operator
  `fr.operators.Fourier` (doc 03); users never type the class names —
  the factory spellings `mx.fourier(...)` are the API. Reprs still
  print the concept-note form `Fourier(x, origin=Center)`.
- **`origin` is always explicit** (owner decision): no
  `origin=None`-means-center default on `mx.fourier(...)` /
  `mx.sine(...)` / `mx.cosine(...)` / `mz.chebyshev(...)` — same
  origin-mixup rationale as the `StructuredMesh1D` factories
  (section 3.2).
- There is deliberately **no `space.wavenumbers`**: wavenumbers and
  mode indices are grid-materialized (`grid.wavenumbers(space)`,
  section 2.7); the space is only the key.
- BC structure of a pure coefficient space is the origin's
  (`self.bc is self.origin.bc`); Galerkin spaces below own their BC
  directly.

### GalerkinSpace and ExtendedGalerkinSpace

One-line role: modal bases with BCs baked in (Shen-type), homogeneous
and inhomogeneous variants (sections 2.2, 3.6; validation 6.2).

- Kind: concrete, final (both)
- Static or dynamic: static (interned)
- Iteration: designed-for (spelling `mx.galerkin(bc=...)` fixed by
  10.2)
- Concept refs: sections 2.2, 3.2, 3.5, 3.6; validation 6.2

```python
class GalerkinSpace(CoefficientSpace):
    """Homogeneous modal basis with baked-in BCs (Shen bases on
    ChebyshevMesh)."""

    # bc: this space's own BCStructure (not the origin's)
    # shape: (m - bc.n_constraints,) where m is the modal count of
    # the underlying basis (m = n + 1 on ChebyshevMesh)


class ExtendedGalerkinSpace(CoefficientSpace):
    """Inhomogeneous variant: homogeneous modes + boundary modes."""

    @property
    def homogeneous(self) -> GalerkinSpace:
        """The homogeneous space this extends (constitutive, like
        origin; the two are related by an inclusion operator)."""
        ...

    # shape: (m,) — (m - k) homogeneous modes + k boundary modes
```

Notes:

- Obtained as `mz.galerkin(bc=fr.BC.DIRICHLET)` and
  `mz.galerkin(bc=..., extended=True)`; the `extended=` spelling is
  proposed here, not fixed by the notes.
- The BC structure determines the free modes and hence the shape
  (section 3.5: "Chebyshev-Shen basis with 2 BCs: n - 2" in the
  table's counting); it also routes `diff` to the Shen recurrence
  with BCs built into the basis (sketch 4.4).
- The extended space realizes the section 3.6 Galerkin row: boundary
  values are ordinary coefficients of the k boundary modes (shenfun
  `BCGeneric` precedent; lifting and tau methods were the rejected
  alternatives). Nothing is conditional on a flag: homogeneous and
  extended are distinct interned jit/dispatch keys, related by an
  inclusion operator (doc 03).
- `homogeneous` is a constitutive attribute (part of the defining
  data), not a relational property in the section 2.2 rejected sense
  — same status as `CoefficientSpace.origin`.

### ConstantSpace

One-line role: one-DOF broadcast factor replacing `topo=False` axes
(section 3.3).

- Kind: concrete, final
- Static or dynamic: static (interned; one per (mesh, scalars))
- Iteration: 1
- Concept refs: sections 3.3, 3.13; sketches 4.5, 4.8

```python
class ConstantSpace(FunctionSpace):
    """Constant along this mesh factor: a single broadcast DOF."""

    # shape == (1,); bc is the free structure; obtained as
    # mesh.constant (any mesh type)
```

Notes:

- Broadcasting a `ConstantSpace` factor against a full factor is
  exact and unambiguous — one of the two sanctioned exceptions to
  strict algebra (with `fr.Real -> fr.Complex` promotion). The
  broadcast itself is implemented in field arithmetic (doc 02); the
  space only marks the axis.
- It is the codomain of `integrate` along a factor (section 3.13) and
  the broadcast vehicle for per-factor `Symbol`s and coordinate
  fields across a product (sections 2.5, 2.7).
- `ConstantSpace` factors do not appear in `init=` keyword signatures
  (section 3.10, sketch 4.8).
- Distinct from `PointValues` on a boundary mesh: a `ConstantSpace`
  is a geometry-less bulk reduction that *does* broadcast into the
  interior; a trace space must not (section 3.6).
- Operators along a `ConstantSpace` axis are identity or trivially
  defined — the structural end of "partial topo not supported".

---

## Open questions

Former questions 1 (default `origin`), 2 (name binding time), and 4
(ChebyshevMesh cell family) are **closed** by owner decisions recorded
in the body: coefficient factories take an explicit `origin` — no
default (`StructuredMesh1D` notes); coordinate names are mandatory at
mesh construction, `bind_names` is deleted, and `fr.Grid(meshes=...)`
only validates duplicate-free names (`Mesh` notes); the
`ChebyshevMesh` family is restricted to `outer`/`lobatto` +
coefficient/Galerkin spaces, no cell family (`ChebyshevMesh` notes).

Still open:

1. **Robin / mixed BCs** (stays open, owner directive; do not resolve
   yet). Constraints for the eventual decision: a float BC parameter
   in the static interning key means a **full recompile per parameter
   value** under the Phase-3 single jit, and it **forecloses
   autodiff through — and module updates of — BC parameters**. The
   candidate resolution is: only the DOF-count-changing *structure*
   (a `BC.ROBIN` member) enters the space key, while the float
   coefficients are dynamic data living where BC data already lives
   (module-owned trace fields, section 3.6), consumed by
   `ghost_fill`/basis assembly at trace time. That candidate ties
   into the same static-structure-must-hold-no-values tension as the
   unstructured-mesh question below; decide them coherently.
2. **Bulk geometry of unstructured meshes** (owner-flagged, requires
   a careful rethink before any unstructured work). Connectivity and
   vertex coordinates are bulk array data; the cluster rule forbids
   large static arrays (memory via immortal interned objects,
   compile time via baked constants), yet the mesh is supposed to be
   a static descriptor. Candidate directions: static topology
   *fingerprint* + dynamic grid-materialized connectivity; a
   dedicated host-side geometry store the grid owns; or relaxing
   descriptor-hood for this mesh family. Nothing unstructured may be
   built until this is resolved.
3. **Boundary mesh of 2D factors.** `UnstructuredMesh.boundary` needs
   a 1D polyline/curve mesh type, and — since the sphere is not a
   closed manifold in practice (chart boundaries toward the poles,
   `SphereMesh` notes) — `SphereMesh.boundary` needs pole-cap
   latitude circles as boundary curves too. Not designed here;
   `PointMesh` only covers the 1D-factor case iteration 1 needs.
4. **Dedicated unstructured space classes.** Vertex/edge/cell spaces
   are speced as `NodalSpace` instances with new `NodeSet` tags;
   whether dispatch ergonomics want dedicated classes (`Vertex`,
   `Edge`, ...) like the interval family has is left to the
   implementing iteration.
