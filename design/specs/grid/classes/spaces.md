---
status: normative
date: 2026-07-07
---

# Grid abstraction redesign — Class designs: function spaces

Part of the framework2 class designs; see [`README.md`](README.md) for the document map. Module layout, cluster-wide rules, and static markers are in [`meshes.md`](meshes.md).

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

    @property
    def layout(self) -> Layout | None:
        """Negotiated device layout, or None for a bare space
        (doc 02 owns the semantics; section 5.1)."""
        ...

    @property
    def bare(self) -> FunctionSpace:
        """The layout-free interned variant (self if bare)."""
        ...

    def with_layout(self, layout: Layout | None) -> Self:
        """The interned variant carrying ``layout`` (grid-minted)."""
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
  descriptors of section 2.2 — plus, like `variance`, the `layout`
  *when set* (section 5.1): mesh factories mint bare spaces (`layout
  is None`, iteration-1 keys unchanged); laid-out variants are
  grid-minted, so single-factor spaces participate in the layout
  protocol and 1D grids work uniformly. Derived coordinate quantities
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
- **Only Dirichlet-type BC structure reduces the shape** (owner
  decision, 2026-07-07), by `bc.n_constraints` *when the constrained
  boundary DOF is in the node set*: a Dirichlet condition eliminates
  a boundary *value* DOF, so Dirichlet `Outer` has shape (n - 1,)
  (the same DOF set as BC-free `Inner`, yet a distinct interned
  space — types differ, and that is fine); Dirichlet `Center` keeps
  (n,) — no boundary node — but selects DST-II as its compatible
  coefficient basis (section 3.2). **Neumann structure never reduces
  the shape**: a Neumann condition constrains a derivative
  combination, not a nodal DOF — Neumann `Outer` keeps all n + 1
  nodes, which is exactly what makes DCT-I shape-honest (n + 1
  cosine modes k = 0..n; this resolves the DCT-I inconsistency found
  during implementation — the earlier blanket over non-free kinds
  was a doc bug). BC-free `Outer` keeps
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
    # half-spectrum, rfft layout); (n,) if fr.Complex.
    # In a multi-axis real transform only the FIRST-transformed
    # factor keeps a real origin (half spectrum); later stages
    # target as_complex() origins — full spectrum (section 5.1,
    # transform planner, doc 03).


class SineSpace(CoefficientSpace):
    """DST coefficients of a Dirichlet-structured bounded origin."""

    # DST-I of Dirichlet Inner: (n - 1,); DST-II of Dirichlet
    # Center: (n,)


class CosineSpace(CoefficientSpace):
    """DCT coefficients of a Neumann-structured bounded origin."""

    # DCT-II of Neumann Center: (n,); DCT-I of Neumann Outer:
    # (n + 1,) — Neumann never reduces the origin shape (owner
    # decision 2026-07-07, NodalSpace shape note)


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
  indices (doc 04). In products at most one factor — the
  first-transformed axis of the planned schedule (section 5.1) —
  carries a real origin and hence the half spectrum; later stages
  target complexified origins (full spectrum), and the value
  invariant becomes conjugate symmetry on the self-conjugate
  k = 0 / Nyquist *planes* of the halved factor (same seam owners).
  `as_complex()` on a coefficient space returns
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

Former question 5 (multi-factor real Fourier transforms) is
**closed** by the layout/transform decisions of
[§5.1](../04_decomposition.md#51-layout-is-part-of-the-function-space):
per-axis real transforms do not commute — the half spectrum lands on
the first-transformed factor (real origin), later stages target
complexified origins (full spectrum, `FourierSpace` note above), and
the planner's schedule — free to reorder `axes` for speed — fixes the
choice statically per grid.

Still open:

1. **Robin / mixed BCs** (stays open, owner directive; do not resolve
   yet — a resolution is *proposed* as decision R3 of
   [`../../../plans/active/boundary_plan.md`](../../../plans/active/boundary_plan.md), which supersedes
   this directive only when signed). Constraints for the eventual decision: a float BC parameter
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
4. **BC-free bounded spaces: exterior values untouchable**
   (owner-flagged, 2026-07-07). Should `BC.NONE` on a bounded axis
   mean "no operation may read beyond the boundary" — replacing the
   one-sided extrapolation ghost fill with per-row legality
   (exterior-needing signatures exist only on BC-structured spaces)
   plus explicit opt-in one-sided stencil rows? Full analysis,
   motivation (the fill is inconsistent under composition), and
   migration cost in
   [`../../../plans/active/bc_free_boundaries.md`](../../../plans/active/bc_free_boundaries.md); decide
   together with the Robin/mixed question above (both hinge on what
   BC structure the space key carries vs what stays dynamic) — the
   joint resolution is proposed in
   [`../../../plans/active/boundary_plan.md`](../../../plans/active/boundary_plan.md) (R1-R4).
4. **Dedicated unstructured space classes.** Vertex/edge/cell spaces
   are speced as `NodalSpace` instances with new `NodeSet` tags;
   whether dispatch ergonomics want dedicated classes (`Vertex`,
   `Edge`, ...) like the interval family has is left to the
   implementing iteration.
