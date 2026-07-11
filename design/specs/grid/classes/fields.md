---
status: normative
date: 2026-07-07
---

# Grid abstraction redesign — Class designs: fields

Part of the framework2 class designs; see [`README.md`](README.md) for the document map. Shared strict-algebra semantics, `TensorProductSpace`, and error types are in [`product_spaces.md`](product_spaces.md).

---

### FieldMetadata

Pure annotation for naming and I/O, no discretization content
(section 2.4).

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen dataclass |
| Pytree | static aux via its `ScalarField` owner; **annotation-exempt from aux equality** (Phase-2 reconciliation amendment, 2026-07-08 — see the metadata-propagation entry under `ScalarField`) |
| Iteration | 1 |
| Concept refs | 2.4, section 5 (metadata stays name/units/nc-attrs) |

```python
"""Annotation metadata for fields: name, units, nc-attrs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldMetadata:
    """Immutable, hashable annotation for a ScalarField."""

    name: str = "unnamed"                                      # it-1
    long_name: str = "Unnamed"                                 # it-1
    units: str = "n/a"                                         # it-1
    nc_attrs: tuple[tuple[str, str], ...] = ()                 # it-1

    @classmethod
    def create(                                                # it-1
        cls,
        name: str = "unnamed",
        long_name: str = "Unnamed",
        units: str = "n/a",
        nc_attrs: Mapping[str, str] | None = None,
    ) -> FieldMetadata:
        """Convenience constructor accepting a mapping for nc_attrs."""
        ...

    def replace(self, **changes: object) -> FieldMetadata:     # it-1
        """Functional update (dataclasses.replace wrapper)."""
        ...
```

Notes:

- **Removed relative to today** (all subsumed by the function space,
  section 2.4): `position`, `bc_types`, `topo`, `is_spectral`. Also
  removed: the `_flags` dict (`NO_ADV`, `ENABLE_MIXING`, ...) — those
  are model-physics markers, not field annotation; they move to the
  model-side field declarations that `State` subclasses own. The
  serialization helpers shrink accordingly.
- `nc_attrs` is canonically a tuple of pairs so the dataclass stays
  hashable (it sits in the static treedef); `create` accepts a
  mapping and normalizes. `long_name` is retained as ordinary
  nc-style annotation.
- Metadata never influences dispatch, dtype, shape, or algebra — it
  is invisible to sections 3.1–3.13 by construction.

---

### ScalarField

`(grid, function_space, array) + metadata` — the single concrete
field type; every derived array in the design (coordinates,
wavenumbers, masks, metrics) is one of these (sections 2.4, 2.7).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final |
| Pytree | `jaxify, dynamic=("_data",)`; grid/space/metadata static aux (grid identity-hashed; metadata annotation-exempt from aux equality — 2026-07-08 amendment below) |
| Iteration | 1 (core); individual methods tagged |
| Concept refs | 2.4, 2.7, 3.1–3.5, 3.10–3.13, all sketches |

```python
"""The scalar field: (function_space, array) + metadata."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


@partial(fr.utils.jaxify, dynamic=("_data",))
class ScalarField:
    """A discrete scalar field on a tensor-product function space."""

    def __init__(                                              # it-1
        self,
        grid: Grid,
        function_space: SpaceLike,
        data: jax.Array,
        metadata: FieldMetadata | None = None,
    ) -> None:
        """Trusting plumbing constructor (jit-hot): takes
        storage-shaped data, no validation, no copies."""
        ...

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def grid(self) -> Grid:                                    # it-1
        """The grid this field was created on (section 2.7)."""
        ...

    @property
    def function_space(self) -> SpaceLike:                     # it-1
        """The (product) function space of the field."""
        ...

    @property
    def data(self) -> jax.Array:                               # it-1
        """Raw local array at true shape (halo/padding stripped).
        Read-only: the setter is an explicit raising stub guiding to
        ``with_data`` / ``grid.create_field`` (Phase-2 amendment,
        model design D1.5 — a teaching error for the old
        ``f.arr += ...`` mutation habit, not a bare AttributeError)."""
        ...

    @property
    def metadata(self) -> FieldMetadata:                       # it-1
        """Annotation metadata (name/units/nc-attrs)."""
        ...

    @property
    def name(self) -> str:                                     # it-1
        """Shorthand for ``metadata.name``."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:                        # it-1
        """Global true DOF shape, ``function_space.shape``."""
        ...

    @property
    def dtype(self) -> jnp.dtype:                              # it-1
        """Derived storage dtype (from space scalars + basis; 3.1)."""
        ...

    # ================================================================
    #  Functional updates
    # ================================================================

    def with_data(self, data: jax.Array) -> ScalarField:       # it-1
        """Same grid/space/metadata, new true-shape array (routed
        through decomposition.pad; task-1.8 validity zero)."""
        ...

    def with_metadata(self, **changes: object) -> ScalarField: # it-1
        """Same grid/space/data, updated metadata."""
        ...

    # ================================================================
    #  Scalars (Körper) surface — section 3.1
    # ================================================================

    def as_complex(self) -> ScalarField:                       # it-1
        """Explicit promotion onto the complexified space."""
        ...

    @property
    def real(self) -> ScalarField:                             # it-1
        """The real part, an ``fr.Real`` field (identity if real)."""
        ...

    @property
    def imag(self) -> ScalarField:                             # it-1
        """The imaginary part, an ``fr.Real`` field (zero if real)."""
        ...

    def conj(self) -> ScalarField:                             # it-1
        """Complex conjugate on the same space (identity if real)."""
        ...

    # ================================================================
    #  Dispatch sugar — section 3.4
    # ================================================================

    def diff(self, name: str) -> ScalarField:                  # it-1
        """Default derivative along ``name``: (kind="diff", space)."""
        ...

    def to(                                                    # it-1
        self,
        target: ScalarField | SpaceLike,
    ) -> ScalarField:
        """Generic per-axis conversion onto the target's space."""
        ...

    def reshard(self, target: Layout | SpaceLike) -> ScalarField:  # it-1
        """Explicit layout change: sugar over ``Reshard`` (cluster
        03, section 5.1). Never implicit in arithmetic."""
        ...

    def integrate(self, *names: str) -> ScalarField:           # it-1
        """Weighted integral; named factors reduce to ConstantSpace."""
        ...

    def mean(self, *names: str) -> ScalarField:                # it-1
        """Integral divided by the integrated measure (sugar)."""
        ...

    def cumint(                                                # later
        self,
        name: str,
        direction: Literal["forward", "backward"] = "forward",
    ) -> ScalarField:
        """Sugar over the it-1 ``CumulativeIntegral`` operator
        (cluster 03; kind="cumint", 3.13)."""
        ...

    def grad(self) -> VectorField:                             # later
        """Dispatch kind "grad"; components keyed by coordinate name."""
        ...

    def laplacian(self) -> ScalarField:                        # later
        """Dispatch kind "laplacian" (composed default; 3.4)."""
        ...

    # ================================================================
    #  Selection (ROADMAP Phase 1)
    # ================================================================

    def sel(                                                   # later
        self,
        method: Literal["nearest"] | None = None,
        **coords: float,
    ) -> ScalarField:
        """Select at coordinates; factors reduce to ConstantSpace."""
        ...

    def isel(self, **indices: int) -> ScalarField:             # later
        """Integer-index companion of ``sel``."""
        ...

    # ================================================================
    #  Arithmetic — sections 3.1, 3.3, 3.11 (join rule above)
    # ================================================================

    def __add__(self, other: ScalarField | complex) -> ScalarField:
        """Linear; join rule. Python scalars are constant fields."""
        ...                                                    # it-1

    def __radd__(self, other: complex) -> ScalarField: ...     # it-1

    def __sub__(self, other: ScalarField | complex) -> ScalarField:
        """Linear; join rule."""
        ...                                                    # it-1

    def __rsub__(self, other: complex) -> ScalarField: ...     # it-1

    def __neg__(self) -> ScalarField: ...                      # it-1

    def __pos__(self) -> ScalarField: ...                      # it-1

    def __mul__(
        self, other: ScalarField | complex,
    ) -> ScalarField:
        """Scalar: linear scaling. Field: dispatched physical
        product, (kind="multiply", space) (3.11)."""
        ...                                                    # it-1

    def __rmul__(self, other: complex) -> ScalarField: ...     # it-1

    def __truediv__(
        self, other: ScalarField | complex,
    ) -> ScalarField:
        """Scalar: linear scaling. Field: (kind="divide", space);
        default registered on nodal/average spaces only."""
        ...                                                    # it-1

    def __rtruediv__(self, other: complex) -> ScalarField: ... # it-1

    def __pow__(self, exponent: int | float) -> ScalarField:
        """Physical power, (kind="power", space) (2.5 table)."""
        ...                                                    # it-1

    def __abs__(self) -> ScalarField:                          # it-1
        """Pointwise modulus, (kind="abs", space); nodal default."""
        ...

    def __bool__(self) -> bool:
        """Always raises TypeError (fields have no truth value)."""
        ...                                                    # it-1

    # ================================================================
    #  Diagnostics and export
    # ================================================================

    def has_nan(self) -> jax.Array:                            # it-1
        """0-d boolean array: any NaN in the (global) field."""
        ...

    def block_until_ready(self) -> ScalarField:                # it-1
        """Wait for async device work; returns self."""
        ...

    @property
    def xr(self) -> xr.DataArray:                              # it-1
        """xarray export (label/gather rules: cluster 04 Export)."""
        ...

    def __repr__(self) -> str:                                 # it-1
        """Name, space, shape, dtype summary."""
        ...
```

Semantics, invariants, error behavior:

- **Construction paths.** `grid.create_field(space, init=...,
  init_coeff=..., data=...)` is the single user-facing factory
  (section 3.10, fixed anchor); it owns sharding/layout validation
  and calls `__init__`, which is the trusting constructor operators
  use inside jit (they already hold validated shards). **Fields
  always live on laid-out spaces** (section 5.1): `create_field`
  accepts a bare space and attaches the decomposition's default
  layout (a laid-out space is honored as given); the layout is read
  from `f.function_space.layout`, and bare-space comparisons go
  through `f.function_space.bare`. `__init__`
  performs no validation and no copies. `grid.random.normal(space,
  seed)` and the coordinate accessors `grid.evaluation_nodes(space)`
  / `grid.wavenumbers(space)` return `ScalarField`s through the same
  plumbing (cluster 04).
- **The grid is carried, spaces are not grid-aware** (section 2.7):
  `f.diff("x")` resolves `(kind="diff",
  f.function_space.factor("x"))` in the grid's merged
  `OperatorRegistry`, reached as `grid.dispatch` (cluster 03,
  `framework2/grid/operators/registry.py`, owns resolution; the key shape is
  fixed here). Every grid-mediated accessor is reachable from a
  field via `f.grid`.
- **dtype is derived, never stored** (sections 2.4, 3.1): real +
  Fourier factor ⇒ complex Hermitian half-spectrum storage; real +
  sine/cosine ⇒ real; any complex factor ⇒ complex. `dtype` is a
  read-only report of that derivation (concretely: the array's
  dtype, which `create_field` guarantees consistent).
- **Hermitian value invariant** (3.2): on coefficient spaces of real
  origin, the half-spectrum *shape* removes the conjugate half, but
  realness at the self-conjugate modes (k = 0, Nyquist) is a *value*
  constraint the shape cannot encode. `grid.create_field` with
  `init_coeff=` or `data=` therefore **projects the imaginary part
  at self-conjugate modes** on construction, and operators must
  preserve the invariant (real-linear operators do so
  automatically, 3.1). The random-draw side — real-only draws at
  self-conjugate indices — is owned by cluster 04.
- **Arithmetic** follows the join rule of the shared-semantics
  section (grid identity first, then the space join). Exact raises:
  operands from different grids ⇒ `GridMismatchError`;
  `f + g` / `f - g` / `f * g` / `f / g` with
  no join ⇒ `SpaceMismatchError` (e.g. `Right(x)` vs `Center(x)`,
  or `Fourier(origin=Right)` vs `Fourier(origin=Center)` — sketch
  4.3). Python scalars in `+`/`-` are treated as fields on the
  all-`ConstantSpace` product and enter the same join (so adding a
  constant to a coefficient-space field is the exact zero-mode
  update, `# later`, matching lift 1); scalars in `*`/`/` are plain
  linear scaling on any space (representation-independent, 3.11).
  A complex Python scalar promotes the result space.
- **`f * g` is the dispatched physical product** (3.11): the `*`
  overloading table of section 2.5 is fixed — space `*` space is the
  tensor product, field `*` field the physical product, `Symbol`
  algebra is cluster 03's. Realizations: nodal ⇒
  `CollocationProduct` (`# it-1`), coefficient ⇒ `Convolution`
  (`# later`), average ⇒ quadrature product — the second-order
  shortcut ships as the `("multiply", CellAvg)` default (`# it-1`,
  cluster 03), higher-order quadrature `# later`. Cluster 03 also
  registers the `"divide"`, `"power"`, and `"abs"` kinds backing the
  dunders below. `f ** n` follows the same
  table; the coefficient-space default is `# later` — cluster 03
  carries the designed-for `("power", Fourier(origin))` →
  repeated-`Convolution` registry row (integer `n >= 1`) that this
  dunder will resolve to. The coefficient-wise product is
  *never* `*`: it is the explicit `Hadamard` operator, and applying
  a `Symbol` to a field is callable Hadamard multiply (cluster 03).
- **`to(target)`** accepts a field (`g.to(f)`), a full product
  space, or — decision — a *single factor space* as shorthand for
  "convert that factor only, keep the rest"
  (`g.to(mx.center)` ≡ `g.to(g.function_space.replace(x=mx.center))`).
  Per-axis dispatch reads the conversion kind from the source→target
  factor relationship (section 3.4); the explicit family-pair → kind
  matrix is:

  | Source factor → target factor | Kind | Status |
  |-------------------------------|------|--------|
  | nodal → nodal | `"interpolate"` | it-1 |
  | average → nodal/face | `"reconstruct"` | it-1 |
  | average → average | dual-family transfer (cluster 03) | it-1 |
  | coefficient → coefficient, same basis / different origin | `"interpolate"` (exact phase shift, 3.2) | it-1 |
  | nodal → average | `"average"` (quadrature projection) | later (DispatchError until registered) |
  | nodal ↔ coefficient | **raises `SpaceMismatchError`** — a `.to` is not a transform; use `fr.operators.Fourier(grid, axes=...).forward/.backward` | — |

  A registered operator whose codomain differs from the requested
  factor raises `SpaceMismatchError`. **One target per kind**: from
  a given source factor, exactly one codomain per kind is reachable
  via `.to` (the registered operator fixes its own codomain, 3.4);
  any other target requires an explicit operator instance or a
  registry override. `to` onto the identical space returns `self`.
- **`integrate(*names)`** (section 3.13): no names ⇒ all factors.
  Signature per factor: `S(x) → ConstantSpace(x)`; already-constant
  factors are identity. Weights come from the space's
  quadrature-measure fields via the grid (uniform `dx`,
  Clenshaw-Curtis, Jacobian weights — cluster 04); weights compose
  per mesh. The result *broadcasts back* by lift 1, so
  `f - f.integrate("x")` is legal with no extra API. There is no
  unweighted `sum`/`max`/`min` field method — raw DOF reductions are
  the array escape hatch (`f.data.sum()`), per 3.13.
- **`cumint`**: the underlying `CumulativeIntegral` operator is
  **iteration 1** (hydrostatic parity, cluster 03) with fixed
  codomains — `CellAvg(n) → Outer(n+1)`, periodic `Center → Right`,
  and bounded `Center → Outer` (information-preserving; the earlier
  `Inner` codomain discarded the total and was dropped by cluster
  03), the discrete-FTC partial inverse of `flux_diff` (3.9/3.13).
  On periodic meshes the input must be mean-zero for the cumulative
  integral to be single-valued, and the integration constant is
  fixed per `direction` by cluster 03's convention (zero at the
  start face). Only this method *sugar* is tagged `# later`;
  explicit operator application covers iteration-1 needs.
- **`sel`/`isel`** (`# later`): reduce
  the named factors to `ConstantSpace` — a slice at `x = a` has no
  x-extent, which is exactly what `ConstantSpace` encodes, and the
  broadcast lift makes `f - f.sel(z=0.0)` work. `sel` requires an
  exact node match unless `method="nearest"`; on coefficient factors
  both raise `ValueError` (no physical coordinate — transform back
  first), and on **average factors both raise too**: averages have
  no position (§2.2 — a `CellAvg` DOF is a functional over the cell,
  not a value at a point); reconstruct to a nodal space first.
  Boundary *data* is not `sel`: trace fields live on
  boundary meshes (section 3.6), not on `ConstantSpace`.
- **Metadata propagation** (decision): operations that re-represent
  the *same quantity* keep metadata (`with_data`, `to`, `real`,
  `imag`, `as_complex`, `conj`, `sel`/`isel`, transforms); operations
  that produce a *different quantity* (all binary arithmetic, `diff`,
  `integrate`, `grad`, ...) return default metadata — no unit
  algebra is attempted. Users re-annotate via `with_metadata`. This
  default-metadata rule applies to **bare `ScalarField` ops only**:
  `VectorField`/`State` componentwise arithmetic *preserves* each
  component's metadata, because component names are structural there
  (see the scan-stability rule under `VectorField`).
- **Amended (Phase-2 reconciliation, 2026-07-08): metadata is
  annotation-exempt from pytree structure.** `FieldMetadata` stays in
  the static aux — it must survive flatten/unflatten — but jaxify
  grows an *annotation* aux category **excluded from aux equality**:
  two fields differing only in metadata have equal treedefs.
  Documented consequences: `lax.scan`/`vmap`/jit caching are
  metadata-insensitive, and objects returned from jitted functions
  carry *trace-time* metadata — host code wanting authoritative names
  reads container keys or the model's `FieldTable`, never
  round-tripped field metadata. The "same quantity keeps, new
  quantity resets" rule above **stands unchanged** — it is now purely
  annotational, never structural. The same fix direction applies to
  `to` on the converting path: the landed implementation returns the
  registered operator's output with default metadata where this
  document says "keep" — an annotational bug of the same class, no
  longer a structural one. Rationale:
  [`../../../plans/done/phase1_findings.md`](../../../plans/done/phase1_findings.md) contract finding 2
  (arithmetic on a *named* field changes the treedef; scan rejects
  named-field carries), widened by the Phase-2 reconciliation audit —
  `replace`/`map`/`add` and the model's replace-gated stage writes
  reproduced the treedef break *inside `State` carries* (fresh
  components inserted without re-attached metadata), and the aux hash
  being keys-only meant metadata differences also caused **silent
  recompiles** rather than errors.
- **No comparisons** (decision): `<`, `<=`, `>`, `>=` are not
  defined (elementwise comparisons are `f.data` territory); `==` is
  identity (pytree/jaxjit friendly); `__bool__` raises to catch
  `if f:` bugs early.
- **Storage contract** (jointly with cluster 04): the dynamic leaf
  `_data` is **storage-shaped** — halo-extended and stagger-padded
  per the space's layout (sections 5, 5.1; the layout is part of the
  function space, so transform outputs legally live in their pencils
  and mixing layouts in arithmetic is a `SpaceMismatchError`) —
  while `.data` is
  the **true-shape view** with halo and padding stripped (3.5).
  `with_data` and `grid.create_field(..., data=...)` accept
  *true-shape* arrays and route them through `decomposition.pad`
  (zero-filled ghost slots). Consumption-side halo contract (task
  1.8; owned by cluster 04, cross-ref): fields carry an internal
  per-name ghost validity (`halo_valid`, static aux in the treedef);
  constructed fields claim zero, the operator base syncs an operand
  exactly when an application needs more validity than it claims
  (memoized onto the operand), and kernel results carry their
  construction seam's claim. An operator never reads an invalid
  halo. Sync machinery, halo-extended layout, and the tracer-field
  dry run are specified in cluster 04.
- **`.xr` export is specified in cluster 04's "Export" subsection**
  (coordinate-label rules, xgcm staggered-dim naming, wavenumber
  coords, the multi-device gather path); the `xr` property here is
  only the field-side entry point delegating to it.
- **Removed relative to `FieldBase`/`ScalarField` today**: `fft` /
  `ifft` methods (transforms are grid-bound operators,
  `fr.operators.Fourier(grid, axes=...).forward/.backward`, sketch
  4.3), `sync` / `apply_water_mask` (grid/immersed-domain concerns,
  3.7), `set_zero` / `set_random` (functional construction),
  `get_mesh` (removed accessor, section 2 terminology note),
  `extend`/`topo` (ConstantSpace, 3.3), `sum`/`max`/`min` (escape
  hatch), `norm_l2` / `dot` (model-side diagnostics on explicit
  common spaces), `unpad` (below the operator layer).

---

### VectorField

Thin, metric-free collection of `ScalarField`s on different-but-
related spaces (section 2.4).

| Aspect | Value |
|--------|-------|
| Kind | concrete, subclassable (`State` subclasses it) |
| Pytree | jaxified; component mapping dynamic (leaves live in the components) |
| Iteration | 1 |
| Concept refs | 2.4, 2.5 (State.map consumer), 6.3, sketch 4.9 |

```python
"""A named collection of scalar fields (no metric)."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


@partial(fr.utils.jaxify, dynamic=("_components",))
class VectorField:
    """Thin container of ScalarFields; carries no metric."""

    def __init__(                                              # it-1
        self,
        components: (Mapping[str, ScalarField]
                     | Iterable[ScalarField]),
    ) -> None:
        """Build from a name→field mapping or an iterable of named
        fields; all components must share one grid."""
        ...

    # ================================================================
    #  Component access
    # ================================================================

    @property
    def components(self) -> Mapping[str, ScalarField]:         # it-1
        """Read-only name→field view, in declaration order."""
        ...

    @property
    def component_names(self) -> tuple[str, ...]:              # it-1
        """Component names in declaration order."""
        ...

    @property
    def grid(self) -> Grid:                                    # it-1
        """The common grid of all components."""
        ...

    def __getitem__(self, key: str | int) -> ScalarField:      # it-1
        """Component by name or positional index."""
        ...

    def __iter__(self) -> Iterator[ScalarField]:               # it-1
        """Iterate over component fields in declaration order."""
        ...

    def __len__(self) -> int:                                  # it-1
        """Number of components."""
        ...

    def __contains__(self, name: str) -> bool:                 # it-1
        """Whether a component of that name exists."""
        ...

    # ================================================================
    #  Functional surface
    # ================================================================

    def map(                                                   # it-1
        self,
        fn: Callable[[ScalarField], ScalarField],
    ) -> Self:
        """Apply ``fn`` to each component on its own space (2.4)."""
        ...

    def replace(self, **components: ScalarField) -> Self:      # it-1
        """Functional update of named components; re-attaches the
        incumbent component's metadata by key (2026-07-08 amendment
        below)."""
        ...

    def add(self, **contributions: ScalarField) -> Self:       # it-1
        """Functional accumulate: per component ``self[k] + v``
        through the metadata-preserving path (2026-07-08 amendment
        below; the literal ``replace(**{k: self[k] + v})`` spelling
        loses annotation and is superseded); unknown name ->
        MissingComponentError listing components. (Phase-2
        amendment, model design D1.5: the composer's primitive for
        summing tendency-contribution dicts.)"""
        ...

    # ================================================================
    #  Arithmetic (componentwise delegation; join rule per component)
    # ================================================================

    def __add__(self, other: Self | complex) -> Self: ...      # it-1
    def __radd__(self, other: complex) -> Self: ...            # it-1
    def __sub__(self, other: Self | complex) -> Self: ...      # it-1
    def __rsub__(self, other: complex) -> Self: ...            # it-1
    def __neg__(self) -> Self: ...                             # it-1
    def __pos__(self) -> Self: ...                             # it-1

    def __mul__(
        self, other: Self | ScalarField | complex,
    ) -> Self:
        """Componentwise: scalar scaling, ScalarField broadcast
        product, or component-by-component product."""
        ...                                                    # it-1

    def __rmul__(self, other: ScalarField | complex) -> Self:  # it-1
        ...

    def __truediv__(
        self, other: Self | ScalarField | complex,
    ) -> Self: ...                                             # it-1

    def __pow__(self, exponent: int | float) -> Self: ...      # it-1

    # ================================================================
    #  Diagnostics and export
    # ================================================================

    def has_nan(self) -> jax.Array:                            # it-1
        """0-d boolean array: any NaN in any component."""
        ...

    def block_until_ready(self) -> Self: ...                   # it-1

    @property
    def xr(self) -> xr.Dataset:                                # it-1
        """Dataset of the components' xarray exports."""
        ...

    def __repr__(self) -> str: ...                             # it-1
```

Semantics, invariants, error behavior:

- **Thin means thin** (section 2.4): no metric, no axis semantics,
  no inner product. The container does not know which component
  "belongs to" which coordinate; that association — like component
  *variance* — is carried by the component **spaces**. A covariant
  and a contravariant velocity component live on distinct factor
  spaces, distinguished by the designed-for
  `FunctionSpace.variance` descriptor (cluster 01), so the
  strict algebra rejects mixing variances exactly as it rejects
  mixing staggered positions, with no vector-level machinery. Index
  raising/lowering is an explicit metric-consuming operator reading
  the grid-owned metric (clusters 03/04); the sphere needs no
  metric-aware vector type.
- **Consequently there is no `dot` / `@` and no `div` method.**
  A naive componentwise dot of C-grid velocities is illegal under
  the strict algebra (components live on different spaces; their
  products cannot be summed without explicit conversion), and a
  correct inner product or divergence is metric- and axis-aware —
  an *operator* (`div: edge-normal → cell` on unstructured meshes,
  metric-aware forms on the sphere; sections 3.4, 6.3, 6.4). This
  intentionally drops `FieldBase.dot`, `norm_l2`, `norm_of_diff`,
  and `VectorField.div` from the field surface.
- **Component access convention** follows current FRIDOM:
  `vec["u"]` / `vec[0]` via `__getitem__`, iteration yields fields,
  and *named attribute access is added by subclasses as explicit
  properties* (as `nonhydro.State.u` does today). No dynamic
  `__getattr__` fallback — keeps the surface explicit, IDE-friendly,
  and safe under jaxify.
- **Constructor validation**: iterable input takes names from each
  field's `metadata.name`; duplicate names ⇒ `ValueError`; differing
  grids ⇒ `GridMismatchError`. When the duplicated name is the
  default `"unnamed"` (two components built without metadata), the
  message special-cases: it tells the user to *name the components*
  (`FieldMetadata.create(name=...)` / `f.with_metadata(name=...)`)
  rather than reporting a generic duplicate. Component *spaces* are
  unconstrained — the whole point is that components live on
  different spaces (2.4).
- **Componentwise arithmetic**: `vec op vec` requires identical
  component-name tuples (order included) ⇒ `ValueError` otherwise
  (a container-shape error, not a space error); each component pair
  then follows the ScalarField rule (grid identity, then the join),
  so per-component `GridMismatchError`s/`SpaceMismatchError`s
  propagate. `vec * field` broadcasts one
  `ScalarField` against every component (the classic
  `f_cor * velocity` with `f_cor` on a ConstantSpace-in-z product,
  sketch 4.5); `vec * scalar` is linear scaling.
- **Componentwise arithmetic preserves metadata (scan stability).**
  Unlike bare `ScalarField` arithmetic, every componentwise op keeps
  each component's metadata: component names are *structural* (they
  key the pytree), so the ScalarField default-metadata rule would
  make the scan carry `z_new = z + dt * dz` lose its names, change
  the treedef, and break `lax.scan`/`jit` round trips. Required test
  (model smoke level):
  `jax.tree_util.tree_structure(step(z)) == tree_structure(z)`.
- **Amended (Phase-2 reconciliation, 2026-07-08): container ops
  re-attach incumbent metadata by key.** Componentwise arithmetic
  continues to preserve component metadata as above (under the
  annotation-exempt rule this is no longer load-bearing for the
  treedef — it keeps component annotation authoritative). `add` is
  specified to run through the same metadata-preserving path (the
  docstring's former literal spelling is superseded), and
  `replace`/`map` — and with them the model's replace-gated stage
  application, which routes through `replace` — re-attach the
  *incumbent* component's metadata to the incoming field, keyed by
  component name. A genuinely new component (no incumbent under that
  key) keeps the metadata it was given. Rationale: the reconciliation
  audit found `replace`/`map` inserting fresh components *without*
  re-attaching metadata, so every replace-gated stage write and the
  composer's add path broke carry-treedef stability on step 1 of any
  multistep tendency ring — see the `ScalarField` metadata amendment
  for the full finding.
- **`map` is the functional surface consumed by eigenmode objects
  and spectra-based ICs** (sketch 4.9): `fn` receives each component
  on its own space and must return a `ScalarField`; the result keeps
  names and order. `map` never inspects spaces — per-component
  space changes (e.g. transforms) are fine and land in the returned
  collection.
- **Pytree note**: the component mapping is flattened **keyed, in
  component-declaration order**, with names in the static treedef;
  renaming or re-keying a component changes the treedef (retrace),
  mirroring the metadata rule for scalars. This relies on the jaxify
  flatten-order **prerequisite** stated in the shared pytree section
  (declaration-order tuples instead of today's unordered `set`).

---

### TensorField

Rank-2 companion of `VectorField` for stress/strain-type diagnostics;
designed-for, kept deliberately brief (section 2.4).

| Aspect | Value |
|--------|-------|
| Kind | concrete |
| Pytree | jaxified; component mapping dynamic |
| Iteration | designed-for |
| Concept refs | 2.4 |

```python
"""A rank-2 collection of scalar fields."""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr


@partial(fr.utils.jaxify, dynamic=("_components",))
class TensorField:
    """Thin rank-2 container of ScalarFields; carries no metric."""

    def __init__(                                              # later
        self,
        components: Mapping[tuple[str, str], ScalarField],
    ) -> None:
        """Build from an (i, j)-name → field mapping; one grid."""
        ...

    @property
    def components(                                            # later
        self,
    ) -> Mapping[tuple[str, str], ScalarField]:
        """Read-only (i, j) → field view."""
        ...

    def __getitem__(                                           # later
        self, key: tuple[str | int, str | int],
    ) -> ScalarField:
        """Component by (row, col) name or index pair."""
        ...

    def map(                                                   # later
        self,
        fn: Callable[[ScalarField], ScalarField],
    ) -> Self:
        """Apply ``fn`` to each component on its own space."""
        ...

    # componentwise +, -, unary -, scalar * / **, ScalarField
    # broadcast — same delegation pattern as VectorField.
```

Notes: same thin-container rules as `VectorField` — no metric, no
trace/contraction/transpose-with-metric methods (those are operators
consuming the grid metric). Symmetric-storage optimizations are an
implementation concern below this surface. Not implemented in
iteration 1; nothing in the cluster's design depends on it.

---

### State (contract only)

Model-side specialization: **`State` *is* a `VectorField`** (section
2.4). The model side is out of scope for the grid redesign; this
section records only the inherited surface plus the extension
contract that the grid cluster promises to support.

| Aspect | Value |
|--------|-------|
| Kind | concrete per model package, subclass of `VectorField` |
| Pytree | inherited from `VectorField` |
| Iteration | 1 (nonhydro port, ROADMAP Phase 1); constructor details deferred to Phase 2 |
| Concept refs | 2.4, 2.5 (eigenmode reuse), sketch 4.9 |

```python
"""Model state vector (model package, e.g. fridom.nonhydro)."""
from __future__ import annotations

import fridom.framework2 as fr


class State(fr.grid.VectorField):
    """State vector of a model; physics-carrying VectorField."""

    # Inherited and used as-is: components, __getitem__, __iter__,
    # map, replace, componentwise arithmetic, has_nan, xr.
    # Where the design notes say ``State.map`` (eigenmode scaling,
    # sketch 4.9), the surface is the inherited VectorField.map.

    # Extension contract (model-side):
    #   - canonical prognostic components exposed as explicit
    #     properties (z.u, z.v, z.w, z.b), successor of today's
    #     nonhydro.State properties;
    #   - user-registered extra fields (tracers, diagnostics) become
    #     ordinary named components, declared model-side (the
    #     successor of mset custom fields; flags like NO_ADV move
    #     here from FieldMetadata);
    #   - physics diagnostics (ekin, pot_vort, cfl, ...) are
    #     properties/methods written in the field algebra of this
    #     document — explicit .to conversions, dispatched products,
    #     integrate — never against CENTER/FACE assumptions (6.2);
    #   - physics parameters are module-owned, never grid-owned
    #     (2.6): scalar parameters resolve through the Phase-2
    #     parameter table (provides/requires, model design D2),
    #     spatially-varying ones are AUXILIARY state components —
    #     State itself holds no parameters and no module
    #     back-reference (Phase-2 amendment; supersedes the earlier
    #     "model settings/parameter objects" phrasing).
    ...
```

Constraints this cluster imposes on `State` authors: the component
set and names must be stable over a model run (pytree treedef) —
componentwise arithmetic preserves metadata precisely so that the
scan carry keeps its structure; the required test
`tree_structure(step(z)) == tree_structure(z)` (see `VectorField`)
is part of every model port;
tendency construction is functional (`replace`, `map`, arithmetic) —
there is no in-place component mutation in the new design; eigenmode
objects reuse `State` with components on per-variable coefficient
spaces (section 2.5), which works because `VectorField` places no
constraint on component spaces. The constructor signature (what
replaces `mset`) is owned by the Phase 2 composition redesign, not by
this document.

---

## Open questions

Closed by review: the broadcast dispatch entry
(`("broadcast", ConstantSpace)` → `ConstantBroadcast`, cluster 03);
the `CumulativeIntegral` codomains and integration-constant
convention (cluster 03); the grid's pytree status (fully static,
identity-hashed — the pytree-child alternative is recorded as
rejected above); the metadata/scan collision (componentwise
arithmetic preserves metadata); and the jaxify flatten-order
question, which is upgraded to a stated **prerequisite** in the
shared pytree section, not left open.

1. **`to` single-factor shorthand**: `g.to(mx.center)` (replace one
   factor, keep the rest) is proposed here for ergonomics; confirm it
   does not blur the "target space must be named explicitly" line the
   notes draw elsewhere (mandatory `space` args, 3.10).
2. **Metadata propagation rule (scalar level)**: the "same quantity
   keeps metadata, new quantity resets" split for bare `ScalarField`
   ops is a pragmatic default (the vector/state level is decided:
   preserved); fine-tune the exact method list during the nonhydro
   port (ROADMAP Phase 1). Since the 2026-07-08 amendment the split
   is purely annotational (never structural), so fine-tuning it can
   no longer break treedefs.
3. **Migration mutation shim** — *resolved by the Phase-2 model
   design (D1.5, `design/specs/model/01_concepts.md`)*: ports go
   fully functional immediately; the shims are raising teaching
   errors only (`ImmutableStateError` from property setters and
   `__setitem__`, plus the raising `data` setter above) — a
   "temporarily working" mutation shim is rejected as silently wrong
   under jit and unable to catch the dominant `f.arr += ...`
   pattern anyway.
