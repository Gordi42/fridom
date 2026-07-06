# Grid abstraction redesign — Class designs: product spaces and fields

Part of the grid redesign notes; see [`../00_overview.md`](../00_overview.md)
for the document map. Status: draft class design, no implementation.
Signatures are the intended public API for `framework2.grid`; the
numbered concept sections remain the normative reference.

This document owns the **TensorProductSpace and Field cluster**:
`TensorProductSpace`, `SpaceMismatchError`, `FieldMetadata`,
`ScalarField`, `VectorField`, `TensorField`, and the `State` contract.
Neighboring clusters are referenced by name only:

- `Mesh`, `FunctionSpace` families, `ConstantSpace`, `fr.Real` /
  `fr.Complex` — [`01_meshes_and_spaces.md`](01_meshes_and_spaces.md).
- Operators, transforms, `Symbol`, the `OperatorRegistry` —
  [`03_operators.md`](03_operators.md).
- `Grid`, decomposition, `grid.create_field`, coordinate accessors,
  `ImmersedDomain`, metrics —
  [`04_grid_and_decomposition.md`](04_grid_and_decomposition.md).

Iteration tags used in the skeletons: `# it-1` (implemented in
iteration 1) and `# later` (designed-for, deferred; see
[`../07_iteration1_api.md`](../07_iteration1_api.md)).

---

## Module placement

The code lives in `fridom.framework2.grid` (part of the new parallel
`fridom.framework2` package), renamed to `fridom.framework.grid` at
cutover. Layout:

| Class | Module | Transitional import |
|-------|--------|---------------------|
| `SpaceMismatchError` | `framework2/grid/errors.py` | `fr.grid.SpaceMismatchError` |
| `GridMismatchError` | `framework2/grid/errors.py` | `fr.grid.GridMismatchError` |
| `TensorProductSpace` | `framework2/grid/spaces/tensor_product.py` | `fr.grid.TensorProductSpace` |
| `FieldMetadata` | `framework2/grid/fields/metadata.py` | `fr.grid.FieldMetadata` |
| `ScalarField` | `framework2/grid/fields/scalar_field.py` | `fr.grid.ScalarField` |
| `VectorField` | `framework2/grid/fields/vector_field.py` | `fr.grid.VectorField` |
| `TensorField` | `framework2/grid/fields/tensor_field.py` | `fr.grid.TensorField` |
| `State` | model packages (e.g. `fridom.nonhydro.state`) | `nh.State` |

All `__init__.py` files follow the lazypimp convention. After the
final rename these names are re-exported at the `fr.` top level
(replacing today's `fr.ScalarField` etc.); users never spell the
submodule paths.

---

## Shared semantics: lifts, joins, and the strict algebra

All binary field arithmetic in this cluster is governed by one rule,
stated here once and referenced from every dunder. It implements
sections [3.1](../02_rules.md#31-strict-space-algebra) and
[3.3](../02_rules.md#33-constantspace-replaces-topo-with-automatic-broadcast).

**Sanctioned per-factor lifts.** Exactly two implicit conversions
exist; both are exact, so neither hides a numerics choice:

1. **Constant broadcast** (section 3.3): a `ConstantSpace` factor on
   mesh m lifts to any factor space on mesh m. Realization is the
   `ConstantBroadcast` operator registered at
   `("broadcast", ConstantSpace)` (cluster 03): the nodal/average
   entries are the plain array broadcast (`# it-1`); the
   coefficient-factor entry is the exact delta embedding into the
   zero mode (`# later` — a constant is *not* a constant coefficient
   array).
2. **Real → complex promotion** (section 3.1): a factor lifts to its
   `as_complex()` variant. Realization is a dtype cast where the
   storage dtype changes at all (the dtype is derived per section
   2.4, never stored).

**The join.** For two product spaces `A`, `B` on the same grid, the
*join* `A ∨ B` exists iff `A` and `B` have factors on the same meshes
and, per mesh, the factors are identical or related by a chain of the
two lifts. The join is the per-factor least upper bound (full factor
beats `ConstantSpace`; complex variant beats real). Because spaces are
interned, all of this is identity comparison plus two `is`-checks per
factor — cheap and jit-static.

**The rule.** Binary ops first require **grid identity**:
`f.grid is g.grid`, else `GridMismatchError` — meshes (and therefore
interned spaces) may legally be shared across grids, so spaces alone
cannot distinguish operands living under different decompositions.
Then `f + g`, `f - g`, `f * g`, `f / g` compute the join of
the operand spaces, lift both operands to it, and apply the operation
there. If the join does not exist, they raise `SpaceMismatchError`.
Nothing else is implicit: staggering conversion, inter-origin phase
shifts, and transforms are always explicit (`.to`, operators).

**Pytree treatment (whole cluster).** Spaces, the grid, and
`FieldMetadata` are pure *static aux*: spaces are interned singletons
used as jit/dispatch keys (section 2.2) and never appear as pytree
leaves; metadata is a frozen, hashable dataclass in the treedef; the
**grid is fully static** — `Grid` defines explicit *identity*
`__eq__`/`__hash__` (cluster 04), exactly like spaces, so grid
identity keys the jit cache. Its attachments (`ImmersedDomain`,
`CoordinateMapping`) are static descriptors that materialize arrays
on demand at trace time; time-dependent geometry is module-owned
state fields consumed through explicit-data accessors (cluster 04
owns the details). Field *arrays* are the only dynamic leaves.
Concretely: `ScalarField` is
`@partial(fr.utils.jaxify, dynamic=("_data",))`; `VectorField` /
`TensorField` are jaxified with their component mapping dynamic (the
`ScalarField` children carry the leaves). Consequence: a change of
space, grid, or metadata changes the treedef and retriggers jit
tracing — intended for spaces and grids (they are the cache key) and
harmless for metadata because component names are stable within a
run.

**Rejected alternative (review round 2):** carrying the grid as a
pytree *child* of the field with dynamic attachment leaves (immersed
fraction, mapping parameter fields). Reviewers showed it fails three
ways: it is **cyclic** — the immersed fraction is itself a
`ScalarField` carrying `_grid`, and jax pytrees have no cycle
detection, so flattening recurses to a `RecursionError`; it is
**leaf-duplicating** — N fields in a jitted signature would ship N
copies of the fraction array as arguments; and it is **incompatible
with module updates under a traced scan** — a module cannot swap a
leaf buried inside every field's grid mid-trace. The static grid has
a direct payoff in this cluster: jit round trips preserve grid
*identity* (unflattening never clones grids), so `VectorField`'s
same-grid validation and the `f.grid is g.grid` precondition stay
sound inside and outside jit.

**Prerequisite (jaxify flatten order).** Today's `fr.utils.jaxify`
keeps dynamic attribute names in an unordered `set`, so flatten
order depends on `PYTHONHASHSEED` — a latent multi-host bug
(leaf-order mismatch across processes) even before framework2.grid. This is a
stated **prerequisite**, not an open question: jaxify must store
dynamic attrs in declaration order (a tuple) before framework2.grid lands, and
`VectorField` additionally needs keyed flattening in
component-declaration order (see its pytree note).

**Fields are immutable.** Construction is functional (section 3.10;
mutating `f.set(...)` was rejected there). There is no `.arr = ...`
setter; updates go through `with_data` / `with_metadata` /
`VectorField.replace`. This replaces the mutation-based surface of the
current `FieldBase` (`sync` in-place, `set_zero`, `set_random` — the
latter superseded by `grid.random.normal(space, seed)`).

---

### SpaceMismatchError

The strict-algebra exception: raised when operands' function spaces
cannot be joined by the sanctioned lifts.

| Aspect | Value |
|--------|-------|
| Kind | concrete, final |
| Pytree | n/a (exception type) |
| Iteration | 1 |
| Concept refs | 3.1, 3.3, sketches 4.1, 4.3 |

```python
"""Exception for cross-space field operations."""
from __future__ import annotations


class SpaceMismatchError(TypeError):
    """Raised when fields on incompatible function spaces are combined."""

    def __init__(
        self,
        msg: str,
        *,
        left: object | None = None,
        right: object | None = None,
        operation: str | None = None,
        mismatched_names: tuple[str, ...] = (),
    ) -> None:
        """Store the offending spaces for programmatic inspection."""
        ...

    left: object | None       # space of the left operand
    right: object | None      # space of the right operand (None if unary)
    operation: str | None     # "+", "*", "to", "forward", ...
    mismatched_names: tuple[str, ...]   # factor names that differ
```

Notes:

- Subclasses `TypeError`: the operand *combination* is unsupported,
  the moral analogue of `unsupported operand type(s)`.
  `mismatched_names` lists exactly the factor names whose factors
  differ beyond the sanctioned lifts; the intended message format is
  a **per-factor diff over those names** plus the conversion hint,
  e.g. `cannot add fields: x: Right vs Center (y, z agree); use
  .to(...) for an explicit conversion`.
- Raised by: all binary arithmetic when the join does not exist; by
  `.to` when the registered conversion's codomain does not equal the
  requested target factor (section 3.4); and by operator application
  to a field outside the operator's domain (cluster 03 imports the
  same exception from `framework2/grid/errors.py` — one exception type for one
  rule).
- Not raised for a *missing dispatch entry* (e.g. `f / g` on a
  coefficient space with no registered `"divide"`): that is a
  registry-resolution error owned by cluster 03. The distinction:
  `SpaceMismatchError` means "illegal by the algebra", the dispatch
  error means "legal but no default registered".

---

### GridMismatchError

Raised when binary field operations combine fields created on
different grids.

| Aspect | Value |
|--------|-------|
| Kind | concrete, final |
| Pytree | n/a (exception type) |
| Iteration | 1 |
| Concept refs | 2.6, 2.7, section 5 |

```python
"""Exception for cross-grid field operations."""
from __future__ import annotations


class GridMismatchError(TypeError):
    """Raised when fields on different grids are combined."""

    def __init__(
        self,
        msg: str,
        *,
        left: object | None = None,
        right: object | None = None,
        operation: str | None = None,
    ) -> None:
        """Store the offending grids for programmatic inspection."""
        ...

    left: object | None       # grid of the left operand
    right: object | None      # grid of the right operand
    operation: str | None     # "+", "*", "to", ...
```

Notes: lives beside `SpaceMismatchError` in `framework2/grid/errors.py` and is
checked *before* the space join. Rationale: meshes — and therefore
interned spaces — may legally be shared across grids (same factors,
same names, different decomposition or dispatch defaults), so the
space-join check alone cannot detect operands living under different
decompositions. Because the grid is static aux with identity hashing
(pytree section above) and jit round trips never clone grids, the
`f.grid is g.grid` check is exact inside and outside jit.

---

### TensorProductSpace

Flat, associative, interned product of per-mesh factor spaces —
the space a multi-dimensional field lives on (section 2.3).

| Aspect | Value |
|--------|-------|
| Kind | concrete, final (not subclassed; variability lives in factors) |
| Pytree | static — interned, hashable by identity, never a leaf |
| Iteration | 1 |
| Concept refs | 2.2, 2.3, 3.1, 3.3, 3.5, sketches 4.1, 4.3–4.8 |

```python
"""Flat tensor product of per-mesh factor spaces."""
from __future__ import annotations


class TensorProductSpace:
    """Flat, interned product of factor spaces, one per mesh."""

    def __init__(self, factors: tuple[FunctionSpace, ...]) -> None:
        """Plumbing constructor; not interned — use ``of`` or ``*``."""
        ...

    # ================================================================
    #  Construction (interned)
    # ================================================================

    @classmethod
    def of(                                                    # it-1
        cls,
        *spaces: FunctionSpace | TensorProductSpace,
    ) -> FunctionSpace | TensorProductSpace:
        """Normalize, validate, and intern a product of spaces."""
        ...

    def __mul__(                                               # it-1
        self,
        other: FunctionSpace | TensorProductSpace,
    ) -> TensorProductSpace:
        """Tensor product (flat, associative): ``self ⊗ other``."""
        ...

    # ================================================================
    #  Properties
    # ================================================================

    @property
    def factors(self) -> tuple[FunctionSpace, ...]:            # it-1
        """The per-mesh factor spaces, flat, in coordinate order."""
        ...

    @property
    def names(self) -> tuple[str, ...]:                        # it-1
        """All coordinate names, concatenated across factors."""
        ...

    @property
    def shape(self) -> tuple[int, ...]:                        # it-1
        """True global DOF shape: concatenated factor shapes (3.5)."""
        ...

    @property
    def ndim(self) -> int:                                     # it-1
        """Number of array axes, ``len(self.shape)``."""
        ...

    @property
    def scalars(self) -> Scalars:                              # it-1
        """The Körper (``Scalars`` enum, cluster 01): ``fr.Complex``
        if any factor is complex, else ``fr.Real``."""
        ...

    # ================================================================
    #  Factor access and derived products
    # ================================================================

    def factor(self, name: str) -> FunctionSpace:              # it-1
        """Return the factor space contributing coordinate ``name``."""
        ...

    def replace(                                               # it-1
        self, **by_name: FunctionSpace,
    ) -> TensorProductSpace:
        """New interned product with named factors substituted."""
        ...

    def as_complex(self) -> TensorProductSpace:                # it-1
        """The product of ``factor.as_complex()`` for all factors."""
        ...

    def __iter__(self) -> Iterator[FunctionSpace]:             # it-1
        """Iterate over the factor spaces."""
        ...

    def __len__(self) -> int:                                  # it-1
        """Number of factors (meshes), not coordinate names."""
        ...

    def __repr__(self) -> str:                                 # it-1
        """Render as ``Center(x) ⊗ Center(y)``."""
        ...

    # __eq__ / __hash__ deliberately NOT overridden: interning makes
    # equality an identity comparison (section 2.2 / 2.3).
```

Semantics and invariants:

- **Normalization on construction** (`of`, and `*` which routes
  through it): nesting is collapsed — any `TensorProductSpace`
  argument contributes its factors, never itself — so no factor is a
  product; factors keep left-to-right order; the result is looked up
  in the intern table keyed by the factor identity tuple, so
  `a * (b * c)` *is* `(a * b) * c` *is* the same object. The intern
  table is a `weakref.WeakValueDictionary` keyed on the factor-id
  tuple: unreferenced products are collected, meshes are not pinned
  by the table, and no state leaks across tests. Flatness is
  at the mesh level: a 2D `SphereMesh` factor stays one entry while
  contributing two names (section 2.3).
- **Duplicate coordinate names are rejected**: `of` raises
  `ValueError` (not `SpaceMismatchError` — this is a construction
  error, not an algebra violation) naming the duplicates. Since names
  are per mesh, this also rejects two factors on the same mesh.
- **The empty product is rejected** (`ValueError`). There is no unit
  object; "trivial along mesh m" is spelled per mesh as
  `m.constant` (`ConstantSpace`, section 3.3).
- **A lone factor space is usable wherever a product is expected.**
  Decision: `TensorProductSpace.of(s)` with a single factor returns
  `s` itself — the product of one thing is that thing. There is no
  1-factor wrapper, so identity never depends on whether a space went
  through a product: on a 1D grid `f.function_space is mx.center`
  holds. The product surface used by this document (`factors`,
  `names`, `factor(name)`, `shape`, `scalars`, `as_complex()`) is
  therefore a shared **product protocol** that single factor spaces
  implement too (`s.factors == (s,)`, `s.factor(name)` returns `s`
  for its own names). The protocol defaults for factors (`factors`,
  `names`, `factor(name)`) are defined on the `FunctionSpace` base
  (cluster 01); this document owns the protocol's meaning.
  Consumers should type against the union
  `FunctionSpace | TensorProductSpace` (a `TypeAlias`, e.g.
  `SpaceLike`, exported from `framework2.grid`).
- `factor(name)` raises `KeyError` for unknown names. It is the fixed
  anchor used by eigenvalue queries:
  `u_hat.function_space.factor("x")` (sketch 4.6).
- `replace` is the codomain-building primitive for operators and
  reductions (`diff` maps the x factor `Center → Right`; `integrate`
  maps it to `ConstantSpace`). Substituting via *any* name of a
  multi-name (2D-mesh) factor replaces that whole factor; the
  replacement must live on the same mesh (`ValueError` otherwise).
- `shape` concatenates per-factor true shapes (section 3.5): masked
  domains keep full product shape (section 3.7), coefficient spaces
  of real origins contribute half-spectrum extents (section 3.2).
- `scalars` is derived, not stored: `fr.Complex` iff any factor's
  scalars are complex (`fr.Real` / `fr.Complex` are the module-level
  aliases of the `Scalars` enum members, cluster 01 — enum members,
  not classes). Mixed products (`Complex(x) ⊗ Real(y)`) are
  legal intermediate states; the promotion join complexifies only the
  factors that need it.
- Products hold **no grid reference** and no arrays (section 2.7);
  they are pure static keys. Whether a product's factors match a
  given grid's meshes is validated by `grid.create_field` (cluster
  04), not here.

Rejected alternatives (recorded in the notes): nested binary
products (breaks flat names and interning, 2.3); relational
properties like `space.shift()` / `space.spectral` (2.2); a
"wet-DOFs-only" space (3.7).

---

### FieldMetadata

Shrunken successor of today's `FieldMetadata`: pure annotation for
naming and I/O, no discretization content (section 2.4).

| Aspect | Value |
|--------|-------|
| Kind | concrete, frozen dataclass |
| Pytree | static (part of the treedef via its `ScalarField` owner) |
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
| Pytree | `jaxify, dynamic=("_data",)`; grid/space/metadata static aux (grid identity-hashed) |
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
        """Raw local array at true shape (halo/padding stripped)."""
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
        through decomposition.pad + grid.sync)."""
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
  use inside jit (they already hold validated shards). `__init__`
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
- **No comparisons** (decision): `<`, `<=`, `>`, `>=` are not
  defined (elementwise comparisons are `f.data` territory); `==` is
  identity (pytree/jaxjit friendly); `__bool__` raises to catch
  `if f:` bugs early.
- **Storage contract** (jointly with cluster 04): the dynamic leaf
  `_data` is **storage-shaped** — halo-extended and stagger-padded
  per the negotiated per-mesh layout (section 5) — while `.data` is
  the **true-shape view** with halo and padding stripped (3.5).
  `with_data` and `grid.create_field(..., data=...)` accept
  *true-shape* arrays and route them through `decomposition.pad` and
  `grid.sync`, so stored halos are always valid. Iteration-1 halo
  contract (owned by cluster 04, cross-ref): operator inputs may
  assume valid halos, and **every operator application returns a
  synced field**; eliding redundant syncs along traced operator
  chains is the designed-for optimization. Sync machinery,
  halo-extended layout, and the tracer-field dry run are specified
  in cluster 04.
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
        """Functional update of named components."""
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
    #   - physics parameters come from model settings/parameter
    #     objects, never from the grid (2.6).
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

Closed by cross-review rounds 1–2: the broadcast dispatch entry
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
   port (ROADMAP Phase 1).
3. **Migration mutation shim**: old model code mutates `z.u`; decide
   whether ports go fully functional immediately (`replace`) or a
   temporary deprecation shim on `State` properties is worth it.
