---
status: normative
date: 2026-07-06
---

# Grid abstraction redesign — Class designs: product spaces and fields

Part of the grid redesign notes; see [`../00_overview.md`](../00_overview.md)
for the document map. Status: implemented (Phase 1 landed; kept as the normative reference).
Signatures are the intended public API for `framework2.grid`; the
numbered concept sections remain the normative reference.

This document owns the **TensorProductSpace and Field cluster**:
`TensorProductSpace`, `SpaceMismatchError`, `FieldMetadata`,
`ScalarField`, `VectorField`, `TensorField`, and the `State` contract.
Neighboring clusters are referenced by name only:

- `Mesh`, `FunctionSpace` families, `ConstantSpace`, `fr.Real` /
  `fr.Complex` — [`meshes.md`](meshes.md), [`spaces.md`](spaces.md).
- Operators, transforms, `Symbol`, the `OperatorRegistry` —
  [`operators_base.md`](operators_base.md).
- `Grid`, decomposition, `grid.create_field`, coordinate accessors,
  `ImmersedDomain`, metrics —
  [`grid.md`](grid.md), [`decomposition.md`](decomposition.md).

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
*join* `A ∨ B` exists iff `A` and `B` carry the **same layout**
(section 5.1 — the lifts never touch it; the same bare space in two
layouts raises `SpaceMismatchError` with a reshard hint, and **no
implicit reshard exists in field arithmetic**) and `A` and `B` have
factors on the same meshes
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

**Rejected alternative:** carrying the grid as a
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
(pytree section above), the `f.grid is g.grid` check is exact inside
and outside jit.

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

    @property
    def layout(self) -> Layout | None:                         # it-1
        """Negotiated device layout, or None for a bare space
        (section 5.1)."""
        ...

    @property
    def bare(self) -> TensorProductSpace:                      # it-1
        """The layout-free interned variant (self if bare)."""
        ...

    def with_layout(                                           # it-1
        self, layout: Layout | None,
    ) -> TensorProductSpace:
        """The interned variant carrying ``layout`` (grid-minted)."""
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
  through a product: on a 1D grid `f.function_space.bare is
  mx.center` holds. The product surface used by this document
  (`factors`, `names`, `factor(name)`, `shape`, `scalars`,
  `as_complex()`, `layout`, `bare`, `with_layout`) is
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
- **Layout is an optional defining attribute** (section 5.1): the
  intern key is the factor-id tuple *plus the layout when set* — the
  `variance` precedent, so bare keys are unchanged. `Layout` is a
  pure combinatorial value (cluster 04), so laid-out spaces remain
  grid-reference-free static keys. Mesh factories and `of`/`*` mint
  bare spaces only; laid-out variants come from the grid seams —
  `create_field` attaches the default layout to bare arguments, the
  operator application path threads the domain layout, and `Reshard`
  (cluster 03) is the explicit layout-changing operator.
  `factor(name)` returns **bare** factors (dispatch keys never see
  layouts); `replace` and `as_complex` preserve the layout. A
  layout-only mismatch renders as `same space, layouts differ:
  {x: 'p0' vs local, ...}; use .reshard(...)`.
- Products hold **no grid reference** and no arrays (section 2.7);
  they are pure static keys. Whether a product's factors match a
  given grid's meshes is validated by `grid.create_field` (cluster
  04), not here.

Rejected alternatives (recorded in the notes): nested binary
products (breaks flat names and interning, 2.3); relational
properties like `space.shift()` / `space.spectral` (2.2); a
"wet-DOFs-only" space (3.7).

---

