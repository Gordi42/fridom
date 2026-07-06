# Grid abstraction redesign — Class designs: operators

Part of the grid redesign notes; see [`../00_overview.md`](../00_overview.md)
for the document map. Status: draft class design, no implementation.
Signatures are the intended public API for `framework.grid2`; the
numbered concept sections remain the normative reference.

> **Operator algebra folded in (D8).** The operator *algebra* —
> composition `@`, sums, `c * A` scaling, `Identity`/`Zero`/`Block`,
> axis binding `op["x"]`, tuple signatures — is designed in the sibling
> note set
> [`../../operator_design/`](../../operator_design/00_overview.md)
> (authored on `dev`); the reconciliation decisions are in
> [`operator_algebra_merge.md`](operator_algebra_merge.md). Those
> decisions are now applied to the **base hierarchy**
> ([Operator algebra](#operator-algebra), bind-only `__call__`,
> `bound_axis`), the **composed operators** (algebra-derived factories),
> and the **registry** below. Bind-only axis naming (`op["x"]` replaces
> the `axis=` keyword) rippled through every operator's `_apply`
> signature — now uniformly `_apply(self, f)`, with the separable
> `_apply_factor(self, f, axis)` unchanged; that pass is applied
> throughout this document.

This document owns the **Operator cluster**: the operator base
hierarchy, the concrete stencil/nodal operators, transforms, `Symbol`,
the binary product operators, composed operators, and the dispatch
registry class. Sibling documents own the seams referenced here:
`Mesh` / `FunctionSpace` / `ConstantSpace` / `fr.Real` / `fr.Complex`
([`01_meshes_and_spaces.md`](01_meshes_and_spaces.md)),
`TensorProductSpace` / `ScalarField` / `VectorField` / `State` /
`SpaceMismatchError`
([`02_product_spaces_and_fields.md`](02_product_spaces_and_fields.md)),
and `Grid` / decomposition / `grid.evaluation_nodes` /
`grid.wavenumbers` / measure and metric fields / `HaloTracer`
([`04_grid_and_decomposition.md`](04_grid_and_decomposition.md)).

---

## Module placement

Transitional package `fridom.framework.grid2` (renamed to
`framework.grid` once the old grid is deleted,
[§8](../00_overview.md#8-migration-strategy)). Free-standing operators
are re-exported as the collection namespace `fr.operators` via the
usual lazypimp `__init__.py`.

```
src/fridom/framework/grid2/operators/
    __init__.py           # lazypimp; aliased as fr.operators
    base.py               # Operator, UnaryOperator, BinaryOperator,
                          # SeparableOperator, OperatorRequirements,
                          # EigenbasisError; algebra objects: Identity,
                          # Zero, Composite, SeparableComposite,
                          # OperatorSum, ScaledOperator, Block, Dispatched
    registry.py           # OperatorRegistry, DispatchError
    symbol.py             # Symbol
    finite_difference.py  # FiniteDifference
    interp.py             # LinearInterp
    reconstruct.py        # LinearReconstruction, WenoReconstruction
    flux_diff.py          # FluxDifference, DualFluxDifference,
                          # FaceDifference, FVDerivative
    spectral.py           # SpectralDerivative, PhaseShift, SincShift
    transform.py          # Transform (ABC)
    fourier.py            # Fourier
    trig.py               # Sine, Cosine
    chebyshev.py          # Chebyshev
    products.py           # CollocationProduct, Hadamard, Convolution,
                          # ConstantBroadcast, Divide, Power, Abs,
                          # Where
    integrate.py          # Integral, CumulativeIntegral
    composed.py           # Laplacian, Gradient, Divergence, Curl,
                          # RaiseIndex, LowerIndex  (designed-for)
    dealias.py            # PadFactor, degree()
    combinators.py        # transform_once          (designed-for)
```

All classes below follow the repo conventions
([`AGENTS.md`](../../../AGENTS.md)): PEP 604 unions,
`from __future__ import annotations`, `@fr.utils.jaxify` registration.
Skeletons omit docstring bodies and implementations; `Grid`,
`FunctionSpace`, `TensorProductSpace`, `ScalarField`, `VectorField`
are the sibling-cluster types.

**Pytree rule for the whole cluster**
([§2.7](../01_concepts.md#27-where-coordinate-data-lives)): operators
are **static structure** — order, stencil pattern, axes, pad factor are
part of the dispatch/jit key. Spacing-dependent *coefficient values*
are never stored; they are derived from the grid's measure fields
(`grid.measure(space, name=...)`, doc 04, iteration 1) at trace time.
The dynamic-leaf carriers in this cluster are `Symbol` (`_data`) and a
`ScaledOperator` with a *field* coefficient ([Operator algebra](#operator-algebra),
D8) — every other operator, including all other algebra objects, is
fully static.
The grid itself is **fully static** (G1, doc 04): `Transform._grid`
is static structure (plans, layouts, refined meshes) with no dynamic
pytree leaves; all grid-materialized data is read at trace time
through the grid accessors (and, where per-field, through the
operand's `f.grid`). Consequently **registry entries must be
array-free**: an operator never stores per-grid arrays — it derives
them at trace time — in contrast to today's `_dx1` state in
`grid/cartesian/finite_differences.py`.

---

## Base hierarchy

The split is: one root ABC (`Operator`) carrying the surface every
operator shares (codomain resolution, per-factor requirements,
eigenvalues); two arity ABCs (`UnaryOperator`, `BinaryOperator`)
fixing the call signatures; one workhorse intermediate
(`SeparableOperator`) for separable 1D kernels, which is where almost
every concrete stencil operator lives. A mixin-based design
(`SeparableMixin` + free combination) was rejected: arity and
separability are not independent axes in practice (all separable
kernels are unary), and a linear hierarchy keeps the shared template
machinery — the final `__call__` with its halo-tracer interception,
and the `⊗`-lifting — in exactly one place each.

On top of these sit the **algebra objects**
([Operator algebra](#operator-algebra) below): the composites, sums,
blocks, scalings, and placeholders that `@` / `+` / `*` / `Block(...)`
/ `Dispatched(...)` build — the derived sixth operator kind of
operator_design §2.1, ordinary operators themselves. They follow the
merge decisions in
[`operator_algebra_merge.md`](operator_algebra_merge.md); two of those
shape the base surface directly. **Axis naming is bind-only** (D2):
`op["x"]` is the sole way to name an axis — there is no `axis=` call
keyword — so "can this operator name an axis?" is a matter of type
(`SeparableOperator` overrides `__getitem__`; whole-space operators
inherit a raising default). And **operators are interned** (D6): a
bound variant or a composite is canonicalized on its static structure,
because the cluster's identity-hash invariant needs structurally-equal
operators to be the same object.

### Operator

Root ABC: a typed, free-standing, parameterized map between function
spaces ([§2.5](../01_concepts.md#25-operator--typed-maps-between-spaces)).

| | |
|---|---|
| Kind | ABC |
| Pytree | static (jaxify; no dynamic leaves) |
| Iteration | 1 |
| Concept refs | §2.5, §2.7, §3.4, §5 |
| Module | `grid2.operators.base` |

```python
@fr.utils.jaxify
class Operator(ABC):
    """Typed map between function spaces; free-standing and grid-free."""

    #: suggested registry kind for convenience registration; the
    #: authoritative kind is always the registry key (section 3.4)
    dispatch_kind: ClassVar[str | None] = None

    @abstractmethod
    def codomain(
        self, *domains: FunctionSpace,
    ) -> FunctionSpace | tuple[FunctionSpace, ...]:
        """Resolve the codomain from the domain space(s); a tuple for
        direct-sum (vector/tensor) signatures (§3.1)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """Per-factor decomposition requirements (halo, layout)."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """Diagonal symbol relative to a diagonalizing basis."""
        ...

    # ------------------------------------------------------------
    #  Algebra (operator_algebra_merge.md D1, T2; §3.2/§3.4/§3.5)
    # ------------------------------------------------------------
    def __matmul__(
        self, other: Operator | tuple[Operator, ...],
    ) -> Operator:
        """Composition ``(A @ B)(f) == A(B(f))``; dispatches on operand
        arity (T2). Separable, axis-compatible operands yield a
        ``SeparableComposite``, else a ``Composite``."""
        ...

    def __rmatmul__(self, other: object) -> Operator:
        """A tuple on the left (``(A1, A2) @ P``) has no meaning: raise."""
        ...

    def __add__(self, other: Operator) -> Operator: ...   # OperatorSum
    def __sub__(self, other: Operator) -> Operator: ...   # A + (-1) * B
    def __neg__(self) -> Operator: ...                    # (-1) * A
    def __mul__(self, c: ScalarField | complex) -> Operator: ...   # c * A
    def __rmul__(self, c: ScalarField | complex) -> Operator: ...
    def __pow__(self, n: int) -> Operator: ...            # n-fold chain

    # ------------------------------------------------------------
    #  Axis binding (§2.3): default is "nothing to bind"
    # ------------------------------------------------------------
    def __getitem__(self, axis: str) -> Operator:
        """Bind a coordinate axis. Whole-space operators have a fixed
        signature and raise; ``SeparableOperator`` overrides."""
        raise TypeError(
            f"{type(self).__name__} has a fixed signature; nothing to bind"
        )
```

Notes:

- **The signature is a resolver, not stored state.** An operator's
  `(domain, codomain)` signature is exposed as `codomain(domain)`:
  concrete operators resolve the codomain per domain factor (e.g.
  `FiniteDifference.codomain(Center(mx)) == Right(mx)` on a periodic
  mesh). Unsupported domains raise `SpaceMismatchError` (owned by
  doc 02). Storing an explicit `(domain, codomain)` pair was rejected:
  operators are axis-agnostic separable kernels reused across meshes,
  so the signature must be a function of the incoming space.
- **Tuple (direct-sum) signatures** (§3.1): the resolver returns a
  *tuple* of spaces for vector-/tensor-valued operators (one per
  component), the scalar case being the length-1 tuple. A
  `VectorField` supplies the operand tuple; signature equality is
  componentwise space identity **by position** (names are metadata).
  No new space type is introduced — the tuple is a signature notion,
  fields on it stay ordinary `VectorField`s (§3.1).
- **The dunders build the algebra objects** of
  [Operator algebra](#operator-algebra): `@` a `Composite` /
  `SeparableComposite`, `+`/`-` an `OperatorSum`, scalar/field `*` a
  `ScaledOperator`, `Block([...])` a block matrix, `Dispatched(kind)` a
  registry placeholder. They are **shallow eager structures** — no
  expression graphs, no algebraic rewriting (operator_design §3.11);
  the only normalizations are chain-flattening, `Identity` elision, and
  `Zero`-dropping. Iteration split (D7): `@`, `Identity`, `Dispatched`,
  and bound axes are **iteration 1** (the FV derivative needs them);
  `+`/`-`/`*` (sums, scaling), `Zero`, `OperatorSum`, `ScaledOperator`,
  and `Block` are **designed-for**.
- `requirements` defaults to `OperatorRequirements()` (halo 0, any
  layout); the grid's halo-accounting trace
  ([§5](../04_decomposition.md#5-domain-decomposition)) reads it per
  applied factor and accumulates depth along un-synced chains.
- `eigenvalues` and the **block expansion are optional capabilities**
  (B4): a nonlinear or nonlinear-tuple-signature operator that answers
  neither is legal (the base `eigenvalues` raises `EigenbasisError`;
  there is no mandatory block method). Only linear operators over a
  diagonalizing basis carry symbols; only linear tuple operators expand
  into blocks. `eigenvalues(grid, space)` is queried per coefficient
  *factor* space, never a `(kx, ky, kz)` tuple; composites compose
  per-factor symbols (§3.7). It is grid-mediated because eigenvalues
  derive from `grid.wavenumbers(space)` and the metric measures; the
  operator stays grid-free until this call.
- **Operators are interned (D6).** Bound variants (`op["x"]`) and
  algebra objects are canonicalized on their static structure (class,
  order, bound axes, factor identities, coefficient *placement* — not
  values), because the cluster's identity-hash invariant
  (`__eq__`/`__hash__` return `self is other`, README) requires
  structurally-equal operators to be the same object for jit caching.
  This resolves operator_design §5.1 toward interning.

### OperatorRequirements

Frozen value type describing what an operator demands from the
decomposition along one factor.

| | |
|---|---|
| Kind | final frozen dataclass |
| Pytree | static value |
| Iteration | 1 |
| Concept refs | §2.5, §5 |
| Module | `grid2.operators.base` |

```python
@dataclass(frozen=True)
class OperatorRequirements:
    """Per-factor decomposition requirements of an operator."""

    #: ghost-layer depth needed along this factor's axis
    halo: int = 0
    #: "any"       — works on a sharded axis via halo exchange
    #: "local"     — the factor's axis must be undistributed
    #: "transpose" — distributed via transpose-based scheduling (FFT)
    layout: Literal["any", "local", "transpose"] = "any"
    #: informational: the operator performs a cross-shard reduction
    #: (negotiation treats it as no-constraint)
    collective: bool = False
```

### EigenbasisError

```python
class EigenbasisError(TypeError):
    """Raised when eigenvalues are queried outside a diagonalizing basis."""
```

Iteration 1 (the error type ships with the base class even while the
`Symbol` machinery is designed-for, so the base `eigenvalues` has a
defined failure mode from day one).

### UnaryOperator

| | |
|---|---|
| Kind | ABC |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §2.5 |
| Module | `grid2.operators.base` |

```python
class UnaryOperator(Operator, ABC):
    """Operator applied to a single field: ``op(f)`` (bind axes first)."""

    @abstractmethod
    def codomain(
        self, domain: FunctionSpace,
    ) -> FunctionSpace | tuple[FunctionSpace, ...]:
        """Resolve the codomain from the single domain space."""
        ...

    @final
    def __call__(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Template: validate, intercept halo tracers, delegate."""
        ...

    @abstractmethod
    def _apply(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Apply to real (non-tracer) operands (subclass hook)."""
        ...
```

The base `__call__` is a **concrete, final template method**: it
validates the operand space and resolves the codomain, **intercepts
doc 04's `HaloTracer` operands** — recording
`requirements(domain).halo` per factor and returning the codomain
tracer without touching data — and delegates real fields to the
abstract `_apply`. Subclasses implement `_apply` only; no
per-operator tracer code exists anywhere
([§5](../04_decomposition.md#5-domain-decomposition)'s
author-effort-free claim made structural).

**Bind-only axis naming (D2).** There is no `axis=` call keyword and no
`**kwargs`: an axis is named only by binding, `op["x"](f)` (§2.3). So
"can this operator name an axis?" is a matter of **type** —
`SeparableOperator` overrides `__getitem__`, whereas whole-space
operators (composed `grad`/`Laplacian`, transforms) inherit the raising
default and are simply applied `op(f)`, the axis fixed by their
signature. This deletes the old `axis`/`ValueError` validation
entirely; the "which axis?" question becomes a binding question,
resolved in `SeparableOperator._apply`. Operators that previously took
a call-site parameter through `**kwargs` (`ConstantBroadcast`'s `to=`)
bind it the same way (`to[target]`, D3). This rewrites every concrete
`_apply(self, f, axis, **kwargs)` signature to `_apply(self, f)` — the
separable `_apply_factor(self, f, axis)` is unchanged, since there
`axis` is the resolved axis, always a string — and the pass is applied
throughout this document.

### BinaryOperator

| | |
|---|---|
| Kind | ABC |
| Pytree | static |
| Iteration | 1 (base; concrete products below carry their own status) |
| Concept refs | §2.5, §3.11 |
| Module | `grid2.operators.base` |

```python
class BinaryOperator(Operator, ABC):
    """Operator applied to two (or more) fields: ``op(f, g)``."""

    @abstractmethod
    def codomain(
        self, domain_a: FunctionSpace, domain_b: FunctionSpace,
    ) -> FunctionSpace:
        """Resolve the codomain from the two domain spaces."""
        ...

    @final
    def __call__(
        self, f: ScalarField, g: ScalarField, *more: ScalarField,
    ) -> ScalarField:
        """Template: validate, intercept halo tracers, delegate."""
        ...

    @abstractmethod
    def _apply(
        self, f: ScalarField, g: ScalarField, *more: ScalarField,
    ) -> ScalarField:
        """Apply to real (non-tracer) operands (subclass hook)."""
        ...
```

The same final template pattern as `UnaryOperator`: `__call__`
validates/unites the operand spaces, intercepts `HaloTracer`
operands, and delegates to `_apply`. The variadic `*more` slot serves
the n-ary elementwise operators (`Hadamard`, `Where`); strict
binaries reject extra operands. `codomain(a, b)` applies the two
sanctioned implicit exceptions of the strict algebra before demanding
equality: `ConstantSpace` broadcast
([§3.3](../02_rules.md#33-constantspace-replaces-topo-with-automatic-broadcast))
and `fr.Real -> fr.Complex` promotion
([§3.1](../02_rules.md#31-strict-space-algebra)); anything else is a
`SpaceMismatchError`.

**Binary operators join the `@` algebra with two rules** (T2,
operator_design §3.8), asymmetric because a binary has *two inputs but
one output*:

- **Post-composition wraps the single output**: `A @ P` for binary `P`
  is the binary operator `(f, g) -> A(P(f, g))`.
- **Pre-composition takes a tuple, one unary per input**:
  `P @ (B1, B2)` is `(f, g) -> P(B1(f), B2(g))`; the sugar `P @ B` is
  `P @ (B, B)`, and the n-ary form `P @ (B1, ..., Bn)` serves the
  `*more` operands.
- **`(A1, A2) @ P` is an error** — one output, nothing to distribute a
  left tuple over; it raises via `Operator.__rmatmul__`.

This is what makes `Convolution = trim @ CollocationProduct @
pad_inverse` a literal expression of the algebra (§3.12): the padded
inverse pre-composes both operands, the product runs on the finer
space, the trimming transform post-composes. A binary-headed composite
reuses this template (its `_apply` arity follows the chain's inner
factor); **no named-operand form** (`P.compose(left=, right=)`) is
added — the positional tuple already covers the n-ary case, and named
left/right does not generalize past arity 2 (revisit only if flux-module
porting shows it error-prone).

### SeparableOperator

Intermediate base for separable 1D kernels — the shape of almost every
concrete operator on tensor grids: the kernel acts on one factor, and
the product lifting `kernel ⊗ identity` is provided once, here.

| | |
|---|---|
| Kind | ABC |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §2.3, §2.5, §3.5 |
| Module | `grid2.operators.base` |

```python
class SeparableOperator(UnaryOperator, ABC):
    """Separable 1D kernel lifted per factor: ``kernel ⊗ identity``."""

    #: bound coordinate axis; None = unbound (axis-agnostic 1D kernel).
    #: Part of the STATIC structure (jit / dispatch identity, §2.3).
    bound_axis: str | None = None

    @final
    def __getitem__(self, axis: str) -> Self:
        """Bind to a coordinate axis: an interned static variant (D5/D6).
        Rebinding an already-bound kernel raises."""
        ...

    def _rebind(self, axis: str) -> Self:
        """Interned structural copy with ``bound_axis`` set."""
        ...

    def _apply(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """axis = ``bound_axis``; else the field's sole bindable factor;
        else raise. Then run ``_apply_factor`` per shard."""
        ...

    @abstractmethod
    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Per-factor signature. Unbound: 1D factor -> 1D factor
        (``Center -> Right``). Bound: full product -> product with the
        bound factor transformed, identity elsewhere."""
        ...

    # ------------------------------------------------------------
    #  Extension contract (subclass hook)
    # ------------------------------------------------------------
    @abstractmethod
    def _apply_factor(self, f: ScalarField, axis: str) -> ScalarField:
        """Apply the 1D kernel along the (resolved) factor axis."""
        ...
```

Notes:

- **Binding is the only way to name an axis (D2).** The inherited
  final `__call__` delegates to `_apply`, which resolves the axis:
  `bound_axis` if set (via `op["x"]`), else the field's sole
  non-`ConstantSpace`, non-trivial factor if unambiguous, else a
  `ValueError` asking for an explicit bind. `op["x"]` returns an
  **interned static variant** with `bound_axis` set (D6); rebinding
  (`op["x"]["y"]`) raises; binding distributes over the algebra
  (`(A @ B)["x"] == A["x"] @ B["x"]`, §2.3). Application along a
  `ConstantSpace` factor is the identity
  ([§3.3](../02_rules.md#33-constantspace-replaces-topo-with-automatic-broadcast)).
- **`codomain` shape depends on binding state.** Unbound, the resolver
  is 1D-factor -> 1D-factor (reused across meshes); bound, it takes the
  full product space and transforms only the bound factor (the
  `⊗ identity` lift made concrete). An unbound separable kernel
  composed with another (`flux_diff @ reconstruct`) is still a 1D
  kernel — a `SeparableComposite` ([Operator algebra](#operator-algebra),
  D5) — bindable later.
- Kernels are **slice-based over halo-extended storage, never
  roll-based** ([§3.5](../02_rules.md#35-shape-is-a-property-of-the-space)):
  `_apply_factor` reads stencil-offset slice windows of the local
  array and writes the true-shape output; shape changes
  (`Outer(n+1) -> CellAvg(n)`) are just differing slice lengths.
  Periodic vs bounded is not a kernel concern — it lives in the
  halo-fill mode.
- **Execution model (G4).** `_apply`/`_apply_factor` bodies run
  per-shard under a decomposition-supplied `shard_map`, provided by
  the base `_apply` and the decomposition layer — kernel authors
  never see it. Pad, halo sync, and kernel form **one shard-local
  region per application**. The iteration-1 sync contract: operator
  inputs have valid halos, and **every operator application returns a
  synced field**; sync-elision along traced chains is designed-for;
  halo-0 operators skip the sync structurally. Storage is
  halo-extended: `_data` is storage-shaped, `.data` the true-shape
  view (docs 02/04).
- Spacing enters through the grid-owned measure fields
  (`grid.measure(space, name=...)`, doc 04; reachable as the field
  carries its grid, §2.7), read inside `_apply_factor` at trace time —
  **no hardcoded `dx` constants**. A uniform mesh constant-folds.

---

## Operator algebra

The **derived operators** (operator_design §2.1): the objects `@`, `+`,
`*`, `Block(...)`, and `Dispatched(...)` build. They are ordinary
operators — callable, registrable, halo-accountable, symbol-bearing
where linear — and shallow eager structures with no expression graphs
or algebraic rewriting (§3.11). Merge decisions:
[`operator_algebra_merge.md`](operator_algebra_merge.md) D1–D8, B1–B4,
T2. All live in `grid2.operators.base`. Iteration split (D7): `Identity`,
`Composite`, `SeparableComposite`, `Dispatched` are **iteration 1** (the
FV derivative needs them); `Zero`, `OperatorSum`, `ScaledOperator`,
`Block` are **designed-for**.

**Pytree amendment.** `ScaledOperator` with a *field* coefficient is a
**second dynamic-leaf carrier** in this cluster (the coefficient field),
alongside `Symbol` — updating the module-placement rule's "the only
dynamic-leaf carrier is `Symbol`". Its static part is the algebraic
structure; the coefficient is the leaf (operator_design §2.2). Every
other algebra object is fully static.

### Identity / Zero

```python
@final
@fr.utils.jaxify
class Identity(Operator):                       # iteration 1
    """Neutral element of @ (A @ Identity == A); elided on normalize."""
    def codomain(self, *domains): ...           # domains unchanged


@final
@fr.utils.jaxify
class Zero(Operator):                           # designed-for
    """Neutral of + and absorbing of @ (Zero @ A == A @ Zero == Zero)."""
    def codomain(self, *domains): ...
```

`Identity` is what an unbound axis contributes in the `⊗ identity`
extension (§2.3) and what operators along a `ConstantSpace` axis reduce
to (§3.3); halo 0, unit symbol. `Zero`'s consumer is the block layout —
structural zeros in `curl` and sparse system matrices — contributing no
computation, no halo, a zero symbol.

### Composite

```python
@final
@fr.utils.jaxify
class Composite(UnaryOperator):                 # iteration 1
    """Flat right-to-left factor chain; whole-space, not bindable.

    A binary-headed chain (Convolution) presents a binary signature and
    reuses the BinaryOperator template instead.
    """
    def __init__(self, factors: tuple[Operator, ...]) -> None:
        """Flat chain; no nested composites (associativity, §3.2)."""
        ...
    def codomain(self, domain): ...             # thread the chain
    def requirements(self, domain): ...         # per-axis SUM of factor halos (§3.6)
    def eigenvalues(self, grid, space): ...     # product of factor symbols (§3.7)
    def _apply(self, f): ...                    # apply factors right-to-left
```

Whole-space, so it inherits the raising `__getitem__` — a mixed-axis or
transform-bearing chain (`fd["x"] @ fd["y"]`, `div @ grad`,
`fourier.forward @ fd["x"]`) is not bindable. Composition is
associative and flat; a composite normalizes to a single factor chain,
eliding `Identity` and dropping `Zero`. Halo **sums** along the chain
(§3.6). A symbol exists iff every factor is linear with a symbol on the
compatible bases, and is their mode-wise product (§3.7); nonlinear
factors (WENO, limiters) make the composite symbol-free.

### SeparableComposite

```python
@final
@fr.utils.jaxify
class SeparableComposite(SeparableOperator):    # iteration 1
    """A chain of separable kernels on one axis; itself separable."""
    def __init__(self, factors: tuple[SeparableOperator, ...]) -> None: ...
    def codomain(self, domain): ...             # thread the chain per factor
    def requirements(self, domain): ...         # SUM of factor halos (§3.6)
    def eigenvalues(self, grid, space): ...     # PRODUCT of factor symbols (§3.7)
    def _apply_factor(self, f, axis): ...       # run each factor on `axis`
```

Produced by `@` when both operands are separable **and axis-compatible**
(both unbound, or bound to the same axis); different bound axes give a
whole-space `Composite` instead. Being a `SeparableOperator`, it
inherits `bound_axis`, `__getitem__`, the axis-resolving `_apply`, and
the per-shard template — so binding distributes for free
(`(flux_diff @ reconstruct)["x"]` just sets its `bound_axis`, which is
`flux_diff["x"] @ reconstruct["x"]`). It overrides only `_apply_factor`
(chain each factor on the axis), `requirements` (sum), and `eigenvalues`
(product). This is the object the FV derivative default is
([`FVDerivative`](#fvderivative), D1).

### OperatorSum / ScaledOperator

```python
@final
@fr.utils.jaxify
class OperatorSum(Operator):                    # designed-for
    """Flat term list: (A + B)(f) == A(f) + B(f); same signature."""
    def __init__(self, terms: tuple[Operator, ...]) -> None: ...
    def codomain(self, *domains): ...           # the terms' common signature
    def requirements(self, domain): ...         # per-axis MAX (§3.6)
    def eigenvalues(self, grid, space): ...      # SUM of term symbols (§3.7)
    def _apply(self, f): ...


@final
@partial(fr.utils.jaxify, dynamic=("coeff",))
class ScaledOperator(Operator):                 # designed-for
    """Coefficient-scaled operator: (c * A)(f) == c * A(f)."""
    #: complex scalar, or a ScalarField on ``A.codomain`` (dynamic leaf)
    def codomain(self, *domains): ...           # A.codomain
    def requirements(self, domain): ...         # halo(A) (product is pointwise)
    def eigenvalues(self, grid, space): ...      # scales A's symbol iff c constant
    def _apply(self, f): ...
```

Sums read the same input in parallel, so halo **maxes** (§3.6); `Zero`
terms drop on normalization. `ScaledOperator` is the §3.4 machinery: a
field coefficient multiplies the *output* (so `c` lives on
`A.codomain`, `ConstantSpace` broadcast making plain scalars the
constant case), and it is what spells the terrain-following derivative
`ddx_z = fd["x"] - c_metric * fd["sigma"]` (§3.8, sketch 4.4) — the
metric coefficient a dynamic leaf read through `grid.metric`. A genuine
field coefficient breaks translation invariance, so the scaled operator
has **no symbol** unless `c` is constant (§3.7).

### Block

```python
@final
@fr.utils.jaxify
class Block(Operator):                          # designed-for
    """Block matrix of scalar-signature operators (§3.5, B1)."""
    def __init__(self, blocks: Sequence[Sequence[Operator]]) -> None:
        """m x n grid; ``Zero`` for structural zeros, ``Identity`` on
        the diagonal where needed."""
        ...
    def codomain(self, *domains): ...           # tuple codomain (§3.1)
    def requirements(self, domain): ...         # per block-row max_j(halo compose) (§3.6)
    def eigenvalues(self, grid, space): ...      # a per-mode block matrix of symbols (§3.7)
    def _apply(self, f): ...                    # D(f)_i = sum_j D_ij(f_j)
```

A whole-space, tuple-signatured operator, not bindable. `@` is
**block-matrix multiplication** ((A @ B)_ik = sum_j A_ij @ B_jk), which
is why `laplacian = div @ grad` type-checks ((1×n) @ (n×1) = 1×1) and
its single block is the `OperatorSum` of per-axis `SeparableComposite`
second derivatives (B1). `grad` is a column, `div` a row, `curl` a
matrix with `Zero` blocks in 3-D. Its **entries are the bound scalar
operators** of the separable layer; the block expansion is **metadata
computed on demand** for halo and symbol queries, never materialized as
a rewritten operator (§3.11). Nonlinear tuple-signature operators
(kinetic energy `(u, v) -> ke`) are **not** blocks — they carry a tuple
signature with a direct `_apply` and decline block/symbol (B4).

### Dispatched

```python
@final
@fr.utils.jaxify
class Dispatched(Operator):                     # iteration 1
    """A registry kind placeholder, resolved when the grid is known (D4)."""
    def __init__(self, kind: str) -> None: ...
    def __getitem__(self, axis: str) -> Dispatched:
        """Carry a pending axis bind onto the (later) resolved operator."""
        ...
    def resolve(
        self, registry: OperatorRegistry, space: SpaceLike,
    ) -> Operator:
        """Replace the placeholder with the registered operator."""
        ...
    def __call__(self, f: ScalarField | VectorField):
        """Standalone use: resolve against ``f.grid.dispatch``, then apply."""
        ...
    def codomain(self, *domains): ...           # from the resolved target
```

**One object, two roles** (D4). As a **chain factor** inside a
registered default (`flux_diff @ Dispatched("reconstruct")`), it is
resolved at **model assembly** against the merged registry (grid
defaults + module overrides), then baked concrete — no per-application
lookup inside chains, so a module's `reconstruct` override propagates
into the FV derivative (sketch 4.2). As the **user verb**
(`diff = Dispatched("diff")`, `diff["x"](f)`), it resolves against
`f.grid.dispatch` at **application** — exactly what `f.diff("x")` did.
The unifying rule: *resolve when the registry/grid becomes known.*
Resolution precedence is ordinary dispatch (space-specific > kind-only
> default, [registry](#operatorregistry)). The field-sugar forwarders
(D3) route through this verb: `f.diff(axis) == fr.operators.diff[axis]
(self)`, `f.integrate(axis) == fr.operators.integrate[axis](self)`;
`f.to(target)`'s target-space binding spelling is a D3 open detail.

---

## Symbol

### Symbol

Diagonal operator on a coefficient space; the eigenvalue/filter/mask
primitive ([§2.5](../01_concepts.md#25-operator--typed-maps-between-spaces),
[§3.11](../02_rules.md#311-field-operations-linear-ops-and-the-product-problem)).

| | |
|---|---|
| Kind | final concrete |
| Pytree | static space tags, **dynamic** `_data` leaf |
| Iteration | designed-for (first consumer: the nonhydro port's spectral pressure solver and projections — the `discrete_spectral_operators` successor) |
| Concept refs | §2.5, §3.2, §3.11, §3.12, sketch 4.6 |
| Module | `grid2.operators.symbol` |

```python
@final
@partial(fr.utils.jaxify, dynamic=("_data",))
class Symbol:
    """Diagonal operator on a coefficient space (eigenvalues, filters)."""

    def __init__(
        self,
        space: SpaceLike,
        data: jax.Array,
        codomain: SpaceLike | None = None,
    ) -> None:
        """Wrap per-mode values ``data`` as a diagonal on ``space``."""
        ...

    @classmethod
    def from_field(cls, f: ScalarField) -> Symbol:
        """Build a symbol from a coordinate ScalarField."""
        ...

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def space(self) -> SpaceLike:
        """Domain coefficient space of the diagonal."""
        ...

    @property
    def codomain(self) -> SpaceLike:
        """Codomain space (equals ``space`` unless retagging)."""
        ...

    @property
    def data(self) -> jax.Array:
        """The per-mode diagonal values (dynamic leaf)."""
        ...

    def conj(self) -> Symbol:
        """Complex-conjugate symbol (adjoint of the diagonal)."""
        ...

    def inverse(self, where_zero: complex = 0.0) -> Symbol:
        """Pseudo-inverse; zero diagonal entries become where_zero."""
        ...

    # ------------------------------------------------------------
    #  Application (Hadamard multiply)
    # ------------------------------------------------------------
    def __call__(self, f: ScalarField) -> ScalarField:
        """Apply to a field: elementwise (Hadamard) multiply."""
        ...

    # ------------------------------------------------------------
    #  Diagonal (elementwise) algebra — never the physical product
    # ------------------------------------------------------------
    def __mul__(self, other: Symbol | complex) -> Symbol: ...
    def __rmul__(self, other: complex) -> Symbol: ...
    def __add__(self, other: Symbol | complex) -> Symbol: ...
    def __radd__(self, other: complex) -> Symbol: ...
    def __sub__(self, other: Symbol | complex) -> Symbol: ...
    def __neg__(self) -> Symbol: ...
    def __pow__(self, p: int) -> Symbol: ...
    def __truediv__(self, other: Symbol | complex) -> Symbol: ...
    def __rtruediv__(self, other: complex) -> Symbol: ...
    def __matmul__(self, other: Symbol) -> Symbol: ...
```

Notes:

- **A `Symbol` is not a `ScalarField`** and not a `ScalarField`
  subclass: its `*`, `**`, `+`, `1 / .` are the diagonal algebra
  (composition/inverse of diagonal operators), never the physical
  product/convolution (§3.11). `1 / lap` is `__rtruediv__`;
  `k ** 2` is `__pow__`. Making it a field with special flags was
  rejected in the notes; making it an `Operator` subclass was
  considered and rejected here: operators are static structure while
  a `Symbol` carries a dynamic data leaf, and no axis machinery or
  registry membership applies. It *implements* the unary-apply
  protocol (`__call__`, `space`/`codomain`) so solvers can treat it as
  a diagonal map.
- **Broadcast across product factors is diagonal-operator extension**
  (`Identity ⊗ D`), *not* doc 02's field lift: a per-factor symbol
  returned by `fd.eigenvalues(grid, fourier_x)` lives on
  `Fourier(x) ⊗ Constant(y) ⊗ Constant(z)`, and extending it to the
  full product **replicates** its diagonal along every lifted factor
  (the same `kx` value acts on every `ky` row). Although both cite
  §3.3, this is deliberately **not** doc 02's lift 1, which embeds a
  constant *function* — on coefficient factors a delta into the zero
  mode. Delta embedding is correct for functions and wrong for
  diagonal *operators*: a delta-lifted `kx ** 2` would act only on
  the `ky = 0` row, and the Laplacian broadcast sum `kx**2 + ky**2`
  (sketch 4.6) would be wrong everywhere else. `+`/`*` of symbols on
  broadcast-compatible spaces replication-extend both operands to the
  union product first, then combine elementwise.
- **Retagging symbols.** Most symbols have `codomain is space`. A
  first-derivative FD symbol (`Fourier(origin=Center) ->
  Fourier(origin=Right)`) and the `PhaseShift` symbol retag: they are
  diagonal in the shared mode index but change the space tag.
  Elementwise `*`/`+` require matching `(space, codomain)` pairs
  (after broadcast); *chained composition across retags* is the
  explicit `a @ b` (`b.codomain == a.space`), keeping composition
  order visible. This is why the honest discrete Laplacian symbol is
  `bwd @ fwd`, not `fwd ** 2`; `Laplacian.eigenvalues` does this
  internally, and `__pow__` demands `codomain is space`.
- Carried contents (all the same type, per §3.11): operator
  eigenvalues, spectral filters, the 2/3 truncation mask (§3.12),
  inter-origin phase shifts (§3.2), the `sinc(k dx / 2)`
  cell-averaging factor (§3.2/§3.9).
- **Zero-mode division.** Bare `1 / s` keeps jax semantics: a zero
  diagonal entry yields `inf`/`nan` that propagates. The explicit
  pseudo-inverse is `s.inverse(where_zero=0.0)` — zero entries of the
  diagonal are replaced by `where_zero` in the inverse — so the
  spectral solve `lap.inverse()(-div_hat)` regularizes `k = 0`
  without caller-side masking; any other regularization stays
  caller-side.
- Construction path: from grid coordinate fields —
  `Symbol.from_field(grid.wavenumbers(space))`, then algebra
  (`1j * k`, `2 / dx * ...`). The classmethod takes the `ScalarField`
  itself; §2.5's "built from their `.data`" is provenance language
  (the symbol adopts the field's array as its diagonal), not the
  argument type. The coordinate accessors themselves stay
  `ScalarField`s (§2.7); `from_field` is the sanctioned crossing.

---

## Stencil / nodal operators

All are `SeparableOperator`s: grid-free, axis-agnostic 1D kernels.
Per-factor signatures are written `Domain -> Codomain`; on bounded
meshes the resolver picks the bounded variant as listed.

Per owner directive (G2), the FV/average family is **model-complete
in iteration 1**: every conversion, derivative, product, and
transform a flux-form C-grid model needs resolves from the default
table — see the flux-form closure walk after `FaceDifference` below.

### FiniteDifference

Staggered finite-difference derivative of configurable order; the
default `"diff"` entry on nodal spaces.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §2.5, §2.7, §3.4, §3.5, sketch 4.6 |
| Module | `grid2.operators.finite_difference` |

```python
@final
class FiniteDifference(SeparableOperator):
    """Staggered finite-difference derivative (order 2, 4, ...)."""

    dispatch_kind: ClassVar[str | None] = "diff"

    def __init__(self, order: int = 2) -> None:
        """Create an FD kernel of the given even order."""
        ...

    @property
    def order(self) -> int:
        """Order of accuracy of the stencil."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """diff: Center -> Right | Inner; Right/Outer/Inner -> Center."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = order // 2, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """i k_hat symbol on a Fourier factor space (retagging)."""
        ...
```

Notes:

- Per-factor signatures: periodic mesh `Center -> Right`,
  `Right -> Center`; bounded mesh `Center -> Inner`
  (n -> n - 1), `Outer -> Center` (n + 1 -> n), `Inner -> Center`.
  Nodal only — the FV derivative on average spaces is `FVDerivative`.
- Stencil *pattern* (from `order`) is static identity; the spacing
  denominators are the **dual center-to-center / primal cell-width
  measure fields** read from the field's grid at trace time
  (`grid.measure(space, name=...)`, doc 04; §2.7) — a uniform mesh
  constant-folds them. This replaces today's
  `_dx1 = 1 / grid.dx` module state
  (`grid/cartesian/finite_differences.py`).
- `eigenvalues(grid, fourier_x)` exists for Fourier factors
  (constant-coefficient FD on a periodic mesh) and returns the exact
  discrete symbol (order 2: `i * 2 sin(k dx / 2) / dx` times the
  inter-origin phase — a retagging symbol). On non-diagonalizing
  factors (Chebyshev): `EigenbasisError`. This is the
  `discrete_spectral_operators.k_hat` successor.

### LinearInterp

Two-point staggering interpolation; the default `"interp"` entry on
nodal spaces.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §3.1, §3.4, sketch 4.1 |
| Module | `grid2.operators.interp` |

```python
@final
class LinearInterp(SeparableOperator):
    """Second-order two-point interpolation between nodal node sets."""

    dispatch_kind: ClassVar[str | None] = "interp"

    def __init__(self, target: NodeSet | None = None) -> None:
        """Create the kernel; ``target`` overrides the default codomain."""
        ...

    @property
    def target(self) -> NodeSet | None:
        """Explicit target node set, or None for the default table."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """interp: Center <-> Right (periodic); Center -> Inner,
        Outer/Inner -> Center (bounded); target= selects Outer."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 1, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """one_hat averaging symbol on a Fourier factor (retagging)."""
        ...
```

Notes:

- Fixed codomain per §3.4: the registered operator fixes its own
  codomain; a `.to(target)` whose target disagrees is a space error.
  A *call-time* target argument was rejected; alternative codomains
  are **per-instance**, selected by the `target=` constructor knob
  and registered explicitly by whoever needs them.
- Per-factor defaults (`target=None`): periodic `Center <-> Right`;
  bounded `Center -> Inner` (interior faces) and
  `Outer/Inner -> Center`. The periodic `Center <-> Right` rows are
  **periodic-only**: on a bounded mesh `Right` lacks the left
  boundary face, and accepting it would silently impose a one-sided
  boundary treatment.
- **`Center -> Outer` variant** (`target=NodeSet.OUTER`, iteration 1):
  interior faces by the two-point mean; the two boundary faces are
  filled by one-sided linear extrapolation from the two nearest
  centers (the BC-free `Outer` boundary DOFs receive extrapolated
  values; boundary-data-owning modules overwrite them, §3.6). Not a
  default row — same key as the `Inner` default — so modules register
  it explicitly.
- Higher-order centered interpolation (today's
  `PolynomialInterpolation(order=...)`) is a designed-for sibling
  `PolynomialInterp(order)` in the same module; not spelled out here
  because it adds only the `order` parameter to this exact surface.
- `eigenvalues` is the `one_hat` successor: order 2 gives
  `cos(k dx / 2)` times the inter-origin phase.

### LinearReconstruction

Second-order conversions inside the average family
(nodal-at-face <-> dual/primal averages); the default `"reconstruct"`
entry.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §3.9, sketch 4.7 |
| Module | `grid2.operators.reconstruct` |

```python
@final
class LinearReconstruction(SeparableOperator):
    """2nd-order average <-> point-value conversion (FV family)."""

    dispatch_kind: ClassVar[str | None] = "reconstruct"

    def __init__(self, target: NodeSet | None = None) -> None:
        """Create the kernel; ``target`` overrides the default codomain."""
        ...

    @property
    def target(self) -> NodeSet | None:
        """Explicit target node set, or None for the default table."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """CellAvg -> Right (periodic) | Inner (bounded);
        Right/Outer/Inner -> CellAvg; FaceAvg <-> Center dual."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 1, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """sinc-corrected averaging symbol on a Fourier factor."""
        ...
```

Notes:

- Covers both directions of sketch 4.7's table at second order:
  average-to-point (`CellAvg -> Right/Inner`, Shu two-point mean) and
  evaluate-to-average (`Outer -> CellAvg`, trapezoid mean). At second
  order both collapse to two-point means — the classic C-grid
  average — but they are *distinct signatures*, and higher-order
  members of the family (`ShuReconstruction(order=...)`,
  designed-for) genuinely differ per direction.
- Bounded default is `CellAvg -> Inner` (interior faces, the
  no-normal-flow C-grid staggering of sketch 4.7); the
  `CellAvg -> Outer` variant is `target=NodeSet.OUTER` with the same
  boundary-extrapolation treatment as `LinearInterp` — an explicit
  instance, not a default row.
- `"reconstruct"` (average <-> point value) is a distinct kind from
  `"interp"` (nodal -> nodal), per sketch 4.2; `.to` picks the kind
  from the source/target family (§3.4).

### WenoReconstruction

Nonlinear upwind-biased reconstruction; the module-override
archetype (sketch 4.2).

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 on periodic axes (parity with today's `weno_interpolation.py`); bounded-axis boundary biasing designed-for |
| Concept refs | §3.9, sketch 4.2 |
| Module | `grid2.operators.reconstruct` |

```python
@final
class WenoReconstruction(SeparableOperator):
    """WENO average-to-point reconstruction (CellAvg -> face values)."""

    dispatch_kind: ClassVar[str | None] = "reconstruct"

    def __init__(
        self,
        order: int = 5,
        bias: Literal["left", "right"] = "left",
    ) -> None:
        """Create a WENO kernel of the given odd order and bias."""
        ...

    @property
    def order(self) -> int:
        """Formal order of the WENO reconstruction."""
        ...

    @property
    def bias(self) -> Literal["left", "right"]:
        """Upwind bias side of the reconstruction."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """reconstruct: CellAvg -> Right | Outer (biased)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = order // 2 + 1, layout "any"."""
        ...
```

Notes:

- Nonlinear, so no `eigenvalues` (base raises `EigenbasisError`) —
  correct and automatic.
- Upwinding is a *pair* of biased instances; flux-splitting advection
  modules hold both and select by velocity sign (the sign selection
  itself is the `("select", ...)` kind, `Where` below). The
  alternative — a velocity-consuming ternary operator — was rejected
  as a module-level (physics) concern, consistent with §3.9
  ("advection schemes override reconstruction, not diff").
- **Iteration 1 is periodic-only**: the wide stencil is valid on
  wrap-around halos. Bounded axes need reduced-stencil boundary
  biasing (one-sided smoothness indicators near the wall) —
  designed-for; a bounded-axis registration of the iteration-1
  instance is a space error, not a silent fallback.

### FluxDifference

The exact FV flux difference — the discrete Gauss theorem
([§3.9](../02_rules.md#39-finite-volume-semantics-the-average-family-and-the-fv-derivative)).

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §2.7, §3.9, sketch 4.7 |
| Module | `grid2.operators.flux_diff` |

```python
@final
class FluxDifference(SeparableOperator):
    """Exact flux difference: Outer -> CellAvg (discrete Gauss)."""

    dispatch_kind: ClassVar[str | None] = "flux_diff"

    def __init__(self) -> None:
        """Create the exact face-difference kernel."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """flux_diff: Outer(n+1) | Inner(n-1) -> CellAvg(n) (bounded);
        Right(n) -> CellAvg(n) (periodic only)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 1, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """2 i sin(k dx / 2) / dx = i k sinc(k dx / 2) on Fourier."""
        ...
```

Notes:

- Signature `Outer -> CellAvg` with the **primal cell-width measure as
  denominator**: `(u_{i+1/2} - u_{i-1/2}) / w_i`, where `w` is the
  cell-width measure field *on `CellAvg`/`Center`*
  (`grid.measure(space, name=...)`, doc 04; §2.7, §3.9) — never a
  scalar `dx`. `Inner -> CellAvg` is the homogeneous
  no-normal-flow variant (zero boundary fluxes); inhomogeneous
  boundary fluxes occupy the boundary DOFs of an `Outer`-space flux
  field (§3.6) — no extra parameter here.
- The `Right(n) -> CellAvg(n)` row is **explicitly periodic-only**:
  on a bounded mesh `Right` lacks the left boundary face, so
  accepting it would silently impose a one-sided no-flux boundary.
  Bounded domains must present fluxes on `Outer` (explicit boundary
  fluxes) or `Inner` (homogeneous).
- **Exactness is the contract**: summing the output against the cell
  measure telescopes to boundary fluxes; the eigenvalue identity
  `i k sinc(k dx / 2)` = "average of d/dx" is the §3.9 validation
  hook tested against `05_validation.md`'s FV claims.

### DualFluxDifference

The discrete Gauss theorem on the **dual cells**: fluxes known at
cell centers, differenced into dual-cell averages — the
momentum-control-volume derivative of sketch 4.7.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 (G2: momentum advection on the staggered box) |
| Concept refs | §2.7, §3.9, sketch 4.7 |
| Module | `grid2.operators.flux_diff` |

```python
@final
class DualFluxDifference(SeparableOperator):
    """Exact dual-cell flux difference: Center -> FaceAvg."""

    dispatch_kind: ClassVar[str | None] = "flux_diff"

    def __init__(self) -> None:
        """Create the exact dual-cell face-difference kernel."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """flux_diff: Center -> FaceAvg (exact FTC);
        CellAvg -> FaceAvg (O(dx^2) identification)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 1, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """i k sinc(k w / 2) on the dual mesh (Fourier factors)."""
        ...
```

Notes:

- **Exact-FTC semantics on the dual mesh**: the dual cell around face
  `i + 1/2` spans `[x_i, x_{i+1}]`, so
  `(1 / w_{i+1/2}) int_dual du/dx = (u(x_{i+1}) - u(x_i)) / w_{i+1/2}`
  — exact when the domain holds **point values at centers**
  (`Center -> FaceAvg`), with `w` the dual cell-width measure field
  on `FaceAvg` (`grid.measure`, §2.7). Summing against the dual
  measure telescopes to the outermost centers — the mimetic property,
  exactly parallel to `FluxDifference`.
- The `CellAvg -> FaceAvg` row carries the **declared O(dx^2)
  identification** of cell averages with midpoint values (§3.9,
  sketch 4.7's "explicit approximate conversion") — fluxes computed
  on average spaces difference into the momentum control volume
  without a silent retag.
- Registered under the same `"flux_diff"` kind, keyed by the center
  domains — no collision with `FluxDifference`'s face-domain rows.

### FaceDifference

The FV pressure gradient: two-point difference of cell values landing
on face point values — signature `diff: CellAvg -> face space` in the
review's notation, registered under its own kind.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 (G2: the C-grid pressure-gradient term) |
| Concept refs | §2.7, §3.9, sketch 4.7 |
| Module | `grid2.operators.flux_diff` |

```python
@final
class FaceDifference(SeparableOperator):
    """FV pressure gradient: CellAvg -> face point values."""

    dispatch_kind: ClassVar[str | None] = "face_diff"

    def __init__(self) -> None:
        """Create the two-point dual-spacing difference kernel."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """face_diff: CellAvg -> Right (periodic) | Inner (bounded)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 1, layout "any"."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """2 i sin(k dx / 2) / dx times the sinc/phase retag (Fourier)."""
        ...
```

Notes:

- `(p_{i+1} - p_i) / d_{i+1/2}` with `d` the **dual center-to-center
  spacing measure** on `Right`/`Inner` (`grid.measure`, §2.7). The
  codomain is a *point-value* face space (where the C-grid momentum
  DOFs live, sketch 4.7), so the operator carries the **documented
  O(dx^2) identification** on both ends: cell averages read as
  midpoint values, and the two-point difference lands as the face
  point value. This is the declared-conversion discipline of §3.9,
  not a silent equivalence.
- A dedicated kind (`"face_diff"`) rather than `("diff", CellAvg)`:
  a registry entry fixes one codomain, and `("diff", CellAvg)` is
  §3.9's normative `FVDerivative` composition (`CellAvg -> CellAvg`).
  The pressure-gradient term resolves `("face_diff", CellAvg)`.

**Flux-form closure (G2).** The sketch-4.7 C-grid step closes under
the default table: the advecting velocity reaches the flux point via
per-axis `("reconstruct", ...)` rows (`v.to(u_space)`); the flux
itself is `u * q` via the `("multiply", CellAvg/FaceAvg)`
second-order shortcut; the tracer flux divergence is per-axis
`("flux_diff", Outer/Right/Inner) -> CellAvg`; momentum advection
differences center fluxes into the staggered box via
`("flux_diff", Center/CellAvg) -> FaceAvg` (`DualFluxDifference`);
the Coriolis reconstruction `u.to(v)` is the `reconstruct` rows in
both directions; and the pressure gradient is
`("face_diff", CellAvg) -> Right/Inner` (`FaceDifference`).

### FVDerivative

The FV derivative `flux_diff ∘ reconstruct` — the default `"diff"`
entry on average spaces. Under D1 it is **not a bespoke class** but a
factory that builds the algebra chain; the result is an ordinary
`SeparableComposite`.

| | |
|---|---|
| Kind | factory function -> `SeparableComposite` |
| Pytree | static (the composite) |
| Iteration | 1 |
| Concept refs | §3.4, §3.9; merge D1/D4/D5 |
| Module | `grid2.operators.flux_diff` |

```python
def FVDerivative(
    reconstruct: SeparableOperator | None = None,
) -> SeparableComposite:
    """flux_diff @ (reconstruct or Dispatched("reconstruct")).

    With ``reconstruct=None`` the reconstruction is a ``Dispatched``
    placeholder resolved once at model assembly (D4); an explicit
    kernel pins the composition. Both are same-axis separable factors,
    so the chain is a ``SeparableComposite`` (CellAvg -> CellAvg): it
    binds with ``["x"]``, its halo is the sum of the factor halos
    (§3.6), and ``f.diff("x")`` on an average space resolves to it.
    """
    return FluxDifference() @ (reconstruct or Dispatched("reconstruct"))
```

Notes:

- The reconstruction is a **`Dispatched("reconstruct")` placeholder
  resolved once at model assembly** (D4/D5), not the old apply-time
  late binding — so a module override of `"reconstruct"` (sketch 4.2)
  still propagates into what `f.diff("x")` does on average spaces (the
  reason advection schemes override reconstruction, not diff), but the
  baked chain is fully concrete and static, with no per-application
  registry lookup.
- Halo composes additively as an un-synced chain — but now *because*
  `SeparableComposite.requirements` sums its factor halos (§3.6), not
  via bespoke code. The registered default is the composite object
  itself, so the halo-accounting trace sees it with no special casing.

---

## Spectral (coefficient-space) operators

### SpectralDerivative

Exact derivative on coefficient spaces; the only `"diff"` choice
there.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §3.2, §3.4, §6.1 |
| Module | `grid2.operators.spectral` |

```python
@final
class SpectralDerivative(SeparableOperator):
    """Exact derivative in coefficient space (i k multiply / recurrence)."""

    dispatch_kind: ClassVar[str | None] = "diff"

    def __init__(self) -> None:
        """Create the spectral derivative."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Fourier -> Fourier (same origin); Sine -> Cosine; Cheb recurrence."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 0, layout "local" (whole-axis coefficient access)."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """i k on a Fourier factor space (non-retagging)."""
        ...
```

Notes:

- All three paths are **iteration 1** (U4: all coefficient spaces and
  transforms ship day one): the Fourier diagonal `i k` multiply —
  origin preserved (spectral differentiation does not stagger) — the
  sine/cosine bc-flip, and the Chebyshev recurrence. The bc-flipping
  signature `d/dx : SineCoeff -> CosineCoeff` (§3.2) replaces today's
  `SpectralDiff` metadata mutation.
- **Mode-index bookkeeping (sine/cosine pairs).** The index maps are
  explicit and shape-honest (§3.5):
  - *II-type pair (Center origin):* DST-II carries sine modes
    `k = 1..n`, DCT-II cosine modes `k = 0..n-1` (scipy layout:
    DST-II array index `j = 0..n-1` holds mode `k = j + 1`; DCT-II
    index `j` holds `k = j` — the forward map is the array shift
    `j -> j + 1`). `d/dx : Sine -> Cosine` maps sine-`k` to
    cosine-`k` for `k = 1..n-1`, **annihilates the top sine mode**
    `k = n` (its cosine image vanishes identically at the center
    nodes), and never populates cosine `k = 0`. The reverse
    `d/dx : Cosine -> Sine` annihilates the constant `k = 0` and
    lands inside sine `k = 1..n-1`; the top sine mode is never
    populated.
  - *I-type pair (`Inner` Dirichlet <-> `Outer` Neumann):* DST-I
    carries sine modes `k = 1..n-1` (scipy index `j = k - 1`), DCT-I
    cosine modes `k = 0..n` (index `j = k`). `d/dx : Sine -> Cosine`
    lands in cosine `k = 1..n-1` (neither `k = 0` nor `k = n` is
    populated); the reverse annihilates the constant `k = 0` **and**
    the Nyquist cosine `k = n` (its sine image vanishes identically
    at the interior faces) and lands in sine `k = 1..n-1`.
- `layout "local"`: the Chebyshev recurrence couples all modes of the
  factor; the Fourier diagonal case could relax this, but one
  conservative declaration keeps the negotiation simple (revisit knob
  noted in doc 04).

### PhaseShift

Exact inter-origin conversion between Fourier coefficient spaces —
the `"interp"` entry in coefficient space
([§3.2](../02_rules.md#32-coefficient-representations-are-separate-spaces),
sketch 4.3).

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §3.2, §3.4, §6.1, sketch 4.3 |
| Module | `grid2.operators.spectral` |

```python
@final
class PhaseShift(SeparableOperator):
    """Inter-origin e^{i k s dx} shift between Fourier spaces."""

    dispatch_kind: ClassVar[str | None] = "interp"

    def __init__(self, to: NodeSet = NodeSet.CENTER) -> None:
        """Shift to the Fourier space of the given origin node set."""
        ...

    @property
    def to(self) -> NodeSet:
        """Target origin node set (NodeSet.CENTER, NodeSet.RIGHT, ...)."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Fourier(origin=A) -> Fourier(origin=<to>) on the same mesh."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 0, layout "any" (pure diagonal multiply)."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """The retagging phase symbol e^{i k s dx} itself."""
        ...
```

Notes:

- Diagonal: applying it *is* applying its own symbol. The default
  registry entry `("interp", Fourier(origin=Right)) ->
  PhaseShift(to=NodeSet.CENTER)` makes `u_hat.to(w_hat)` in
  sketch 4.3 work; other targets are explicit instances (the `.to`
  sugar errors if the registered codomain does not match the target,
  §3.4).
- **Exactness caveat (real origins, even n).** On the Hermitian
  half-spectrum of a `fr.Real` origin with even `n`, the
  `e^{i k dx/2}` shift is *not* an exact spectrum automorphism: the
  Center-origin Nyquist coefficient is real, and multiplying it by
  `±i` leaves no valid rfft layout — indeed the Nyquist cosine
  sampled at centers vanishes identically at the faces. `PhaseShift`
  therefore **zeroes the Nyquist mode** on real-origin even-`n`
  factors; this is the one non-exact DOF, documented here and in the
  matching §3.2 caveat added to `02_rules.md` in this change set. On
  complex origins and odd `n` the shift is exact and unitary.
- **One-directional seeding.** Only
  `("interp", Fourier(o != center)) -> PhaseShift(to=NodeSet.CENTER)`
  is seeded: a single-codomain entry cannot express per-target
  defaults, so Center -> staggered conversions require explicit
  `PhaseShift(to=NodeSet.RIGHT)`-style instances. Sketch-4.9-style
  eigenmode assembly constructs its inter-origin shifts explicitly
  for the same reason.
- The target is doc 01's `NodeSet` enum member, not a space object —
  the operator stays mesh-agnostic; `codomain` resolves the member
  against the domain's mesh. A string token (`"center"`) is **not**
  accepted, uniform with the no-string-keys rule for interned space
  identities.
- Identity when the origin already matches. On a purely collocated
  spectral grid (§6.1) no entry ever fires — the `DummyInterpolation`
  pathology disappears structurally.

### SincShift

Diagonal conversion between **average-origin** and nodal-origin
Fourier spaces — the named `sinc(k dx / 2)` factor of §3.2/§3.9.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 (G2: average-origin coefficient spaces are day one) |
| Concept refs | §3.2, §3.9 |
| Module | `grid2.operators.spectral` |

```python
@final
class SincShift(SeparableOperator):
    """sinc(k dx / 2) conversion between average and nodal origins."""

    dispatch_kind: ClassVar[str | None] = "interp"

    def __init__(self, to: NodeSet = NodeSet.CENTER) -> None:
        """Convert to the Fourier space of the given origin node set."""
        ...

    @property
    def to(self) -> NodeSet:
        """Target origin node set (nodal or average)."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Fourier(origin=cell_avg/face_avg) <-> Fourier(origin=nodal)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 0, layout "any" (pure diagonal multiply)."""
        ...

    def eigenvalues(
        self, grid: Grid, space: FunctionSpace,
    ) -> Symbol:
        """The retagging sinc (times phase) symbol itself."""
        ...
```

Notes:

- Cell-averaging is convolution with a top-hat (§3.2), so an
  average-origin spectrum differs from the nodal-origin one by
  `sinc(k dx / 2)`: nodal -> average multiplies, average -> nodal
  divides. Where the origins are additionally offset by half a cell
  (`face_avg -> center`), the diagonal composes the sinc factor with
  the corresponding phase — one operator, one symbol.
- Exists because `PhaseShift(to=...)` targets **nodal** origins only;
  average origins dispatch here (default rows below). The division is
  **invertible on the resolved band**: `sinc(k dx / 2)` has its first
  zero at `k dx = 2 pi`, beyond the Nyquist `k dx = pi`
  (`sinc(pi / 2) = 2 / pi`), so no resolved mode is annihilated.

---

## Transforms

Transforms are the **deliberate exception** to grid-freedom (§2.5):
they bind the grid at construction because they need the domain
decomposition and FFT plan up front, and they are applied through
`forward`/`backward`.

### Transform

| | |
|---|---|
| Kind | ABC |
| Pytree | static structure; `_grid` is a fully static reference (plans, layouts, refined meshes — G1) |
| Iteration | 1 |
| Concept refs | §2.5, §3.2, §3.12, sketches 4.3, 4.10 |
| Module | `grid2.operators.transform` |

```python
@fr.utils.jaxify
class Transform(UnaryOperator, ABC):
    """Grid-bound change of representation: nodal <-> coefficient."""

    dispatch_kind: ClassVar[str | None] = "transform"

    def __init__(
        self,
        grid: Grid,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid; transform along ``axes`` (default: all)."""
        ...

    # ------------------------------------------------------------
    #  Properties
    # ------------------------------------------------------------
    @property
    def grid(self) -> Grid:
        """The bound grid (decomposition + plans)."""
        ...

    @property
    def axes(self) -> tuple[str, ...]:
        """Coordinate names this transform acts along."""
        ...

    @property
    def pad(self) -> PadFactor | None:
        """Dealiasing pad factor, or None for the plain transform."""
        ...

    # ------------------------------------------------------------
    #  Application
    # ------------------------------------------------------------
    @abstractmethod
    def forward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Nodal/average -> coefficient (trims if padded)."""
        ...

    @abstractmethod
    def backward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Coefficient -> origin nodal space (padded if ``pad``)."""
        ...

    def _apply(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Delegate to ``forward`` (registry-uniform application)."""
        ...

    # ------------------------------------------------------------
    #  Space resolution
    # ------------------------------------------------------------
    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """Forward target: the per-origin coefficient space."""
        ...

    def backward_space(self, domain: FunctionSpace) -> FunctionSpace:
        """Backward target: origin space, refined by ``pad`` if set."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 0, layout "transpose" (distributed transforms)."""
        ...
```

Notes:

- **Per-origin coefficient handling** (sketch 4.3): `codomain` maps
  each named-axis factor `s` to `s.mesh` Fourier/…-space *with origin
  `s`* (`mx.fourier(origin=...)` — the fixed space-factory anchor).
  `backward` needs no target argument: the origin is constitutive of
  the coefficient space (§3.2), so the round trip is unambiguous.
- `forward`/`backward` on a `VectorField`/`State` map componentwise
  via the inherited `VectorField.map` (sketch 4.9); each component
  keeps its own per-origin coefficient spaces.
- **Dealiasing is a property of the transform** (§3.12): with
  `pad=degree(p)`, `backward` lands in the finer nodal space
  (`Fourier(N) -> Center((p+1)/2 N)`, a genuine first-class space)
  and `forward` from that finer space trims back to the unpadded
  coefficient space. Both space families are fixed at construction:
  the finer nodal space lives on the refined mesh obtained via
  doc 01's `StructuredMesh1D.refined(factor)` (iteration 1 on
  `IntervalMesh`, `Fraction` factor), called once when the transform
  binds the grid.
- **Padded-`forward` codomain exception.** The padded `forward` takes
  the *refined* mesh's `Center((3/2) N)` and lands on the **coarse**
  space's coefficient space — a deliberate exception to the
  per-origin codomain rule (the codomain's origin is the coarse
  `Center(N)`, not the refined domain). The transform stores both
  space families at construction and derives the coarse target
  through doc 01's `refined_from` parent link on the refined mesh.
- **Deviation callout.** §2.5 says transforms are applied through
  `.forward`/`.backward` "rather than" `op(field, axis=...)`; the
  inherited `__call__` (via `_apply` delegating to `forward`)
  deviates from that letter for registry uniformity — a
  `("transform", space)` entry must be applicable through the
  uniform operator calling convention. User code is still expected
  to write `forward`/`backward`; the deviation is recorded here
  explicitly.

### Fourier

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static structure (bound grid) |
| Iteration | 1 (`pad=` included: parity with `FFTPadding`); `truncation_mask` designed-for (with `Symbol`) |
| Concept refs | §3.1, §3.2, §3.12, sketches 4.3, 4.10, 4.11 |
| Module | `grid2.operators.fourier` |

```python
@final
class Fourier(Transform):
    """FFT-family transform to per-origin Fourier coefficient spaces."""

    def __init__(
        self,
        grid: Grid,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid and plan (r)FFTs along ``axes``."""
        ...

    def forward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """rfft for fr.Real spaces, full fft for fr.Complex spaces."""
        ...

    def backward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Inverse transform to the (padded) origin nodal space."""
        ...

    def truncation_mask(
        self,
        space: FunctionSpace | TensorProductSpace,
        keep: Fraction = Fraction(2, 3),
    ) -> Symbol:
        """0/1 truncation-filter Symbol on ``space`` (2/3 rule)."""
        ...
```

Notes:

- **Scalars drive the realization with no special-casing** (§3.1,
  sketch 4.11): `fr.Real` origins produce the Hermitian half-spectrum
  coefficient space (rfft layout as *shape*, §3.2), `fr.Complex`
  origins the full spectrum. The `RFFTPressureSolver` bypass
  disappears: the rfft *is* the dispatched default.
- **Average origins are ordinary origins** (G2): `forward` from
  `CellAvg`/`FaceAvg` spaces is the (r)fft of the stored averages,
  landing on `Fourier(origin=cell_avg)`-style coefficient spaces —
  the origin tag carries the `sinc` relationship to the nodal
  origins, converted explicitly by `SincShift` (§3.2/§3.9).
- `truncation_mask` returns the fixed 0/1 diagonal
  `Fourier(N) -> Fourier(N)` of §3.12's 2/3 rule as a `Symbol`
  (applied as a Hadamard); it lives on the transform because the
  retained-band bookkeeping does.
- Distributed operation is transpose-based (jaxDecomp-style), declared
  through `requirements` and negotiated by the grid (doc 04); the
  transform API is deliberately rich enough that solvers no longer
  bypass it ([§5](../04_decomposition.md#5-domain-decomposition)).

### Sine / Cosine

DST/DCT transforms for bounded nodal spaces with Dirichlet/Neumann
structure.

| | |
|---|---|
| Kind | concrete (final), two classes |
| Pytree | static structure (bound grid) |
| Iteration | 1 (U4: all coefficient spaces and transforms are day one; bounded-axis parity with today's `cartesian/fft.py` DST/DCT paths) |
| Concept refs | §3.2, §3.5, §3.6 |
| Module | `grid2.operators.trig` |

```python
@final
class Sine(Transform):
    """DST transform; type (I/II) selected by the origin node set."""

    def __init__(
        self,
        grid: Grid,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid and plan DSTs along ``axes``."""
        ...

    def forward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Nodal Dirichlet space -> sine coefficient space."""
        ...

    def backward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Sine coefficients -> the origin nodal space."""
        ...


@final
class Cosine(Transform):
    """DCT transform; type selected by the origin node set."""

    def __init__(
        self,
        grid: Grid,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid and plan DCTs along ``axes``."""
        ...

    def forward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Nodal Neumann space -> cosine coefficient space."""
        ...

    def backward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Cosine coefficients -> the origin nodal space."""
        ...
```

Notes:

- The DST-I vs DST-II choice is **not a parameter**: it follows from
  the origin space (Dirichlet `Inner` -> DST-I with n - 1 modes,
  Dirichlet `Center` -> DST-II with n modes, §3.2/§3.5) — coefficient
  counts equal space shapes by construction, replacing today's
  position-driven if/else in `cartesian/fft.py`.
- A single `Trig` class multiplexing sine/cosine by BC was rejected:
  the two have different codomain families and `d/dx` couples them
  (`Sine <-> Cosine` under `SpectralDerivative`), so distinct classes
  keep signatures honest.
- The mode-index alignment between the sine and cosine coefficient
  families (scipy type-I/II conventions) is fixed in the
  `SpectralDerivative` bookkeeping notes above; the transforms and
  the derivative share those index maps.

### Chebyshev

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static structure (bound grid) |
| Iteration | 1 (U4: `ChebyshevMesh` and its spaces are iteration 1 per doc 01) |
| Concept refs | §3.2, §6.2 |
| Module | `grid2.operators.chebyshev` |

```python
@final
class Chebyshev(Transform):
    """Chebyshev transform (Gauss-Lobatto nodes <-> Cheb coefficients)."""

    def __init__(
        self,
        grid: Grid,
        axes: tuple[str, ...] | str | None = None,
        pad: PadFactor | None = None,
    ) -> None:
        """Bind the grid and plan Chebyshev transforms along ``axes``."""
        ...

    def forward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Gauss-Lobatto nodal -> Chebyshev/Shen coefficient space."""
        ...

    def backward(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Chebyshev/Shen coefficients -> the origin nodal space."""
        ...
```

Shen-basis codomains (BCs baked in, shape n - k) follow the origin
space's BC structure (§3.6). `requirements` declares
`layout "transpose"` along its axes: per the owner directive,
Chebyshev meshes declare **transpose-capable** decomposition traits,
so the transform plans pencil layouts exactly like `Fourier` (the
§6.2 shard-x/y-keep-z-local layout remains a valid negotiation
outcome, no longer a structural restriction).

### PadFactor and `dealias.degree`

| | |
|---|---|
| Kind | final frozen dataclass + factory function |
| Pytree | static value |
| Iteration | 1 (consumed by `pad=`) |
| Concept refs | §3.12, sketch 4.10 |
| Module | `grid2.operators.dealias` |

```python
@dataclass(frozen=True)
class PadFactor:
    """Dealiasing pad factor for padded transform variants."""

    #: refinement ratio of the padded nodal space, e.g. 3/2
    factor: Fraction


def degree(p: int) -> PadFactor:
    """Pad factor (p + 1) / 2 for a degree-p nonlinearity."""
    ...
```

Sketch 4.10 spells this `fr.dealias.degree(2)`; normatively it lives
in `fr.operators.dealias` (only `fr.meshes` / `fr.operators` are
fixed top-level namespaces, §8). A top-level `fr.dealias` alias is an
open question below.

---

## Pointwise and product operators

The three product realizations of §3.11, plus the small pointwise
family behind doc 02's remaining field arithmetic (`/`, `**`,
`abs()`) and the `ConstantSpace` broadcast. The infix `f * g` (a
`ScalarField` dunder, owned by doc 02) is sugar for the dispatched
default physical product: it resolves `("multiply", space)` on the
operands' common space and calls the registered binary operator (see
the product-key resolution rule in the registry section). `Hadamard`
is *never* reachable through `*`.

### CollocationProduct

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 (the `*` default on nodal spaces and — as the 2nd-order shortcut — on average spaces) |
| Concept refs | §3.11, §3.12, sketch 4.10 |
| Module | `grid2.operators.products` |

```python
@final
class CollocationProduct(BinaryOperator):
    """Pointwise physical product on nodal spaces (aliased)."""

    dispatch_kind: ClassVar[str | None] = "multiply"

    def __init__(self) -> None:
        """Create the collocation (elementwise nodal) product."""
        ...

    def codomain(
        self, domain_a: FunctionSpace, domain_b: FunctionSpace,
    ) -> FunctionSpace:
        """Common space after ConstantSpace broadcast / C-promotion."""
        ...

    def _apply(
        self, f: ScalarField, g: ScalarField, *more: ScalarField,
    ) -> ScalarField:
        """Elementwise product of nodal values."""
        ...
```

Aliased unless the operands live on a padded (finer) nodal space —
dealiasing is the bracketing transforms' job, never this operator's
(§3.12). halo 0, layout "any".

Also the iteration-1 `("multiply", CellAvg/FaceAvg)` default: the
pointwise product of averages is the standard **second-order
shortcut** (identifying averages with midpoint values, O(dx^2) — the
§3.9 identification made explicit), covering both members of the
average family. The true quadrature product (reconstruct, multiply,
re-average) remains designed-for and replaces these entries when
higher-order FV lands. All `("multiply", *)` rows hold **one shared
instance** — a requirement of the registry's form-2 resolution (see
the registry notes).

### ConstantBroadcast

The `("broadcast", ConstantSpace)` entry: embeds a `ConstantSpace`
factor into a full factor space — the operator realization of the
§3.3 sanctioned broadcast, resolved by doc 02's field arithmetic
(this closes doc 02's open question 1).

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 on nodal/average factors; exact delta embedding on coefficient factors designed-for |
| Concept refs | §3.3, §3.11 |
| Module | `grid2.operators.products` |

```python
@final
class ConstantBroadcast(SeparableOperator):
    """Embed a ConstantSpace factor into a full factor space."""

    dispatch_kind: ClassVar[str | None] = "broadcast"

    def __init__(self) -> None:
        """Create the broadcast embedding."""
        ...

    def codomain(
        self,
        domain: FunctionSpace,
        to: FunctionSpace | None = None,
    ) -> FunctionSpace:
        """broadcast: ConstantSpace(m) -> ``to`` (target factor)."""
        ...

    def _apply(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Replicate the constant DOF into the bound target factor."""
        ...
```

Unlike other separable kernels, the codomain is not a function of the
domain alone: the target factor is supplied by the *other* operand
when binary `codomain` unites a constant factor with a full one. Under
bind-only (D2) that target is carried by binding rather than a call
keyword — the concrete spelling (`broadcast[target]` vs a `to=`
constructor argument) is the **D3 open detail** (see Open questions) —
a recorded widening of the per-factor signature, analogous to the
composed-operator widening. On nodal/average targets the embedding is
exact replication (halo 0, layout "any", iteration 1). On coefficient
targets it is the exact delta embedding (the constant lands in the
k = 0 / mean mode, scaled by the basis normalization) — designed-for.
**The delta embedding requires a constant mode in the basis**: on
`SineSpace`/Dirichlet-Shen bases no exact constant exists, so the
broadcast raises `SpaceMismatchError` rather than projecting — the
field-lift analogue of doc 02's "lifts are exact" rule.

### Divide / Power / Abs

Pointwise arithmetic behind doc 02's iteration-1 field dunders
(`f / g`, `f ** p`, `abs(f)`). They share `CollocationProduct`'s
semantics — elementwise on local-DOF (nodal) spaces, second-order
shortcut on `CellAvg` *and* `FaceAvg`. `Divide` and `Abs` have **no
coefficient-space entries** (a quotient or modulus of spectra has no
representation-independent realization; transform back first);
`Power` does admit one — an integer power realized by repeated
`Convolution` *is* the representation-independent physical power —
recorded as a designed-for row below.

| | |
|---|---|
| Kind | concrete (final), three classes |
| Pytree | static |
| Iteration | 1 (`divide`/`power` on nodal + `CellAvg`/`FaceAvg`; `abs` on nodal only) |
| Concept refs | §3.11 |
| Module | `grid2.operators.products` |

```python
@final
class Divide(BinaryOperator):
    """Pointwise quotient on nodal/average spaces."""

    dispatch_kind: ClassVar[str | None] = "divide"

    def __init__(self) -> None:
        """Create the pointwise quotient."""
        ...

    def codomain(
        self, domain_a: FunctionSpace, domain_b: FunctionSpace,
    ) -> FunctionSpace:
        """Common space after broadcast / C-promotion."""
        ...

    def _apply(
        self, f: ScalarField, g: ScalarField, *more: ScalarField,
    ) -> ScalarField:
        """Elementwise quotient of nodal values."""
        ...


@final
class Power(BinaryOperator):
    """Pointwise power on nodal/average spaces."""

    dispatch_kind: ClassVar[str | None] = "power"

    def __init__(self) -> None:
        """Create the pointwise power."""
        ...

    def codomain(
        self, domain_a: FunctionSpace, domain_b: FunctionSpace,
    ) -> FunctionSpace:
        """Common space; scalar exponents leave the space unchanged."""
        ...

    def _apply(
        self, f: ScalarField, p: ScalarField | float,
    ) -> ScalarField:
        """Elementwise ``f ** p`` (scalar or same-space exponent)."""
        ...


@final
class Abs(UnaryOperator):
    """Pointwise absolute value / complex modulus on nodal spaces."""

    dispatch_kind: ClassVar[str | None] = "abs"

    def __init__(self) -> None:
        """Create the pointwise absolute value."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """abs: S -> the fr.Real variant of S."""
        ...

    def _apply(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Elementwise magnitude (whole-space, not bindable)."""
        ...
```

`Abs` is the one scalar-*changing* pointwise op (`fr.Complex ->
fr.Real`, the conjugation criterion of §3.1); `f ** 2` on a field
resolves `("power", space)` and is the physical square per §2.5's
`**` table, never `Symbol.__pow__`. **`Abs` is nodal-only by
design**: a sign change inside a cell makes `|avg(u)| != avg(|u|)`,
so the second-order average-space shortcut would be *silently* wrong
— unlike products, where the shortcut error is a controlled O(dx^2).

### Where

The `("select", ...)` kind: an elementwise ternary select, so that
upwind flux-sign selection stays **inside the operator layer** where
doc 04's halo tracer can see it — a raw `jnp.where` on `.data` would
be invisible to the halo-accounting trace.

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 (nodal + `CellAvg`/`FaceAvg`) |
| Concept refs | §3.11, §5, sketch 4.2 |
| Module | `grid2.operators.products` |

```python
@final
class Where(BinaryOperator):
    """Elementwise ternary select on same-space operands."""

    dispatch_kind: ClassVar[str | None] = "select"

    def __init__(self) -> None:
        """Create the elementwise select."""
        ...

    def codomain(
        self,
        domain_cond: FunctionSpace,
        domain_a: FunctionSpace,
        domain_b: FunctionSpace,
    ) -> FunctionSpace:
        """Common space of the branches (cond broadcast-united)."""
        ...

    def _apply(
        self, cond: ScalarField, a: ScalarField, b: ScalarField,
    ) -> ScalarField:
        """``where(cond, a, b)`` on aligned local data."""
        ...
```

Ternary via the binary template's `*more` slot; `codomain` widens to
three domains (recorded, like the composed-operator widening). All
operands share a space after the sanctioned implicit exceptions;
halo 0, layout "any". Flux-splitting advection combines it with the
biased `WenoReconstruction` pair — sign selection routed through
`("select", space)` is what makes the whole upwind path visible to
the halo tracer.

### Hadamard

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | designed-for (first consumer: spectra-based ICs, sketch 4.9) |
| Concept refs | §3.11, sketch 4.9 |
| Module | `grid2.operators.products` |

```python
@final
class Hadamard(BinaryOperator):
    """Coefficient-wise (elementwise) product on one coefficient space."""

    dispatch_kind: ClassVar[str | None] = None   # never a `*` default

    def __init__(self) -> None:
        """Create the coefficient-wise product."""
        ...

    def codomain(
        self, domain_a: FunctionSpace, domain_b: FunctionSpace,
    ) -> FunctionSpace:
        """The (identical, broadcast-united) coefficient space."""
        ...

    def _apply(
        self, f: ScalarField, g: ScalarField, *more: ScalarField,
    ) -> ScalarField:
        """Elementwise multiply of two or more same-space fields."""
        ...
```

Notes:

- Elementwise in coefficients = convolution in physical space; it has
  no representation-independent meaning, so it is **always named
  explicitly and never spelled `*`** (§3.11). Not registered under
  `"multiply"` by design.
- Variadic call via the binary template's `*more` slot: sketch 4.9's
  amplitude-and-phase scaling is `Hadamard()(f, a, r)` — associative
  elementwise products fuse into one call. The instance-call spelling
  is the **accepted decision**
  (§3.11 of `02_rules.md` is amended accordingly in the same change
  set): operators are instances constructed from parameters and then
  applied, uniformly across the cluster (a `__new__` returning a
  field would break the "operators are static structure" pytree rule
  and the registry's instance-holding contract).
- Applying a `Symbol` to a field *is* this operation;
  `Symbol.__call__` is specified to be behaviorally identical to
  `Hadamard()(field_view_of_symbol, f)` with the codomain retag.

### Convolution

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static structure (holds a grid-bound transform) |
| Iteration | designed-for (the `*` default on coefficient spaces, §6.1) |
| Concept refs | §3.11, §3.12 |
| Module | `grid2.operators.products` |

```python
@final
class Convolution(BinaryOperator):
    """Physical product realized in coefficient space (dealiased)."""

    dispatch_kind: ClassVar[str | None] = "multiply"

    def __init__(self, transform: Transform) -> None:
        """Define via a padded transform pair (reads its pad factor)."""
        ...

    @property
    def transform(self) -> Transform:
        """The padded transform bracketing the product."""
        ...

    def codomain(
        self, domain_a: FunctionSpace, domain_b: FunctionSpace,
    ) -> FunctionSpace:
        """The (identical) coefficient space of both operands."""
        ...

    def _apply(
        self, f: ScalarField, g: ScalarField, *more: ScalarField,
    ) -> ScalarField:
        """trim_forward o CollocationProduct o pad_backward."""
        ...
```

Defined as the composite of §3.12 — it adds no primitive. Because it
holds a `Transform` it is grid-bound *by containment*; the registry
entry is grid-owned anyway, so this costs nothing. Re-transforms
shared operands; the transform-once combinator below is the
performant multi-term idiom.

### transform_once

| | |
|---|---|
| Kind | pure higher-order function (combinator) |
| Pytree | n/a (function) |
| Iteration | designed-for |
| Concept refs | §3.12, sketch 4.10 |
| Module | `grid2.operators.combinators` |

```python
def transform_once(
    transform: Transform,
    fn: Callable[..., dict[str, ScalarField]],
    **operands: ScalarField,
) -> dict[str, ScalarField]:
    """One padded backward per operand, ``fn`` on the finer nodal
    space, one forward+trim per named output."""
    ...
```

Eager, boilerplate-free, transform-once by construction; `fn`
receives the padded nodal fields as keyword arguments matching
`operands` and returns named nodal results. Automatic scheduling over
a lazy expression graph is **rejected** (§3.12); XLA CSE remains the
backstop for redundant backwards.

---

## Reductions

### Integral

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 (`f.integrate` is day-one surface, §10.4) |
| Concept refs | §2.7, §3.3, §3.13 |
| Module | `grid2.operators.integrate` |

```python
@final
class Integral(SeparableOperator):
    """Weighted integral of a factor, landing in ConstantSpace."""

    dispatch_kind: ClassVar[str | None] = "integrate"

    def __init__(self) -> None:
        """Create the quadrature-weighted reduction."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """integrate: S(m) -> ConstantSpace(m); Constant -> Constant."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 0, layout "any", collective = True."""
        ...
```

Contracts the field against the space's quadrature-weight measure
field (`grid.measure(space, name=...)`, doc 04, iteration 1; uniform
`dx`, Clenshaw-Curtis, Jacobian weights — §3.13), read at trace time.
The designed-for mapping-metric accessor `grid.metric` is *not* used
here — measures and mapping metrics are distinct grid accessors.
Exact on average spaces. The result broadcasts back via
`ConstantSpace` (§3.3), so `f - f.integrate("x")` stays in the strict
algebra. There is no unweighted `sum` operator (`f.data.sum()` is the
escape hatch). The cross-shard sum is declared through
`collective=True` (informational, no layout constraint).

Scope of the default rows: **nodal and average factors only**.
Coefficient factors deliberately have no `("integrate", ...)` default
— resolving one raises `DispatchError` with the guidance "transform
back first" — because the honest coefficient-space integral is the
zero-mode extraction `L * c0`, recorded as a **designed-for**
coefficient row (it needs per-basis normalization bookkeeping).
Related Parseval note for spectral diagnostics: reductions over
real-origin **half-spectra** must double-weight the
non-self-conjugate modes (Parseval on the rfft layout); that
weighting belongs to the designed-for coefficient rows, never to
`f.data.sum()` escape hatches.

### CumulativeIntegral

| | |
|---|---|
| Kind | concrete (final) |
| Pytree | static |
| Iteration | 1 (parity: hydrostatic-pressure-style running integrals) |
| Concept refs | §3.9, §3.13 |
| Module | `grid2.operators.integrate` |

```python
@final
class CumulativeIntegral(SeparableOperator):
    """Running integral along a factor (discrete-FTC partial inverse)."""

    dispatch_kind: ClassVar[str | None] = "cumint"

    def __init__(
        self, direction: Literal["forward", "backward"] = "forward",
    ) -> None:
        """Create a running integral in the given direction."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """cumint: Center/CellAvg -> Outer (bounded);
        Center/CellAvg -> Right (periodic, mean-zero input)."""
        ...

    def requirements(
        self, domain: FunctionSpace,
    ) -> OperatorRequirements:
        """halo = 0, layout "local" (scan along the axis)."""
        ...
```

Notes:

- The staggering change (running integral lands on a face space — the
  partial inverse of `flux_diff`) is this operator's decision, per
  §3.13's explicit delegation.
- **Bounded codomain is `Outer` (n + 1)** — information-preserving:
  landing on `Inner` would drop *both* boundary faces and discard the
  total integral (the final running value). The boundary face fixed
  by the integration constant holds an exact zero; the opposite face
  holds the total.
- **Periodic `Center/CellAvg -> Right` requires mean-zero input** —
  the running integral of a nonzero-mean field is not periodic. This
  is a documented **caller invariant** (checked at most by an
  optional debug assertion, not by the type system); remove the mean
  first via `f - f.integrate("x")` (§3.13).
- **Integration-constant convention:** `direction="forward"` fixes
  the integral to zero at the domain-start face (the first `Outer`
  DOF; on periodic axes the face preceding the first cell) and
  accumulates rightward; `direction="backward"` fixes zero at the
  domain-end face and accumulates leftward — the hydrostatic
  integrate-down-from-the-surface case.

---

## Composed operators (vector calculus)

Generic **dispatch kinds**, not special slots (§3.4), and under D1
**algebra-derived**: the default entries on separable grids are
`Block`s and chains over `"diff"`/`"interp"`/`"flux_diff"` (B1), *not*
bespoke classes. `grad`/`div`/`curl`/`laplacian` have **grid-dependent
block shape** (2-D vs 3-D), so they are `Dispatched`-family (B2): the
registered default for the kind is a builder that, at model assembly,
expands against the grid's axis family into a `Block` whose entries are
`Dispatched("diff")` bound per axis (late-bound, so module overrides
propagate). On non-separable meshes (sphere, unstructured) models
register primitive metric-aware mesh-level operators under the *same
kinds* (§6.3, §6.4) — not blocks; nothing here assumes the registered
entry is separable or block-structured. The factory names
(`fr.operators.Gradient(...)` etc.) return that builder; the kind and
the factory resolve to the same object. There is no distinct
`Laplacian` *type* — `isinstance(op, Laplacian)` no longer exists (D1).

### Gradient / Divergence / Curl / Laplacian

| | |
|---|---|
| Kind | factory functions -> `Block` (assembly-resolved, B2) |
| Pytree | static (the block / composite) |
| Iteration | designed-for |
| Concept refs | §2.4, §3.4, §6.3, sketch 4.6/4.7; merge D1/B1/B2 |
| Module | `grid2.operators.composed` |

```python
def Gradient(order: int | None = None) -> Operator:
    """The "grad" builder (B1/B2). At model assembly it expands, over
    the grid's axis family, into a column Block
    ``[[d["x"]], [d["y"]], ...]`` — scalar ``S`` -> the staggered vector
    ``(Right⊗Center, Center⊗Right, ...)`` — each entry
    ``Dispatched("diff")`` (or ``FiniteDifference(order)``) bound to
    that axis.
    """
    ...


def Divergence(order: int | None = None) -> Operator:
    """The "div" builder: a row Block ``[[d["x"], d["y"], ...]]`` —
    the staggered vector -> the common cell space. The FV C-grid ``div``
    (sketch 4.7) resolves ``("flux_diff", factor)`` per axis, exact by
    type.
    """
    ...


def Curl(order: int | None = None) -> Operator:
    """The "curl" builder: a matrix Block with ``Zero`` structural zeros
    — the staggered vector -> its dual-staggered curl."""
    ...


def Laplacian(order: int = 2) -> Operator:
    """``div @ grad`` — block-matmul ((1×n) @ (n×1) = 1×1) to a Block
    whose single block is the ``OperatorSum`` of per-axis
    second-derivative ``SeparableComposite``s (B1)."""
    return Divergence(order) @ Gradient(order)
```

Notes:

- **These build the dispatch defaults; they are not privileged types**
  (D1, operator_design §3.5). The block *expansion* — the per-axis
  entries the halo (per block-row, §3.6) and the symbol matrix (§3.7)
  walk — is computed **on demand, never materialized** as a rewritten
  operator (§3.11). This **replaces the rejected `VectorOperator` ABC
  framing**: grad/div/curl are `Block`s with tuple signatures (§3.1),
  not unary operators with a widened `codomain` — the ABC is unneeded
  for a different reason than before.
- **Laplacian's symbol** falls out of `Block.eigenvalues` on the 1×1
  block: the `OperatorSum` of per-factor symbols — `bwd @ fwd` per
  factor, each **replication-extended** across the remaining factors
  and summed (the diagonal-operator broadcast of the `Symbol` notes,
  *not* doc 02's delta field lift; sketch 4.6). Raises
  `EigenbasisError` unless *every* factor diagonalizes (fully periodic
  grids); mixed grids fall back to banded per-column solves outside
  this operator. This is the D8-preserved `Symbol` subtlety — now
  living in the block/chain symbol calculus rather than a bespoke
  `Laplacian.eigenvalues`.
- Entries are late-bound `Dispatched("diff")`/`("flux_diff")` per axis,
  so module overrides propagate (as for `FVDerivative`). On a sphere /
  unstructured mesh the *same kinds* hold primitive metric-aware
  registrations (`div: edge-normal -> cell`) instead of blocks; nothing
  here assumes separability or block structure of the registered entry.

### RaiseIndex / LowerIndex

| | |
|---|---|
| Kind | concrete (final), two classes |
| Pytree | static |
| Iteration | designed-for (sphere/curvilinear, §6.3) |
| Concept refs | §2.4, §3.8, §6.3 |
| Module | `grid2.operators.composed` |

```python
@final
class RaiseIndex(UnaryOperator):
    """Metric-consuming index raising: covariant -> contravariant."""

    dispatch_kind: ClassVar[str | None] = "raise_index"

    def __init__(self) -> None:
        """Create the index-raising operator."""
        ...

    def codomain(
        self, *domains: TensorProductSpace,
    ) -> tuple[TensorProductSpace, ...]:
        """Contravariant-variance component spaces."""
        ...

    def _apply(
        self, f: ScalarField | VectorField,
    ) -> VectorField:
        """Contract with the inverse metric from ``grid.metric``."""
        ...


@final
class LowerIndex(UnaryOperator):
    """Metric-consuming index lowering: contravariant -> covariant."""

    dispatch_kind: ClassVar[str | None] = "lower_index"

    def __init__(self) -> None:
        """Create the index-lowering operator."""
        ...

    def codomain(
        self, *domains: TensorProductSpace,
    ) -> tuple[TensorProductSpace, ...]:
        """Covariant-variance component spaces."""
        ...

    def _apply(
        self, f: ScalarField | VectorField,
    ) -> VectorField:
        """Contract with the metric from ``grid.metric``."""
        ...
```

Component variance is a property of the component's *space* (§2.4),
so these are ordinary space-changing operators: they read the
grid-owned metric (`grid.metric(name, space)`, §3.8 — possibly
dynamic/time-dependent, never cached) and retag variance. The vector
stays thin; one metric owner (the grid).

---

## The operator dispatch registry

### OperatorRegistry

The dispatch table behind `f.diff("x")`, `f.to(...)`, `f * g`, and
every other kind ([§3.4](../02_rules.md#34-generic-operator-dispatch)).
The registry class is specified here; its *placement* — the grid owns
one as `grid.dispatch`, seeded with the defaults below — is doc 04's
seam ([§2.6](../01_concepts.md#26-grid--the-assembly-object)).

| | |
|---|---|
| Kind | final concrete |
| Pytree | static container; entries are **array-free** operators (G1) |
| Iteration | 1 |
| Concept refs | §2.6, §3.4, §5, sketch 4.2 |
| Module | `grid2.operators.registry` |

```python
#: SpaceLike = FunctionSpace | TensorProductSpace (alias owned by
#: doc 02) — keys may name a factor space or a full product space
#: registry key: kind-only or (kind, space)
DispatchKey: TypeAlias = str | tuple[str, SpaceLike]


class DispatchError(KeyError):
    """Raised when no operator is registered for (kind, space)."""


@final
@fr.utils.jaxify
class OperatorRegistry:
    """Layered (kind, space) -> Operator dispatch table."""

    def __init__(
        self,
        defaults: Mapping[DispatchKey, Operator] | None = None,
    ) -> None:
        """Create a registry from a default entry table."""
        ...

    # ------------------------------------------------------------
    #  Mapping surface
    # ------------------------------------------------------------
    def __getitem__(self, key: DispatchKey) -> Operator:
        """Exact-entry lookup (no precedence fallback)."""
        ...

    def __setitem__(self, key: DispatchKey, op: Operator) -> None:
        """Register/override an entry (setup-time only, pre-trace)."""
        ...

    def __contains__(self, key: DispatchKey) -> bool:
        """Whether an exact entry exists for the key."""
        ...

    def items(self) -> Iterator[tuple[DispatchKey, Operator]]:
        """Iterate all entries (halo-accounting trace consumer)."""
        ...

    # ------------------------------------------------------------
    #  Dispatch
    # ------------------------------------------------------------
    def resolve(self, kind: str, space: SpaceLike) -> Operator:
        """Resolve with precedence; raise DispatchError if absent."""
        ...

    def merge(
        self, overrides: Mapping[DispatchKey, Operator],
    ) -> OperatorRegistry:
        """Pure layered copy with ``overrides`` on top (assembly)."""
        ...
```

**Entry semantics** (§3.4, fixed by the notes):

- Keys are either a kind alone (`"reconstruct"` — fires for every
  space the kind can fire on) or `(kind, space)` with
  `space: SpaceLike` — a factor space *or* a full product space
  (needed for binary products and composed operators). Spaces are
  interned/hashable, so keys hash cheaply and resolution is
  identity-based.
- `resolve(kind, space)` precedence for a **factor space**, outermost
  layer first:
  1. override `(kind, space)`,
  2. override kind-only,
  3. default `(kind, space)`,
  4. default kind-only,
  5. `DispatchError`.
- **Product-key resolution** (`space` a `TensorProductSpace`; this is
  how `f * g` resolves `("multiply", common_space)`, where the common
  space is generally a product): resolution walks three key *forms*
  in order, and within each form overrides beat defaults exactly as
  above:
  1. **exact product key** — `(kind, product_space)` (the interned
     product itself);
  2. **per-factor fallback** — drop `ConstantSpace` factors, resolve
     `(kind, factor)` for every remaining factor (form-1/form-3
     layering per factor); if all factors resolve to the **same
     operator instance**, that instance is the result; if they
     resolve to *different* operators, `DispatchError` — a genuinely
     mixed product (e.g. nodal ⊗ coefficient) has no well-defined
     single product operator and requires an explicit exact product
     key;
  3. **kind-only entry**, then `DispatchError`.
  The iteration-1 tables only ever need form 2, and it imposes a
  **seeding requirement**: every `("multiply", *)` row — nodal *and*
  average — must hold **one shared `CollocationProduct` instance**,
  otherwise a mixed nodal ⊗ average operand (legal at second order)
  fails the same-instance test. The recorded **trap**: upgrading the
  `CellAvg` entry to the true quadrature product breaks form-2
  resolution for exactly those mixed products; the fix then is
  resolution by **operator equality/compatibility class** instead of
  instance identity — designed-for, not needed while the shortcut
  family is one instance. Form 1 exists so a spectral grid can pin
  `Convolution` to one specific product space.
- **Modules carry plain override dicts** (`self.dispatch["reconstruct"]
  = weno`, sketch 4.2), merged into the grid registry during model
  assembly via `merge` — pure, returning a layered registry; the
  original defaults are never mutated. Grid-side, doc 04's
  `grid.merge_overrides(...)` is the wrapper that calls
  `OperatorRegistry.merge` and swaps the grid's held instance. The
  merge *call site* is the open Phase 2 question (§3.4); this class
  only provides the mechanism.
- **`Dispatched` placeholders resolve at merge time** (D4). A default
  may be a chain or block containing `Dispatched("reconstruct")` /
  `Dispatched("diff")` holes ([`FVDerivative`](#fvderivative),
  [`Gradient`](#gradient--divergence--curl--laplacian)); `merge`
  resolves every hole against the *merged* registry (defaults +
  overrides) with ordinary precedence, and the grid-dependent block
  builders (`grad`/`div`/`curl`) expand over the grid's axis family at
  the same moment (B2). After the merge the entries are fully concrete
  and static — **no per-application registry lookup inside chains**;
  standalone user-verb `Dispatched` (`diff["x"](f)`) instead resolves
  at application against `f.grid.dispatch` (same rule, later moment).
- **Transform rows are lazy factories** (G6; doc 04's grid-lifecycle
  subsection): the `("transform", ...)` defaults are seeded as
  factories, and the grid-bound instance is constructed on first
  `resolve`, *after* provisional decomposition negotiation — breaking
  the `Grid.__init__` -> `Fourier(grid)` -> `grid.decomposition`
  construction cycle. `resolve` returns the constructed instance;
  `items()` reports factory entries to the halo trace with their
  declared requirements.
- **Per-factor transform resolution:**
  `resolve("transform", factor_space)` returns the registry's single
  all-axes instance; per-axis-subset transforms
  (`Fourier(grid, axes=("x",))`) are explicit construction, never
  registry variants.
- **The two `diff` paths do not space-commute:** nodal diff staggers
  (`Center -> Right`) while spectral diff preserves origin
  (`Fourier(o) -> Fourier(o)`), so mixing them in one tendency
  requires explicit `.to`/phase shifts; the strict algebra surfaces
  the mismatch loudly instead of silently misaligning half-cell
  offsets.
- After the merge, the grid derives per-mesh halos by running the
  halo-accounting trace against the resolved entries
  ([§5](../04_decomposition.md#5-domain-decomposition)); `items()` and
  `Operator.requirements` are the surface it consumes.
- `__setitem__` is setup-time mutation only (building the default
  table, user tweaks before the first trace); after assembly the
  registry is frozen structure: doc 04 specifies that the registry is
  treated as **frozen after decomposition negotiation** and is part
  of the grid's static dispatch identity. A fully immutable
  builder-pattern registry was considered and rejected as ceremony —
  the freeze point is enforced by the grid, not the container;
  whether the grid hands out a distinct frozen *view object* or
  relies on this convention is a doc 04 detail.

**Default entry table** (iteration-1 rows unless marked; seeded by
the base `Grid` constructor from its meshes' space families, doc 04 —
the cartesian subclass adds nothing):

| Key | Default operator | Per-factor signature |
|-----|------------------|----------------------|
| `("diff", Center)` | `FiniteDifference(order=2)` | `Center -> Right` (periodic) / `Inner` (bounded) |
| `("diff", Right/Outer/Inner)` | `FiniteDifference(order=2)` | `-> Center` |
| `("diff", CellAvg)` | `flux_diff @ Dispatched("reconstruct")` (a `SeparableComposite`, = `FVDerivative()`) | `CellAvg -> CellAvg` |
| `("diff", Fourier(o))` | `SpectralDerivative()` | `Fourier(o) -> Fourier(o)` |
| `("diff", Sine/Cosine coeff)` | `SpectralDerivative()` | `Sine <-> Cosine` |
| `("diff", Chebyshev coeff)` | `SpectralDerivative()` | recurrence, same family |
| `("interp", nodal)` | `LinearInterp()` | `Center <-> Right` (periodic); `Center -> Inner`, `Outer/Inner -> Center` (bounded) |
| `("interp", Fourier(o = staggered nodal))` | `PhaseShift(to=NodeSet.CENTER)` | `-> Fourier(origin=Center)` |
| `("interp", Fourier(o = cell_avg/face_avg))` | `SincShift(to=NodeSet.CENTER)` | `-> Fourier(origin=Center)` |
| `("reconstruct", CellAvg)` | `LinearReconstruction()` | `CellAvg -> Right` (periodic) / `Inner` (bounded) |
| `("reconstruct", Right/Outer/Inner)` | `LinearReconstruction()` | `-> CellAvg` |
| `("flux_diff", Outer/Inner)` | `FluxDifference()` | `-> CellAvg` |
| `("flux_diff", Right)` | `FluxDifference()` | `-> CellAvg` (periodic only) |
| `("flux_diff", Center/CellAvg)` | `DualFluxDifference()` | `-> FaceAvg` (dual cells) |
| `("face_diff", CellAvg)` | `FaceDifference()` | `CellAvg -> Right` (periodic) / `Inner` (bounded) |
| `("transform", Real nodal/average)` | `Fourier(grid, ...)` (lazy; rfft path) | `-> Fourier(o)` half-spectrum |
| `("transform", Complex nodal/average)` | same lazy instance (fft path) | `-> Fourier(o)` full |
| `("transform", bounded Dirichlet)` | `Sine(grid, ...)` (lazy) | `-> DST-I/II coeff` |
| `("transform", bounded Neumann)` | `Cosine(grid, ...)` (lazy) | `-> DCT coeff` |
| `("transform", Chebyshev nodal)` | `Chebyshev(grid, ...)` (lazy) | `-> Cheb/Shen coeff` |
| `("multiply", nodal)` | `CollocationProduct()` (one shared instance) | same space |
| `("multiply", CellAvg/FaceAvg)` | same `CollocationProduct()` instance (2nd-order shortcut; true quadrature product *designed-for*) | same space |
| `("multiply", Fourier(o))` | `Convolution(padded Fourier)` *(designed-for)* | same space |
| `("power", Fourier(o))` | repeated `Convolution`, integer `n >= 1` *(designed-for)* | same space |
| `("broadcast", ConstantSpace)` | `ConstantBroadcast()` (nodal/average targets; coefficient delta embedding *designed-for*, absent on Dirichlet bases) | `ConstantSpace -> target factor` |
| `("divide", nodal/CellAvg/FaceAvg)` | `Divide()` | same space |
| `("power", nodal/CellAvg/FaceAvg)` | `Power()` | same space |
| `("abs", nodal)` | `Abs()` | `-> fr.Real variant` |
| `("select", nodal/CellAvg/FaceAvg)` | `Where()` | same space |
| `("integrate", nodal/average factor)` | `Integral()` | `-> ConstantSpace` |
| `("integrate", coefficient factor)` | none — `DispatchError` ("transform back first"); zero-mode extraction `L * c0` *(designed-for)* | `-> ConstantSpace` |
| `("cumint", Center/CellAvg)` | `CumulativeIntegral()` | `-> Outer` (bounded) / `Right` (periodic, mean-zero) |
| `"laplacian"` | `Laplacian(order=2)` builder -> `div @ grad` Block *(designed-for)* | `S -> S` |
| `"grad"` / `"div"` / `"curl"` | `Gradient()` / `Divergence()` / `Curl()` builders -> `Block` over the grid's axes (B2) *(designed-for)* | composed (tuple sig) |
| `"raise_index"` / `"lower_index"` | `RaiseIndex()` / `LowerIndex()` *(designed-for, metric grids)* | variance retag |

Reserved kinds whose default operators live in other clusters:
`"discretize"` / `"assign_coeff"` (grid field factory, §3.10, doc 04),
`"ghost_fill"` (halo-fill mode, §3.6, doc 04). Rows are written with
node-set family names; concretely the grid seeds one entry per factor
space instance of its meshes **and of adopted refined-mesh space
families** — a padded transform's refined mesh adopts the parent
mesh's rows with the *same operator instances* (doc 04's normative
adoption paragraph, G10) — spaces are interned, so this stays a small
finite table.

**Sugar wiring** (owned by doc 02's field classes, listed here for
the contract). Under D3 the named field methods are **thin forwarders
to the operator verbs**, not a parallel path: `f.diff("x")` is
`fr.operators.diff["x"](f)` where `diff = Dispatched("diff")` resolves
`("diff", f.function_space.factor("x"))` against `f.grid.dispatch` and
applies the bound operator (`op["x"](f)`, bind-only — no `axis=`
keyword); `f.integrate("x")` forwards the same way. `f.to(target)`
reads the conversion kind from the per-axis source/target family
relationship and resolves `(kind, source_factor)` (its target-space
binding spelling is a D3 open detail). The **arithmetic dunders stay
on the field** (Python syntax): `f * g` resolves
`("multiply", common_space)` via the product-key rule above, and
`f / g`, `f ** p`, `abs(f)` resolve `("divide", ...)`,
`("power", ...)`, `("abs", ...)` the same way; doc 02's `where`
sugar resolves `("select", common_space)`; binary `codomain`
resolution consults `("broadcast", ConstantSpace)` when uniting a
constant factor with a full one. The registered operator always
fixes its own codomain; `.to` errors if it disagrees with the target
(§3.4 — no space-pair keys exist).

---

## Open questions

1. **Top-level `fr.dealias` alias.** Sketch 4.10 writes
   `fr.dealias.degree(2)`; §8 fixes only `fr.meshes`/`fr.operators`.
   Ship `fr.operators.dealias` only, or add the top-level alias?
2. **`layout` declaration granularity.** `SpectralDerivative` declares
   `"local"` conservatively even though its Fourier-diagonal case is
   layout-agnostic. Whether `OperatorRequirements` needs a per-space
   (not per-operator) refinement is a doc 04 negotiation detail —
   mirrored in doc 04's own open questions.
3. **Retagging-symbol ergonomics.** The `space`/`codomain` pair and
   `@` composition on `Symbol` implement the notes' diagonal algebra
   for origin-changing diagonals (FD first derivative, `PhaseShift`);
   sketch 4.6's `fd.eigenvalues(...) ** 2` shorthand is realized in the
   block/chain symbol calculus (`Block.eigenvalues` -> `OperatorSum` of
   `bwd @ fwd` per factor, D8), not a bespoke `Laplacian.eigenvalues`.
   Confirm this reading is acceptable before the `Symbol` iteration
   lands, or restrict first-derivative symbols to solver-internal use.
4. **D3 forwarder loose ends.** The `f.to(target)` forwarder's
   target-space binding spelling (`to[target]` vs a `To(space)`
   builder), and whether `Dispatched` needs a public constructor or
   only the seeded verbs (`diff`, `integrate`, ...). Small; folds into
   the model-facing sugar work.

Resolved by the operator-algebra merge (D8,
[`operator_algebra_merge.md`](operator_algebra_merge.md)): the operator
*algebra* is folded into the base hierarchy, composed operators, and
registry above — composition `@`/`Composite`/`SeparableComposite`, axis
binding `op["x"]` (bind-only, replacing the `axis=` keyword),
`Identity`/`Zero`/`OperatorSum`/`ScaledOperator`/`Block`/`Dispatched`,
tuple signatures, and the algebra-derived composed operators. The
bind-only signature pass (`_apply(self, f)`) is applied to every
operator in this document; `ConstantBroadcast`'s target-binding
spelling is the one residual (D3, question 4 above).

Resolved since the first draft: the refined-mesh handle for padded
transforms is closed — doc 01 defines
`StructuredMesh1D.refined(factor)` (iteration 1 on `IntervalMesh`,
`Fraction` factor), cited in the `Transform` notes above. The
registry freeze-point question is likewise closed into the registry
notes (frozen after decomposition negotiation, part of the grid's
static dispatch identity; view-object-vs-convention is a doc 04
detail).
