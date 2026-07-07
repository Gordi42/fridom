# Grid abstraction redesign — Class designs: operators

Part of the grid redesign notes; see [`../00_overview.md`](../00_overview.md)
for the document map. Status: draft class design, no implementation.
Signatures are the intended public API for `framework2.grid`; the
numbered concept sections remain the normative reference.

> **Operator algebra folded in (D8).** The operator *algebra* —
> composition `@`, sums, `c * A` scaling, `Identity`/`Zero`/`Block`,
> axis binding `op["x"]`, tuple signatures — is designed in the sibling
> note set
> [`../operator_algebra/`](../operator_algebra/00_overview.md); it is
> applied to the **base hierarchy**
> ([Operator algebra](#operator-algebra), bind-only `__call__`,
> `bound_axis`), the **composed operators** (algebra-derived factories),
> and the **registry** below. Bind-only axis naming (`op["x"]` replaces
> the `axis=` keyword) makes every operator's `_apply` signature
> uniformly `_apply(self, f)`, with the separable
> `_apply_factor(self, f, axis)` unchanged.

This document owns the **Operator cluster**: the operator base
hierarchy, the concrete stencil/nodal operators, transforms, `Symbol`,
the binary product operators, composed operators, and the dispatch
registry class. Sibling documents own the seams referenced here:
`Mesh` / `FunctionSpace` / `ConstantSpace` / `fr.Real` / `fr.Complex`
([`meshes.md`](meshes.md), [`spaces.md`](spaces.md)),
`TensorProductSpace` / `ScalarField` / `VectorField` / `State` /
`SpaceMismatchError`
([`product_spaces.md`](product_spaces.md), [`fields.md`](fields.md)),
and `Grid` / decomposition / `grid.evaluation_nodes` /
`grid.wavenumbers` / measure and metric fields / `HaloTracer`
([`grid.md`](grid.md), [`decomposition.md`](decomposition.md)).

---

## Module placement

The code lives in `fridom.framework2.grid` (part of the new parallel
`fridom.framework2` package), renamed to `fridom.framework.grid` at
cutover ([§8](../00_overview.md#8-migration-strategy)). Free-standing operators
are re-exported as the collection namespace `fr.operators` via the
usual lazypimp `__init__.py`.

```
src/fridom/framework2/grid/operators/
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
operator_algebra §2.1, ordinary operators themselves. Two decisions
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
| Module | `framework2.grid.operators.base` |

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
  expression graphs, no algebraic rewriting (operator_algebra §3.11);
  the only normalizations are chain-flattening, `Identity` elision, and
  `Zero`-dropping.
- `requirements` defaults to `OperatorRequirements()` (halo 0, any
  layout); the grid's halo-accounting trace
  ([§5](../04_decomposition.md#5-domain-decomposition)) reads it per
  applied factor and accumulates depth along un-synced chains.
- **Layout threading**
  ([§5.1](../04_decomposition.md#51-layout-is-part-of-the-function-space)):
  `codomain` resolvers see and return **bare** spaces — operator
  authors never touch layouts. The shared application path re-attaches
  the domain's layout to the resolved codomain by default; the only
  layout transitions are lowering-inserted `Reshard` nodes (below),
  so a requirement-driven layout change is visible in the applied
  operator's codomain.
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
  This resolves operator_algebra §5.1 toward interning.

### OperatorRequirements

Frozen value type describing what an operator demands from the
decomposition along one factor.

| | |
|---|---|
| Kind | final frozen dataclass |
| Pytree | static value |
| Iteration | 1 |
| Concept refs | §2.5, §5 |
| Module | `framework2.grid.operators.base` |

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

### Reshard / Sync and requirements-driven lowering

The two data-movement operators of
[§5.1](../04_decomposition.md#51-layout-is-part-of-the-function-space),
and the plan-time pass that places them.

| | |
|---|---|
| Kind | concrete, final (both) |
| Pytree | static structure (bound grid, endpoint layouts) |
| Iteration | 1 |
| Concept refs | §5, §5.1 |
| Module | `framework2.grid.operators.movement` |

```python
@final
class Reshard(UnaryOperator):
    """Explicit layout change: identity on the bare space."""

    def __init__(self, grid: Grid, target: Layout) -> None:
        """Bind grid and target layout (grid-bound, like transforms)."""
        ...

    def codomain(self, domain: SpaceLike) -> SpaceLike:
        """``domain.bare.with_layout(target)``; bare domains raise."""
        ...


@final
class Sync(UnaryOperator):
    """Halo-exchange node; inserted by the base/lowering only."""

    def codomain(self, domain: SpaceLike) -> SpaceLike:
        """Identity — halo validity is storage, not space identity."""
        ...
```

Notes:

- **`Reshard` is user-reachable** (`f.reshard(...)` sugar, doc 02):
  the explicit conversion the layout-strict algebra demands, same
  status as `PhaseShift` between origins. Kernel =
  `Decomposition.redistribute`; `target` must be in the negotiated
  layout vocabulary (§5.1); object-level halo rule: resets
  accumulated trace depth on the moved axes. Identity when
  `domain.layout == target` (elided like `Identity`).
- **`Sync` is internal-only**: users and kernel authors never spell
  it — a user-facing sync would leak the storage layer into the
  semantic layer. It realizes the consumption-side contract (task
  1.8, cluster 04): the base inserts it **before** a kernel whose
  operand's halo validity is below the application's requirement
  (and memoizes the result onto the operand). It performs the
  BC-structured/`("ghost_fill", space)` edge fill through the
  grid, and is a structural no-op where the negotiated width is 0.
- **Requirements-driven lowering** (§5.1): at bind/registration time
  a single pass walks a composite tracking the current layout and
  inserts both node kinds from `OperatorRequirements` — `.layout`
  demands locality the current layout lacks → `Reshard` (target
  chosen by shortest path with lookahead over the negotiated layout
  graph, edges weighted by moved data volume); `.halo` → `Sync`
  placement per the object-level halo rules. Inserted nodes carry
  explicit endpoints; apply time infers nothing. This produces a
  *lowered form beside* the user-built composite — it is not
  algebraic rewriting, which stays rejected (operator_algebra §3.11).
- **Strictness boundary** (§5.1): field arithmetic never reshards;
  an operator application may contain requirement-driven reshards,
  and the resulting layout is visible in its codomain.

### UnaryOperator

| | |
|---|---|
| Kind | ABC |
| Pytree | static |
| Iteration | 1 |
| Concept refs | §2.5 |
| Module | `framework2.grid.operators.base` |

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
`**kwargs`: an axis is named only by binding, `op["x"](f)` (§2.3), so
"can this operator name an axis?" is a matter of **type** —
`SeparableOperator` overrides `__getitem__`, whereas whole-space
operators (composed `grad`/`Laplacian`, transforms) inherit the raising
default and are simply applied `op(f)`, the axis fixed by their
signature. The "which axis?" question becomes a binding question,
resolved in `SeparableOperator._apply`. Every concrete `_apply` is
`_apply(self, f)`; the separable `_apply_factor(self, f, axis)` is
unchanged, since there `axis` is the resolved axis, always a string.

### BinaryOperator

| | |
|---|---|
| Kind | ABC |
| Pytree | static |
| Iteration | 1 (base; concrete products below carry their own status) |
| Concept refs | §2.5, §3.11 |
| Module | `framework2.grid.operators.base` |

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
operator_algebra §3.8), asymmetric because a binary has *two inputs but
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
| Module | `framework2.grid.operators.base` |

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
  region per application**. The consumption-side sync contract (task
  1.8, cluster 04): the base makes operand halos valid *before* the
  kernel exactly when the application's requirement exceeds the
  operand's claimed validity; kernel results carry their
  construction seam's validity claim, so periodic chains elide
  exchanges; halo-0 applications never sync. Storage is
  halo-extended: `_data` is storage-shaped, `.data` the true-shape
  view (docs 02/04).
- Spacing enters through the grid-owned measure fields
  (`grid.measure(space, name=...)`, doc 04; reachable as the field
  carries its grid, §2.7), read inside `_apply_factor` at trace time —
  **no hardcoded `dx` constants**. A uniform mesh constant-folds.

---

## Operator algebra

The **derived operators** (operator_algebra §2.1): the objects `@`, `+`,
`*`, `Block(...)`, and `Dispatched(...)` build. They are ordinary
operators — callable, registrable, halo-accountable, symbol-bearing
where linear — and shallow eager structures with no expression graphs
or algebraic rewriting (§3.11). All live in
`framework2.grid.operators.base`. Iteration split (D7): `Identity`,
`Composite`, `SeparableComposite`, `Dispatched` are **iteration 1** (the
FV derivative needs them); `Zero`, `OperatorSum`, `ScaledOperator`,
`Block` are **designed-for**.

**Pytree amendment.** `ScaledOperator` with a *field* coefficient is the
cluster's **second dynamic-leaf carrier** (the coefficient field),
alongside `Symbol`: its static part is the algebraic structure, the
coefficient is the leaf (operator_algebra §2.2). Every other algebra
object is fully static.

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
([`FVDerivative`](operators_stencils.md#fvderivative), D1).

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
> default, [registry](operators_composed.md#operatorregistry)). **Seeded verbs vs. the
constructor (D3b):** the standard verbs are module-level singletons on
`fr.operators` — `diff`, `integrate`, `interpolate` (`= Dispatched(
"diff")` etc.) — the discoverable, ergonomic surface that the field
forwarders route through (`f.diff(axis) == fr.operators.diff[axis]
(self)`). The `Dispatched(kind)` constructor stays **public as the
extension escape-hatch**: a module registering a custom kind can expose
`Dispatched("mykind")["x"]` as its own verb; an unknown kind is a clean
`DispatchError` at application, so the open constructor is safe. Ordinary
use goes through the seeded verbs. `f.to` is deliberately *not* a
`Dispatched` verb — it is the multi-kind resolver of D3a.

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
| Module | `framework2.grid.operators.symbol` |

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

