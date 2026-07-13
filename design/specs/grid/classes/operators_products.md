---
status: normative
date: 2026-07-13
---

# Grid abstraction redesign — Class designs: pointwise, product and reduction operators

Part of the framework2 class designs; see [`README.md`](README.md) for the document map. Base hierarchy and the operator algebra are in [`operators_base.md`](operators_base.md).

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
| Module | `framework2.grid.operators.products` |

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
| Module | `framework2.grid.operators.products` |

```python
@final
class ConstantBroadcast(SeparableOperator):
    """Embed a ConstantSpace factor into a full factor space."""

    dispatch_kind: ClassVar[str | None] = "broadcast"

    def __init__(self, to: FunctionSpace | None = None) -> None:
        """Create the embedding; ``to`` is the target factor (static,
        interned space), supplied by the unification machinery."""
        ...

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """broadcast: ConstantSpace(m) -> ``self.to`` (target factor)."""
        ...

    def _apply(
        self, f: ScalarField | VectorField,
    ) -> ScalarField | VectorField:
        """Replicate the constant DOF into the target factor."""
        ...
```

Unlike other separable kernels, the codomain is not a function of the
domain alone: the target factor is supplied by the *other* operand
when binary `codomain` unites a constant factor with a full one. Under
D3a the target is a **constructor parameter** (`ConstantBroadcast(to=
target)`, the target a static interned space) built by that unification
machinery — **not** a `["x"]` binding, which stays axis-only; `_apply(
self, f)` reads the target off `self`. A recorded widening of the
per-factor signature, analogous to the composed-operator widening. On nodal/average targets the embedding is
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
| Module | `framework2.grid.operators.products` |

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
| Module | `framework2.grid.operators.products` |

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
| Module | `framework2.grid.operators.products` |

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
  (§3.11 of `../02_rules.md` is amended accordingly in the same change
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
| Module | `framework2.grid.operators.products` |

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
| Module | `framework2.grid.operators.combinators` |

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
| Module | `framework2.grid.operators.integrate` |

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
The mapping-metric accessor `grid.metric` (landed with ROADMAP 3.4) is
*not* used here — measures and mapping metrics are distinct grid
accessors.
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
| Module | `framework2.grid.operators.integrate` |

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

