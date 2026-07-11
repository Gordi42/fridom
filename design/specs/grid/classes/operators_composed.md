---
status: normative
date: 2026-07-06
---

# Grid abstraction redesign — Class designs: composed operators and the dispatch registry

Part of the framework2 class designs; see [`README.md`](README.md) for the document map. Base hierarchy and the operator algebra are in [`operators_base.md`](operators_base.md).

---

## Composed operators (vector calculus)

Generic **dispatch kinds**, not special slots (§3.4), and under D1
**algebra-derived**: the default entries on separable grids are
`Block`s and chains over `"diff"`/`"interpolate"`/`"flux_diff"` (B1), *not*
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
`Laplacian` *type* and no `isinstance(op, Laplacian)` (D1).

### Gradient / Divergence / Curl / Laplacian

| | |
|---|---|
| Kind | factory functions -> `Block` (assembly-resolved, B2) |
| Pytree | static (the block / composite) |
| Iteration | designed-for |
| Concept refs | §2.4, §3.4, §6.3, sketch 4.6/4.7; merge D1/B1/B2 |
| Module | `framework2.grid.operators.composed` |

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
  (D1, operator_algebra §3.5). The block *expansion* — the per-axis
  entries the halo (per block-row, §3.6) and the symbol matrix (§3.7)
  walk — is computed **on demand, never materialized** as a rewritten
  operator (§3.11). This **replaces the rejected `VectorOperator` ABC
  framing**: grad/div/curl are `Block`s with tuple signatures (§3.1),
  not unary operators with a widened `codomain`.
- **Laplacian's symbol** falls out of `Block.eigenvalues` on the 1×1
  block: the `OperatorSum` of per-factor symbols — `bwd @ fwd` per
  factor, each **replication-extended** across the remaining factors
  and summed (the diagonal-operator broadcast of the `Symbol` notes,
  *not* doc 02's delta field lift; sketch 4.6). Raises
  `EigenbasisError` unless *every* factor diagonalizes (fully periodic
  grids); mixed grids fall back to banded per-column solves outside
  this operator. This `Symbol` subtlety lives in the block/chain symbol
  calculus, not a bespoke `Laplacian.eigenvalues`.
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
| Module | `framework2.grid.operators.composed` |

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
| Module | `framework2.grid.operators.registry` |

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
  `Dispatched("diff")` holes ([`FVDerivative`](operators_stencils.md#fvderivative),
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
| `("interpolate", nodal)` | `LinearInterp()` | `Center <-> Right` (periodic); `Center -> Inner`, `Outer/Inner -> Center` (bounded) |
| `("interpolate", Fourier(o = staggered nodal))` | `PhaseShift(to=NodeSet.CENTER)` | `-> Fourier(origin=Center)` |
| `("interpolate", Fourier(o = cell_avg/face_avg))` | `SincShift(to=NodeSet.CENTER)` | `-> Fourier(origin=Center)` |
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
the contract). Under D3 the **single-kind** field methods are **thin
forwarders to seeded operator verbs**, not a parallel path:
`f.diff("x")` is `fr.operators.diff["x"](f)` where `diff =
Dispatched("diff")` resolves `("diff", f.function_space.factor("x"))`
against `f.grid.dispatch` and applies the bound operator (`op["x"](f)`,
bind-only — no `axis=` keyword); `f.integrate("x")` and the
interpolation verb `fr.operators.interpolate["x"]` forward the same way.
`f.to(target)` is **not** such a verb: it is a *multi-kind resolver*
(D3a) that reads the conversion kind — `interpolate` / phase shift /
`transform` — from the per-axis source/target family relationship,
resolves `(kind, source_factor)`, and delegates to that single-kind
verb; it stays a field method (delegating, so no drift) and introduces
**no space-keyed binding** (`["x"]` is axis-only, everywhere). The
**arithmetic dunders stay on the field** (Python syntax): `f * g`
resolves
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
Resolved (D8): the operator *algebra* is folded into the base
hierarchy, composed operators, and registry above, and the D3 forwarder
residual is closed (D3a/D3b) — `f.diff`/`f.integrate` and the
`interpolate` verb are forwarders to seeded `Dispatched` verbs, `f.to`
stays a multi-kind resolver, subscript `["x"]` is axis-only (non-axis
parameters like targets and orders are constructor arguments), and the
`Dispatched(kind)` constructor stays public as the extension escape-hatch.

Resolved since the first draft: the refined-mesh handle for padded
transforms is closed — doc 01 defines
`StructuredMesh1D.refined(factor)` (iteration 1 on `IntervalMesh`,
`Fraction` factor), cited in the `Transform` notes above. The
registry freeze-point question is likewise closed into the registry
notes (frozen after decomposition negotiation, part of the grid's
static dispatch identity; view-object-vs-convention is a doc 04
detail).
