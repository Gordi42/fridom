---
status: normative
date: 2026-07-06
---

# Operator algebra merge — decisions

Status: **implemented decisions** (Phase 1 landed; kept as the normative reference)
Author: Silvano Rosenau (with AI-assisted brainstorming)
Date: 2026-07-06

This note records the decisions for merging the **operator algebra**
([`../../operator_algebra/`](../../operator_algebra/00_overview.md),
authored on `dev`) into the **operator class design**
([`operators_base.md`](operators_base.md), authored on
`grid-redesign-classes`). The two note sets overlap in subject but sit
at different layers: `operator_algebra` is a rules-layer note set (the
algebra: composition `@`, sums `+`, field-coefficient scaling `c * A`,
`Identity`/`Zero`/`Block`, axis binding `op["x"]`, tuple signatures),
while `operators_base.md` is the class design.

These decisions are the reconciliation, now applied to
`operators_base.md` (D8). All code snippets are illustrative,
not normative; decision numbers `D1`…`D8` are stable identifiers.

Where a decision resolves an `operator_algebra` open thread or overturns
an `operators_base.md` rejected-alternative, it says so explicitly.

---

## Summary

| # | Decision | Resolves |
|---|----------|----------|
| D1 | Composed operators are **algebra-derived** (thin factories over `@`/`Block`/`Dispatched`), not bespoke `_apply` classes. | conflict C1 |
| D2 | **Bind-only axis naming**: drop the `axis=` call keyword; `op["x"]` is the sole way to name an axis. | conflict — `axis=` vs binding |
| D3 | Field methods (`f.diff`, `f.integrate`, `f.to`) become **thin forwarders** to operators; transforms keep `.forward`/`.backward`; arithmetic dunders stay on the field. | drift seam |
| D4 | `Dispatched(kind)` is **one object** serving both the internal chain placeholder and the user-facing verb; resolution happens *when the grid becomes known*. | conflict C2 |
| D5 | `@` produces a **`SeparableComposite`** (is-a `SeparableOperator`) when operands are separable and axis-compatible, else a whole-space `Composite`. `bound_axis` is a static field; `__getitem__` is the sole binding point. | sub-question 1 |
| D6 | Bound operators and composites are **interned** (forced by the identity-hash invariant). | operator_algebra §5.1 |
| D7 | **Iteration split**: a small same-axis-composition slice is iteration 1; the rich algebra (blocks, tuple signatures, sums, symbol matrices) is designed-for. | timing |
| D8 | Rewrite scope: only the base hierarchy, composed operators, and registry sections of `operators_base.md` change. | — |
| B1 | `Block` is a **distinct algebra node** (a grid of operators), sibling to `Composite`/`OperatorSum`; whole-space, not bindable; entries are the bound scalar operators of D5; block-matmul reduces to D5 chains. | — |
| B2 | `grad`/`div`/`curl`/`laplacian` have **grid-dependent block shape**, so they are `Dispatched`-family: assembly-resolved builders that emit a `Block` over the grid's axes. | operator_algebra §3.5 |
| B3 | **`map` is a verb, `Block` is a noun** — keep both, with a rule for which to write. | operator_algebra §5.6 |
| B4 | Nonlinear tuple-signature operators need **no new class**; the base treats block-expansion/symbols as **optional** capabilities. Tensor blocks: flatten now, defer index-aware addressing. | operator_algebra §3.5, §5.5 |
| T2 | Binary composition: the **positional tuple** `P @ (B1, …, Bn)` is the one spelling; `(A1, …) @ P` is an error; no named-operand form. | operator_algebra §5.2 |
| T3 | Mid-chain syncs are **numerically transparent**, so auto-insertion is *permitted in principle* — an explicit, reportable decomposition-layer optimization, not silent. | operator_algebra §5.3 |
| T4 | A **`Symbol` is never a chain factor**; static diagonal *operators* are, deriving their symbol at trace time. Refines operator_algebra §3.7. | operator_algebra §5.4 |
| D3a | `f.to` stays a **multi-kind resolver** (field method); `["x"]` is **axis-only**; non-axis parameters (targets, orders) are constructor args. `interpolate` is the composable single-kind verb. | D3 residual |
| D3b | Standard verbs (`diff`/`integrate`/`interpolate`) are **seeded** on `fr.operators`; `Dispatched(kind)` stays a **public extension escape-hatch**. | D3 residual |

---

## D1. Composed operators are algebra-derived

`Laplacian`, `Gradient`, `Divergence`, `Curl`, and `FVDerivative` stop
being bespoke classes with hand-rolled `_apply` methods that
"compose per factor and sum" internally. They become **thin factories
that build an algebra expression**:

```python
diff_fv   = flux_diff @ Dispatched("reconstruct")   # the FV derivative
grad      = Block([[fd["x"]], [fd["y"]]])            # scalar -> vector column
laplacian = div @ grad                               # (1 x n) @ (n x 1) = 1 x 1
```

Halo (§3.6) and symbols (§3.7) come from the chain/block, not from
per-operator code — `Laplacian.eigenvalues`'s hand-rolled `bwd @ fwd`
sum is deleted and reappears as the general block/chain symbol
calculus (see D8). This honors `operator_algebra` §3.5's "these are
**constructions of the dispatch defaults, not privileged objects**",
and matches `operators_base.md`'s own framing ("generic dispatch kinds,
not special slots").

The named spellings survive only as **module-level factories** for
ergonomics and as dispatch labels; `fr.operators.laplacian(order=2)`
builds the composite above. Nothing may rely on `isinstance(op,
Laplacian)` — the result is a `Composite`, not a distinct type. This
overturns `operators_base.md`'s "Gradient/Divergence/Curl stretch the
unary `codomain` to tuples" framing and its rejected `VectorOperator`
ABC: they are `Block`s with tuple signatures (D7), which is a different
reason the ABC is not needed.

## D2. Bind-only axis naming

An axis is named **only** by binding: `op["x"](f)`. The call-site
`axis=` keyword is dropped everywhere — a stricter stance than
`operator_algebra` §2.3, which keeps `fd(f, axis="x")` as sugar.

- `UnaryOperator.__call__` loses its `axis: str | None` parameter and
  all "passed axis where forbidden / omitted where required"
  validation.
- "Can I name an axis on this operator?" becomes a matter of **type**,
  not a runtime check: bindable operators override `__getitem__`;
  everything else inherits a `__getitem__` that raises. Bindability
  *is* separability.

The payoff is that composition becomes a first-class, halo-accountable
object where it was previously invisible:

```python
f.diff("x").diff("x")        # today: two dispatched applies, a sync between,
                             #        no operator object to account
(diff["x"] @ diff["x"])(f)   # one wide-stencil object, halo = 2, one sync
```

Unbound application is still allowed when the operand has **exactly one
bindable factor** (the kernel auto-resolves that axis); otherwise it
raises, asking for an explicit bind. This is `operators_base.md`'s
existing "axis optional on effectively-1-D fields" convenience,
re-expressed without the keyword.

## D3. Field methods become thin forwarders

The field-sugar methods are kept for readability but redefined as
one-line forwarders, so there is **no second implementation** that can
drift from the registry:

```python
class ScalarField:
    def diff(self, axis):      return fr.operators.diff[axis](self)
    def integrate(self, axis): return fr.operators.integrate[axis](self)
    def to(self, target):      return fr.operators.to[target](self)   # spelling TBD
```

- **Transforms are the exception** and keep their `.forward`/
  `.backward` surface with grid-bound construction
  (`Fourier(grid, axes=...)`). They are *not* forced into the `[...]`
  spelling: a transform binds the grid at construction and has two
  directions, both of which the `op["x"]` form fights.
- **Arithmetic dunders stay on the field** (`f + g`, `f * g`, `f / g`,
  `f ** p`, `abs(f)`) — they are Python syntax the field must overload;
  they remain dispatch sugar over `("multiply", …)` etc. as
  `operators_base.md` specifies. A fully "operators-only" field is
  therefore impossible anyway, which is itself the argument for keeping
  the forwarder layer uniform rather than deleting `diff`/`integrate`.

This deletes the "doc 02 field sugar must be kept in sync with the
registry" seam: the sugar is now an alias, not a parallel path.

## D4. `Dispatched` is one object, resolved when the grid is known

`Dispatched(kind)` (`operator_algebra` §3.10) serves two roles with one
object:

- as a **chain factor** inside a registered default
  (`flux_diff @ Dispatched("reconstruct")`);
- as the **user-facing verb** (`diff["x"](f)`, where `diff` is
  `Dispatched("diff")`).

The apparent tension — §3.10 says `Dispatched` resolves "at assembly,
no per-application lookup" — is not real once the actual rule is
stated: **`Dispatched` resolves when the grid becomes known.** For a
registered chain that is model-assembly time (the composite is baked
concrete, no per-application lookup inside it). For an ad-hoc
`diff["x"](f)` the grid arrives with `f`, so resolution is at
application — exactly what `f.diff("x")` does today. Same principle,
different moment; this must be written down so "assembly-only" is not
misread as forbidding apply-time resolution.

This resolves conflict C2: `FVDerivative`'s apply-time
`reconstruct=None` late binding is replaced by the `Dispatched`
placeholder, resolved once at assembly for the registered default.

## D5. Separable-composite typing

`Operator.__matmul__` branches on its operands and produces one of two
composite types:

```python
def __matmul__(self, other):
    chain = _flatten(self, other)          # associativity; elide Identity (§3.2/3.3)
    if _all_separable_one_axis(chain):     # every factor separable AND axes compatible
        return _intern(SeparableComposite(chain))
    return _intern(Composite(chain))       # whole-space; base __getitem__ raises
```

`_all_separable_one_axis` holds iff every factor is a
`SeparableOperator`/`SeparableComposite` **and** their bound axes are
compatible (all unbound, or all bound to the same axis). Consequences:

| Composition | Result | Bindable? |
|-------------|--------|-----------|
| `flux_diff @ reconstruct` (both unbound) | `SeparableComposite` | yes (later) |
| `fd["x"] @ fd["x"]` (same axis) | `SeparableComposite` bound to `x` | already bound |
| `fd["x"] @ fd["y"]` (different axes) | `Composite` (whole-space) | no — a mixed derivative |
| `fourier.forward @ fd["x"]` | `Composite` | no |

The **bound axis is a static field** on `SeparableOperator`
(`bound_axis: str | None`), not a `BoundOperator` wrapper: in the
bind-only world binding is the common case, so a per-application
wrapper is the wrong shape, and §2.3 already calls binding "part of the
static structure." `op["x"]` returns an interned static variant (D6).
`__getitem__` is defined on `SeparableOperator`; the base
`Operator.__getitem__` raises, so whole-space operators and general
`Composite`s reject binding by type.

`SeparableComposite` **is-a** `SeparableOperator`, so it inherits
`bound_axis`, `__getitem__`, the axis-resolution `_apply`, and the
per-shard template; it only overrides `_apply_factor` (run each factor
on the axis), `requirements` (sum factor halos), and `eigenvalues`
(product of factor symbols). Binding distributes for free:
`(flux_diff @ reconstruct)["x"]` sets the composite's `bound_axis`,
which is exactly `flux_diff["x"] @ reconstruct["x"]`.

## D6. Interning is forced by the identity-hash invariant

`operators_base.md` and the classes README fix that operators are
identity-hashed (`__eq__`/`__hash__` return `self is other`). For jit
caching to hit, two separately-constructed `fd["x"]` objects — or two
`flux_diff @ reconstruct` chains — must therefore be **the same
object**. So `_rebind` (binding) and `__matmul__`/`__add__` (algebra)
construct through an **intern cache** keyed on static structure (class,
order, bound axes, factor identities, coefficient *placement* — not
values).

This **resolves `operator_algebra` §5.1** ("intern composites, or is
structural equality enough? Leaning: structural equality first") in
favor of interning: the class-design invariant removes the choice —
identity-hashing demands a canonical interned instance per structure.

The other `operator_algebra` open threads remain open: §5.2 (asymmetric
binary composition ergonomics), §5.3 (mid-chain sync insertion), §5.4
(`Symbol`-in-chain normalization), §5.5 (tensor flattening vs indexed
blocks), §5.6 (`VectorField.map` vs diagonal `Block`).

## D7. Iteration split

Making `FVDerivative` algebra-derived (D1) promotes composition into
the **iteration-1** surface, but only a small slice — because
`FVDerivative` is a *same-axis chain of two separable kernels*, which
is still a 1-D separable kernel (§2.3).

**Iteration 1** (what `FVDerivative` and the `("diff", CellAvg)`
default need day one):

- `Operator.__matmul__` and the `Composite`/`SeparableComposite` pair;
- `Dispatched` (D4) and its assembly-time resolution in `merge`;
- `Identity` (chain normalization) and chain/sum halo accounting;
- `bound_axis` + `__getitem__` for the same-axis case.

**Designed-for** (lands with its consumers, all already designed-for in
`operators_base.md`):

- `Block` and tuple/direct-sum signatures — with `grad`/`div`/`curl`/
  `laplacian`;
- mixed-axis binding `fd["x"] @ fd["y"]` and `Zero` blocks;
- `+` / `c * A` field-coefficient sums — with terrain-following
  derivatives (§3.8);
- symbols of composites (chain products, block symbol matrices) — with
  the spectral solvers (sketch 4.6).

This keeps iteration 1 bounded even under the algebra-derived choice:
one small composition primitive is added; the rich algebra defers to
consumers that were already deferred.

## D8. Rewrite scope

The bulk of `operators_base.md` is untouched — the concrete separable
kernels (`FiniteDifference`, interpolation, reconstruction, the flux-diff
family), transforms, `Symbol`, the pointwise/product operators, and
reductions all sit *below* the algebra. Three sections change:

1. **Base hierarchy** — add the algebra dunders (`@`, `+`, `-`, `*`,
   `**`) and `__getitem__` to `Operator`; drop `axis` from
   `UnaryOperator.__call__`; add `bound_axis` + `_apply` axis
   resolution to `SeparableOperator`; add `Composite`,
   `SeparableComposite`, `Identity`, `Zero`, `Dispatched` (iteration 1)
   and `Block`, `OperatorSum`, `ScaledOperator` (designed-for);
   generalize `codomain` to tuple signatures.
2. **Composed operators** — rewrite `Laplacian`/`Gradient`/
   `Divergence`/`Curl`/`FVDerivative` from bespoke classes into the
   factories of D1; rewrite the rejected-`VectorOperator` note.
3. **Registry** — the `("diff", CellAvg)` and composed-operator rows
   hold chains/blocks; `Dispatched` resolution folds into `merge` at
   assembly, replacing `FVDerivative`'s apply-time late binding.

Cross-linking: add the reciprocal link from `operators_base.md` (and
grid-redesign §2.5) back to `../../operator_algebra/`, so the class doc
names its normative parent.

Symbol note preserved through the move: the block/chain symbol calculus
must keep the `operators_base.md` `Symbol` rule that broadcast across
product factors is **replication-extension** (`Identity ⊗ D`), *not* a
delta field-lift — otherwise the Laplacian symbol sum `kx² + ky²` is
wrong off the `ky = 0` row.

---

## Block and tuple signatures

D5 fixed the *separable* (same-axis) side of the algebra. These fix the
whole-space, multi-axis side — the block operators. The framing is that
**block-matmul reduces to the D5 machinery**: on a 2-D C-grid,

```python
grad      = Block([[diff["x"]], [diff["y"]]])       # 2x1 column, entries are D5 bound ops
div       = Block([[dflux["x"], dflux["y"]]])        # 1x2 row
laplacian = div @ grad                               # (1x2) @ (2x1) = 1x1
#  single block = dflux["x"] @ diff["x"] + dflux["y"] @ diff["y"]
#                 └─ SeparableComposite (D5) ─┘   └─ SeparableComposite (D5) ─┘
#                          └──────────── OperatorSum ────────────┘
```

so sketch 4.6's Laplacian "broadcast sum" is *derived*, and its symbol
`-(kx² + ky²)` is the `OperatorSum` of the per-axis chain symbols.

### B1. `Block` is a distinct algebra node

`Block` is a **grid of operators**, a sibling of `Composite` (chain)
and `OperatorSum` (term list) under the three structural node kinds of
`operator_algebra` §3.11. It is whole-space and tuple-signatured
(§3.1): applied to a `VectorField` as `D(f)_i = sum_j D_ij(f_j)`, with
`Zero` for structural zeros and `Identity` on the diagonal where
needed. It is **not bindable** (inherits the raising `__getitem__` of
D5). Its **entries are the bound scalar operators of D5**, and its
block expansion — the per-entry chains that halo (§3.6, per block-row
`max_j(halo(A_ij) + halo(B_jk))`) and the symbol matrix (§3.7) walk —
is **computed on demand, never materialized** as a rewritten operator
(§3.11). `grad` is a column, `div` a row, `curl` a matrix with `Zero`
blocks in 3-D. Iteration: designed-for (lands with grad/div/curl/
laplacian, per D7).

### B2. grad/div/curl/laplacian are `Dispatched`-family (grid-dependent shape)

The block *shape* of these operators depends on the grid — 2-D grad is
a 2-row column, 3-D a 3-row column — so they cannot be fixed `Block`
literals. They are **assembly-resolved builders**: the registered
default for the `"grad"`/`"div"`/`"curl"`/`"laplacian"` kind expands, at
model assembly against the grid's axis family, into a `Block` whose
entries are `Dispatched("diff")["x"]` / `("interpolate")` / `("flux_diff")`
per axis. This keeps two D-decisions intact: the entries are late-bound
(D1 — a module's `reconstruct`/`diff` override propagates into the
block), and there is **one placeholder concept** (D4 — `Dispatched`
resolved when the grid becomes known), now covering both the "resolve a
kind" hole and the "expand over the grid's axes" builder. Before
assembly they are symbolic kind-placeholders; after, concrete blocks.
On a sphere / unstructured mesh the *same kinds* hold primitive
metric-aware mesh-level registrations instead (grid-redesign §6.3/§6.4),
which are not blocks — nothing here assumes the registered entry is
separable or block-structured.

### B3. `map` is a verb, `Block` is a noun

`VectorField.map(fn)` (a field method, iteration 1) and
`Block(diag(op, …))` (an operator) are **not competing spellings** —
one is eager application, the other is an operator object:

| | `vec.map(op)` | `Block(diag(op, …))` |
|---|---|---|
| Result | a **field** (eager) | an **operator** (a noun) |
| Per-component resolution | **re-dispatches** per component's space | **fixed** at construction |
| Error timing | at application, per component | at assembly, by signature |
| Composes into `@` / registry | no | yes |
| Halo / symbol as one object | no | yes |

They coincide only when all components share a space; the moment
components differ (the point of `VectorField`) `map` re-dispatches and
`Block` does not. **Rule for module code:** reach for `map` for eager,
polymorphic componentwise work ("do this to each component now" — the
common case); reach for `Block` *only* when the componentwise action
must itself be an operator (a factor in a larger `@`/block expression,
a registered default, or a unit whose halo/symbol is queried). Pure
diagonal blocks are rare — most blocks are non-diagonal (grad column,
div row, curl matrix) — and appear mainly to lift a scalar operator
into a vector-operator expression. Smell test: building
`Block(diag(op, op))` just to apply it once to a vector means you
wanted `map`. This resolves `operator_algebra` §5.6.

### B4. Nonlinear tuple operators need no new class

Genuinely multi-input operators (kinetic energy `(u, v) -> ke`, a WENO
flux assembly) are *neither* `map` (they combine components, not act
per-component) *nor* `Block` (nonlinear, no block structure). They need
no dedicated type: they are either plain module code
(`ke = 0.5 * (u.to(c)**2 + v.to(c)**2)`) or, when they must be
operators, a general `Operator` subclass with a **tuple `codomain` and
a direct `_apply` that declines block/symbol**. The only base-class
requirement is the one D7 already introduces: **treat block-expansion
and symbols as optional capabilities, not mandatory** — a
tuple-signature operator that answers neither is legal (the base
`eigenvalues` already raises `EigenbasisError`, and there is no
mandatory block method). This must be stated so nobody assumes every
tuple-signature operator is a `Block`. This closes `operator_algebra`
§3.5's "opaque" case.

**Tensor blocks (§5.5): flatten now, defer.** The `TensorField` spec
already uses `(i, j)` keys with `map`; `operator_algebra` §3.1 flattens
multi-indices into the positional tuple and keeps the index structure
as metadata. Index-aware `Block` addressing (a `Block` keyed by
multi-index) earns its keep only when the first tensor consumer
(strain -> stress) is ported — matching `operator_algebra`'s own timing.
So: positional/flattened signature now, index-aware blocks deferred.

---

## Open-thread resolutions (operator_algebra §5.2–§5.4)

Two of these are *forced* by decisions the class design already made;
the third is reframed by one observation. None needs a new class.

### T2. Binary composition uses the positional tuple form (§5.2)

The pre/post rules of `operator_algebra` §3.8 are asymmetric because a
binary operator has **two inputs but one output**:

- **Pre-composition takes a tuple** (one operator per input):
  `P @ (B1, B2)` is `(f, g) -> P(B1(f), B2(g))`; the sugar `P @ B` is
  `P @ (B, B)`. It generalizes to the n-ary elementwise operators
  (`Hadamard`, `Where`, whose `__call__` is `(f, g, *more)`):
  `P @ (B1, B2, B3)`.
- **Post-composition cannot take a tuple**: `A @ P` is
  `(f, g) -> A(P(f, g))`, and there is only one output to post-compose.
  So **`(A1, A2) @ P` is an error** — confirmed, and it already is one
  mechanically (a tuple left-operand dispatches to `P.__rmatmul__`,
  which rejects it).

**No named-operand form** (`P.compose(left=, right=)`) is added: it
only makes sense at arity 2, whereas the positional tuple covers the
n-ary operators too, and it is what makes
`Convolution = trim @ CollocationProduct @ pad_inverse` expressible
(pad pre-composes both operands via the `P @ B` sugar). Revisit only if
porting real flux modules shows the positional form is error-prone — it
is pure sugar, cheap to add later.

Feeds D8: the base `Operator.__matmul__` **dispatches on operand
arity** — unary@unary -> unary composite; unary@binary -> binary
composite (post); binary@tuple or binary@unary -> binary composite
(pre).

### T3. Mid-chain sync insertion is permitted in principle (§5.3)

The observation that settles this: **a mid-chain halo sync is
numerically transparent — bit-identical results.** Whether the chain
accumulates halo to depth 3 and syncs once, or syncs at depth 1 /
applies / syncs / applies, the ghost cells hold the same neighbor
values at every read; a sync is a *copy*, not arithmetic, so there is
no floating-point reassociation and the interior results are identical.

That puts mid-chain syncs in a **different category** from what
grid-redesign §3.11 forbids. "What you wrote is what runs" blocks
*numerics-changing* rewrites (operation reorder, aliasing, distributing
`@` over `+`). A sync changes neither the result nor the operator
structure — it is a communication-scheduling detail *below* the
operator algebra. So, unlike algebraic rewriting (categorically
forbidden), auto-insertion is **permitted in principle**.

Decision: **the grid may insert mid-chain syncs, but as an explicit,
reportable optimization owned by the decomposition layer, not silent
magic.** The crossover (ghost-layer width vs. communication cost) is a
decomposition cost-model knob; the tendency author keeps the ability to
place explicit syncs. Two scoping notes: it is **downstream of
sync-elision** (itself designed-for — in iteration 1 every application
already syncs, so no chain accumulates and the question is moot day
one), and its true owner is **doc 04 / grid-redesign §5**; the merge
note only records the principle.

### T4. A `Symbol` is never a chain factor (§5.4)

Forced by a class-design decision already taken: **a `Symbol` is not an
`Operator`** (`operators_base.md` rejected the subclass — operators are
static structure, a `Symbol` carries a dynamic `_data` leaf; it has its
own diagonal algebra `Symbol * Symbol`, `Symbol @ Symbol`, and
`Symbol(f)` = Hadamard). A `Symbol` therefore **cannot be a factor in
an operator `@` chain** — the chain is static structure and a Symbol has
a dynamic leaf.

What goes in the chain is a **static diagonal operator**
(`SpectralDerivative`, `PhaseShift`, `SincShift`, a spectral filter)
that *derives* its Symbol from `grid.wavenumbers` at trace time and
applies it as Hadamard inside `_apply`. The raw `Symbol` stays
solver-facing (`op.eigenvalues(...)`, `1 / lap`, the spectral solve).

So the §5.4 question ("should `S @ A` normalize?") **dissolves**: `S`
is never on the operator side of `@`, the two `@`s (operator
composition, Symbol diagonal composition) never mix, and there is no
normalization rule to write. This **refines** `operator_algebra` §3.7's
"a Symbol may appear as a factor in a chain" into "a diagonal
*operator* appears as the factor; the Symbol is what its `.eigenvalues`
returns" — a wording fix to apply when §3.7 / `operators_base.md` are next
edited. The only excluded case is a runtime-data diagonal (a
learned/data-driven filter) not derivable from the grid; out of scope,
and if it ever arrives it is an explicit Hadamard step, not a chain
factor.

---

## D3 forwarder resolution (D3a / D3b)

D3 made field methods thin forwarders, leaving the `f.to` /
interpolation spelling and the `Dispatched` constructor open. The
resolution rests on separating **two "binding-like" needs** that D3
conflated: *axis binding* (`op["x"]`, key = a coordinate name — which
mesh factor a separable kernel acts on) and *parameterization* (order,
pad factor, **target space** — what the operator is). A target space is
a parameter, not an axis, so it belongs in the constructor;
overloading `["x"]` to accept a space is a category blur, rejected.

### D3a. `to` is a resolver; subscript is axis-only

`diff`/`integrate` are **single-kind** verbs (one dispatch kind,
axis-bindable, composable) — good forwarders. `f.to(target)` is
different: a **multi-kind resolver** that reads *which* conversion the
source->target family relationship needs (`interpolate` / phase shift /
`transform`) and delegates to that single-kind verb. It does not fit
the verb mold, so **it stays a field method** — thin, delegating to the
registry (no drift, since it owns only the `(source, target) -> kind`
mapping, which lives nowhere else). The **composable** conversion is
the single-kind verb `interpolate` (`= Dispatched("interpolate")`,
axis-bindable like `diff`); `f.to` is the convenience over it.
Consequently **`["x"]` means axis binding only, everywhere**, and
non-axis parameters are constructor arguments:
`FiniteDifference(order)`, and `ConstantBroadcast(to=target)` (its
target supplied internally by binary-`codomain` unification, a static
interned space; `_apply(self, f)` reads it off `self`).

The dispatch kind is renamed `"interp"` -> `"interpolate"` so the verb
and kind match (as `diff` <-> `"diff"`); the class name `LinearInterp`
is unchanged (class name != kind is already normal, cf.
`FiniteDifference` / `"diff"`).

### D3b. Seeded verbs, public `Dispatched` constructor

The standard verbs are module-level singletons on `fr.operators` —
`diff`, `integrate`, `interpolate` — the discoverable surface the field
forwarders route through. The `Dispatched(kind)` constructor stays
**public as the extension escape-hatch**: a module registering a custom
kind can expose `Dispatched("mykind")["x"]` as its own verb; an unknown
kind is a clean `DispatchError` at application, so the open constructor
is safe. Ordinary use goes through the seeded verbs.

---

## Status

**D8 executed** (prior round): `operators_base.md`'s base hierarchy,
composed operators, and registry are rewritten onto the algebra — the
`Operator` dunders and `__getitem__`, bind-only `UnaryOperator`,
`SeparableOperator.bound_axis`, the new "Operator algebra" section
(`Identity`/`Zero`/`Composite`/`SeparableComposite`/`OperatorSum`/
`ScaledOperator`/`Block`/`Dispatched`), the algebra-derived
`FVDerivative`/`Gradient`/`Divergence`/`Curl`/`Laplacian` factories,
and the registry's `Dispatched`-at-merge resolution. The bind-only
`_apply(self, f)` pass is applied to every operator in the document.
The T4 wording fix to `operator_algebra` §3.7 is in.

**D3a/D3b closed** (this round): `f.to` is a resolver, `["x"]` is
axis-only, non-axis parameters are constructor args, the interpolation
kind/verb is `interpolate`, and the `Dispatched` constructor is a public
escape-hatch behind the seeded verbs. Applied to `operators_base.md`'s
registry sugar note, `Dispatched`, and `ConstantBroadcast`, plus the
`"interp"` -> `"interpolate"` rename across the class docs.

## Still open (next)

Every merge decision (D1–D8, B1–B4, T2–T4, D3a/D3b) is recorded and
applied to the notes. What remains is **implementation** (ROADMAP
Phase 1), not design — turning these class specs into `framework2.grid`
code.
