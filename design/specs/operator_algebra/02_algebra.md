---
status: normative
date: 2026-07-06
---

# Operator algebra — Rules

Part of the operator design notes; see
[`00_overview.md`](00_overview.md) for the document map, and
[`01_taxonomy_and_binding.md`](01_taxonomy_and_binding.md) for the
taxonomy and axis binding.

---

## 3. The algebra

### 3.1 Signatures: space tuples (direct sums)

An operator's domain and codomain are **ordered tuples of product
spaces** — finite direct sums `S_1 ⊕ S_2 ⊕ ... ⊕ S_n`. The scalar
signature of grid-redesign
[section 2.5](../grid/01_concepts.md#25-operator--typed-maps-between-spaces)
is the length-1 case; nothing changes for scalar operators.

- **A `VectorField` supplies the tuple.** A vector field is a
  collection of scalar fields on *different but related* spaces
  (grid-redesign
  [section 2.4](../grid/01_concepts.md#24-field)); its
  ordered component-space tuple *is* the operand type. So on a C-grid:

  ```
  grad : (Center⊗Center)                    -> (Right⊗Center, Center⊗Right)
  div  : (Right⊗Center, Center⊗Right)       -> (Center⊗Center,)
  curl : (Right⊗Center, Center⊗Right)       -> (Right⊗Right,)
  ```

- **Order is the type; names are metadata.** Component names
  (`u`, `v`, `w`, `b`; tensor multi-indices `xx`, `xy`, ...) belong to
  the field/state, like `name`/`units` (grid-redesign section 2.4).
  Signature equality is componentwise space identity *by position*.
  A `TensorField` flattens its multi-index into the ordered tuple; the
  index structure is metadata on the collection.
- **The strict algebra lifts componentwise.** `U + V` on vector
  fields requires equal space tuples; the sanctioned exceptions
  (`ConstantSpace` broadcast, real -> complex promotion) apply per
  component. Applying an operator to a field whose space tuple does
  not match the domain is a `SpaceMismatchError`.
- **No new space type is introduced.** The direct sum is a *signature*
  notion (an ordered tuple), not a new kind of `FunctionSpace`:
  fields on it are ordinary `VectorField`s, transforms and dispatch
  stay per component, and the decomposition never sees it. Reifying a
  first-class `DirectSumSpace` was considered and rejected as
  machinery without a consumer — the tuple is enough for typing, and
  `VectorField` already carries it.
- **Mixed tuple lengths are just signatures.** Scalar-to-vector
  (`grad`), vector-to-scalar (`div`, kinetic energy), and
  vector-to-vector (`curl`, a rotation) operators all fall out; a
  "scalar" operand is the 1-tuple, and a scalar operator applied to a
  `VectorField` componentwise is `VectorField.map(op)`, not a
  signature extension.

### 3.2 Composition: `C = A @ B`

`@` is operator composition, read right-to-left like function
application:

```python
C = A @ B          # C(f) == A(B(f))
```

- **Typing**: `C.domain = B.domain`, `C.codomain = A.codomain`, valid
  iff `B.codomain == A.domain` (componentwise tuple identity,
  section 3.1). Where both signatures are concrete — bound operators,
  grid-bound transforms, resolved dispatch entries — the check is
  **eager** at composition time; separable kernels whose spaces
  resolve only per field defer the check to application/dispatch
  time. Either way a mismatch is the same `SpaceMismatchError` as
  `f + g` on different spaces — the strict algebra extended to
  operators.
- **Associative and flat**: `A @ (B @ C) == (A @ B) @ C`; a composite
  normalizes to a single flat factor chain (no nested composites),
  mirroring the flat `TensorProductSpace` (grid-redesign
  [section 2.3](../grid/01_concepts.md#23-tensorproductspace-and-named-coordinates)).
  The flat chain is what halo accounting (section 3.6) and symbols
  (section 3.7) walk.
- **Not commutative**, obviously; and no reordering is ever performed
  (section 3.11) — even for factors that would commute (bound
  operators on different axes).
- **Composites are ordinary operators**: callable, bindable while
  unbound (section 2.3), registrable as dispatch entries
  (section 3.10), and further composable.
- **Linearity is not assumed.** `(A @ B)(f) = A(B(f))` is plain
  function composition and holds for nonlinear factors (WENO
  reconstruction, flux limiters). Only the *block expansion*
  (section 3.5) and the symbol calculus (section 3.7) require
  linearity, and they simply do not exist for nonlinear factors.
- `A ** n` for a non-negative integer is sugar for the n-fold chain
  `A @ A @ ... @ A` (`A ** 0 == Identity`); on `Symbol`s `**` keeps
  its diagonal-algebra meaning — the two coincide, since composition
  of diagonal operators is elementwise multiplication.

### 3.3 Identity, zero, and normalization

Two trivial operators complete the algebra:

- **`Identity`** — the neutral element of `@`. Signature: any space to
  itself (resolved at use). Chains elide it on normalization
  (`A @ Identity == A`). It is what an unbound axis contributes in the
  `⊗ identity` extension (section 2.3) and what operators along a
  `ConstantSpace` axis reduce to (grid-redesign
  [section 3.3](../grid/02_rules.md#33-constantspace-replaces-topo-with-automatic-broadcast)).
- **`Zero`** — the neutral element of `+` and absorbing element of
  `@` (`Zero @ A == A @ Zero == Zero`, `A + Zero == A`). Sums drop
  zero terms on normalization. Its consumer is the block-operator
  layout (section 3.5), where structural zeros are the norm (`curl`,
  sparse system matrices); a `Zero` block contributes no computation,
  no halo, and a zero symbol.

Normalization is **structural only**: flattening nests of the same
node kind, eliding `Identity`, dropping `Zero` terms. No algebraic
rewriting (no distribution of `@` over `+`, no factoring, no
cancellation) ever happens — what you wrote is what runs
(section 3.11).

### 3.4 Linear combinations and coefficient scaling

Operators form a module over fields (in the algebraic sense):

```python
D = A + B            # D(f) == A(f) + B(f)
D = A - B            # sugar for A + (-1) * B
D = -A               # (-1) * A
D = 2.0 * A          # scalar coefficient
D = c * A            # c a ScalarField: coefficient field
```

- **Typing**: all terms of a sum must have the same (resolved)
  signature, which is the sum's signature; checked eagerly where
  concrete, else at application — as for `@`.
- **Coefficient fields multiply the *output***: `(c * A)(f) ==
  c * A(f)`, so `c` must live on `A.codomain` (per component for
  tuple codomains); the `ConstantSpace` broadcast applies, making
  plain scalars the constant special case. The multiplication is the
  physical product `*` of grid-redesign
  [section 3.11](../grid/02_rules.md#311-field-operations-linear-ops-and-the-product-problem),
  dispatched per space. Input-side scaling is spelled explicitly as
  a chain: `A @ (c * Identity)`.
- **This is the §3.8 machinery.** Terrain-following derivatives are
  now literal code
  (`ddx_z = fd["x"] - c_metric * fd["sigma"]`, sketch
  [4.4](03_api_sketches.md#44-terrain-following-derivative-a-sum-with-a-field-coefficient)),
  with the metric coefficient a dynamic pytree leaf (section 2.2)
  read through `grid.metric` — time-dependent metrics trace through
  with no special casing (grid-redesign
  [section 3.8](../grid/02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted)).
- **The `*`/`@` disambiguation table grows three rows** (extending
  grid-redesign
  [section 2.5](../grid/01_concepts.md#25-operator--typed-maps-between-spaces)):

  | Spelling                    | Meaning                                  |
  |-----------------------------|-------------------------------------------|
  | scalar `*` operator         | scaled operator (constant coefficient)    |
  | field `*` operator          | scaled operator (field coefficient on the codomain) |
  | operator `+` operator       | operator sum (same signature)             |
  | operator `@` operator       | composition (section 3.2)                 |
  | `Symbol` `@` `Symbol`       | == `Symbol * Symbol` (diagonal composition) |

  `field * field` (physical product), `space * space` (tensor
  product), and the `Symbol` diagonal algebra are unchanged.

### 3.5 Block operators: the linear case

For **linear** operators between space tuples, the natural structure
is a **block matrix of scalar-signature operators**: an operator
`D : S_1 ⊕ ... ⊕ S_n -> T_1 ⊕ ... ⊕ T_m` is an `m x n` grid of blocks
`D_ij : S_j -> T_i`, applied as

```
D(f)_i = sum_j D_ij(f_j)
```

- **`@` is block-matrix multiplication.** For block operators the
  composition rule of section 3.2 *is* matmul:
  `(A @ B)_ik = sum_j A_ij @ B_jk` — which is why `@` is the right
  spelling. `laplacian = div @ grad` type-checks
  (`(1 x n) @ (n x 1) = 1 x 1`) and its single block is the sum of
  per-axis second-derivative chains — grid-redesign sketch 4.6's
  broadcast sum, now derived instead of asserted.
- **Construction**: `fr.operators.Block([[...], [...]])` with
  operator entries, `Zero` for structural zeros, `Identity` on the
  diagonal where needed. `grad` is a column (`n x 1`), `div` a row
  (`1 x n`), `curl` a matrix with `Zero` blocks in 3D. These are
  *constructions of the dispatch defaults*, not privileged objects
  (grid-redesign
  [section 3.4](../grid/02_rules.md#34-generic-operator-dispatch):
  the default entry *is* an operator object).
- **The block expansion is metadata, not a rewrite.** A chain
  `div @ grad` stays a two-factor chain as written; its block
  expansion is *computed on demand* by halo accounting and the symbol
  calculus, never eagerly materialized as a rewritten operator
  (section 3.11).
- **Nonlinear tuple-signature operators are opaque.** A nonlinear
  vector-input operator (kinetic energy `(u, v) -> ke`, a WENO flux
  assembly) has a tuple signature (section 3.1) but **no block
  structure**; it composes by `@` under plain function-composition
  typing and simply does not answer block/symbol queries.

### 3.6 Halo accounting

The per-axis halo of an algebraic operator follows two rules, matching
grid-redesign
[section 5](../grid/04_decomposition.md#5-domain-decomposition):

- **Composition sums**: an un-synced chain reads through the stacked
  stencils, so `halo(A @ B) = halo(A) + halo(B)` per axis. A composite
  therefore slices into one wider halo with **one** sync in front,
  instead of one sync per factor.
- **Sums (and blocks) max**: the terms of `A + B` read the *same*
  input in parallel, so `halo(A + B) = max(halo(A), halo(B))` per
  axis. For block operators, per block-row:
  `halo((A @ B)_ik) = max_j (halo(A_ij) + halo(B_jk))`.
- `Identity` and `Zero` contribute zero halo; a coefficient scaling
  `c * A` contributes `halo(A)` (the product is pointwise).

The automatic **halo-accounting trace** (grid-redesign section 5) is
unchanged: composites present the same operator interface, so the
tracer field flows through their factors and accumulates exactly these
numbers with no composite-specific code — the object-level rules above
just let the grid query a composite's demand *without* a trace (e.g.
when a composite is registered as a dispatch default), and let a long
chain advertise its total up front. Whether the grid may insert
mid-chain syncs to cap the accumulated width is an open thread
([section 5.3](04_open_threads.md)).

### 3.7 Symbols of composites

The eigenvalue machinery of grid-redesign
[section 2.5](../grid/01_concepts.md#25-operator--typed-maps-between-spaces)
extends compositionally. First, one clarification the composition
rules force: a `Symbol` is a **mode-diagonal map between coefficient
spaces** — domain and codomain share mode indexing but may differ in
origin (the exact phase shift `e^{i k dx/2}` *is* a Symbol from
`Fourier(origin=Center)` to `Fourier(origin=Right)`). "Diagonal" means
mode-by-mode, not endomorphic. Today's `k_hat` — which bundles the
derivative eigenvalue with the staggering phase — is exactly such an
origin-changing symbol.

With that:

- **Chains multiply**: `(A @ B).eigenvalues(grid, coeff_space)` is the
  mode-wise product of the factor symbols, each queried on the
  coefficient space of *its own* domain — the chain of coefficient
  spaces follows the chain of nodal spaces through the per-origin
  coefficient spaces of grid-redesign
  [section 3.2](../grid/02_rules.md#32-coefficient-representations-are-separate-spaces),
  so origin bookkeeping is carried by the factor symbols and the
  product is well-typed end to end.
- **Sums add**: same domain/codomain coefficient spaces required.
- **Blocks are symbol matrices**: a linear block operator yields a
  block matrix of symbols — a per-mode small matrix. This is the
  primitive the model-side eigenmode objects consume (per-mode system
  matrices); the eigenmode assembly itself stays model-side and out
  of scope (grid-redesign section 2.5).
- **Scaling**: `c * A` has a symbol iff `c` is constant
  (`ConstantSpace` coefficient) — then it scales `A`'s symbol. A
  genuine field coefficient breaks translation invariance and with it
  diagonalizability; the composite then answers "no symbol", exactly
  like a factor on a non-diagonalizing basis.
- **Existence is per factor**: a composite has a symbol iff *every*
  factor has one relative to the compatible bases (grid-redesign
  section 2.5's rule, applied factor-wise). Nonlinear factors never
  do.
- **The FV exactness check becomes executable** (grid-redesign
  [section 3.9](../grid/02_rules.md#39-finite-volume-semantics-the-average-family-and-the-fv-derivative)):
  the symbol of `flux_diff @ reconstruct` at second order is
  `i k sinc(k dx / 2)` — the product of the factor symbols — and can
  be asserted in tests against the composed operator (sketch
  [4.6](03_api_sketches.md#46-verifying-fv-exactness-through-composed-symbols)).

Applying a `Symbol` to a field remains a `Hadamard` multiply;
`Symbol @ Symbol` equals the diagonal product `*` (section 3.4). A raw
`Symbol` is **not** a factor in an operator `@` chain: a `Symbol`
carries a dynamic data leaf while the chain is static structure (the
class design fixes `Symbol` as *not* an `Operator`). What appears in a
chain is a **static diagonal operator** — `SpectralDerivative`,
`PhaseShift`, `SincShift`, a spectral filter — that derives its symbol
from the grid at trace time and applies it as `Hadamard`; the raw
`Symbol` stays solver-facing (`op.eigenvalues`, `1 / lap`). So there is
no `Symbol @ Operator` and no normalization rule to write (thread 5.4,
resolved; class-design `../grid/classes/operator_algebra_merge.md` T4).

### 3.8 Binary operators in chains

Binary operators (`CollocationProduct`, `Convolution`, `Hadamard`;
grid-redesign
[section 3.11](../grid/02_rules.md#311-field-operations-linear-ops-and-the-product-problem))
join the algebra with two rules:

- **Post-composition wraps the result**: `A @ P` for binary `P` is the
  binary operator `(f, g) -> A(P(f, g))`.
- **Pre-composition takes a tuple, one operator per operand**:
  `P @ (B_1, B_2)` is `(f, g) -> P(B_1(f), B_2(g))`; the sugar
  `P @ B` (single unary right-hand side) means `P @ (B, B)` — the
  same operator applied to each operand, which is the common
  symmetric case.

Under these rules the `Convolution` definition of grid-redesign
[section 3.12](../grid/02_rules.md#312-dealiasing) is a
literal expression of the algebra:

```python
Convolution = trim_transform @ CollocationProduct @ pad_inverse_transform
```

— the padded inverse transform applies to *both* operands
(pre-composition sugar), the product runs on the finer nodal space,
and the trimming forward transform wraps the result. Halo and typing
rules extend per operand; symbols do not (a binary operator is not
mode-diagonal).

The tuple form covers only per-operand *unary* pre-composition;
anything fancier (operand permutation, sharing) is written as a plain
function — the transform-once combinator (grid-redesign section 3.12)
remains the tool for multi-term scheduling.

### 3.9 Transforms in chains

Grid-bound transforms keep their deliberate exception status
(construction binds the grid; application is `.forward`/`.backward`,
grid-redesign section 2.5) — and join the algebra anyway:
**`t.forward` and `t.backward` are themselves ordinary unary
operators** with concrete signatures (the direction-selected halves of
the transform pair). `Convolution` above composes them; a de-aliased
pseudo-spectral pipeline is a chain of `t.backward`, products, and
`t.forward`. Their signatures are concrete (the grid is known), so
chains containing them are eagerly type-checked at composition time
(section 3.2).

### 3.10 Dispatch integration

Composites and the dispatch registry (grid-redesign
[section 3.4](../grid/02_rules.md#34-generic-operator-dispatch))
interlock in both directions:

- **Composites are registrable.** A dispatch default *is* an operator
  object; nothing stops that object being a chain, a sum, or a block.
  The separable-grid defaults are exactly that:
  `("diff", CellAvg) -> flux_diff @ reconstruct-default`,
  `("laplacian", space) -> div-block @ grad-block`.
- **Kind placeholders resolve at assembly time.** The FV `diff`
  default must pick up a module's `reconstruct` override
  (grid-redesign sketch
  [4.2](../grid/03_api_sketches.md#42-custom-operator-module-local-override))
  without runtime late binding. A chain factor may therefore be a
  **kind placeholder**, `fr.operators.Dispatched("reconstruct")`,
  which is resolved against the *merged* registry (grid defaults +
  module overrides) during model assembly — the same moment the
  overrides are merged. After assembly the composite is fully
  concrete and static; there is no per-application registry lookup
  inside chains. Placeholders are resolved with the same precedence
  as ordinary dispatch (space-specific over kind-only over default).
- **`.to` is unchanged**: it stays per-axis single-kind dispatch; it
  neither produces nor consumes the algebra's objects at its surface
  (though a registered conversion entry may of course *be* a
  composite).

### 3.11 No expression graphs (rejected alternatives)

The algebra's objects are **shallow, eager structures**: a flat factor
chain, a flat term list, a block grid. Rejected, deliberately:

- **Automatic algebraic rewriting** (distributing `@` over `+`,
  reordering commuting factors, cancellation, folding
  `forward @ backward`): the structure you wrote is the structure
  that runs. Rationale: rewrites silently change numerics (operation
  order, aliasing, where syncs land), the win is XLA's job (CSE,
  fusion), and debuggability of "what runs" beats cleverness. The
  only structural normalizations are those of section 3.3.
- **Lazy operator/field expression graphs** (Dedalus/shenfun style):
  already rejected once for fields and transform scheduling
  (grid-redesign sections
  [3.10](../grid/02_rules.md#310-discretizing-continuous-functions),
  [3.12](../grid/02_rules.md#312-dealiasing)); the operator
  algebra does not reintroduce them through the back door. A chain is
  applied factor by factor when called, eagerly, under jit.
- **A canonical matrix/assembled form**: operators stay matrix-free;
  the block view (section 3.5) is a typing and query structure, not
  an assembly into arrays.
