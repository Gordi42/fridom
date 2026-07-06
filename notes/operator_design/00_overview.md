# Operator algebra design

Status: **draft** (design phase, no implementation yet)
Author: Silvano Rosenau (with AI-assisted brainstorming)
Date: 2026-07-06

This note set designs the **operator algebra** of the new grid
abstraction: how operators compose (`C = A @ B`), how they add and
scale (`A + B`, `c * A`, `field * A`), how their signatures generalize
to vector- and tensor-valued fields, and how composites interact with
the machinery the grid redesign already fixed — halo accounting,
`Symbol` eigenvalues, and the dispatch registry.

It is a sibling of the grid redesign notes
([`../grid_redesign/00_overview.md`](../grid_redesign/00_overview.md))
and shares their conventions: all code snippets are **illustrative,
not normative**; section and sketch numbers are stable identifiers.
The operator *concept* — free-standing, typed maps between function
spaces — is fixed in grid-redesign
[section 2.5](../grid_redesign/01_concepts.md#25-operator--typed-maps-between-spaces)
and is not re-litigated here; this set adds the algebra on top of it.

---

## Document map

| File | Contents |
|------|----------|
| [`00_overview.md`](00_overview.md) | Motivation (section 1), precedents — this file. |
| [`01_taxonomy_and_binding.md`](01_taxonomy_and_binding.md) | Operator taxonomy recap, static/dynamic split, axis binding `op["x"]` (section 2). |
| [`02_algebra.md`](02_algebra.md) | The algebra (section 3): tuple signatures (vector/tensor operands), composition `@`, identity/zero, linear combinations, block operators, halo, symbols, binary operators, transforms, dispatch integration, rejected alternatives. |
| [`03_api_sketches.md`](03_api_sketches.md) | Non-normative API sketches (section 4). |
| [`04_open_threads.md`](04_open_threads.md) | Open threads (section 5). |

---

## 1. Motivation

### 1.1 Composition is already everywhere — implicitly

The grid redesign fixes operators as typed maps
`(domain_space -> codomain_space)` but leaves composition informal.
Four normative places already *are* compositions:

1. **The FV derivative** factors into exact + approximate
   (grid-redesign
   [section 3.9](../grid_redesign/02_rules.md#39-finite-volume-semantics-the-average-family-and-the-fv-derivative)):
   `diff = flux_diff o reconstruct`.
2. **`Convolution`** is *defined* as a composite (grid-redesign
   [section 3.12](../grid_redesign/02_rules.md#312-dealiasing)):
   `trim_transform o CollocationProduct o pad_inverse_transform`.
3. **`grad`/`div`/`laplacian` dispatch defaults** on separable grids
   are "a composition over `diff`/`interpolate`" (grid-redesign
   [section 3.4](../grid_redesign/02_rules.md#34-generic-operator-dispatch)).
4. **Halo accounting** is specified precisely for "un-synced
   composition chains" (`f.diff("x").diff("x")` needs
   `halo_1 + halo_2`, grid-redesign
   [section 5](../grid_redesign/04_decomposition.md#5-domain-decomposition)) —
   a rule about composites with no object to attach itself to.

This note set gives these an explicit, uniform spelling — `@` for
composition, `+`/`*` for the linear structure — so that composites are
**ordinary operators**: callable, registrable in the dispatch table,
halo-accountable, and symbol-bearing like any primitive.

### 1.2 Vector- and tensor-valued signatures

The unary signature `(domain_space -> codomain_space)` covers scalar
fields only, yet the most important derived operators are not
scalar-to-scalar: `grad` maps a scalar field to a *vector* field whose
components live on **different** staggered spaces (the C-grid point of
grid-redesign [section 2.4](../grid_redesign/01_concepts.md#24-field)),
`div` maps that vector field back, `curl` maps vectors to vectors, and
a strain-rate or stress operator produces a *tensor* field. Section
[3.1](02_algebra.md#31-signatures-space-tuples-direct-sums) generalizes
signatures to **tuples of product spaces** (finite direct sums), and
section [3.5](02_algebra.md#35-block-operators-the-linear-case) shows
that linear operators between such tuples are **block matrices of
scalar operators**, for which `@` is literally block-matrix
multiplication — `laplacian = div @ grad` becomes a type-checked
identity rather than a convention.

### 1.3 Sums with field coefficients are required, not optional

Terrain-following coordinates (grid-redesign
[section 3.8](../grid_redesign/02_rules.md#38-boundaries-iii-terrain-following-boundary-fitted))
demand that physical-space operators be expressible as **sums of
separable operators with field coefficients**:

```
d/dx|_z  =  d/dx|_sigma  -  (sigma H_x / H) d/dsigma
```

Without `A + B` and `field * A` in the algebra, that rule has no
spelling. The same linear structure gives the separable Laplacian
(sum of per-axis second derivatives, grid-redesign sketch
[4.6](../grid_redesign/03_api_sketches.md#46-operator-eigenvalues-for-exact-spectral-solvers))
and the metric-aware operators on spheres.

### 1.4 What this is *not*

This is **not** a lazy expression-graph system. Composites and sums
are shallow, eager structural objects — a flat tuple of factors or
terms — with no automatic simplification, distribution, or scheduling
([section 3.11](02_algebra.md#311-no-expression-graphs-rejected-alternatives)).
The grid redesign already rejected deferred fields and automatic
transform scheduling (grid-redesign sections
[3.10](../grid_redesign/02_rules.md#310-discretizing-continuous-functions),
[3.12](../grid_redesign/02_rules.md#312-dealiasing)); the operator
algebra follows the same line.

## Precedents

- **FRIDOM's own deleted experiment** — an earlier
  `framework/operator/operator.py` defined an abstract `Operator` with
  `_ComposeOperator`, `_AddOperator`, `_SubOperator`,
  `_ScalarMulOperator` wrappers before being removed ("moving stuff to
  backup"). It validated the appetite for the algebra but predated
  typed signatures, so composition could not be checked; this design
  is its typed successor.
- **scipy `LinearOperator`** — matrix-free operators with `@`/`+`/
  scalar `*` returning wrapper operators; the closest API precedent
  for the linear structure (no function-space typing).
- **shenfun / Dedalus v3** — operator expressions over tensor-product
  bases; both build (lazily evaluated) expression trees, which we
  deliberately do not adopt — only the typed-composition idea.
- **FEEC / Firedrake (UFL)** — form language with typed function
  spaces; the block-operator view of vector-valued maps mirrors their
  mixed function spaces, without adopting the form compiler.
