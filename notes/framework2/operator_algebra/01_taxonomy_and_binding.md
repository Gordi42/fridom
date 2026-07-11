# Operator algebra — Taxonomy and axis binding

Part of the operator design notes; see
[`00_overview.md`](00_overview.md) for the document map. The algebra
itself is in [`02_algebra.md`](02_algebra.md).

---

## 2. Taxonomy and axis binding

### 2.1 Operator kinds (recap)

The grid redesign already distinguishes five operator kinds; the
algebra must treat all of them uniformly:

| Kind | Examples | Signature shape |
|------|----------|-----------------|
| separable 1D kernels | `FiniteDifference(order=2)`, `LinearInterp()`, `WenoReconstruction(order=5)` | 1D factor space -> 1D factor space, axis-agnostic until applied/bound |
| mesh-level primitives | metric-aware `grad`/`div` on a sphere, `div: edge-normal -> cell` on an unstructured mesh | full product (or tuple) signature, axis fixed by the mesh |
| binary operators | `CollocationProduct`, `Convolution`, `Hadamard` | `(domain_a, domain_b) -> codomain` |
| grid-bound transforms | `Fourier(grid, axes=...)` | nodal space <-> coefficient space, via `.forward`/`.backward` |
| `Symbol`s | operator eigenvalues, spectral filters, phase shifts | mode-diagonal map between coefficient spaces |

All of these participate in the algebra of section 3; the composite
and sum objects it introduces are a sixth, *derived* kind that is
itself an ordinary operator.

### 2.2 Static structure, dynamic coefficients

The pytree treatment follows grid-redesign
[section 2.7](../01_concepts.md#27-where-coordinate-data-lives)
unchanged:

- An operator's **structure** — its class, order, stencil pattern,
  factor chain / term list / block layout, bound axes — is **static**
  and part of its jit/dispatch identity.
- Its **coefficient values** — stretched-mesh stencil weights, metric
  fields, the field coefficients of a scaled operator
  ([section 3.4](02_algebra.md#34-linear-combinations-and-coefficient-scaling))
  — are **dynamic pytree leaves**, derived from grid-materialized
  arrays at trace time.

Consequently a composite (`A @ B`), a sum (`A + B`), and a
field-scaled operator (`c * A`) are jaxified objects whose static part
is the algebraic structure and whose leaves are the coefficient
fields. Two composites are equal (for jit caching) iff their
structures are equal — the same rule as for spaces.

### 2.3 Axis binding: `op["x"]`

A separable 1D kernel is axis-agnostic; today's application form
names the axis at the call site (`fd(f, axis="x")`, grid-redesign
[section 2.5](../01_concepts.md#25-operator--typed-maps-between-spaces)).
Composition needs a way to fix the axis *before* application — a
mixed derivative `d^2/dx dy` composes two kernels acting on
*different* axes, which the call-site keyword cannot express.

**Subscripting binds the axis**: `fd["x"]` is the *axis-bound* form
of `fd` —

- **Semantics**: the extension of the 1D kernel to product spaces by
  identity on all other factors — the `(mesh-x operator) ⊗ identity`
  reading of grid-redesign
  [section 2.3](../01_concepts.md#23-tensorproductspace-and-named-coordinates)
  made into an object.
- **Application**: bound operators are callable without the keyword:
  `fd["x"](f)`. The unbound call `fd(f, axis="x")` remains sugar for
  `fd["x"](f)`.
- **The key is a coordinate name**, resolved to the mesh factor that
  owns it (a 2D sphere factor is bound by either of its names,
  `op["lon"]` and `op["lat"]` resolving to the same factor — but a 1D
  kernel is only bindable to a 1D factor, so on multi-name factors
  only mesh-level primitives bind).
- **Rebinding is an error**: `fd["x"]["y"]` raises. Binding an
  operator whose axis is already fixed by its signature (mesh-level
  primitives, grid-bound transforms, full-product composites) also
  raises — there is nothing left to bind.
- **Binding distributes over the algebra**:
  `(A @ B)["x"] == A["x"] @ B["x"]`,
  `(A + B)["x"] == A["x"] + B["x"]`, and `(c * A)["x"] == c * A["x"]`.
  An *unbound* composition of 1D kernels is therefore still a 1D
  separable kernel — applied with `axis=` or bound later — while a
  mixed-axis composite must bind its factors first
  (`fd["x"] @ fd["y"]`).
- **Bound operators are static**: `fd["x"]` adds the axis name to the
  static structure (section 2.2); binding never touches coefficient
  leaves.

Per-axis quantities of the bound form specialize accordingly: the
halo requirement of `fd["x"]` is `fd`'s 1D halo on the x mesh and zero
elsewhere; its eigenvalue symbol on a product coefficient space is the
1D symbol broadcast across the other factors via `ConstantSpace`
(grid-redesign
[section 3.3](../02_rules.md#33-constantspace-replaces-topo-with-automatic-broadcast),
sketch
[4.6](../03_api_sketches.md#46-operator-eigenvalues-for-exact-spectral-solvers)).

`Symbol`s are never bound: they are already tagged with their
coefficient factor space and broadcast across the product by
`ConstantSpace` (grid-redesign
[section 3.11](../02_rules.md#311-field-operations-linear-ops-and-the-product-problem)).
