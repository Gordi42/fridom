---
status: frozen
date: 2026-07-18
---

# Coefficient-space field arithmetic — semantics of product/power rows

Research record (see [`README.md`](README.md) for status). Answers the
former open-roadmap question "coefficient-space product/power rows —
needs an owner call" and records the owner ruling of 2026-07-18. The
normative outcome lives in the spec
([`../specs/grid/classes/operators_products.md`](../specs/grid/classes/operators_products.md),
"Coefficient spaces — ruled 2026-07-18"); this file is the evidence
and the options behind that ruling.

## 1. The question

Should coefficient-space `ScalarField`s (Fourier / cosine / sine
spectra) get elementwise product, quotient, power, and absolute-value
rows, and if so with what semantics?

The obstruction is representational. Field arithmetic is *function*
arithmetic: `(f * g)` must be the field whose values are the pointwise
product of the two represented functions. In a coefficient space an
elementwise multiply of the stored arrays is a **convolution** of the
represented functions, not their product — so an elementwise multiply
cannot share the `("multiply", space)` kind that nodal spaces use for
the pointwise product without lying about what `*` means. The same
obstruction rules out a shared quotient (`("divide", space)`) and
absolute value (`("abs", space)`): a quotient or per-mode magnitude of
two spectra has no representation-independent meaning. Scalar-add and
constant-broadcast into a coefficient space are a *different*
obstruction — they are the exact zero-mode update, not a field
operation, and not even representable when the basis has no constant
mode (a Sine basis).

## 2. Ruling (owner-ratified 2026-07-18, Silvano, in chat)

Coefficient-space `ScalarField`s form a **vector space, not an
algebra**. Only transform-commuting operations are field arithmetic:
add/subtract of same-space fields, and scalar multiply/divide. Those
already work. Elementwise multiply/divide/power/abs of two
coefficient-space fields is a convolution or a spectral diagnostic,
not a field, so `("multiply"|"divide"|"power"|"abs",
coefficient-space)` registry rows are **permanently absent by
design** — no longer an "iteration 1" deferral.

Pointwise per-mode coefficient manipulation is **diagonal-operator
algebra** and lives on `Symbol` (`src/fridom/spatial/operators/symbol.py`),
whose `*`/`+`/`**`/`1/.` are the diagonal algebra and whose `Symbol ×
field` apply is guarded to fields constant on every transformed
factor — the one regime where a coefficient multiply is exact.
`Convolution` (the true function product in coefficient space) and the
zero-mode `ConstantBroadcast` remain **reserved, distinct kinds**,
unbuilt until a consumer exists. Note that a constant is not even
representable in a Sine basis, so scalar-add can never be a
basis-uniform field op.

## 3. Evidence: machinery state

The linear (vector-space) operations are already exact and legal, and
deliberately bypass the registry:

- `_linear_combine` (`src/fridom/spatial/fields/scalar_field.py`)
  combines same-space operands elementwise for `+`/`-`; on a
  coefficient space this is the exact spectral sum. Pinned by
  `test_coefficient_space_linear_ops_are_elementwise`
  (`tests/spatial/fields/test_scalar_field.py`).
- `_scalar_scale` (same module) scales by a Python/0-d scalar for
  `*`/`/` on any space; on a coefficient space this is the exact
  per-mode scaling (linearity of the transform). Pinned by
  `test_scalar_scaling_is_legal_on_coefficient_spaces`.

The blocked set was exactly the non-linear elementwise family —
multiply, divide, power, abs — plus scalar/constant add (the
"zero-mode update"). The product spec's `Divide` already refused
coefficient rows by design
([`../specs/grid/classes/operators_products.md`](../specs/grid/classes/operators_products.md),
"Divide / Power / Abs": "a quotient or modulus of spectra has no
representation-independent realization; transform back first"). The
dispatch marker is the `CoefficientSpace` ABC
(`src/fridom/spatial/spaces/coefficient.py:34`); every guard tests
`isinstance(factor, CoefficientSpace)`.

## 4. Evidence: consumer census (new stack, 2026-07-18)

A sweep of the new stack for sites that manipulate coefficient fields
found **zero** wanting a coefficient×coefficient product. The classes:

- ~22 **diagonal-multiplier** sites (`1/k²`, `1/dsqr`, `N²(z)`
  weights) — all a per-mode diagonal applied to a spectrum, i.e.
  `Symbol × field`, not a field×field product. Served by the
  `Symbol × field` apply whose guard requires the field be constant on
  every transformed factor (`src/fridom/spatial/operators/symbol.py:548`),
  the exact regime where a coefficient multiply is well-defined; this
  is the pressure-solve `1/dsqr` / `N²(z)` weight path.
- ~8 **scalar scalings** — `_scalar_scale`, already legal.
- ~9 `**p` sites — all on `Symbol` diagonals (diagonal powers), never
  on a field spectrum.
- ~4 **additive-constant** sites (Helmholtz shift etc.) — all on
  `Symbol` diagonals, not field scalar-add.
- ~12 **conjugated reductions / contractions** (energy, Parseval-style
  spectral norms) where the honest level is raw `.data`, not a field
  op; `energy.py` explicitly refuses field-valued weights in
  coefficient space.

The old stack followed the identical diagonal-multiplier-only pattern:
`spectral_pressure_solver.py` spells the solve `(-div.fft() *
k_squared_inv).ifft()` — a diagonal multiplier applied to a spectrum,
transformed back, never a spectrum×spectrum product.

## 5. Evidence: external precedent

Every mature typed-spectral library either forces the product to grid
space or names the convolution explicitly; none silently spells a
coefficient×coefficient product `*` and calls it the function product:

- **Dedalus v3** forces product evaluation to grid space via deferred
  `Multiply` operators — a spectral field product is realized by
  transforming to grid, multiplying, transforming back.
- **Chebfun** `.*` is the *true* function product, computed as a
  correct coefficient-space convolution (Toeplitz/circulant FFT; Olver
  & Townsend 2013) — the honest `Convolution`, not an elementwise
  array multiply.
- **Shenfun** spectral `Function` inherits ndarray `__mul__` and
  silently computes the elementwise-array convolution — precisely the
  trap this ruling forecloses (the array multiply reads as a product
  but means a convolution).
- Raw-array spectral codes (jax-cfd spectral, SpectralDNS) have no
  typed coefficient object at all, so the question does not arise for
  them.

Standard pseudospectral doctrine: products are formed in physical
space with 2/3- or 3/2-rule dealiasing; only **diagonal** operators
apply in coefficient space. The ruling encodes exactly this doctrine
in the type system.

## 6. Options considered

1. **Elementwise rows under the same kinds** (`("multiply",
   coefficient-space)` = array multiply): refuted. Zero consumers, it
   is the Shenfun trap, and it breaks the invariant that field
   arithmetic is function arithmetic (the array multiply is a
   convolution, not the product).
2. **Function-product semantics** — Dedalus-style auto-transform, or
   Chebfun-style coefficient convolution under `*`: no consumer, and
   hiding transforms behind `*` is against the explicit-over-auto-magic
   house rule. `Convolution` stays reserved and named if a consumer
   ever appears.
3. **Leave the question open**: carries a decided question in
   `open.md` indefinitely; rejected.
4. **Ratify the vector-space / `Symbol` split** — coefficient fields
   are a vector space, per-mode diagonal algebra lives on `Symbol`,
   convolution and zero-mode broadcast stay reserved distinct kinds.
   **Chosen.**

## 7. Consequences shipped

- Taught-error rewording of the coefficient-space dunders and
  broadcast guards (`src/fridom/spatial/fields/scalar_field.py`): each
  now raises a `DispatchError` "… by design …" pointing at `Symbol` or
  the nodal transform, and the pinned tests
  (`tests/spatial/fields/test_scalar_field.py`) match the new
  distinctive fragments ("convolution", "quotient of spectra", "Symbol
  algebra", "spectral diagnostic", "zero-mode", "broadcast").
- Spec note (§Coefficient spaces — ruled 2026-07-18) and the
  `products.py` module-docstring line.
- Roadmap close (`design/roadmap/open.md` → `done.md`); the Phase-2
  grid follow-ups plan closes with it.

Left open deliberately:

- Migrating the few hand-rolled broadcast sites (e.g.
  `src/fridom/nonhydro2/initial_conditions.py:836`) onto `Symbol`
  application — cleanup, unforced.
- A named `spectrum()`-style diagnostic if `|f̂|²` ergonomics are ever
  wanted — a diagnostic surface reading `.data`, **not** an
  `("abs", coefficient-space)` row.
- The zero-mode update (`ConstantBroadcast` into a coefficient space)
  only if a real consumer appears, and then per-basis (Sine excluded,
  having no constant mode).
