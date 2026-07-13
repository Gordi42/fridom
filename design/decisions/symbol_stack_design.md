---
status: accepted
date: 2026-07-08
supersedes: ../plans/done/operator_symbols_plan.md
partially_superseded_by: commit d3309640 (2026-07-09)
---

# The symbol stack — dynamic symbols, mixed transforms, banded axes

> **Partially superseded, 2026-07-09.** Decisions 1–3 shipped: the scalar
> `Symbol` type, `SpectralSolve`
> (`spatial/operators/spectral_solve.py`, carrying the nonhydro pressure
> solve), layout-faithful `eigenvalues`, and the banded primitive
> (`spatial/operators/banded.py`). What did **not** survive is the
> `BlockSymbol` layer — the claim that eigenvectors/projectors *are*
> symbols (a rank-1 `BlockSymbol`) died when the block IR was deleted in
> `d3309640`; see
> [`blocksymbol_l_assembly.md`](blocksymbol_l_assembly.md). Read the
> `BlockSymbol`-of-`Banded` rung below as a proposal that no longer has a
> substrate, not as a committed target. The `Banded` rung itself is still
> live work, tracked as phase I of
> [`../plans/active/projection_eigenmode_roadmap.md`](../plans/active/projection_eigenmode_roadmap.md).

**Status: design decisions, 2026-07-08 (owner-driven discussion).**
Refines [`../plans/done/operator_symbols_plan.md`](../plans/done/operator_symbols_plan.md) (§3-5 the
Symbol type + solver) and
[`blocksymbol_l_assembly.md`](blocksymbol_l_assembly.md) (resolves its
real-FFT/staggering boundary). Triggered by the nonhydro pressure
solver: the hand-rolled `discrete_laplace_symbol` in
`nonhydro2/modules/pressure.py` must give way to
`Laplacian → eigenvalues → inverse` for *whatever* difference operator
the grid uses.

## Decisions

1. **`SpectralSolve` object (not a raw three-liner).** The pressure
   module states the elliptic operator and applies the solver; the
   transform→symbol→inverse→transform-back is hidden.
2. **The vertical weight lives in the operator.** The pressure Laplacian
   is `Div @ Diag(1, 1, 1/dsqr) @ Grad`, built from the grid's own
   diff operators; `dsqr` folds into the symbol. Because `dsqr` rides
   `ctx.params` as a traced-but-constant leaf, the scaling happens at
   the **symbol** level (`Symbol._data` is a dynamic leaf, so a traced
   scalar is fine) — *not* via `ScaledOperator.eigenvalues`, which
   rightly refuses a non-constant operator coefficient.
3. **`eigenvalues` is layout-faithful.** `op.eigenvalues(grid, space)`
   produces a symbol **relative to the coefficient `space` passed**,
   reading `grid.wavenumbers(space, axis)` per factor — so it matches
   whatever transform layout the field uses (rfftn: one axis halved,
   the rest full; full-complex; partial). A symbol is only meaningful
   relative to a basis; this makes that explicit and dissolves the
   Wave-9B "composed half-spectra don't match the rfftn field" wall.
4. **Partial transforms + mixed `Fourier ⊗ Nodal` spaces are
   first-class.** This is the load-bearing commitment; everything below
   rides on it.

## The dynamic-data policy

**Operators stay static** (interned, array-free, jit-cache-stable) —
that is a feature, not the problem. **`Symbol` is the dynamic layer**:
all live data flows through symbols, never into operators. Two
consequences:

- **Coefficients enter by `Symbol × field`** (below), not by a
  field-valued operator.
- **Eigenvectors/projectors *are* symbols.** A per-mode projector
  `P_s = q_s ⟨q_s,·⟩_M` is a rank-1 `BlockSymbol` carrying the
  eigenvector as data. Expressing projections as block symbols (not
  hand-assembled `State`s) collapses the H0/H1 staggering-phase
  patchwork into one representation.

## The stack (four layers, each `@final`)

- **Operator** — static structure; emits symbols via `.eigenvalues`.
- **`Symbol`** — diagonal in *every* axis. Transformed axes: per-mode
  diagonal. Non-transformed axes: `Constant` (broadcast) **or a
  data-carrying physical axis** (a variable coefficient). Apply =
  Hadamard; invert = reciprocal.
- **`Banded`** — diagonal in the transformed axes, a **local stencil
  (band)** in one non-transformed axis (`∂y`, `∂z`, wall rows). The
  `grid/operators/banded.py` kernel promoted to a first-class operator
  with banded matvec / Thomas solve, O(N) per mode.
- **`BlockSymbol`** — dense `m×m` over field **components**; nests over
  `Banded` for the general system.

Coupling in a non-transformed axis is a ladder: **diagonal** (variable
coefficient `c(y)`) → **banded** (local stencil) → **dense**
(Chebyshev/global). `Symbol` owns the diagonal rung; `Banded` the
banded rung; a dense `y`-operator is `Banded` with full bandwidth.

### `Symbol × field` — the coefficient rule

`kx · FFT_x(c(y) f) = kx · c(y) · FFT_x(f)` because multiplication by a
coefficient constant in `x` commutes with `FFT_x`. So:

> `Symbol.__mul__(field)` is legal **iff `field` is `Constant` on every
> *transformed* factor of the symbol**; it may vary on the symbol's
> `Constant`/nodal factors. The result adopts the field's space there
> (`Constant(y) → Nodal(y)`), and applies in the mixed representation
> (Fourier in `x`, physical in `y`).

`dsqr` is the degenerate case (constant everywhere); `f(y)`, `N²(z)`,
a wall profile are the general case — one code path.

### Nesting: `BlockSymbol` of `Banded`, densify only at `eigh`

The general variable-coefficient, wall-bounded system is a `BlockSymbol`
whose `(i,j)` entries are `Banded` `y`-operators — block over
components (sparse: Coriolis `u↔v`, buoyancy `w↔b`), banded over space.
Keep both sparsities for **solves** (block-Thomas, near-linear). Only at
the **`eigh` boundary** materialize the per-mode `(m·N_y)×(m·N_y)` dense
matrix (eigenvectors are global in `y`). A flat `(m·N_y)` matrix
everywhere would densify both and cost O((m·N_y)³) — rejected.

## The realized-map algebra — grades, `@`, explicit materialization

Everything that maps `ScalarField → ScalarField` is **one concept** with a
domain/codomain **space tag** and a `@` (function composition, well-typed
iff `B.codomain == A.space`). The tag *encodes the representation*
(`Nodal` / `Fourier` / mixed `Fourier ⊗ Nodal`), so a representation
mismatch — a spectral `Symbol` against a physical stencil with no
transform between — is a **tag error, caught for free**. The three
things differ only in **exposed structure**, a lattice:

- **symbolic operator** (recipe) — grid-free, static, composes to a
  `Composite`; materialized via `.eigenvalues(grid, space)`.
- **structured realized** (`Symbol`, `Banded`, `BlockSymbol`) —
  grid-bound, dynamic data; composes by Hadamard / band / matmul; cheap
  inverse.
- **opaque realized** — an arbitrary callable chain; a valid linear map
  but no exploitable structure, no cheap inverse.

`A @ B` yields the **weakest grade** of its operands (a lattice meet).
Consequences:

- **`SpectralSolve` is not a class** — it is a *composition* of realized
  maps, `backward @ symbol.inverse() @ forward`. The Phase-D′
  `SpectralPressureSolver` should become a thin constructor returning
  that value (a follow-up reframe, not a rewrite). **The implementation
  path is [`../plans/active/composition_refactor_plan.md`](../plans/done/composition_refactor_plan.md)**
  — the realized-map category + the shared composition core.
- **Materialization is explicit (decision A, 2026-07-08).** A bare
  `symbol @ recipe` **raises** a taught error pointing at
  `recipe.eigenvalues(grid, space)`. Crossing recipe → structured binds
  a grid *and* asserts diagonalizability; both must be visible, never
  silent. `.eigenvalues` is the one labelled door.
- **Physical↔spectral duality.** For a diagonalizable `A`,
  `transform @ A_physical == A_symbol @ transform` is the
  change-of-representation rule; `Symbol × field` (the `kx·c(y)` case) is
  its coefficient-carrying special case.
- **Symmetry with `StateTransform`.** This realized-map algebra on
  `ScalarField` mirrors the `StateTransform` algebra on `State`
  (grid/model-bound, `@`/`+`/`.complement`, applies) — the same pattern
  at two levels, bridged by materialization. *(Open: whether the two are
  literally one abstraction parameterized by field-vs-state, or two
  parallel ones — deferred.)*

## What this dissolves

1. **Pressure solver** — `SpectralSolve(Div @ Diag(1,1,1/dsqr) @ Grad)`;
   layout-faithful `.eigenvalues(coeff_space).inverse()`; no hand-written
   `k̂²`; works for any diff scheme; `dsqr` as a symbol-level coefficient.
2. **Projections** — eigenvectors/projectors as `BlockSymbol`s; one
   staggered representation instead of three.
3. **β-plane / variable `N²` / walls** — `Symbol×f(y)` (diagonal) +
   `Banded` (stencil) + `BlockSymbol` (components), nested, in the mixed
   representation — the object `boundary_emission` builds by hand.

## Re-scoping (see the roadmap)

- **Pressure solver (near-term):** layout-faithful `eigenvalues` +
  `Symbol × field` + `SpectralSolve` with the `dsqr` metric → retire
  `discrete_laplace_symbol`. Bounded; refines roadmap Phase D / Wave 9B.
- **Phase I (general tier):** the `Banded` type (promote `banded.py`) +
  `BlockSymbol`-of-`Banded` nesting + mixed-representation eigenmodes.
  Replaces the "needs Chebyshev/Shen" framing — Chebyshev/Shen is one
  `Banded` instance; the design is the ladder + nesting above.
