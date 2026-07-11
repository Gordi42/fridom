---
status: active
date: 2026-07-08
superseded_by: ../../decisions/symbol_stack_design.md
---

# Operator symbols — the spectral-solve substrate (design + plan)

**Status: design synthesis + staged plan, awaiting owner sign-off on
the decisions in §7 (2026-07-08).** Owning class doc for the type:
[`../../specs/grid/classes/operators_base.md`](../../specs/grid/classes/operators_base.md) §"Symbol"
(`:801-902`) — this note refines that sketch and adds the solver
architecture. Backed by this session's research on the symbol type,
the spectral solver, and (deferred) the eigen layer.

> **Refined by [`../../decisions/symbol_stack_design.md`](../../decisions/symbol_stack_design.md)
> (2026-07-08):** `Symbol` is the dynamic layer (coefficients enter via
> `Symbol × field`; operators stay static); `eigenvalues` is
> layout-faithful to the coefficient space passed; partial transforms +
> mixed `Fourier ⊗ Nodal` spaces are first-class; and the banded /
> non-transformed axis is a separate `Banded` type. Read that note for
> the current stack; §4-5 here stand as the diagonal-case detail.

## Scope

**In scope:** the scalar `Symbol` type + its algebra, and the spectral
solver it powers (Poisson / Helmholtz, and the mixed diagonal × banded
vertical solve).

**Deferred — handled later (seam left open):** the block/eigenvalue
layer — `BlockSymbol`, per-mode eigendecomposition, `Eigenmodes` /
`from_model`, and the Vortical/Wave/Divergence projections + Propagator
(ROADMAP 2.8). This note does **not** settle any of that. The only
forward-compat commitment it makes: scalar `Symbol` stays `@final`, and
the future block case is a **separate** type — so nothing here has to
change when the eigen layer lands.

## 1. The framing (scoped)

Everything the solver needs is a **per-mode operation on a diagonal
operator in a diagonalizing (Fourier / sine / cosine) basis**,
expressed as a `Symbol`:

| Task | Per-mode object | Operation |
|---|---|---|
| Poisson / Helmholtz / pressure solve | scalar `Symbol` `−|k|²` (or `−(k²+λ)`) | `Symbol.inverse(where_zero)`, Hadamard-applied |
| IMEX implicit vertical solve | banded `(1 − dt·γ·L_z)` | per-column banded solve |
| mixed `Fourier(x,y) ⊗ Chebyshev(z)` | horizontal `Symbol` + vertical band | per-mode banded z-solve |

The diagonal abstraction holds until a factor is non-diagonalizing
(Chebyshev), where it degrades to a banded vertical solve — the one
place `Symbol` stops (§5). (The `m > 1` block generalization is the
deferred eigen layer.)

## 2. Current seam state

- The **composition algebra is already coded** against a `Symbol` that
  doesn't exist yet, and starts working the moment `Symbol` implements
  `@`/`+`/`*`: `_chain_eigenvalues` = product (`base.py:1531-1541`),
  `OperatorSum.eigenvalues` = sum (`:989-995`),
  `ScaledOperator.eigenvalues` = scale iff scalar coeff (`:1050-1056`).
- `Operator.eigenvalues(grid, space)` raises `EigenbasisError` in the
  base (`:197-226`); every concrete operator's `eigenvalues` currently
  inherits the raising base ("designed-for until the Symbol cluster
  lands").
- `operators/symbol.py` is a 9-line stub.

So implementing scalar `Symbol` is small and high-leverage: it lights
up the already-wired algebra and delivers the pure-diagonal solver.

## 3. `Symbol` — the type

A thin wrapper over one materialized diagonal array.

- `@partial(fr.utils.jaxify, dynamic=("_data",))`; **static** aux =
  `(space, codomain)` interned tags, **dynamic** leaf = `_data`, the
  per-mode diagonal `jax.Array`.
- **Eager materialization** inside `eigenvalues(grid, space)` — the
  grid is always present there (needs `grid.wavenumbers`/`grid.measure`),
  and the formulas are the one-line `k_hat`/`one_hat` successors of v1
  `discrete_spectral_operators.py`. No lazy expression tree.
- **Broadcast = JAX size-1 broadcasting.** `grid.wavenumbers(space)`
  returns the diagonal pre-shaped with `ConstantSpace` (shape-`(1,)`)
  factors, so a per-factor symbol's `_data` is `[1,…,n_k,…,1]` and the
  `Identity ⊗ D` diagonal-operator extension is plain jnp broadcasting.
  This is why a `Symbol` is **not** a `ScalarField`: same array,
  opposite lift rule (no delta-into-zero-mode field lift).
- **Retagging** (`codomain ≠ space`, e.g. FD `Center→Right`) is carried
  by the two tags; the leaf is diagonal in the shared physical mode
  index. Composition `A @ B` requires `B.codomain == A.space`,
  multiplies leaves, sets `space=B.space, codomain=A.codomain`. `+`/`*`
  require matched `(space, codomain)` and factor-wise union the space
  tags (`Constant ⊗ X → X`) before broadcasting the leaves — this is
  the `kx² + ky²` Laplacian sum.
- **The honest discrete Laplacian is `bwd @ fwd`** (real `−k̂²`), not
  `fwd**2`; `**` forbids itself when `codomain ≠ space`.
- **`inverse(where_zero=0.0)`** is the solve primitive: maps every
  **structural** zero — the Poisson `k=0` nullspace *and* the rfft
  Nyquist-zeroed modes — to `where_zero` via an exact `== 0` test
  (these zeros are set exactly by the operators, not rounding), and
  flips the tags. No caller-side masking.
- **Nyquist / real-FFT:** on an `fr.Real` origin with even `n`, origin-
  staggering symbols (`PhaseShift`, `SincShift`, FD/interp) zero the
  Nyquist leaf entry (the one documented non-exact DOF). `inverse` must
  not divide it; `__call__` into an `fr.Real` field lands on the field
  factory's reality projection rather than re-implementing it.
- `__call__(f)` = Hadamard multiply on the strict same-space
  precondition, emitting on `codomain` (reuse the `spectral.py`
  `_diagonal_result` pattern).

## 4. Iteration-1 symbol table

`k = grid.wavenumbers(space)`, `dx = grid.measure(space)`; order-2
successors of `discrete_spectral_operators.py`.

| Operator | leaf (order 2) | retag? | domain → codomain |
|---|---|---|---|
| `FiniteDifference` fwd | `i·2sin(k dx/2)/dx · e^{i k dx/2}` (`i·k_hat`) | yes | `Fourier(Center) → Fourier(Right)` |
| `Laplacian` = `bwd @ fwd` | `−2(1−cos k dx)/dx²` (real) | no | `Fourier(Center) → Fourier(Center)` |
| `LinearInterp` | `cos(k dx/2) · e^{i k dx/2}` (`one_hat`) | yes | `Fourier(Center) → Fourier(Right)` |
| `SpectralDerivative` | `i k` | no | `Fourier(o)→Fourier(o)`; `Sine→Cosine`; Cheb→raise |
| `PhaseShift(to)` | `e^{i k δ dx}` (Nyquist-zeroed real even-n) | yes | `Fourier(A) → Fourier(to)` |
| `SincShift(to)` | `sinc(k dx/2)` (× phase if offset) | yes | `Fourier(avg) ↔ Fourier(nodal)` |
| `FluxDifference` | `i k sinc(k dx/2)` | yes | `Fourier(Right) → Fourier(CellAvg)` |
| `DualFluxDifference` | `i k sinc(k w/2)` (dual `w`) | yes | `Fourier(Center) → Fourier(FaceAvg)` |
| `FaceDifference` | `2i sin(k dx/2)/dx · sinc/phase` | yes | `Fourier(CellAvg) → Fourier(face)` |
| `LinearReconstruction` | sinc-corrected averaging | yes | `Fourier(CellAvg) → Fourier(face)` |
| `Fourier.truncation_mask` | `0/1` band mask (2/3 rule) | no | `Fourier(o) → Fourier(o)` |

**`EigenbasisError` boundary (raise, correct):** Chebyshev (recurrence
couples all modes → non-diagonalizing); nonlinear / WENO / limiters;
`ScaledOperator` with a **field** coefficient (breaks translation
invariance). A composite has a symbol iff *every* factor does on
compatible bases.

## 5. The diagonal → banded boundary: `SpectralSolve`

**Pure-diagonal solve (all Fourier/sine/cosine)** is complete with
`Symbol` alone: build the elliptic operator → `.eigenvalues(grid,
coeff_space)` (product/sum/scale composition) →
`sym.inverse(where_zero)` → wrap in forward/backward transforms. This
is a faithful, better-typed port of v1 `spectral_pressure_solver.py` /
`rfft_pressure_solver.py` (the manual `jnp.where(k²==0, 0, 1/k²)`
becomes `Symbol.inverse`; the hand-picked rfft axis + DST/DCT axes
become the space+transform layer). Helmholtz `(∇²−λ)` = `sym − λ`
diagonal, no nullspace when `λ≠0`.

**Mixed `Fourier(x,y) ⊗ Chebyshev(z)`** block-diagonalizes into a
per-`(kx,ky)`-mode banded solve `(L_z + s(kx,ky)·I) p̂ = rĥs`. Design:

- **`SpectralSolve`** — a grid-bound operator that at construction runs
  `elliptic.eigenvalues(grid, space)` **per factor**, catching
  `EigenbasisError` to partition factors into diagonalizing (folded
  into a horizontal `Symbol` shift) vs non-diagonalizing (a vertical
  banded operator whose band it assembles per mode). If every factor
  diagonalizes it degrades to the pure-`Symbol.inverse` path — one
  class, static (compile-time) partition. Reject: overloading `Symbol`
  with a band (breaks its Hadamard identity); forcing it through
  `Block`.
- **Pencil schedule** rides the existing `layout_for`/`redistribute`
  transform machinery: horizontal transforms per axis, then z-local for
  the batched banded solve, then inverse transforms. Single-jit-clean;
  the partition and pencil order are static. The `RFFTPressureSolver`
  hand-rolled transposes become negotiated reshards.
- **Shared primitive with IMEX.** The banded z-solve is the **same**
  "assemble band + batched banded solve on a local axis" kernel as the
  IMEX `(1 − dt·γ·L_z)⁻¹` vertical-diffusion solve
  (`model/implicit.py`), differing only in batch index (`(kx,ky)` modes
  vs `(x,y)` columns) and the added scalar shift. **Lift that primitive
  out of `model/implicit.py` into a `grid/operators/banded` module** so
  the pressure solve, the implicit vertical diffusion, and the implicit
  free-surface 2-D Helmholtz all consume one kernel.
- **API:** a module (e.g. the nonhydro pressure projection, a CONSTRAINT
  stage) constructs `SpectralSolve(elliptic, grid=...)` once at setup
  and applies it in the stage; no hand-built shardings (the "no bypass"
  rule). An optional `grid.solve(elliptic, rhs)` thin wrapper for
  one-shot host-side use.

## 6. Staging

- **S1 — scalar `Symbol`** (the "Wave 3B" cluster): the type + algebra
  (`@`/`+`/`*`/`**`/`inverse`/`conj`/`__call__`), the per-operator
  `eigenvalues` formulas (§4), materialization on grid wavenumbers.
  Validate against `../../specs/grid/05_validation.md` eigenvalue identities (`i k sinc`
  = "average of d/dx", the `bwd @ fwd` real Laplacian, etc.). Delivers
  the pure-diagonal pressure solver.
- **S2 — `grid/operators/banded` primitive + `SpectralSolve`**: lift the
  banded-solve kernel from `model/implicit.py`; build `SpectralSolve`
  (diagonal/banded partition + pencil schedule). Delivers the mixed
  Fourier×Chebyshev pressure/implicit solves. Feeds ROADMAP 2.7's
  pressure projection.
- **S3 (deferred) — the eigen/projection layer** (`BlockSymbol`,
  `Eigenmodes`, projections): out of scope here, handled later.

## 7. Decisions for owner sign-off (scoped)

- **A. Scalar `Symbol` `@final`; the future block case is a separate
  type.** (The block type itself is deferred.)
- **B. Eager materialization** of `Symbol._data` at `eigenvalues()`
  call — vs lazy. Recommended eager.
- **C. `SpectralSolve` as the diagonal/banded composition root, and lift
  the banded-solve primitive from `model/implicit.py` into
  `grid/operators`** so the pressure solve and IMEX share one kernel.
  This crosses the model/grid boundary — wants explicit sign-off.
- **D. Restrict `SpectralSolve` to separable elliptic operators** in
  iteration 1 (variable-coefficient / terrain-following → deferred
  iterative solver).
- **E. `inverse` uses exact `== 0`** (structural zeros), no floating
  tolerance; `__add__`/`__mul__` own the factor-wise space-union;
  `__call__` into `fr.Real` delegates to the field reality projection.

(Deferred with the eigen layer: the eigh-vs-eig / energy-metric choice,
analytic-vs-numeric `Eigenmodes`, and the `from_model` linear-operator-
matrix surface requirement — none decided here.)
