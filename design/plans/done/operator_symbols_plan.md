---
status: done
date: 2026-07-13
superseded_by: ../../decisions/symbol_stack_design.md
---

# Operator symbols — the spectral-solve substrate (plan, closed)

**Outcome (2026-07-13).** The plan's own scope shipped: the scalar
`Symbol` type + its algebra (S1) and the `SpectralSolve` /
`operators/banded` substrate (S2) are in the tree and carry the
production nonhydro pressure solve. The deferred block/eigen layer (S3)
was built and then **withdrawn** — it is not pending work. What is left
of the design lives in other records:
[`../../decisions/symbol_stack_design.md`](../../decisions/symbol_stack_design.md)
(the `Banded` rung, mixed `Fourier ⊗ Nodal`, `BlockSymbol`-of-`Banded`),
[`composition_refactor_plan.md`](composition_refactor_plan.md) (the
realized-map algebra `SpectralSolve` now rides), and
[`fv_nonhydro_scoping.md`](../active/fv_nonhydro_scoping.md) §F0 (the four missing
finite-volume `eigenvalues`). This file is kept for the symbol table in
§3 and the decision record in §4.

## 1. Landed

Scalar `Symbol` (S1) — `src/fridom/spatial/operators/symbol.py`,
`@final`, `jaxify(dynamic=("_data",))`, eager materialization:

- Type + algebra (`@`/`+`/`*`/`**`/`inverse`/`conj`/`__call__`), the
  `(space, codomain)` retag tags, exact-`== 0` structural-zero
  pseudo-inverse — `258d5b09` (wave 9A), tests in
  `tests/spatial/operators/test_symbol.py`.
- `Identity.eigenvalues` + cross-axis (broadcast) composition —
  `bd613d02`; the tag validator `compose_spaces`/`union_spaces` —
  `04f59868`; `magnitude`/`sqrt` — `2852d134`; derived-shift alignment
  across the trig mode tables — `8fac3f6b`.
- `Symbol × field` coefficient rule + 0-d-array scalars (the `dsqr`
  metric enters at symbol level, not via `ScaledOperator`) — `be036e7d`,
  `309bbdd6`.
- Per-operator `eigenvalues` today: `FiniteDifference`, `LinearInterp`,
  `SpectralDerivative`, `PhaseShift`, `SincShift`, `MetricScaled`
  (`9c4fb988`), plus `Identity` / `Composite` / `SeparableComposite` /
  `OperatorSum` / `ScaledOperator` in `base.py` — the composition
  algebra that was dormant is live.

`SpectralSolve` + banded primitive (S2):

- `operators/banded.py` — the tridiagonal assemble/solve kernel lifted
  out of `model/implicit.py`, which now imports it (decision C) —
  `0002031b`; tests in `tests/spatial/operators/test_banded.py`.
- `SpectralSolve` — pure-diagonal elliptic solve; first as a class
  (`be036e7d`), then reframed as the composition
  `backward @ symbol.inverse() @ forward` over the `RealizedMap`
  protocol (`efe9c312`, `0ce2663f`), with `ComposedTransform` for mixed
  per-component products (`d7802707`).
- The hand-rolled `discrete_laplace_symbol` is gone: `nonhydro2/modules/
  pressure.py` and `mapped_pressure.py` build `Div @ Diag(...) @ Grad`
  and invert it through `SpectralSolve` (`be036e7d`, `e55640fd`).
- `spatial/symbols.py` — the `GridSymbols` kit (`diff`/`interp`/`move`
  symbols per axis), `ModeChart`, `rayleigh_dual` (`052023eb`,
  `d7802707`, `8433e956`); consumed by both models' `eigenmodes.py`.
- Everything moved package with the split `framework2 → fridom.spatial`
  / `fridom.model` (`0260d779`).

## 2. Withdrawn / superseded (not pending)

- **S3, the block/eigen layer.** `BlockSymbol` + `BlockMatrix.
  eigenvalues` + the linear-block term IR were implemented (`85e865f2`,
  `026f4c62`) and then **deleted** (`d3309640`): nothing consumed the
  analytic block spectral path — both models keep their own
  `eigenmodes.py` (written against scalar `Symbol`/`GridSymbols`), and
  the model-agnostic numeric probe (`numeric_eigenpairs`) covers the
  generic case. Any future block layer starts from
  `symbol_stack_design.md` (`BlockSymbol` of `Banded`, densified only at
  the `eigh` boundary), not from §5 here.
- **The diagonal/banded partition inside one `SpectralSolve` class.**
  Superseded: the symbol stack makes `Banded` a *type* (the ladder
  diagonal → banded → dense) and `SpectralSolve` a composition of
  realized maps. `SpectralSolve` today is diagonal-only; the mixed
  `Fourier(x,y) ⊗ Chebyshev(z)` solve is not built, and its design now
  belongs to the symbol stack's Phase I, not to this plan.
- **Decision D (variable coefficients → deferred iterative solver)**
  resolved the other way in practice: terrain-following /
  variable-coefficient pressure is the matrix-free preconditioned CG
  (`operators/krylov.py`, `1d8f17a5`, `64d1d9ae`, scan-converted in
  `c022d79e`), preconditioned *by* a `SpectralSolve`.

## 3. Remaining (owned elsewhere)

- **Four finite-volume `eigenvalues`** — `FluxDifference`,
  `DualFluxDifference`, `FaceDifference`, `LinearReconstruction` still
  inherit the raising base (their docstrings say "designed-for"). Closed
  forms, kept here as the reference
  ([`fv_nonhydro_scoping.md`](../active/fv_nonhydro_scoping.md) §F0 cites them):

  | Operator | leaf (order 2) | domain → codomain |
  |---|---|---|
  | `FluxDifference` | `i k sinc(k dx/2)` | `Fourier(Right) → Fourier(CellAvg)` |
  | `DualFluxDifference` | `i k sinc(k w/2)` (dual `w`) | `Fourier(Center) → Fourier(FaceAvg)` |
  | `FaceDifference` | `2i sin(k dx/2)/dx · sinc/phase` | `Fourier(CellAvg) → Fourier(face)` |
  | `LinearReconstruction` | sinc-corrected averaging | `Fourier(CellAvg) → Fourier(face)` |

  (`Fourier.truncation_mask` as a 0/1 band symbol is likewise unbuilt,
  and unclaimed.)
- **The `Banded` type + mixed `Fourier ⊗ Chebyshev` solve** —
  `symbol_stack_design.md`, Phase I. `banded.py` is still a pair of
  kernel functions, not an operator; `Chebyshev` correctly raises
  `EigenbasisError`.
- **The realized-map composition core** —
  [`composition_refactor_plan.md`](composition_refactor_plan.md).

The `EigenbasisError` boundary as designed still holds: Chebyshev,
nonlinear/WENO/limiters, and `ScaledOperator` with a field coefficient
raise; a composite has a symbol iff every factor does.

## 4. Decisions §7 — outcome

- **A. Scalar `Symbol` `@final`, block case a separate type** — held.
  `Symbol` is `@final`; the separate `BlockSymbol` came and went without
  touching it.
- **B. Eager materialization** at `eigenvalues()` — held; no lazy
  expression tree.
- **C. Lift the banded primitive from `model/implicit.py` into
  `operators/banded`** — done; `implicit.py` consumes it.
- **D. Restrict `SpectralSolve` to separable elliptic operators** — held,
  and the escape hatch is the PCG solver rather than a future
  variable-coefficient `SpectralSolve` (§2).
- **E. `inverse` exact `== 0`; `+`/`*` own the factor-wise space union;
  `__call__` into `fr.Real` delegates to the field reality projection** —
  all held as coded.
