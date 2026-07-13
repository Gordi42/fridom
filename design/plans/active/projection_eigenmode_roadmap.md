---
status: active
date: 2026-07-13
---

# Projection / eigenmode build roadmap (dependency-ordered)

**Status (2026-07-13): phases A–H have landed; only phase I remains,
and it is unscheduled.** The roadmap stays open as the tracker for that
last tier — [`composition_refactor_plan.md`](../done/composition_refactor_plan.md)
explicitly hands the `Banded` / mixed-representation work here — and as
the record of which decisions the build confirmed and which it reversed.

The energy-metric design it sequenced is in
[`projection_eigenmode_plan.md`](../done/projection_eigenmode_plan.md) (P1–P4);
the symbol substrate in
[`operator_symbols_plan.md`](../done/operator_symbols_plan.md) (S1–S2) and
[`../../decisions/symbol_stack_design.md`](../../decisions/symbol_stack_design.md).

## Landed

- **A — `EnergyMetric` + State inner product.** `model/energy.py`:
  diagonal `M`, `apply`/`inner`/`norm`, `EnergyMetric.from_model`
  sourcing the weights from the model energy (scalar *and* profile
  weights). `258d5b09`
- **C — scalar `Symbol` + per-operator `eigenvalues`.** `@final`
  `spatial/operators/symbol.py` (`@ + * ** inverse conj __call__`,
  `Symbol × field`); `eigenvalues` filled across the spectral, finite-
  difference, interp and mapped operator families. `258d5b09`
- **B — eigenmodes `p = M q`.** Hand-written `vec_p` gone from both
  models; `p` is `fr.spatial.rayleigh_dual(q, metric)` under the model
  energy metric (`nonhydro2/eigenmodes.py`, `shallowwater2/eigenmodes.py`).
  `0002031b`
- **D — `SpectralSolve` + banded primitive.** `spatial/operators/`
  `spectral_solve.py` + `banded.py`; later refactored to the composition
  `backward @ symbol.inverse() @ forward` (`0ce2663f`). `0002031b`
- **D′ — the pressure solve rides the symbol layer.** The hand-rolled
  `discrete_laplace_symbol` is retired; `nonhydro2/modules/pressure.py`
  builds `∇² = Div @ Diag(1,1,1/dsqr) @ Grad` and inverts it through
  `SpectralSolve`. `be036e7d`
- **E — `StateTransform` algebra.** `model/transforms/`: base,
  signature, info, `@`/`+`/`.complement`, `Identity`/`Shift`/
  `FixedPoint`, `relative_l2`/`assert_idempotent`, plus the model hooks
  (`fr.linearize`, `model.variant`, `model.tendency`). `0002031b`
- **F — Tier-1 projections.** `EigenProjection`/`EigenFunction`/
  `ProjectionFactory` (`model/transforms/projection.py`) and the
  vortical / wave / kelvin / divergence builders in `nh.transforms` /
  `sw.transforms`. `6f8c8556`
- **G — Tier-2 dynamical transforms.** `Propagator`, `TimeAverage`,
  `OptimalBalance` (ramping to the model's nominal rossby number,
  `94c18579`). `6f8c8556`
- **H0 — numeric eigenmodes `eigh(iML, M)`.** `model/eigen.py`:
  transfer-function probe of `fr.linearize(model).tendency` on unit
  impulses, Cholesky-whitened generalized Hermitian eigensolve. The
  nonhydro constraint is handled by probing the Leray projection through
  the public `model.constrain` matvec and forming `P S P` (`048a8356`).
  `6f8c8556`
- **Wall-bounded eigenmodes, numeric route.** `model/eigen_channel.py` +
  `model/_eigenbasis.py`: the dense-column tier for a grid with exactly
  one bounded axis — per periodic wavenumber a dense `(m·N)×(m·N)`
  `eigh(iMS, M)`, serving coefficients that vary along the bounded axis
  (β-plane `f(y)`, `csqr(y)`, `N²`). Wired as `fr.eigenbasis` with both
  model ports (`3c17b94b`, `08368c03`, `5c8f7906`, `b16d294e`,
  `c18ce537`); `numeric_eigenpairs` rejects walled grids with a taught
  error (`f465401e`). This delivers most of what phase I was written for,
  by the numeric path rather than the symbolic/banded one.
- **NNMD, previously descoped, shipped.** `fr.transforms.BalanceExpansion`
  (`86d807b0`, `0338f4e5`, `859eab32`); see
  [`nnmd_rewrite_plan.md`](../done/nnmd_rewrite_plan.md).
- **`variant`/`bind` lifecycle bug fixed** (2026-07-10) and the eigenmode
  **frequency sign flipped to the standard convention** (positive `ω`
  propagates along `+k`, `c9d2d606`).

## Reversed

- **H1 — symbolic `BlockSymbol` L-assembly. Built, then deleted.**
  Landed as `026f4c62` (BlockSymbol `L(k)`, Leray as a block symbol,
  `BlockMatrix.eigenvalues`, the linear-block IR and `TendencyTerm.blocks`)
  and removed wholesale by `d3309640`: nothing consumed it — both models
  keep their analytic `eigenmodes.py`, the production pressure projection
  is a hand-composed `Div @ Lap⁻¹ @ Grad`, and the numeric probe (H0)
  covers the general case with no symbolic metadata at all. **Decision 4
  of the design note (`L` from the operator-algebra `BlockSymbol`) is
  therefore withdrawn**, and with it
  [`../../decisions/blocksymbol_l_assembly.md`](../../decisions/blocksymbol_l_assembly.md)
  and [`linear_term_blocks_plan.md`](../../archive/linear_term_blocks_plan.md) — both
  describe work that was tried and rejected. Any future block algebra
  starts from the `RealizedMap` protocol, not the deleted class.

Decisions 1–3 (`eigh(H,M)` never `eig`; `M` first-class with `p = M q`
derived; analytic Tier 0 *and* numeric Tier 1 feeding one projector) are
all confirmed in the shipped code.

## Remaining — phase I: the banded / mixed-representation tier

Not scheduled. It waits on an actual demand for boundary-trapped or
vertical-structure modes that the dense-column channel engine cannot
serve; the numeric route above already covers a single bounded axis with
variable coefficients. In dependency order:

1. **`Banded` as a first-class `RealizedMap`.**
   `spatial/operators/banded.py` is still free functions
   (`second_difference_matrix`, `apply_along_axis`, `solve_along_axis`),
   consumed only by `model/implicit.py`. Promote it to a
   diagonal-in-transformed / banded-in-one-axis operator with matvec +
   Thomas solve, on the realized-map layer
   ([`composition_refactor_plan.md`](../done/composition_refactor_plan.md)
   shipped that layer; its §4 design sketch, `git show e1e8e537`, is the
   reference for the type).
2. **Mixed `Fourier(x,y) ⊗ Chebyshev/Nodal(z)` solve.** The per-mode
   banded z-solve that `SpectralSolve` documents as deferred. Needs (1);
   the mixed transforms and `Symbol × field` it also needs are already in
   (`d7802707`, `0ce2663f`).
3. **General eigenmodes past the channel tier.** Two bounded axes, or a
   spectral-vertical (Chebyshev) column, densified only at the `eigh`
   boundary. Needs (1)+(2); the spectral-vertical variant additionally
   needs Chebyshev quadrature + a Shen/Galerkin BC-structured basis —
   which the FD-vertical / structure-function path does **not**.

Smaller open items:

- **Rayleigh-quotient frequency/growth diagnostic** for Tier 2
  (`λ = ⟨Lz,z⟩_M / ⟨z,z⟩_M`). Sketched in the design note, never built;
  `fr.spatial.rayleigh_dual` is the dual vector, not the quotient. A
  handful of lines on top of `EnergyMetric.inner` whenever it is wanted.
- **Docs / gallery example for the eigenmode surface** (deferred by owner
  request; do when asked). A sphinx-gallery example around
  `sw.eigenbasis` — β-plane slow-mode filtering as the showcase —
  covering `em.mode` / `eb.mode`, `random_vortical` / `random_waves` /
  `random_state(..., spectral_energy_density=...)`, and the family
  projections. Tracked with the other gallery work in
  [`docs_examples_plan.md`](docs_examples_plan.md).
