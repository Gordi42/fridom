---
status: done
date: 2026-07-13
---

# Projections & eigenmodes — the energy-metric design (record)

**Shipped.** The design below (P1–P4, and the general wall-bounded /
variable-coefficient tier) is realized in `fridom.model` and in both
model packages. This file is kept as the record of *why* the layer is
built around the energy metric; the build order and its phase-by-phase
status live in
[`projection_eigenmode_roadmap.md`](../active/projection_eigenmode_roadmap.md).

## The idea (load-bearing, unchanged)

Do not carry a separate left/projection eigenvector `p`. Make the
model's **energy inner product `⟨·,·⟩_M` first-class**; under it the
linearized operator is skew-adjoint, so eigenmodes are orthogonal and
every projection needs the right vector only:

    P_s z = q_s · ⟨q_s, z⟩_M / ⟨q_s, q_s⟩_M

The old-stack `vec_p` was never independent: `p_s = M q_s` normalized.
Making `M` explicit *derives* `p`, and the same `M` unlocks the
non-Fourier, variable-coefficient, and dynamical cases the
Fourier-diagonal design could not reach.

### Why (the math)

Linear dynamics `∂_t z = L z` conserving quadratic energy
`E = ½⟨z,z⟩_M`, `⟨z₁,z₂⟩_M = Σ_c w_c ∫ z₁_c* z₂_c dV`, `M` Hermitian
positive-definite (per-field weights × quadrature measure). Energy
conservation for all `z` ⟺ **`L` is `M`-skew-adjoint** (`Lᴴ M = −M L`).
Hence eigenvalues are pure-imaginary, distinct-`ω` eigenvectors are
`M`-orthogonal, and the spectral projector needs only `q_s`.

The weights are the model energy: nonhydro `M = diag(1, 1, δ², 1/N²)`
on `(u,v,w,b)`; shallow water `M = diag(1, 1, 1/c²)` on `(u,v,p)` —
literally the `δ²w²`, `b²/N²`, `p²/c²` factors of the energy
diagnostics, i.e. exactly the weights the hand-written `vec_p`
formulas used to bury.

Because `L` is `M`-skew-adjoint, `H := i M L` is **Hermitian** and
`L q = iω q ⟺ H q = μ M q`. Solving the **generalized Hermitian**
eigenproblem (Cholesky-whiten the diagonal `M`, then standard `eigh`)
instead of a non-symmetric `eig` gives real `ω`, `M`-orthonormal `q`,
and an orthonormal basis of every degenerate eigenspace (the vortical
nullspace, the `±ω` pairs) for free. This was decision 1, adopted.

The metric is also the unifying object one tier up: the dynamical
(no-eigenbasis) transforms use `⟨·,·⟩_M` for convergence norms,
base-point coordinates, and the Rayleigh-quotient
`λ = ⟨Lz,z⟩_M / ⟨z,z⟩_M`.

## What shipped

- **`fr.EnergyMetric`** (`model/energy.py`): the per-component weight
  diagonal with `inner`/`norm`, sourced from the model
  (`EnergyMetric.from_model`, `allow_field_weights=True` for profile
  weights). Model energies: `nonhydro2/energy.py`,
  `shallowwater2/energy.py`.
- **`p = M q` is derived, never written.** `fr.spatial.rayleigh_dual`
  (`spatial/symbols.py`) computes `p_c = w_c q_c / Σ_c w_c |q_c|²` from
  the analytic column and the metric weights; the analytic Tier-0
  eigenmodes (`nonhydro2/eigenmodes.py`, `shallowwater2/eigenmodes.py`)
  are operator-sourced (built from the discrete operator symbols) and
  expose `projector(s)` / `mode` / `function` on coefficient states.
- **Numeric Tier-1** (`model/eigen.py`, `numeric_eigenpairs` /
  `NumericEigenmodes`): `L(k)` from a transfer-function probe of
  `fr.linearize(model).tendency`, nonhydro constraints handled by
  probing the Leray projector (`P S P`), then batched `eigh(iML, M)`.
- **Wall-bounded / variable-coefficient tier**
  (`model/eigen_channel.py`, `ChannelEigenbasis` / `channel_eigenpairs`,
  plus `model/_eigenbasis.py` and the per-package
  `channel_eigenmodes.py`): dense-column channel engine, beta-plane
  `f(y)`, profile weights `csqr(y)` / `N²(y)`, family/predicate
  projections, `fr.eigenbasis` dispatch.
- **Tier-1 projections and Tier-2 dynamical transforms**
  (`model/transforms/`): `StateTransform` algebra plus
  `Projection`, `Propagator`, `TimeAverage`, `OptimalBalance`,
  `BalanceExpansion` (the NNMD rewrite); `nonhydro2/transforms.py`,
  `shallowwater2/transforms.py` expose the family projections.
- Tests: `tests/model/{test_energy,test_eigen,test_eigen_channel,
  test_eigenbasis}.py`, `tests/model/transforms/`, and the per-package
  `test_eigenmodes` / `test_channel_eigenmodes` / `test_transforms` /
  `test_walled_eigenmodes` files.

## Decisions, as resolved

1. **`eigh(iML, M)`, never `eig(A)`** — adopted; the numeric and channel
   engines both solve the generalized Hermitian pencil.
2. **`EnergyMetric` first-class, `p = M q` derived** — adopted; no
   hand-written `p` exists in the new stack (`vec_p` survives only in
   the old `framework`/`nonhydro`/`shallowwater` packages, retired by
   the cutover).
3. **Keep the analytic fast path *and* the numeric path, both feeding
   one `M`-based projector** — adopted.
4. **`L` from an operator-algebra `BlockSymbol`** — *reversed*. It was
   built (`framework2 round 6 (H1)`) and then deleted in `d3309640`
   ("drop the linear-block IR and analytic BlockSymbol spectral path"):
   nothing consumed it — the models keep operator-sourced analytic
   eigenmodes, the pressure projection is a hand-composed
   `Div @ Lap⁻¹ @ Grad`, and the numeric probe covers the general case
   without symbolic assembly. Scalar `Symbol` (the spectral solve)
   stays.

## Open

Nothing in this design. Two descendants are tracked elsewhere:

- the `sw.eigenbasis` gallery example (β slow-mode filtering), deferred
  by owner request — `docs_examples_plan.md`;
- the spectral-vertical (Chebyshev/Shen) variant of the general
  eigenbasis, which is a `Banded` operator question —
  `composition_refactor_plan.md`.
