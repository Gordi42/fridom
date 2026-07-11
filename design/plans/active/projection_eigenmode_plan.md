---
status: active
date: 2026-07-08
---

# Projections & eigenmodes — the energy-metric design (research + plan)

**Status: design synthesis, awaiting owner sign-off (2026-07-08).**
Settles the eigen/projection layer deferred by
[`operator_symbols_plan.md`](operator_symbols_plan.md) (S3) and the
open decisions there (§7 "eigh-vs-eig / energy-metric / analytic-vs-
numeric"). Backed by this session's study of two research codebases
that already run the target design — `Adiabatic-Coriolis-Ramping` and
`boundary_emission` — plus the v1 projection layer and the current
`nonhydro2/shallowwater2` eigenmode ports.

## 0. The one idea

Stop carrying a separate **left/projection eigenvector `p`**. Instead
make the model's **energy inner product `⟨·,·⟩_M` first-class**. Under
it the linearized operator is **skew-adjoint**, so eigenmodes are
orthogonal and every projection is

    P_s z = q_s · ⟨q_s, z⟩_M / ⟨q_s, q_s⟩_M          (right vector only)

The v1 `vec_p` was never independent: it is exactly `p_s = M q_s`
normalized. Making `M` explicit *derives* `p`, and the same `M`
unlocks the general (non-Fourier, variable-coefficient, dynamical)
cases the current Fourier-diagonal design cannot reach.

## 1. The math (why `p` disappears)

Linear dynamics `∂_t z = L z`; the model conserves quadratic energy
`E = ½⟨z,z⟩_M`, `⟨z₁,z₂⟩_M = Σ_c w_c ∫ z₁_c* z₂_c dV` with `M`
Hermitian positive-definite (per-field weights `w_c` × quadrature
measure). Energy conservation for all `z` ⟺ **`L` is `M`-skew-adjoint**
`Lᴴ M = −M L`. Therefore:

- eigenvalues are pure-imaginary, `L q_s = iω_s q_s`, `ω_s` real;
- distinct-`ω` eigenvectors are `M`-orthogonal, `⟨q_s,q_{s'}⟩_M = 0`;
- the spectral projector needs only `q_s` (formula above).

**`p = M q`.** Define `p_s = M q_s / ⟨q_s,q_s⟩_M`; then
`P_s z = q_s ⟨p_s, z⟩` (plain Hermitian dot) and `⟨p_s,q_{s'}⟩ = δ`.
So the existing per-wavenumber `project(p, q)` machinery is reused
*unchanged* — we only stop hand-writing `p` and compute `M q` instead.

**Energy weights (already in the code, as `state.ekin/epot`).**
Nonhydro `M = diag(1, 1, δ², 1/N²)` on `(u,v,w,b)`; shallow water
`M = diag(1, 1, 1/c²)` on `(u,v,p)`. These are literally the `δ²w²`
and `b²/N²` (resp. `p²/c²`) factors in the v1 energy diagnostics — and
exactly the `1/c²`/`γ`/`N²` weights the v1 `vec_p` formulas bury by
hand.

**Validated in practice.** `boundary_emission/projection.py` and
`Adiabatic-Coriolis-Ramping/projection.py` both implement precisely
`⟨z,q⟩_M q / ⟨q,q⟩_M` with an explicit `_inner_product` and no `p`;
their tests confirm mutual orthogonality of mode families and complete
reconstruction (`z = Σ_family P_family z`).

## 2. Obtaining the eigenpairs — a generality ladder

The eigenproblem `L q = iω q` restricted to each mode. Periodic axes
diagonalize (FFT → per-wavenumber); a non-periodic axis (wall) or a
non-diagonalizing one (Chebyshev vertical) couples all `N` points
along it, so the per-mode block grows from `m×m` to `(m·N)×(m·N)`
banded/dense. **Same diagonal-vs-banded partition `SpectralSolve`
already makes** ([`operator_symbols_plan.md`](operator_symbols_plan.md)
§5) — the eigen layer is its block-valued twin.

**Tier 0 — analytic diagonal (fast path, current).** All axes
periodic + constant coefficients ⇒ `L(k)` is a small `m×m`; closed-form
`q_s(k)` known (`nonhydro2/shallowwater2 eigenmodes.py`). Keep it, but
reformulate the projector through `M` (drop `vec_p`).

**Tier 1 — numeric per-block eigensolve (the general spectral path).**
Because `L` is `M`-skew-adjoint, `H := i M L` is **Hermitian**
(`Hᴴ = −i Lᴴ M = i M L = H`) and `L q = iω q ⟺ H q = −ω M q`. Solve the
**generalized Hermitian eigenproblem `H q = μ M q`** (`eigh`, or
Cholesky-whiten `M = RᴴR` then standard `eigh` on `R⁻ᴴ H R⁻¹`),
`ω = −μ`. This:
- returns **real `ω`** (non-symmetric `eig` on raw `A` leaks spurious
  growth from round-off — `eigh` cannot);
- returns **`M`-orthonormal `q` automatically**, including an
  orthonormal basis of each degenerate eigenspace (the `ω=0` vortical
  nullspace, the `±ω` pairs) — no ad-hoc degeneracy handling;
- covers **walls / boundary-trapped modes / vertical structure /
  variable coefficients along the block axis** — everything the
  analytic Fourier path cannot.

Tier 0 is the closed-form solution of this *same* eigenproblem, so both
tiers emit `M`-orthonormal `q` and feed one identical projector.

> **Decision 1 (recommend adopt): `eigh(H, M)`, never `eig(A)`.** This
> is the single decision `operator_symbols_plan.md` §7 left open; the
> skew-adjoint structure makes `eigh` both correct and strictly better.

**Tier 2 — dynamical (no eigenbasis).** Fully variable coefficients
coupling all axes (β-plane), or time-dependent ramping where no
eigenbasis is stationary. Use the model integrator:
adiabatic ramping (`forward @ base @ backward`), geostrophic
time-average, optimal balance. Eigenvalue/growth diagnostics via the
**Rayleigh quotient** `λ = ⟨Lz,z⟩_M / ⟨z,z⟩_M` (imaginary part =
frequency, real part = growth) — no eigensolve. `Adiabatic-Coriolis-
Ramping` runs exactly this (`AdiabaticProjection`, `ModeAnalysis`).

**The unifying thread is `M`.** Tier 0/1 projector = `q⟨q,·⟩_M/⟨q,q⟩_M`;
Tier 2 convergence norms, base-point coordinates, and Rayleigh
quotients all use `⟨·,·⟩_M`. One object spans the ladder.

## 3. Non-transformed directions

`⟨z₁,z₂⟩_M` = transform periodic axes (per-mode) → weight components by
`w_c` → **integrate the non-transformed axes with `grid.measure`
quadrature** → sum. The projector stays diagonal in the periodic modes
but becomes a full **column contraction** along non-transformed axes —
the essential step past "diagonal per wavenumber." Non-transformed
axes carry explicit **structure functions** (sine/cosine/boundary-
trapped-exp/linear in the research repos; or the numeric `eigh`
eigenvectors of Tier 1).

## 4. API (framework2)

Four pieces; the last three ride the **signed-off `StateTransform`
algebra** (`../../specs/model/08_state_transforms.md`: `@`, `+`, `.complement =
Identity() − self`, Tier-1 pytree / Tier-2 host).

1. **`fr.EnergyMetric` — a diagonal, self-adjoint, positive
   `StateTransform` `M: State → State`** built from the model's energy
   (the `state.ekin/epot` weights). Exposes `inner(z₁,z₂) =
   integrate(conj(z₁) * M(z₂))` and `norm`. **This is the missing
   first-class object; everything below consumes it.**

2. **`Eigenmodes`** (per-model package) holds `(ω_s, q_s)` States on
   coefficient spaces + a reference to `M`. Constructors:
   - analytic `Eigenmodes(grid, f0=…, n2=…, dsqr=…, discrete=True)`
     (Tier 0, the current ports);
   - numeric `Eigenmodes.from_operator(L, M, grid)` (Tier 1,
     `eigh(H,M)` per block);
   - `from_model(model, at_time=…)` — read `M` from the model energy,
     assemble `L` from the model's **linear tendency operators**, pick
     analytic vs numeric by the factor partition. Keep today's
     structural validation (constant scalars, Fourier-diagonalizable,
     ramp→`at_time`).
   - `q(s)` normalized `⟨q_s,q_s⟩_M = 1`; `p(s) := M(q(s))` (derived,
     for the `vec_p`-compatible projector); `omega(s)`; `projector(s)`.

3. **Tier-1 projections**: `VorticalProjection` / `WaveProjection` /
   `DivergenceProjection` as sums of `em.projector(s)` in the algebra —
   `WaveProjection = P(+1) + P(−1)`, `DivergenceProjection =
   (P_vortical + P_wave).complement`.

4. **Tier-2 dynamical**: `OptimalBalance = forward @ base @ backward`,
   `TimeAverage`, adiabatic `RampProjection` — `Propagator`-based,
   consuming `M` for norms and the Rayleigh-quotient diagnostic.

## 5. Connection to the symbol layer

- scalar **`Symbol`** (S1) = `1×1` per-mode eigenvalue → the pressure/
  Helmholtz **solve** (`Symbol.inverse`).
- **`BlockSymbol`** (S3) = `m×m` per-mode block = the linearized system
  `L(k)`, assembled by the **same product/sum/scale operator algebra**
  already wired for scalar `Symbol` (matrix product/sum). So
  `from_model` can build `L(k)` from the model's linear tendency
  operators instead of hand-coding `A(k)`.
- **`Eigenmodes` = eigendecomposition of a `BlockSymbol` under `M`**
  (`eigh(H,M)`). A non-diagonalizing axis makes the `BlockSymbol`
  banded — the **same partition + banded primitive** as `SpectralSolve`.

So the spectral substrate has two faces on one partition:
`Symbol.inverse` for **solves**, `eigh(·, M)` on the `BlockSymbol` for
**projections**.

## 6. Staging

- **P1 — `EnergyMetric` + reformulate the ported `Eigenmodes` to derive
  `p = M q`** (drop hand-written `vec_p`; keep analytic `q`). Pure
  refactor, no new capability; validate against the existing
  orthogonality/idempotency tests. De-risks `M`.
- **P2 — `BlockSymbol` + numeric `from_operator` (`eigh(H,M)`)** for the
  periodic `m×m` case; verify it reproduces the analytic Tier-0 modes.
- **P3 — non-periodic block axis** (structure functions / banded
  generalized `eigh`): walls, vertical Chebyshev. Reuses the
  `SpectralSolve` partition + banded kernel.
- **P4 — Tier-1 projection `StateTransform`s + Tier-2 dynamical**
  (ramping, time-average, optimal balance).

## 7. Decisions for sign-off

1. **`eigh(H,M)`, not `eig(A)`** — real ω, `M`-orthonormal `q`, free
   degeneracy handling (§2).
2. **`EnergyMetric` (`M`) is first-class; `p` is derived `M q`, never
   hand-written** (§0–1). Subsumes all current `vec_p`.
3. **Keep analytic Tier 0 as the fast path *and* add numeric Tier 1;
   both feed one `M`-based projector** — not either/or (§2).
4. **`from_model` builds `L` from the operator-algebra `BlockSymbol`**
   (elegant, couples to the deferred `BlockSymbol` type) **vs a
   model-provided `linear_operator()`** (pragmatic, decoupled). The one
   genuine fork — recommend `BlockSymbol` long-term, a model-provided
   linearization as the P2 bootstrap.
