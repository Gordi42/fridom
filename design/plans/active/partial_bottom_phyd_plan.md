# Partial-bottom-cell hydrostatic pressure gradient plan

Status: **approved 2026-07-19** (owner, in chat). Third immersed
residual (roadmap "Immersed partial cells — residuals").
Implementation branch: `feat/partial-bottom-phyd`.

## 1. Problem

The hydrostatic model diagnoses the hydrostatic pressure by a
downward cumulative integral with **full-cell** increments
(`hydrostatic/modules/core.py`, `_diagnose_p_hyd`:
`p_hyd = -CumulativeIntegral(direction="down", target="center",
jacobian=...)(b)`), regardless of immersed partial cells. At a
partial **bottom** cell the wet volume occupies only the upper part
of the cell, so the pressure value labeled "cell center" belongs to
a different physical height than the wet volume's actual centroid.
Adjacent columns cut at different depths then difference pressures
at mismatched heights, and with stratification the horizontal
gradient picks up a spurious force at exactly the bottom cells the
partial-cell machinery is supposed to improve: **a resting
stratified ocean over immersed bathymetry self-accelerates**. The
error is O(dz), confined to cut cells (2nd order everywhere else).
This is the classic partial-cell pressure-gradient error; Pacanowski
& Gnanadesikan (1998) is the canonical cure in MOM-family models.

## 2. Decisions

- **PB-D1 (wet first moment).** `ImmersedDomain` gains a memoized
  per-cell **wet-centroid offset** quadrature (suggested name
  `centroid_offset(space)`; final name follows house conventions):
  `delta = (first vertical moment of the wet region)/(wet volume)
  - z_center`, computed with the same Gauss–Legendre machinery as
  `fraction()`. Sealed: `where(theta > 0, moment/(theta*V), 0)`
  (never a live 0/0; dry cells get exactly 0). The volume fraction
  theta alone CANNOT drive the correction: a bottom cut shifts the
  vertical centroid, a lateral cut does not — only the first moment
  distinguishes them, and it makes the lateral-cut correction
  vanish identically by construction. Static geometry: computed at
  setup, never traced.
- **PB-D2 (correction spelling — the gate arbitrates).** The
  corrected horizontal pressure gradient differences neighbor
  pressures at a **common physical height**: reconstruct each
  cell's pressure at the shared face height from its own value
  using the cell-local buoyancy as the hydrostatic slope
  (piecewise-linear-in-cell p, consistent with piecewise-constant
  discrete b), i.e. the correction couples `delta` (PB-D1) and the
  local `b` in the gradient stencil, following the
  `_slope_gradient` template in `pressure_gradient`
  (`hydrostatic/modules/core.py`). This candidate spelling is the
  starting point; **the well-balancedness gate (§3 G3) selects the
  final spelling** — as in the mapped+immersed plan, any
  gate-passing refinement lands without a design round-trip, with
  the deviation recorded in §6.
- **PB-D3 (scope).** Flat z-level and separable stretched-z
  immersed columns. Terrain-chart (+immersed) `p_hyd` correction is
  **deferred**: the current (uncorrected, O(dz)-at-cuts) behavior
  is the status quo there, stays available, and the deferral is
  recorded as a follow-up — no taught error needed, nothing
  regresses. If the stretched-z leg exposes a measure subtlety
  (computational vs physical moment under the column map), the
  implementer may land flat-z first and record the stretched-z
  deferral in §6 rather than block. Requesting the moment on a
  genuine chart (`mapping.column_corrections` truthy) is a taught
  error at the quadrature (never a silent wrong-geometry moment).
- **PB-D4 (rest-state semantics).** "Resting stratified state" on a
  cut grid means b sampled at the **physical wet-centroid heights**
  of the DOFs (the physical-integral precedent: physical positions,
  not computational labels). The G3 gate constructs its IC that
  way; the docstrings state it.
- **PB-D5 (placement + activation).** The correction lives in the
  hydrostatic core's pressure-gradient path, assembled at bind from
  the static `delta` fields, active **iff** the grid carries an
  immersed domain; `delta == 0` everywhere (all-wet, staircase,
  lateral-cut-only) must make the corrected path **bitwise**
  identical to the current one, or the correction term must be
  provably exactly zero there (G1/G2).
- **PB-D6 (differentiability).** Step-path change: ships an
  autodiff regression shard (policy: `jax.grad` of a quadratic loss
  through a short run via `Model.propagator`, FD-match rtol 1e-4).
  The `delta` fields are static, so the only VJP hazards are the
  sealed moment divide (PB-D1) and any divide in the reconstruction
  — double-`where` sealed, never `custom_vjp`.

## 3. Gates (all must pass before merge)

- **G1 all-wet ≡ unimmersed, bitwise.**
- **G2 staircase ≡ current staircase, bitwise** (`theta` in {0,1}
  gives `delta == 0`; the correction is exactly a no-op).
- **G3 well-balancedness (keystone):** b linear in physical z
  (`b = N^2 z`, sampled per PB-D4) over sloping immersed bathymetry
  at rest: the horizontal momentum tendency is **exactly zero**
  (machine zero) — flat z-levels, and stretched-z if in scope.
- **G4 general stratification:** a smooth nonlinear profile (e.g.
  tanh) at rest: PGF error with the correction converges at
  >= 2nd order under refinement and is smaller than without at
  every tested n; record the numbers in §6.
- **G5 autodiff shard** (PB-D6), finite and FD-matching.
- **G6 forced-4 multi-device:** short immersed hydrostatic run
  device-count invariant.
- Mirrored tests for every edited source file; ruff zero; patch
  coverage by construction.

## 4. Stages

- **P0 — spatial:** the PB-D1 moment quadrature in
  `spatial/immersed_domain.py` (+ chart taught error), with its own
  unit gates: analytic bottom-cut cells give the exact analytic
  centroid to quadrature order; lateral cuts give exactly 0;
  all-wet/dry give exactly 0; memoization mirrors `fraction()`.
- **P1 — hydrostatic:** the PB-D2 correction in the core pressure
  gradient + gates G1–G6.

One branch, sequential stages, single implementer.

## 5. Non-goals

Terrain-chart composition (PB-D3, recorded follow-up); nonhydro2
(no `p_hyd` cumsum on its solve path); any change to the
cumulative-integral operator's public semantics.

## 6. Implementation record

(appended as stages land)
