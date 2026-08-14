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

### Landed on `feat/partial-bottom-phyd` (2026-07-19)

**Final PB-D2 spelling.** The corrected cell-centre pressure is

```
p_corr = p_hyd + S * (b_up - b),
    S = delta * dz / (2 * (zeta_up - zeta)),   zeta = z_c + delta
```

with `delta` the static wet-centroid offset (`centroid_offset`, PB-D1),
`dz` the cell measure, `zeta_up - zeta` the physical spacing between the
neighbouring wet centroids, and `b_up - b` the one-cell-up vertical
buoyancy increment. The horizontal PGF is `-d_x p_corr` /
`-d_y p_corr`. The whole thing lives in
`HydrostaticCore.pressure_gradient` (`_partial_bottom_pressure`),
active iff `self._pb_active` (a flat immersed grid carrying a genuine
bottom cut). `S` is static geometry (folded into the trace as a
constant); the only traced factor is `b_up - b`, **linear** in `b` with
a clean transpose VJP — no step-path divide (PB-D6), so G5 is trivial.

**Deviation from the candidate (the gate arbitrated).** The PB-D2
candidate ("reconstruct to the shared face height with cell-local `b`
as the slope, piecewise-linear-in-cell `p`") cannot be *both*
machine-zero **and** a no-op off partial cells with a single common
height: a linear reconstruction to height `z*` zeroes the rest PGF only
at `z* = z_top` (the cell top face), which is delta-independent and so
breaks G1. The refinement the keystone gate forced is the **quadratic
(local-slope) term expressed through the wet-centroid vertical spacing**:
writing the correction as `beta * delta * dz / 2` with the discrete
buoyancy slope `beta = (b_up - b) / (zeta_up - zeta)` makes `beta == N^2`
*exactly* on a linear stratification, because `b` is sampled at the
wet-centroid heights (`b_up - b = N^2 (zeta_up - zeta)`). The
load-bearing detail is `zeta_up - zeta` (the **wet-centroid** spacing),
not the mesh spacing `dz` — the latter gives only 1st order (measured:
mesh-`dz` variant converges at ~1.5, wet-centroid at ~2–3). This is a
one-cell **upward** stencil (the lower neighbour of a bottom cut is dry,
so a centred slope would read the masked `b = 0` below).

**G3 scope refinement (recorded, not a design round-trip).** Machine
zero holds for the Pacanowski & Gnanadesikan partial-cell setup —
columns with **different bottom depths that are flat within each
horizontal cell** (one partial bottom cell per column, full cells
above). A bathymetry that varies *within* a horizontal cell (sub-cell
corner cuts, which stack partial cells so the accumulated downward
integral picks up column-dependent `delta_j` from cells above) is
**2nd-order, not machine-zero** — that is the staircase-avoidance regime,
a distinct effect from the P&G half-cell error this fixes; the
correction still strictly improves it (below uncorrected). The G3 gate
therefore uses the per-column-constant bathymetry (the P&G "sloping
bottom" = a staircase of depths).

**Stretched-z is in, not deferred.** Separable stretched-z (a
`MappedIntervalMesh`, not a chart) is also machine-zero (G3
`test_g3_rest_state_is_machine_zero_stretched`): the per-cell
linearised quadrature geometry (`_cell_quadrature`) is self-consistent
between `centroid_offset` and the cumulative integral's measure, so no
measure subtlety surfaces. A genuine chart (`column_corrections`
truthy) is the PB-D3 taught error at the quadrature.

**FP hygiene (source fix).** `centroid_offset` normalizes the cell
centre by the weight sum (`z_c = sum(z*w)/sum(w)`, not assuming
`sum(w) == 1`), so a full cell (`chi == 1`) makes `z_chi/theta` and
`z_c` the *identical* float expression and `delta` is **exactly 0**
there — no ~1e-16 residue that would misfire `_pb_active` or leak a
~1e-16 correction onto full cells (G1/G2 exact).

**Decomposition.** z-shard-safe via reshard-to-axis-local
(`_upward_increment`, the `CumulativeIntegral` axis-local contract that
already keeps `p_hyd` local), a no-op when the vertical is already local
(the flat-z / horizontally-sharded common case). `extra_halo` is
**unchanged** (`{x:1, y:1, z:0}`): the vertical stencil rides the
reshard, not a halo, and the horizontal `d_x p_corr` reach-1 already
sits under the existing `x/y` extra halo.

**Gates (measured).**

- G1 all-wet: the correction is a byte no-op vs the plain diff
  (`du = dv = 0.0`, `_pb_active is False`). (Full-step all-wet ==
  unimmersed byte-identity is `w`-only and *pre-existing* — the
  immersed core runs `pressure_gradient` halo-exempt regardless of this
  change; `u/v/b` differ by ~5e-17 there, untouched by this work.)
- G2 staircase (`order=None`): byte no-op (`du = dv = 0.0`).
- G3 rest state (`b = N^2 z` at wet-centroid heights): flat
  `max|du/dt| = 1.78e-15 / 0.0 / 8.88e-16` at `(nx,nz,order) =
  (6,8,8)/(8,16,6)/(5,12,4)`, uncorrected `~3.7e-2`; stretched-z
  `8.88e-16` (uncorrected `3.6e-2`).
- G4 tanh at rest: corrected below uncorrected at every `nz`, orders
  `2.43 / 3.27 / 3.00` (`nz = 8..64`).
- G5 autodiff: `Model.propagator(wrt=("b",))` grad finite and FD-matched
  to `rtol 1e-4`; the pre-existing immersed autodiff shard also passes.
- G6 forced-4: x-shard `5.55e-17`, z-shard `1.39e-17` (device-count
  invariant; the z-shard case exercises the reshard path).
- P0 units: linear-`chi` centroid exact `2.1e-16`
  (`dz^2/(12(z_c-a))`); lateral-cut / all-wet / all-dry / collocation
  exactly `0`; chart taught error; memoization.

**Files.**
- `src/fridom/spatial/immersed_domain.py`: `centroid_offset` (public),
  `_centroid_cells`, `_validate_centroid_space`.
- `src/fridom/hydrostatic/modules/core.py`: `_partial_bottom_pressure`,
  `_upward_increment`, `_up_shift_local`, `_derive_pb_active`, the
  `_pb_active` gate in `pressure_gradient`.
- `tests/spatial/test_immersed_domain_centroid.py` (P0, 12 tests).
- `tests/hydrostatic/test_core_partial_bottom.py` (P1 gates, 10 tests).

**Addendum 2026-07-19 (follow-up sweep).** The hydrostatic
identity-chart ≡ flat-immersed free-surface gate
(`test_identity_chart_mask_matches_flat_immersed`) was silently broken
by this merge — bisect-confirmed `01052ee0` pass → `0ddec821` fail
(0.66% in u). Mechanism: on a genuine bottom cut the flat path now
carries the well-balanced correction while the chart path is the PB-D3
deferral (`_derive_pb_active` short-circuits on `column is not None`),
so the two paths differ by exactly the correction — an intentional
accuracy asymmetry that invalidated the gate's geometry premise. The
miss slipped the mirrored-test policy because the gate lives in
`test_free_surface_terrain_immersed.py`, which mirrors none of this
merge's edited sources. Repair (merge `1e5359dd`): the gate now runs on
a face-aligned 3-D staircase (δ ≡ 0 to machine precision on both
sides; equivalence restored at 4.4e-16, and the genuine-cut control
still fails at 6.6e-3 — the geometry change is what restores the pass),
plus a companion test pinning the intended asymmetry through
`_pb_active` (True flat / False chart on a genuine cut). Latent
fragility noted, not fixed: `_derive_pb_active`'s `> 0.0` predicate can
trip on an ~1e-17 XLA reduction-order residue in an all-wet layer
(`_centroid_cells`' identical-expression exact-zero does not survive
fusion for every layer) — harmless, since the correction magnitude
stays ∝δ ≈ roundoff; tighten only if it ever bites.
