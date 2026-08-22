---
status: frozen
date: 2026-08-22
---

# WENO on stretched meshes — how Oceananigans and JAX-Fluids do it, and the route for FRIDOM

Research report (see [`README.md`](README.md) for status). Question:
FRIDOM's WENO refuses a `MappedIntervalMesh`; Oceananigans and
JAX-Fluids both advertise WENO on stretched grids — how do they do it,
and can FRIDOM? Method: source reading of both codes (Oceananigans at
the last stretched-capable commit `e4545016` / v0.96.21, the original
PR #2060 commit `1f7792ac`, and `main` v0.110.19; JAX-Fluids `main`),
a sweep of FRIDOM's own mapped-mesh records (the high-order mapped
plan, the Jacobian spike, the spec amendment, the advection module),
and a throwaway numpy spike on the coastal-upwelling tanh column. AI-
assisted; FRIDOM file:line references are on `dev` at `a1a658b1`.
The spike scripts were throwaway (scratchpad); every load-bearing
number is inlined below.

## Verdict

- **Both reference codes take route (ii) of the high-order mapped
  plan** — per-face precomputed reconstruction coefficients from the
  actual cell geometry (Shu 1998 eq. 2.20, the primitive-function
  Lagrange form), never the computational-space chain rule the plan
  picked as route (i). They differ in how much they make
  non-uniform: Oceananigans stretched only the candidate
  reconstruction coefficients and kept the uniform ideal weights and
  uniform Jiang–Shu smoothness indicators; JAX-Fluids makes all three
  non-uniform — candidate coefficients, ideal weights and smoothness
  indicators, closed-form rational expressions in the neighbouring
  cell sizes, precomputed once per face into broadcast arrays and
  read by a trace-time `if` — i.e. Shu 1998 throughout.
- **Oceananigans removed its stretched WENO in April 2025** (PR #4411,
  v0.96.22): "not used much, does not yield much better results", and
  the per-face tables stood in the way of a kernel rework. Current
  Oceananigans `main` runs uniform coefficients on stretched grids,
  silently — the exact failure FRIDOM's guard exists to prevent.
- **The spike answers which ingredient matters.** On the
  coastal-upwelling tanh column (5:1 cell-size ratio) the candidate
  coefficients are the only indispensable change: uniform rows give
  order 2, non-uniform candidates with *uniform* ideal weights and
  smoothness indicators restore order 5 within 10 % of the fully
  non-uniform scheme. Per-face ideal weights exist exactly (LSQ
  residual 1e-15) and stay positive on every map tried (min 0.027 at
  a 92:1 tanh). Only an **abrupt** cell-size jump (piecewise 1:3)
  separates the variants: there the uniform smoothness indicators
  misfire as if the jump were a front (order 2–3 locally, extra
  dissipation) and only the fully non-uniform set keeps order 5.
- **FRIDOM should implement route (ii), not route (i), for the
  advection modules** — for a reason the plan did not weigh. Route
  (i) as spiked (`mapped_jacobian_spike.md`) replaces the flux
  divisor by the same-row width `W_z ≠ Δz`; in 3-D the pressure
  projection keeps dividing by `Δz`, so a constant tracer acquires
  the tendency `Δw·(1/W_z − 1/Δz)` — a constancy break of
  `|W/Δz − 1|` = 8.7e-4 (n = 32) … 1.6e-5 (n = 256) on the tanh
  column, the very trade the advection module docstring declines for
  the Shu–Osher form (`model/modules/advection.py:53-60`). Route (ii)
  changes only the face values, keeps `flux_diff`'s physical-measure
  divisor, and therefore keeps conservation and constancy exact as
  today. Route (i) remains right for the collocated
  `FiniteDifference(order > 2)` refusal, which has no projection to
  agree with; the two coexist.
- **Implementable, medium effort, no new mathematics.** Every piece
  has a precedent in the tree: host-side geometry-derived weights
  (`reconstruct._wall_face_weights`), memoized per-space static
  fields in the storage frame (`grid.measure`), and a kernel whose
  coefficient taps already broadcast. The work is plumbing — array-
  valued `WenoTables`, a bias-aware coefficient select in the
  per-tap upwind kernel, the `Fallback` wall rungs, the tests — and
  the coefficient generator is ~100 lines of numpy the spike already
  contains.

## 1. Where FRIDOM stands

Three surfaces refuse a mapped factor, all with the
`staggering.mapped_order_hint` clause:

- `WenoReconstruction` (`spatial/operators/weno.py:238`,
  `require_uniform_mesh`) and through it every `Fallback` rung of
  order >= 3 (`spatial/operators/fallback.py:87,402`);
- the module-private `_BiasedFaceReconstruction`
  (`model/modules/advection.py:1435`, mapped guard in its codomain)
  and `UpwindAdvection`/`WENOAdvection` at bind
  (`_supports_mapped = False`, `advection.py:4310`);
- `FiniteDifference(order > 2)` and its one-sided closure.

The kernel is static: `WenoTables` (`weno.py:168`) holds Python
floats baked into the jaxpr, `_weighted_sum` (`weno.py:381`) skips
exact zeros and ones by Python comparison, the right-biased tables
are the reversed left-biased ones (`weno_tables`, `weno.py:279` — a
symmetry that holds on uniform meshes only), and the advection
kernel selects the window taps by the flux sign and applies one
left-biased table (`advection.py:1926-1948`). The advecting-velocity
interpolation of the C-grid modules is a centered uniform Shu row
too (`advection.py:725`).

The standing plan (`../plans/active/high_order_mapped_plan.md`)
chose route (i) — uniform-offset rows in computational space, the
flux difference divided by the same-row discrete Jacobian — for
reuse, and deferred route (ii) (genuinely non-uniform coefficients)
as "low reuse, high and quiet risk: a wrong beta scaling degrades
shock capturing while smooth-order tests still pass". The Jacobian
spike validated (i) in 1-D: design order restored, the 1-D
linear-preservation identity exact. §4 below is what 3-D adds.

## 2. Oceananigans

Source: `src/Advection/{weno_reconstruction, weno_interpolants,
reconstruction_coefficients, stretched_weno_smoothness}.jl` at
`e4545016` (the last stretched-capable commit) and `1f7792ac`
(PR #2060); `main` for the current state.

**Lifecycle.** Introduced by PR #2060 (Silvestri, 2021-11,
`WENO5(grid=grid)`), generalized to arbitrary order in PR #2603
(2022-07), **removed in PR #4411 (2025-04-20, v0.96.22)**. The
removal rationale, verbatim: "I don't think it is used much or, at
the moment, yields much better results. In addition, I want to
heavily rework the advection module … after removing the `coeff_*`
from the advection types we will also be able to remove the
`buffer_scheme` fields. I envision around a factor 1.5–2 performance
improvement." `main` still carries two dead files
(`stretched_weno_smoothness.jl`, the "Shenanigans for stretched
directions" block) that nothing calls. Current Oceananigans therefore
applies uniform coefficients on stretched grids without warning.

**What it did while it existed.**

- *Dispatch.* `WENO(grid; order)` called
  `compute_reconstruction_coefficients(grid, FT, :WENO; order=N)`
  per direction and location. A direction whose node array is a
  range (uniform) or `Flat` yielded `nothing`; a `Vector`-backed
  coordinate yielded a tuple of `order+1` `OffsetArray{NTuple{order,
  FT}}` — one per candidate stencil `r ∈ -1:order-1`, one tuple per
  face. The struct's type parameters (`XT, YT, ZT`) were `Nothing`
  or that tuple type, so the kernel dispatched at compile time:
  `coeff_p(::WENO, bias, ::Val{stencil}, ::Type{Nothing}, ...)`
  returns constants, the generic method reads
  `scheme.coeff_xᶠᵃᵃ[r+2][i]`. Six tables per grid (x/y/z × Face/
  Center); Center tables treat face-located data as averages over
  dual cells bounded by cell centers. Memory: for WENO5, Nz = 100,
  ≈ 20 kB per direction.
- *Coefficients.* Shu 1998 eq. 2.20 literally, in
  `stencil_coefficients(FT, i, r, xr, xi; order)`: for cell-average
  data, `c_rj = Σ_{m=j+1}^{k} [Σ_{l≠m} Π_{q≠m,l} (x_{i+1/2} −
  x_{i−r+q−1/2})] / [Π_{l≠m} (x_{i−r+m−1/2} − x_{i−r+l−1/2})] ·
  Δx_{i−r+j}`. Evaluated in absolute coordinates (no local shift),
  accumulated in `BigFloat`, rounded to `FT`, with the last
  coefficient replaced by `1 − Σ(others)` to force partition of
  unity. The uniform constants come from the same routine on an
  integer lattice at compile time. Left and right biases share one
  table (same cell sets reconstruct the same face; the right stencil
  `s` reads entry `r = buffer − 2 − s`, reversed).
- *Ideal weights.* **Uniform** (Balsara–Shu constants, `C★`), never
  position-dependent, no per-face solve. Silvestri: "those weights …
  are mostly a matter of choice (if the reconstruction is performed
  correctly for all stencils!) … weights will still be between 0 and
  1."
- *Smoothness indicators.* **Uniform** Jiang–Shu / WENO-Z
  (`# _UNIFORM_ smoothness coefficients (stretched smoothness
  coefficients are to be fixed!)`). PR #2060 had an opt-in
  `stretched_smoothness=true` for WENO5 only — Shu's general
  `β_r = Σ_l ∫ Δx^{2l−1} (p_r^{(l)})² dx` over the upwind cell,
  closed-form by parts, stored as a per-face 3×3 quadratic form —
  measured ~4× slower on GPU with "no noticeable benefit (apparently
  β_r for a stretched mesh are very similar to β_r for a uniform
  mesh)", and dropped in the arbitrary-order rewrite.
- *Linear schemes.* Stretched coefficients for `Centered` /
  `UpwindBiased` were computed but **disabled** ("seem to be more
  unstable than constant spacing ones … some tests are needed to
  verify why"); the WENO's own `Centered(order−1)` advecting-velocity
  interpolation stayed uniform.
- *Boundaries.* Order reduction by the recursive `buffer_scheme`
  (`WENO(order−2)`, also built with the grid), `ifelse`-selected per
  face — no one-sided tables. Halo bug #2717: the constructor reads
  coordinates `order` cells beyond the domain, so a stretched grid
  needed a user halo >= buffer; never fixed.
- *Caveats on record.* No convergence study of the uniform-ideal-
  weight combination; no issue about negative weights (the question
  never arose since they were never computed); the removal note
  itself.

## 3. JAX-Fluids

Source: `tumaer/JAXFLUIDS` `main` at `af2b7cf5` (2026-08-05);
the stretched-coefficient module arrived with v0.2.0 (commit
`22921eaa`, 2024-05). MIT licence. The JAX-Fluids 2.0 paper (Bezgin,
Buhendwa & Adams, CPC 308 (2025) 109433) advertises "arbitrary
one-dimensional mesh stretching" and gives the tanh face-position
maps of its channel / boundary-layer cases (β up to 3.5, plus
piecewise geometric outflow buffers) but **describes none of the
non-uniform coefficient algebra and cites no non-uniform-WENO
reference for it**; the formulas below were verified numerically by
the research agent against their definitions (flagged as
verification, not documentation).

- *Mesh model.* `DomainInformation` carries a per-axis
  `is_mesh_stretching` tuple of Python bools and per-axis cell-size
  arrays shaped `(Nx,1,1)` / `(1,Ny,1)` / `(1,1,Nz)` (a `(1,1,1)`
  scalar on a homogeneous axis). Stretching is strictly 1-D per axis
  (`CHANNEL` / `BOUNDARY_LAYER` tanh maps, `PIECEWISE` constant +
  geometric zones with the ratio Newton-solved). The stencils receive
  the **halo-extended** sizes: halo cell sizes follow the physical
  BC (zero-gradient replication for Dirichlet/Neumann/outflow,
  mirror for walls, wrap for periodic), set once because the mesh
  is static.
- *Coefficients* (`stencils/helper_functions.py`,
  `compute_coefficients_stretched_mesh_weno3/5/6`, `_teno6`,
  `_muscl3`): closed-form rational expressions in `Δx_{i−2} … Δx_{i+2}`
  — no Lagrange loop, no solve — e.g. the WENO5 middle candidate
  `c1_0 = −Δx_i Δx_{i+1} (Δx_i + Δx_{i+1}) / [(Δx_{i−1}+Δx_i)(Δx_i+Δx_{i+1})(Δx_{i−1}+Δx_i+Δx_{i+1})]`.
  Verified to be exactly the cell-average (FV) Shu 2.20 rows, for
  every 3-cell candidate, the 4-cell TENO6 candidate and the full
  6-cell polynomial; uniform limits reproduce `(1/3, −7/6, 11/6)`,
  `(1, −8, 37, 37, −8, 1)/60`.
- *Ideal weights.* **Per face, closed form**: for WENO5
  `d_0 = Δx_{i+1} Δ_{i+1:i+2} / (Δ_{i−2:i+1} Δ_{i−2:i+2})`,
  `d_2 = Δ_{i−1:i} Δ_{i−2:i} / (Δ_{i−1:i+2} Δ_{i−2:i+2})`,
  `d_1 = 1 − d_0 − d_2` (`Δ_{a:b}` the summed widths of cells
  `a … b`). Ratios of products of positive lengths, so **positive by
  construction** for WENO3/5 — the spike's exact-existence finding
  with its proof. WENO6-CU / TENO6 back their four weights out by
  matching the outer coefficients of the 6-cell polynomial (a
  recursion, no positivity guarantee, no clipping; 20 000 random
  cell configurations with neighbour ratios up to ~20 found none
  negative, but strong geometric stretching skews them hard:
  ratio 3 gives `(0.58, 0.40, 0.015, 0.000)` — the downwind
  candidate switches itself off).
- *Smoothness indicators.* Full per-face quadratic forms `v·B·v`
  (6 coefficients per 3-cell candidate, `betar_` arrays), verified to
  be Shu's `β = Σ_l Δx_i^{2l−1} ∫_{cell i} (p^{(l)})² dx` with `Δx_i`
  the **central (upwind) cell** and the integral over it — the
  spike's variant D. The 6-cell forms go through a Newton
  divided-difference basis. Round-off hazard on record: a quadratic
  form can evaluate slightly negative, so `WENO5-Z-ADAP` guards with
  `jnp.abs(beta)` (`# NOTE Beta's might be negative due to machine
  precision`); `WENO5-JS-ADAP` squares it anyway; the TENO
  variants carry no guard.
- *Code structure.* Every stretched-capable scheme is a **separate
  class** (`WENO5-JS-ADAP`, `WENO5-Z-ADAP`, `WENO6-CU-ADAP`,
  `TENO5/6-ADAP`, the MUSCL and central `-ADAP` families;
  `is_for_adaptive_mesh = True`); the plain classes stay uniform and
  a stretched mesh with a uniform stencil only raises a
  `RuntimeWarning`. Tables are computed in `__init__` (host, once):
  `cr (3, 3, N+1, 1, 1)`, `betar (3, 6, N+1, 1, 1)`,
  `dr (3, N+1, 1, 1)` per bias and axis, `None` on a uniform axis;
  60 (N+1) doubles for WENO5. Inside the jitted `reconstruct_xi` a
  Python `if is_mesh_stretching[axis]:` picks arrays or the static
  floats, and the **same** arithmetic expression runs on either
  (`cr[0][0] * buffer[s_[0]] + …` broadcasts), with the β spelling
  the only branch. Under `pmap` decomposition a leading device axis
  is indexed with `lax.axis_index`.
- *Bias.* Both biases are stored; the right-biased tables are the
  **left-biased closed forms evaluated on the reversed cell-size
  window** — the mirror trick moved from the coefficient table to
  the geometry, which is where it survives on a non-uniform mesh.
- *Coverage gaps.* No stretched WENO7/9 or TENO8; a noted
  inconsistency that `TENO5-ADAP` abandons its spectrally optimized
  uniform weights `(0.05, 0.55, 0.40)` for the 5th-order set on
  stretched axes (`# TODO do we need to adap self.dr_ …`); no
  stretched-accuracy caveats in the paper or the issue tracker.

## 4. The spike — which ingredient has to go non-uniform

Setup: FV left-biased WENO5 reconstruction of the right face of cell
`i` from cells `i−2 … i+2`, cell averages exact from an
antiderivative, `f = sin(2πz/L) + 0.3 cos(4πz/L + 1)`, L∞ over
interior faces with the WENO-JS critical points masked (the standard
order loss there, not a mesh effect), `eps = 1e-10` (production
`WENO_EPS`). Coefficients from the primitive-function Lagrange basis
in **local, cell-scaled coordinates** (in absolute coordinates the
linear 5-point reference lost 4 digits at n = 256 on a 100 m column
— the conditioning Oceananigans papered over with `BigFloat`); the
smoothness forms from Shu's general definition integrated exactly;
both reproduce the rational Shu rows and the Jiang–Shu 13/12, 1/4
form on a uniform lattice to 1e-16 (the self-check a production
generator should keep). Per-face ideal weights from the exact
embedding of the three candidates into the 5-cell row (least
squares; residual reported).

Variants: **A** uniform rows on the stretched data (the guard lifted
naively); **B** non-uniform candidate coefficients, uniform `d`,
uniform `β` (Oceananigans); **C** non-uniform `c + d`, uniform `β`;
**D** fully non-uniform `c + d + β` (Shu 1998 throughout); **E** the
linear 5-point non-uniform row (upper bound); **F** the chain-rule
variant `p_m(qJ)/p_m(J)` with uniform rows on the J-weighted data.

Order (pairwise, n = 32 → 512), coastal-upwelling tanh column
(`STRETCH = 1.5`, max/min Δz = 5.3 at n = 64):

| variant | orders | error at n = 128 |
|---|---|---|
| A uniform rows | 2.98 2.03 2.00 2.00 | 1.3e-4 |
| B non-uniform c | 4.84 4.97 5.00 5.00 | 3.4e-6 |
| C non-uniform c + d | 4.84 4.97 4.99 5.00 | 3.3e-6 |
| D non-uniform c + d + β | 4.85 4.98 4.99 5.00 | 3.1e-6 |
| E linear 5-point | 4.87 4.96 4.99 5.00 | 6.0e-7 |
| F chain rule | 4.84 4.84 5.05 4.96 | 5.2e-6 |

Ideal weights: min 0.061 (n = 32) → 0.097 (n = 512), deviation from
the uniform (1/10, 6/10, 3/10) halving with `n` (9.8e-2 → 6.4e-3,
i.e. O(h) on a smooth map); residual 1.6e-15 — they exist exactly.
Harsher tanh (`STRETCH = 3`, 92:1): same picture, A → 2.00, B/C/D
4.97–4.99 at the finest pair, min ideal weight 0.027, still positive.

Piecewise map with an abrupt 1:3 cell-size jump at mid-depth (the
JAX-Fluids-style piecewise stretching): A 1.0; **B 4.8 3.6 2.3 2.3;
C 4.8 3.6 2.8 2.0; D 4.8 4.9 5.0 4.9**; E 5.0; F 1.0. The ideal
weights deviate by 0.23 from uniform at the jump at every `n` (an
O(1) geometry, not O(h)). With uniform `β` the undivided differences
across the jump look like a front, the weights leave the full
stencil, and the scheme drops to order 2–3 locally — the "quiet"
failure the plan feared, here visible in a smooth-order test because
the jump is a fixed geometric feature.

ENO on a step (n = 64, tanh): overshoot 2e-21 (A), 9e-16 (B, D),
0.17 (E linear). All WENO variants keep the step monotone; `β`
choice does not show on a clean step.

**Nodal family** (point values at the mapped computational midpoints
`X((j+½)/n)`, Lagrange interpolation to the face `X((i+1)/n)`): on
both tanh maps the *uniform* midpoint rows already interpolate at
order 5 (errors within 10 % of the non-uniform rows) — the face sits
exactly at the computational midpoint, so a point-value interpolation
of `f∘X` on the uniform lattice is the design-order interpolation of
`f` (the spec amendment's `LinearInterp(one_sided)` observation,
generalized: a `derivative = 0` row without a spacing divisor is
grounded on a mapped lattice). Per-face ideal weights exist (residual
1e-15) and stay positive (min 0.022 at `STRETCH = 3`). On the
piecewise jump only the fully non-uniform set keeps order 5.
FRIDOM's nodal family does **not** use these rows: it applies the FV
Shu rows to point values (the Shu–Osher FD reading, which is what
makes the two-point flux difference 5th order at constant velocity);
that reading is exactly what the measure divisor breaks on a
stretched mesh (the plan §1 trap), and §4 is why the cure is the
face value, not the divisor.

## 5. Route (i) against the projection — the 3-D argument

Route (i) keeps the uniform rows and divides the flux difference by
the same-row width `W_i = x̂_{i+½} − x̂_{i−½}` instead of the measure
`Δz_i`. In 1-D with `u ≡ 1` the numerator vanishes before any
divisor enters, so the spike's E1 could not see a difference; in
3-D the advected quantity `q ≡ 1` gives

    A(1) = Δu/Δx + Δv/Δy + Δw/W_z = Δw · (1/W_z − 1/Δz)

because the pressure projection enforces `Δu/Δx + Δv/Δy + Δw/Δz = 0`
with the two-point measure. Measured on the tanh column (linear
upwind-5 row on the centre coordinates, interior cells):
`max |W/Δz − 1|` = 8.7e-4, 2.4e-4, 6.4e-5, 1.6e-5 for n = 32 … 256 —
O(h²), small, and **not zero**, where today's scheme and route (ii)
hold constancy to machine precision. The module docstring already
declines the Shu–Osher flux reconstruction on exactly this ground
("only if the pressure projection enforced the same wide
reconstructed divergence — a different Poisson operator",
`advection.py:53-60`); the W divisor is the same trade in different
clothing. The H7 surface closure (`_surface_correction`,
`advection.py:3686`) could subtract `q·A(1)` in its full-3-D form,
but that converts the interior into advective form and forfeits
exact conservation along the stretched axis — not an improvement.

Route (ii) touches only the face value. The candidate rows sum to one
and the normalized weights sum to one, so the reconstruction of a
constant is exact and `A(1)` is bitwise the two-point divergence
today's centered scheme sees. `flux_diff`, `_mapped_fv_divergence`,
the terrain-following J-weighting of `stretched_terrain_combined.md`
— none of it moves.

Route (i) keeps its place for `FiniteDifference(order > 2)` and its
one-sided closure: a collocated derivative has no projection to agree
with, and the same-row Jacobian is exact on the metric identity there
(spike E4). Retire that refusal with (i), the reconstruction refusals
with (ii).

## 6. Implementing route (ii) in FRIDOM

Everything below has a precedent in the tree; the list is the work,
in dependency order.

1. **Coefficient generator (host, numpy, exact-rational self-check).**
   Input: the face positions of the factor (`mesh.coordinate_map` at
   `k/n`; the nodal Center→Right family needs only the faces too —
   the FV Shu rows on point values, as today; the dual Right→Center
   direction of the velocity components uses the mapped centres as
   dual-cell boundaries, Oceananigans' "ᶜ" tables). Output per face
   and bias: `r` candidate rows of `r` coefficients, `r` ideal
   weights, `r` symmetric `r×r` smoothness forms — 30 floats per
   face at order 5. Local cell-scaled coordinates (the conditioning
   finding above). On a uniform lattice it must reproduce `_shu_row`
   and `_SMOOTHNESS_ROWS`/`_SMOOTHNESS_SCALE` bitwise — the test that
   pins it to the existing kernel. Build-time guard: ideal weights
   positive (raise otherwise; never observed on a monotone map, but
   the positivity is not a theorem for every geometry). Precedent:
   `reconstruct._geometric_value_weights` / `_wall_face_weights`
   (`reconstruct.py:539,575`) — the same host evaluation of
   `coordinate_map` positions, for one wall face. Periodic stretched
   axes unwrap the coordinate across the seam (the Jacobian spike's
   `halo = [x[-pad:] − L, x, x[:pad] + L]`).
2. **Grid-level materializer.** A memoized `grid` accessor beside
   `measure` (`grid.py:1264`), keyed `(space, order, bias, halo)`,
   returning the tables as storage-frame arrays along the factor
   (shape `(1, n + 2·halo, 1)`-style broadcast, sharded and
   halo-extended exactly as a measure field, zero on bounded ghost
   faces the graded ladder patches anyway). Uniform factors return
   `None` → the static tables, so x/y kernels are untouched
   (Oceananigans' `Nothing` dispatch, spelled as Python `None`).
3. **Array-valued `WenoTables`.** `_weighted_sum` drops the
   `weight == 0.0 / 1.0` Python fast paths when a weight is an
   array; `d` becomes an array; for `β` keep the kernel's
   squares-of-weighted-sums spelling by shipping the **Cholesky
   factor** of each candidate's form (`B = LᵀL`, `β = Σ_k (L_k·v)²`
   — `r` rows of `r` coefficients, unit scales; the Jiang–Shu rows
   are one such factorization on a uniform lattice), which keeps
   `β >= 0` exact in floating point and makes JAX-Fluids' `abs`
   guard unnecessary; the window slicer already works on
   broadcastable arrays. XLA fuses the 1-D
   taps into the existing elementwise expression — no new full-size
   array, ~30 small cache-resident operands per fusion at order 5.
   The uniform path stays bit-identical (static floats).
4. **Bias.** The table mirror of `weno_tables` dies on a
   non-uniform mesh (the right-biased rows at face `i` reconstruct
   from cells `i−1 … i+3`); the mirror moves to the geometry, as in
   JAX-Fluids — run the left-biased generator on the reversed,
   negated face window — so one generator yields both tables and
   they stay exact mirrors on a uniform lattice. In the advection
   kernel's per-tap
   select the coefficient taps are selected with the same predicate
   as the data taps (`where(pos, c_left, c_right)` on broadcast
   arrays — inside the fusion, no extra traffic); the measured-slower
   both-then-select spelling (`advection.py:1700`) stays out.
5. **Walls.** Every `Fallback` rung is the same generator at a
   one-sided cell set and a different face index — the ladder's
   `boundary="graded"` path lifts with no new math; the
   `advection.py:725` velocity row gets its non-uniform sibling from
   the same generator (Oceananigans left this one uniform; the spike
   says that costs no order on a smooth map, but there is no reason
   to keep the inconsistency once the generator exists).
6. **Guards.** `require_uniform_mesh`, the `_BiasedFaceReconstruction`
   codomain guard and `_supports_mapped` retire for the
   reconstruction rows; the `FiniteDifference` refusals wait for the
   route-(i) lift. `mapped_order_hint` loses its "biased" clients.
7. **Tests** (mirrored files, cheap grids): generator vs rational
   rows on a uniform lattice (bitwise); standalone `WenoReconstruction`
   order 5 (masked) on the tanh map, order 2 with the static tables
   as the negative control; the piecewise 1:3 map as the β/d
   discriminator; advection at constant velocity order 5, constancy
   (`q ≡ 1` tendency machine zero under a discretely divergence-free
   3-D flow on a stretched column) and conservation to 1e-15; the
   step-overshoot ENO check; one `Model.propagator` autodiff
   regression (the tables are static geometry, so nothing new enters
   the VJP — but the new quadratic-form spelling of β must be
   exercised). JAX-Fluids' generator can serve as an independent
   oracle for the table values (MIT licence; numerical comparison,
   no code copied).

Effort: medium — comparable to the plan's 1–2 week estimate for the
route-(i) lift; the risk is in 2 and 4 (halo/sharding plumbing and
the select seam), not in the numerics, which §4 settles. Minimal
first rung if wanted: step 1–3 with candidate coefficients only
(Oceananigans' subset, uniform `d` and `β`) — but in FRIDOM's
precomputed-table design `d` and `β` cost nothing extra at run time,
unlike Oceananigans' on-the-fly β, so there is no reason to stop
short of the full Shu 1998 set.

## 7. Moving geometry (ALE charts)

Asked 2026-08-22: does route (ii) survive a moving chart? FRIDOM's
moving geometry (`moving_geometry.py`, coordinate-systems stage C4)
keeps the **mesh static** and moves the `CoordinateMapping` chart
`m = M(b, params)`; every metric (`J = dm/db`, `1/J`, the cross
slopes `Z_i`, the face mesh velocity) is re-derived per query from
the `params=` state fields and never cached, and the mapped
advection is the J-weighted conservative form **in base
coordinates** (`_mapped_fv_divergence`: fluxes on base-mesh faces,
`flux_diff` with the static base measure, `J` on the flux, `1/J`
outside); the FV ALE term (`ale_on_fv.md` option C, landed) is the
same shape with the face mesh velocity.

- **Structurally: yes, for free.** The route-(ii) tables are a
  property of the static base-mesh factor `b` (its face positions);
  they carry no `params` seam, so the biased reconstruction sits in
  the chart exactly where the centered `reconstruct` row sits today
  and the motion enters through `J`, `Z_i` and `w` as it already
  does. Frozen-motion bitwise parity holds by construction, and
  `jax.grad` w.r.t. `params` is untouched (the tables are constants;
  the metrics carry the gradient). Route (i) would *not* compose:
  its same-row width `W` would have to move with the chart and would
  disagree with the J-weighted divergence and the ALE bracket — the
  §5 constancy break, now per stage.
- **Column-uniform charts are exact.** The tables are homogeneous of
  degree zero in the cell sizes (ratios of lengths; `β`'s
  `Δx^{2l−1}∫(p^{(l)})²` is scale-free). Verified: `c`, `d` and the
  `β` forms under a scale-and-shift of the whole column agree to
  4e-13. So on every chart whose `J` is constant along the column —
  `H(t)·b`, the free-surface `z*` / σ family `(H + η)·σ(b)`, a static
  terrain `H(x, y)` under a stretched `σ(b)` — the static base-mesh
  tables **are** the current physical-geometry tables at every
  instant, with nothing to recompute.
- **b-dependent deformation in time** (a chart whose `∂M/∂b` changes
  shape along the column as `params` move — a morphing `Y_N(x, t)`
  on a non-linear chart, Lagrangian-style layer motion): the static
  tables are then not the physical-geometry tables (verified: `c`
  changes by O(1) under a `(1 + 0.3 s)` column deformation). The
  face value stays a consistent, ENO-preserving reconstruction along
  `b` but drops to 2nd order for physical cell averages (spike
  variant A's mechanism). Two ways back to design order, both
  module-side with params-threaded metrics like `J` today: (a)
  **J-rescale the static tables** — `c̃_{m,j} = c_{m,j} J̄_j /
  Σ_k c_{m,k} J̄_k` per candidate, the spike's chain-rule variant F,
  order 5 on smooth maps (and charts are smooth; it fails only at
  abrupt jumps, which a chart cannot produce) — one fused rescale
  reading the cell-mean `J`, rows still summing to one so
  constancy/GCL stay exact; or (b) **regenerate the tables per step**
  from the current physical faces `M(b_faces, params)` with a
  `jnp`-traceable generator (the JAX-Fluids closed forms are
  elementwise in five shifted cell-size slices and fuse into the
  kernel; on a chart varying in `x, y` the tables become 3-D fields
  — order-5: 30 of them, fused, roughly a 1.5–2× kernel cost on that
  axis). Write the generator in `jnp` from the start and memoize only
  concrete (static) geometry, the `grid.measure` pattern — then (b)
  is a `params=` overload, not a redesign.
- **GCL / constancy** needs only that the reconstruction reproduces
  constants and that both ALE bracket terms share the same face
  `w`; partition-of-unity rows with normalized weights give the
  first whatever the tables, so the ALE property is unaffected.
  Note the ALE flux term reconstructs `f` with the default two-point
  `reconstruct` row, not the advection's WENO — consistent in the
  constancy sense, not in the dispersion sense; aligning the two is
  a separate, optional choice (MOM6/ROMS use one reconstruction for
  both).

## 8. Open points

- The tables are static geometry; a moving stretched column (the
  `MovingGeometry` state) would need the generator traceable
  (vectorized `jnp` Lagrange products — possible, not needed now:
  `MappedIntervalMesh` holds a static map).
- Strongly stretched *periodic* axes: untested in the spike (the
  motivating columns are bounded).
- The plan's §5 caveat stands: the composite C-grid tendency is 2nd
  order under a varying velocity on any mesh; what (ii) restores is
  the ENO/dispersion behaviour and honest order for the pieces that
  have it.

## 9. Landing note (2026-08-23)

Shipped on `dev` in two branches (merges `9a1edb38` core, `9b95bc8f`
advection), AI-implemented from a shared contract and owner-reviewed
records. Facts that refine the sections above:

- **Mechanism as built**: the widths are a storage-frame co-operand
  (`weno.cell_widths`, the `grid.measure` of the operand's own node
  set — primal cells, or the dual face cells with the wall half cells
  written into the wall ghost slots through a new
  `patch_physical_ends(co_arrays=...)` seam), windowed exactly like
  the data; `weno.nonuniform_tables` derives the tables in pure `jnp`
  on the windows and runs under `jax.ensure_compile_time_eval` on a
  device-local axis, so the ~1050-equation generator folds to
  constants (mapped periodic WENO-5: 184 optimized HLO ops vs 934
  staged, 116 uniform). The one-pass upwind kernel builds both bias
  table sets from the 1-D width windows and `where`-selects the ~21
  entries — not the width taps, which would stage the generator on
  full 3-D arrays. The smoothness indicators ship as `r-1`
  sum-of-squares rows (the form is rank `r-1`; a Cholesky would hit a
  zero pivot).
- **Correction to §4/§6 — only the FV family gains order.** Measured
  at constant velocity on the wavy periodic axis (n = 24/48/96): FV
  upwind-3 / upwind-5 / weno-5 → 3.0 / 5.0 / 5.0; nodal upwind-5 →
  **2.0**. The nodal family's DOFs are point values and its two-point
  flux difference over `Δx_i` is high-order only through the uniform-
  lattice Shu–Osher identity, so no choice of face rows restores its
  order on a stretched axis — only route (i)'s divisor would, at the
  constancy cost of §5. The width-aware rows carry a ~3x larger
  O(h²) constant than the static rows on that map (3.3e-4 vs 1.1e-4
  at n = 128); the owner call is recorded in the plan.
- **Constancy** on three stretched grids: 1.1e-14 / 1.8e-14 / 2.1e-14
  absolute against divergence scales 35 / 42 / 262 (~1.6e-16
  relative) — §5's argument holds in the shipped code.
- **Validated**: two oracles (the spike's numpy polynomials, 1e-11 on
  the beta forms; JAX-Fluids' closed forms, 1e-14 on coefficients over
  500 random windows — compared, not copied), uniform-lattice limits
  to 1e-15, scale invariance, polynomial exactness, one-pass vs
  both-then-select to reversed-summation ulps, ENO on a step, walled
  columns down to one cell, `Model.propagator` autodiff vs FD,
  forced-4 device-count invariance for the primal and the bounded dual
  frames. `ruff` clean; patch coverage left to CI.
- **Left out**: stretched + immersed (taught refusal), biased schemes on
  a mapped column (taught refusal kept), `FiniteDifference(order > 2)`
  (route (i) remains), perf of the table selects (owner-triggered
  guard).
