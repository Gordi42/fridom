# Genuine-chart (J≠1) physical column-equivalence twin — verdict

**Date:** 2026-07-19 · **Status:** closed, record only (no test shipped)
**Follow-up of:** the M5 seam verdict in
[`../plans/active/mapped_immersed_composition_plan.md`](../plans/active/mapped_immersed_composition_plan.md)
**Probe:**
[`artifacts/genuine_chart_twin/probe_twin.py`](artifacts/genuine_chart_twin/probe_twin.py)

## Question

The mapped+immersed composition certifies genuine-chart (J≠1)
correctness through algebraic gates only; the physical
column-equivalence gate rides the J≡1 chart limit. The M5 seam verdict
called the genuine-chart twin "ambiguous (the wet sub-column is an
affine sub-chart, not a plain shallower domain)". Is a well-posed
physical twin constructible after all — and is it a shippable gate?

## Verdict

**The twin is well-posed and constructible, and it confirms the
composition is physically correct — but it is boundary-order-limited,
so no convergence test ships.** Tracer moments converge at clean 2nd
order; the velocity moments — the fields that actually feel the
composed cut-cell projection — converge at a genuine,
solver-independent, noisy ~1st order set by the min-rule face
aperture. The J≡1-limit gate plus the algebraic gates remain the right
certification.

## Construction

Pin the surface flat: chart A is `zp = z·H(x)` over `z ∈ (−1, 0)`
(`H = 1 + 0.3 sin x`, genuine J), with an immersed bottom cutting at
physical height `ζ(x) = −0.5 + 0.12 sin(x + 0.7)` (resolution-scaled
indicator width ∝ dz, `min_fraction=0` — the default 0.1 floor clamps
cut fractions into a non-converging O(1) perturbation). Chart B is the
wet sub-column as its own **single-parameter** chart: `zp = z·D` with
`D(x) = −ζ(x)` — the same `CoordinateMapping` template, no
affine-offset map needed. Comparison needs no interpolation: physical
J-weighted functionals `⟨q,φ⟩ = (θ·q·φ).integrate()` on A vs
`(q·φ).integrate()` on B are grid-independent scalars.

Two mechanical traps (worth recording — both silent):

- **`init` callables see DIFFERENT vertical coordinates on a chart.**
  The immersed indicator receives chart-physical `zp` (the column
  correction substitutes `M(z, H)`), but `create_field(init=...)`
  receives computational z and does NOT apply the chart map. A
  physical IC `f(x, y, zp)` must be composed manually as
  `init(x, y, z) = f(x, y, z·H(x))`; getting this wrong silently
  mis-projects the IC.
- Slivers down to θ ≈ 0.002 appear at `min_fraction=0`; the
  spectral-preconditioned CG (tol 1e-12) converged everywhere and the
  functionals are solver-independent (identical to 3–4 significant
  figures between tol 1e-8 and 1e-13).

## Convergence (nonhydro2, advection off, f0=1, N²=1, dsqr=0.5,
dt=0.01, 20 steps, n = 8→64 with nx=ny=nz)

Tracer (buoyancy) |A−B| physical moments — clean 2nd order:

| functional | \|A−B\| at n=8,16,32,64 | orders |
|---|---|---|
| b:one | 1.97e-3, 5.02e-4, 1.25e-4, 3.02e-5 | 1.98, 2.01, 2.05 |
| b:z | 1.55e-3, 3.94e-4, 9.89e-5, 2.41e-5 | 1.97, 2.00, 2.04 |
| b:cos2x·z² | 1.70e-3, 4.14e-4, 1.05e-4, 2.64e-5 | 2.04, 1.98, 1.99 |

Velocity |A−B| physical moments — noisy ~1st order, some
non-monotonic (all do decrease → correct, but boundary-order-limited):

| functional | orders over 3 doublings |
|---|---|
| u:cos2x·z² | 0.84, 0.66, 0.75 |
| v:cosx | 0.05, 0.77, 0.90 |
| v:one | −1.32, 0.65, 0.64 |
| w:one | 0.02, 0.62, 0.83 |
| w:z | 0.06, 0.58, 0.80 |

The 2nd-order tracer result additionally requires the
resolution-scaled (sharpening) cut: a FIXED-width indicator (0.1
physical — what a bathymetry user would declare) plateaus toward a
smeared-boundary limit (b:cos2x·z² order 1.23 → 0.56).

## Mechanism

- Cell volume fraction θ is GL-quadratured (`θ = ∫Jχ/∫J`) → 2nd
  order → tracer moments 2nd order.
- Face fraction `α = min(θ_adjacent)` (IP-D2 min-transfer, MITgcm
  `hFacW = min(hFacC)`): the bottom-most open z-face is physically
  fully wet but carries the cut cell's volume fraction — an O(1)
  local aperture error on the boundary face → 1st-order, noisy
  velocity convergence. This is the documented MITgcm partial-cell
  topography order, model-independent (the hydrostatic wet-column
  depth `H̃ = ∫αJ dz` uses the same α, so the tracer-2nd/velocity-1st
  split transfers by construction — reasoned inference; nonhydro2
  was probed directly).

## Why no test ships

A max-over-functionals gate reads "2nd order" (2.04, 1.98, 1.99) only
because tracer magnitudes dominate the velocity's 1st-order tail —
certifying the wrong thing, fragile to the IC amplitude ratio, and
bound to surface the 1st-order velocity at higher n or longer runs
(the 1st-order w drives b through `db/dt = −N²w`). A
velocity-inclusive gate is non-monotonic at reachable n. The clean
signal (short-time tracer moments) is projection/quadrature-dominated
(2nd order already at t=0, no solve involved) and is essentially
implied by the shipped θ-mass and all-wet-bitwise gates. This
corroborates the earlier in-tree decision that a genuine-cut-chart
manufactured-solve convergence study is small-cell/preconditioner
limited (`tests/nonhydro2/test_composed_pressure.py`, "established by
inheritance" comment).

A shippable non-flaky variant exists if a genuine-J≠1 physical
certificate is ever specifically wanted: a short-run,
resolution-scaled-cut, tracer-moment-only 2nd-order twin (two
resolutions, monotonic, solver-independent, cheap) — recorded here,
recommended against (marginal over existing gates).
