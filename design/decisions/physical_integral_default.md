---
status: decided + implemented
date: 2026-07-18
---

# The physical integral is the default on every mapped grid

**Status: decision + implementation record, 2026-07-18.** Owner ruling
(Silvano, in chat): on a grid whose `CoordinateMapping` carries an
analytic single-base `maps=` column (terrain following,
`zp = z * H(x, y)`), the seeded reduction rows —
`ScalarField.integrate` / `.mean` and the `cumint` dispatch rows —
contract against the **physical** volume element (the column Jacobian
`d<p>_d<b>`), exactly as embedding `chart=` grids have since stage C2.
The two mapping forms behave identically under the field verbs; the
computational (sigma-frame) reduction remains available through
directly constructed operator instances (`Integral()` /
`CumulativeIntegral()` with no `jacobian=`). Implemented on
`refactor/physical-integral-default`; spec text updated in
`../specs/grid/02_rules.md` §3.13 and
`../specs/grid/classes/operators_products.md`.

## 1. Why

The asymmetry was an accident of construction order, not a design
intent: embedding charts seeded `Integral(jacobian=...)` from birth
(there is no meaningful computational area on a sphere), while the
terrain `maps=` form predated the `jacobian=` seam and kept plain
rows. The consequences surfaced in the `EnergyMetric`-on-charts
roadmap item: every physics-facing consumer of `.integrate()` (energy
inner products, norms, diagnostics reductions) silently computed
sigma-frame integrals on terrain grids — measured on a 20%-amplitude
seamount, the shipped energy metric leaks the conserved quadratic
invariant at ~9e-4 relative skew while the J-weighted form is
machine-zero (3e-16). Mathematically the terrain map *is* a chart
(`(x, y, zp) = F(x, y, z*)` with identity horizontal legs), so the
sane rule is one default for both forms: **the verbs mean physics;
the raw operators mean implementation.**

## 2. The rule

- **Field verbs are physical.** `f.integrate(...)` / `f.mean(...)` on
  any grid whose mapping derives a volume element (an embedding
  chart's `sqrt_g`, or a single-base analytic column's `d<p>_d<b>`)
  weight the reduction by it. A multi-coordinate analytic map derives
  no volume element and keeps computational rows. `mean`'s divisor
  follows the same measure as its numerator on both mapping forms
  (per-column `H(x, y)` for a terrain `mean("z")`, the `sqrt_g` area
  on a chart), double-`where` guarded for `J = 0` padding columns.
- **Raw operators are computational.** A directly constructed
  `Integral()` / `CumulativeIntegral()` bypasses the seeded rows and
  contracts against the plain `grid.measure` quadrature — the escape
  hatch for implementation-layer code whose integrand already carries
  the Jacobian (flux forms, `_physical_depth`-style definitions) or
  whose reduction is algebra rather than physics.
- **Solver internals stay computational, explicitly.** The discrete
  flux-form operators are SPD in `diag(grid.measure)`, not in the
  physical product, so the Krylov default inner product and every
  nullspace/gauge projection and preconditioner coefficient fold are
  pinned to `krylov._computational_integral` / `_computational_mean`:
  `krylov._default_inner`/`_project`, the mapped-pressure
  `_mean_free`/`_mean_coefficients`, and the hydrostatic barotropic
  solve's `_mean_free`/mean-depth fold (`barotropic_pressure.py`).
  Documented in place as deliberate pins.
- **The constant-factor rule is unchanged.** A ConstantSpace factor
  still reduces as the identity (idempotence of `integrate`, the
  strict algebra). A field born constant along a mapped axis (the
  hydrostatic `ps`) picks up no geometry from `integrate`; its depth
  weighting belongs to its consumer (the `EnergyMetric` `ps` weight —
  the open remainder of the roadmap item). This also keeps the
  barotropic volume gate `ps.integrate()` flip-invariant.
- **Ordering.** The column Jacobian varies over the map's *parameter*
  axes (`H(x, y)`), so the verbs front Jacobian-carrying axes in the
  reduction order (`scalar_field._bases_first`); a hand-built
  sequence that collapses a parameter axis before the base axis
  raises a taught error (`jacobian_weight.py`) instead of
  mis-broadcasting.

## 3. What moved (audit summary)

Flipped (physical intent, verb kept): `EnergyMetric.inner` (the
`u`/`v`/`b` legs of the terrain energy — the roadmap item's driver),
`transforms/norms.relative_l2`, all user-facing reductions on mapped
grids. Pinned (algebra): the Krylov/mapped-pressure/barotropic-solve
sites above, plus the per-file `dot`/gauge helpers of the mapped test
suites. Unchanged: every raw-operator site (hydrostatic sigma core,
free-surface flux forms, `_physical_depth`), immersed-only reductions
(immersed grids are never mapped today), split-explicit `mean(z)`
(taught error on terrain). Terrain test oracles that hand-multiplied
`J` before the verb were migrated to the plain verb and now gate the
flip itself. The migration initially surfaced what looked like a
product asymmetry (baroclinic pair computational-exact, barotropic
pair physical-exact). **Corrected same day** by the follow-up probes
([`../research/energy_metric_asymmetry.md`](../research/energy_metric_asymmetry.md)):
the computational-exactness of the baroclinic pair holds only on flat
grids (the smooth terrain gate passes by state-selection accident);
on terrain the pair leaks O(slope) under *both* plain metrics because
the buoyancy equation is missing its slope-advection term — a physics
gap, not a metric property. Only the barotropic physical-exactness
statement stands. The full-suite gate on the merged state: 3617
passed, 0 failed.

## 4. Follow-ups

- `EnergyMetric` `ps` weight `H(x, y)/c^2` and the missing depth
  factor on flat grids with depth != 1 (probe: skew 9.6e-2 at
  depth 2, machine-zero with the corrected weight) — the metric-side
  remainder of the roadmap item, plus the eigen-channel
  `_bounded_measure` physical measure (stretched-z) and a terrain
  taught error replacing the misleading Hermiticity-residual message.
- The terrain buoyancy slope-advection term
  (`−N²(u·Zₓ + v·Z_y)`, missing from `stratification.restoring`) —
  the root of the apparent product asymmetry; research record
  [`../research/energy_metric_asymmetry.md`](../research/energy_metric_asymmetry.md).
