---
status: complete
date: 2026-07-19
---

# Spherical / torus support in the 3-D models — scoping probe

Empirical + code-audit answer to the owner question (2026-07-19):
"Does the hydrostatic model with spherical coordinates (or torus)
already work? How much do we need to change to make it work with the
nonhydrostatic model as well?" All findings are backed by executed
probes (CPU, small grids) plus cited code paths; no repo changes were
made by the probe.

**Bottom line.** Spherical **shallow-water** (`sw2`) works today and
is metric-correct — it is the reference implementation. Spherical
**hydrostatic** does **not** work: `HydrostaticCore` hard-refuses any
embedding chart at bind. Spherical **nonhydro2** does not even
assemble: its velocity declarations are hardwired to Cartesian
`x/y/z`, and its pressure routing has no chart (Laplace–Beltrami)
operator. The generic flux-form advection used by both 3-D models is
**metric-blind on charts** — it silently drops all curvature terms
(latent, fenced by the bind refusals). Roadmap 3.7's "operator
assembly must be written" still holds. Making either 3-D model
spherical is a **campaign**, not a small fix.

Owner ruling (2026-07-19): spherical 3-D models are promoted out of
the long-term goals into the pre-docs pipeline as one feature
("Spherical 3-D models — hydrostatic first, nonhydro after; sw2 is
the metric reference"), preceded by a dedicated planning phase
(`../plans/active/spherical_models_plan.md`).

## 1. Hydrostatic on the sphere — empirically refused

`spherical.Grid` (`src/fridom/spatial/spherical/grid.py:77-108`) is
**2-D only**: `shape=(nlon, nlat)`, two `IntervalMesh` factors under
the 2-coord `lonlat_sphere` chart (`src/fridom/spatial/charts.py:19-60`,
`X(lon, lat) -> R³`). It has no vertical axis, so it cannot host a
3-D hydrostatic model (which needs a bounded `z` for the
`CumulativeIntegral`).

A 3-D spherical grid **is** constructible by hand and builds fine
(probe verified): `fr.spatial.Grid((lon, lat, z),
mapping=lonlat_sphere(a))` → `chart_coords=('lon','lat')`,
`column_corrections={}` (charts declare no analytic column). But every
hydrostatic build on it raises, verbatim
(`src/fridom/hydrostatic/modules/terrain.py:100-106`, reached via
`discover_column`, guard at `terrain.py:98`):

> the hydrostatic terrain-following core supports analytic maps=
> column mappings (sigma coordinates, e.g. zp = z * H(x, y)); this
> grid carries an embedding chart= on ('lon', 'lat') (a curvilinear /
> spherical coordinate system), which the hydrostatic model does not
> model yet (hydrostatic plan §7). Assemble on a maps= terrain grid,
> or a flat / stretched-only grid

This fires for the linear model (`advection=False`), the advective
model, and `hy.comparison_model`, independent of Coriolis. No
rest-state check is reachable.

**Torus:** there is **no torus chart preset** — `charts.py` exports
only `lonlat_sphere`; "torus" appears only in docstrings
(`coordinate_mapping.py:789`, `coriolis.py:1078`). A `lonlat_sphere`
grid with periodic latitude *builds*, but it is geometrically bogus
(wraps through the metric-singular poles, `sqrt_g = a·cosφ → 0`). A
genuine torus needs a hand-written `CoordinateMapping` chart — and
hydrostatic would refuse it anyway (same chart guard). Note for the
plan: a torus chart has no pole singularity, making it an
analytically clean test chart for the metric machinery.

## 2. Metric-terms audit — flux-form advection is metric-blind on charts

The shared flux-form modules (`CenteredAdvection`/`Upwind`/`WENO`,
`src/fridom/model/modules/advection.py`) are a computational-frame
divergence that **silently omits every chart curvature term**. Two
lines decide it:

- `_bind_mapping` (`advection.py:2543-2581`) keys only on
  `mapping.column_corrections`; a chart's is **empty**, so it returns
  early (`:2562-2563`) — the chart is treated as flat.
  (`_require_uniform_factors` doesn't catch it either; the meshes are
  uniform.)
- `_flux_divergence` (`advection.py:2893-3008`): with
  `self._column is None` — **always** on a chart — it returns
  `flux.diff(axis).retag(q)` (`:2974-2975`). No `sqrt_g` weight, no
  `1/sqrt_g` prefactor, no `raise_index`, no curvature term. The
  metric-carrying branch (`:2976-3008`) fires only for single-base
  `maps=` terrain columns, never for an embedding chart.

This is a chain-rule mapping that omits curvature — not an exact
chart formulation. The curvature terms (e.g. `u·v·tanφ/a`) live
nowhere in the flux-form path. It is **latent, not a live bug**: both
3-D models refuse charts before this could bite (the taught-error
discipline holding), and `sw2` does not use the flux-form module for
its momentum.

**Contrast (the correct pattern): `sw2` is metric-exact.** Its
`DynamicalCore`/Sadourny convert physical → contravariant
(`u^i = U_i/√g_ii`), weight fluxes by `sqrt_g`, use metric-aware
`grad → raise_index` and `sqrt_g`-weighted `div`, and Sadourny
carries the **metric self-advection** (curvature) —
`src/fridom/shallowwater2/modules/core.py:54-75, 498-545`; helpers
`to_contravariant`/`weight_flux`/`scale_divergence`. The generic
`Divergence()` operator is itself flat (probe: matched the flat diff
to 0.000); `sw2`'s correctness lives in the flux weighting +
prefactors + contravariant conversion + Sadourny.

**Empirical discriminators (probes):**

- `sw2` solid-body zonal flow on the sphere stays steady
  (`v_max ≈ 1.3e-3` over 60 steps); the shipped test
  `tests/validation/test_spherical_shallowwater.py:197-232` balances
  Coriolis + metric self-advection and converges at 2nd order.
- `sw2` rest state on the sphere: exactly `0.0` for u, v, p over 30
  steps.
- Turning Sadourny advection **off** roughly triples the spurious `v`
  drift (3.5e-3 vs 1.3e-3) — the metric self-advection is a real O(1)
  balance term.
- `sqrt_g = a·cosφ` varies **3.46×** across a ±80° band — the weight
  the flux-form divergence drops.

## 3. Nonhydro on a 3-D spherical chart — cannot assemble

Two independent blockers, verified empirically:

**(a) Cartesian name-binding.**
`nonhydro2.DynamicalCore.field_declarations` **hardcodes** velocity
staggering on `"x"/"y"/"z"` (`src/fridom/nonhydro2/modules/core.py:509-516`).
On a `lon/lat/z` grid, `nh.Model` raises `KeyError: "no factor
contributes coordinate 'x'; this product's names are ('lon', 'lat',
'z')"` (from `bind`, `core.py:554`). The `coords=("x","y","z")`
constructor arg exists but is used **only** in `bind`'s extra-halo
derivation (`:553`) — it is not threaded to the declarations (an
internal inconsistency). Explicit assembly with
`DynamicalCore(coords=("lon","lat","z"))` also fails
(`StopIteration`). Contrast `sw2`, which derives staggering from
`coords` (`shallowwater2/modules/core.py:272-281`) and works.

**(b) No chart pressure operator.** Even with (a) fixed, the
projection routes on `mapping.column_corrections`
(`nonhydro2/modules/core.py:731`); empty on a chart → it falls to the
**flat `SpectralPressureSolver`** (`:743`), the discrete Cartesian
C-grid Laplacian — metric-blind, i.e. not the Laplace–Beltrami. It
would "converge" on the wrong operator. The `MappedPressureSolver`
(which does carry cross-term curvilinear operators) explicitly
**rejects** charts (`nonhydro2/modules/mapped_pressure.py:515-520`):

> the grid's coordinate mapping declares no single-base analytic map
> (no mapped column); the flat SpectralPressureSolver applies instead

So no chart pressure-solve convergence number is reportable — the
model does not build. (Aside: `FPlaneCoriolis`/`BetaPlaneCoriolis`
also reject charts, `model/modules/coriolis.py:458-478`;
`RotationCoriolis` is the chart-capable Coriolis and already works —
not a blocker.)

## 4. Delta list

**(a) Trustworthy spherical HYDROSTATIC:**

1. *3-D spherical grid ergonomics* — `spherical.Grid` is 2-D; add a
   vertical-extrusion convenience (hand-assembly already works).
   **Small.** `spatial/spherical/grid.py`.
2. *Lift the chart refusal in the core* — chart-metric-aware
   `w`-diagnosis (continuity with `sqrt_g`) and pressure-gradient
   stages; the guard at `terrain.py:98` and the DIAGNOSE stages in
   `hydrostatic/modules/core.py` assume flat or `maps=`-column
   metric. **Module → campaign.**
3. *Metric-aware momentum advection* — hydrostatic momentum needs the
   curvature terms flux-form advection omits (chart branch with
   `sqrt_g` weighting + curvature, or a Sadourny-style core).
   **Campaign** (shared with nonhydro). `model/modules/advection.py`.
4. *Barotropic free-surface elliptic on a chart* — the
   variable-coefficient barotropic Helmholtz on a chart is unbuilt
   (`hydrostatic_model_plan.md:367-372`; `ExplicitFreeSurface` is
   explicit and would run; `Implicit`/`SplitExplicit` need it).
   **Module → campaign.** A minimal spherical hydrostatic
   (explicit free surface) ships before any elliptic work.
5. *Coriolis* — `RotationCoriolis`. **None (shipped).**

**(b) Spherical NONHYDRO (on top of (a)):**

1. *Chart-generic velocity declarations* — thread `coords` into
   `field_declarations` (the `sw2` pattern) and expose it on
   `nh.Model`. **Small → module.** `nonhydro2/modules/core.py:509-516`.
2. *Chart Laplace–Beltrami pressure operator* — assemble
   `(1/√g)∂_i(√g g^{ii} ∂_i p)` and solve SPD-PCG. The
   `MappedPressureSolver` PCG/preconditioner/warm-start scaffolding
   is reusable, but it consumes column `J`/slopes (terrain) and
   rejects charts; a chart operator (new solver or a chart branch +
   routing at `core.py:731`) must be written. **Module → campaign.**
3. *Metric-aware momentum advection* — same as (a) 3. **Shared.**
4. *Metric-aware projection div/grad* — `_project` uses flat
   `Divergence()`/`Gradient()`; the chart projection needs the
   `sqrt_g`-weighted divergence RHS and `raise_index` gradient (`sw2`
   precedent). **Module.** `nonhydro2/modules/core.py:739-758`.
5. *Coriolis* — `RotationCoriolis`. **None (shipped).**

## 5. Relation to roadmap 3.7

"Operator assembly must be written" **still holds**: the mapped
CG/multigrid machinery does not serve embedding charts — it keys on
`column_corrections` (single-base `maps=` sigma columns) and
explicitly rejects charts. The sphere's induced metric is not
expressible as single-base analytic columns
(`g_lonlon = a²cos²φ` depends on `lat`, not `lon`), so the chart
Laplace–Beltrami genuinely must be written. What 3.7 is slightly
optimistic about: nothing — the SPD-PCG structure, preconditioners
and warm starts are mature, reusable infrastructure. What 3.7
**understates**: two of the hard blockers are neither C2 metrics nor
C3 elliptic machinery — nonhydro2's hardcoded `x/y/z` declarations
(§3a) and the metric-blind flux-form advection (§2). It also assumes
the 3-D `X(lon,lat,h)` chart; the simpler thin-shell case (2-D
horizontal chart × flat `z`) is equally unbuilt and is the likelier
first step. And "hydrostatic is the likelier first consumer" is
correct in principle but not currently closer — hydrostatic
hard-refuses charts too.

Probe scripts (session artifacts, not checked in): grid construction,
the verbatim hydro/nonhydro/mapped-solver refusals, the sw2
solid-body/rest-state/Sadourny-off discriminators, the flat-operator
comparison.
