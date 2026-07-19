---
status: active
date: 2026-07-19
---

# Spherical 3-D models — hydrostatic first, nonhydro after

Driver (owner, 2026-07-19): spherical 3-D models are promoted out of the
long-term goals into the pre-docs pipeline as one feature — "Spherical
3-D models — hydrostatic first, nonhydro after; `sw2` is the metric
reference" — preceded by this dedicated planning phase
([spherical_models_scoping.md](../../research/spherical_models_scoping.md),
2026-07-19). This plan is design-only: no `src/` or `tests/` code lands
on its branch. It fills in the seams the coordinate-systems campaign
([coordinate_systems_plan.md](../done/coordinate_systems_plan.md), stages
C0–C4) designed for but left unbuilt on the 3-D models, reusing the
shipped chart machinery, the `sw2` metric-correct reference, and the
mapped/multigrid elliptic scaffolding.

Nothing here is a redesign of the chart layer: the induced-metric
derivation, the metric-aware dispatch kinds, the `sw2` seam-conversion
pattern, the `RotationCoriolis`, and the CG/multigrid pressure
scaffolding all exist. What is missing is (1) 3-D chart-grid ergonomics,
(2) a **3-D** metric divergence/gradient (the shipped ones are 2-D
chart-only), (3) metric-aware **momentum** advection (curvature terms),
(4) the two 3-D cores' chart paths, and (5) the chart Laplace–Beltrami
pressure solve.

## 0. Context — what the scoping probe and this plan's probe established

From the scoping record (all findings backed by executed CPU probes plus
cited code paths):

- **Spherical `sw2` works today and is metric-correct** — it is the
  reference. Its correctness lives in the seam conversions
  ([`shallowwater2/chart.py`](../../../src/fridom/shallowwater2/chart.py):
  `to_contravariant` = `U_i/√g_ii` sealed divide at entry,
  `to_physical_tendency` = `√g_ii·du^i` at exit) plus the metric-aware
  dispatch kinds and the Sadourny metric self-advection
  ([`shallowwater2/modules/core.py`](../../../src/fridom/shallowwater2/modules/core.py):54-75,495-573;
  [`sadourny.py`](../../../src/fridom/shallowwater2/modules/sadourny.py):986-1087).
- **Spherical hydrostatic does not work**: `HydrostaticCore` hard-refuses
  any embedding chart at bind
  ([`hydrostatic/modules/terrain.py`](../../../src/fridom/hydrostatic/modules/terrain.py):98-106).
- **Spherical `nonhydro2` does not even assemble**: velocity
  declarations hardwire `x/y/z`
  ([`nonhydro2/modules/core.py`](../../../src/fridom/nonhydro2/modules/core.py):509-517)
  though the `coords=` arg exists and is already used for the halo
  derivation (:553) — an internal inconsistency; and the projection
  routes on empty `column_corrections` to the flat metric-blind
  `SpectralPressureSolver` (:731-746) while `MappedPressureSolver`
  explicitly rejects charts
  ([`mapped_pressure.py`](../../../src/fridom/nonhydro2/modules/mapped_pressure.py):509-520).
- **The shared flux-form advection is metric-blind on charts** and
  silently drops all curvature terms
  ([`model/modules/advection.py`](../../../src/fridom/model/modules/advection.py):
  `_bind_mapping` returns early on empty `column_corrections` :2560-2563;
  `_flux_divergence` returns `flux.diff(axis).retag(q)` when `_column is
  None` :2974-2975). It is **latent, not a live bug** — both 3-D models
  refuse charts before it bites — but it is a fence that must be made
  explicit before S1 opens the chart path.

This plan added one design probe (CPU, tiny grid) that settles the
central thin-shell question:

- A 3-D `(lon, lat, z)` grid under `lonlat_sphere` **builds**;
  `chart_coords == ('lon','lat')`, `column_corrections == {}`.
- `grid.metric(cell_3d, "sqrt_g")` **derives** on a 3-D cell space and is
  **z-independent** (shape `(nlon, nlat, 1)`, range `cos(lat)`) — the
  shallow-atmosphere factorization holds in the machinery already.
- **But** the shipped metric `div`/`grad`/`curl` (the `MetricDivergence`
  / `MetricGradient` builders, [`spatial/operators/composed.py`](../../../src/fridom/spatial/operators/composed.py))
  **reject a 3-D operand**: `_require_chart_axes` (:969-982) raises
  `SpaceMismatchError` because a C-grid velocity on `(lon,lat,z)` varies
  along all three axes while the metric entry is registered for exactly
  `('lon','lat')`; and `div` demands one component per grid axis. So the
  thin-shell "horizontal-metric ⊕ flat-vertical" divergence/gradient
  **must be built** — it is not a free reuse of the 2-D chart operators.
  This is the wide-blast-radius core of S1.

## 1. Scope and strategy

Four axes of "start simple", each ratified as a decision below:

1. **Hydrostatic before nonhydro.** Hydrostatic needs no elliptic solve
   for a minimal ship (explicit free surface), so it reaches a running
   spherical model soonest; nonhydro adds the chart Laplace–Beltrami
   projection. (Scoping delta lists (a) then (b).)
2. **Thin-shell before full 3-D chart.** The chart maps only `(lon,lat)`
   → sphere surface; the vertical is a flat/stretched `z` with `g_zz = 1`
   and `√g` independent of `z` (shallow-atmosphere). The full
   `X(lon,lat,h)` deep metric (`g_λλ = (a+h)²cos²φ`) is deferred (SP-D2).
3. **Centered before biased.** Metric-aware momentum advection lands for
   the order-2 centered scheme first; upwind/WENO on charts follow, and
   stay a taught error until they do — the shipped `_supports_mapped`
   precedent (SP-D5).
4. **`sw2` is the metric reference at every step.** Every phase's gate
   includes a `sw2` cross-check in the thin-shell one-layer limit and the
   identity-chart bitwise reduction; the seam-conversion pattern and the
   PHYSICAL-components invariant are reused verbatim (SP-D7).

The phase spine: **S0** 3-D chart-grid ergonomics → **S1** 3-D
metric-aware momentum advection (the shared, wide-blast-radius phase) →
**S2** hydrostatic chart core + explicit free surface (**a minimal
spherical hydrostatic ships here**) → **S3** barotropic Helmholtz on
charts (implicit / split-explicit) → **S4** nonhydro on charts (chart
Laplace–Beltrami + metric projection + `coords` threading).

## 2. Owner calls — decisions needing sign-off

Decision IDs follow the house `<prefix>-D#` convention (HY-D, FV-D, GM-D,
CS-D, …). The five calls the owner flagged are SP-D1..SP-D5; the
technical rulings SP-D6..SP-D9 are recommended for ratification with the
plan.

### SP-D1 — S1 advection form (THE main call): flux-form taught the chart metric, **not** vector-invariant 3-D. **Recommend: flux-form (option b).**

Two ways to make momentum advection metric-correct on a chart:

- **(a) Vector-invariant / Sadourny extended to 3-D.** Momentum as PV
  flux + kinetic-energy gradient (the Lamb form), with contravariant
  conversion + `√g`-weighted mass fluxes + metric self-advection — the
  `sw2` pattern.
- **(b) Flux-form advection taught the chart metric.** Each velocity
  component is transported as a scalar by the `√g`-weighted flux
  divergence `(1/√g)∂_i(√g F^i)` on the horizontal (charted) axes and the
  plain `∂_z F^z` on the flat vertical, **plus** a curvature/Christoffel
  source term on the momentum equations (the `u·v·tanφ/a`-type terms a
  velocity component carries because it is a vector component, not a
  scalar). Tracers stay flux-form with **only** `√g` weighting — a scalar
  has no Christoffel terms, so tracers need no curvature source (stated
  explicitly, as required).

**Analysis.**

- *Correctness burden.* Both share the entry/exit seam conversion and the
  `√g`-weighted flux divergence. (b) adds one curvature source term per
  momentum component (closed-form Christoffel symbols of the chart,
  derivable from `grid.metric`); the shipped 3-D flux-form transport is
  otherwise untouched. (a) requires writing a **new** scheme: `sw2`'s
  Sadourny is a *2-D single-layer* vorticity/KE scheme (2-D vorticity is
  a scalar); the 3-D vector-invariant form needs the full 3-D vorticity
  vector, the Lamb term `ω×u`, and a vertical-momentum treatment `sw2`
  has no equation for. That is a research-grade port, not a reuse.
- *Energy / enstrophy.* (a) buys exact semi-discrete energy (and, in the
  enstrophy variant, potential enstrophy) — the property `sw2` proves. (b)
  is consistent to truncation order only. **But the flat 3-D models do
  not have exact-energy momentum advection today** — their flux-form
  advection is not an energy-conserving pair with the pressure gradient
  the way `sw2`'s gravity + Sadourny are. Demanding it *only on charts*
  would make the chart path a structurally different scheme from the
  Cartesian path.
- *WENO / upwind under (a).* There is no vector-invariant analog of a
  biased face reconstruction; adopting (a) **abandons WENO/upwind on
  charts entirely**. Under (b) the biased reconstructions are unchanged —
  only the divergence weighting and the momentum source term change — so
  the whole Centered/Upwind/WENO family serves charts (staged, SP-D5).
- *Blast radius on the shared module.* (b) is contained to
  `advection.py`'s `_flux_divergence` chart branch + a new momentum
  curvature term, shared by both 3-D models. (a) means **replacing** the
  3-D advection module with a new scheme and re-deriving its conservation
  proofs.
- *Per-model consistency.* Tracers (both models) need `√g` only;
  momentum needs `√g` + curvature — one flux-form module expresses both
  by adding the source on the momentum terms. Under (a) tracers would
  still be flux-form (no vector-invariant tracer scheme), so the model
  would run *two* advection paradigms.
- *Testability against `sw2`.* A one-layer thin-shell hydrostatic run
  reduces to shallow water; (b)'s momentum must match `sw2`'s steady
  states (solid-body rotation, TC2) to truncation and converge at 2nd
  order — a real cross-check that does not require (b) to *be* Sadourny.

**Decisive argument.** Option (b) **preserves the flat↔chart structural
identity** — the identity-chart-equals-Cartesian bitwise-reduction gate
the entire chart design rests on (`sw2`'s flat limit is bitwise;
[`test_spherical_shallowwater.py`](../../../tests/validation/test_spherical_shallowwater.py):93-174).
The Cartesian 3-D models are flux-form; teaching that same flux-form path
the metric keeps the chart program a neutral (×1/÷1) specialization of
the flat one. Option (a) makes the chart momentum a different scheme from
the Cartesian momentum, forecloses WENO/upwind, needs a novel 3-D scheme,
and is exactly how *no* production spherical primitive-equation model
(MOM, MITgcm, NEMO) discretizes momentum. `sw2` stays vector-invariant
in its own 2-D world; the thin-shell one-layer limit is the cross-check,
not a shared code path.

### SP-D2 — thin-shell (shallow-atmosphere) metric before full 3-D chart. **Recommend: thin-shell first; full `X(lon,lat,h)` deferred.**

- **Thin-shell (recommended):** chart on `(lon,lat)` only, `g_zz = 1`,
  `√g` independent of `z` (probe-confirmed). The 3-D metric divergence
  factorizes to horizontal-metric ⊕ flat-vertical, letting the `sw2` 2-D
  proof carry per z-level. This is the standard ocean-model approximation
  and the likelier first consumer (scoping §5).
- **Full 3-D chart metric `X(lon,lat,h)`:** deep-atmosphere `g_λλ =
  (a+h)²cos²φ` couples height into every horizontal weight; the metric
  operators would have to become fully 3-D-coupled. Deferred as a
  recorded follow-up — a refinement, not a blocker.

### SP-D3 — torus preset chart as a pole-free test chart. **Recommend: yes, add it in S0.**

Add a `torus(major, minor)` factory to
[`charts.py`](../../../src/fridom/spatial/charts.py) alongside
`lonlat_sphere`, the R³-embedded torus `X(u,v) = ((R+r cos v)cos u,
(R+r cos v)sin u, r sin v)` with **both** coordinates periodic,
`orthogonal=True`. Its metric is diagonal and varies (`g_uu = (R+r cos
v)²`, `g_vv = r²`, `√g = r(R+r cos v)`) but is **never zero** (`R>r`) and
has **no walls**. Cheap (~15 lines, mirrors `lonlat_sphere`) and powerful:
it exercises the full metric machinery (varying `√g`, non-unit `g_ii`,
closed manifold) while isolating metric-term bugs from the polar-cap wall
closures — rest states and steady flows on a torus are clean
machine-zero / bitwise gates. The scoping record flagged it explicitly.

### SP-D4 — pole treatment: keep the excluded-cap convention. **Recommend: keep spherical.Grid's closed-lat-walls-before-the-poles; poles-inside is a non-goal.**

`spherical.Grid` already requires a bounded latitude band strictly inside
`(-π/2, π/2)` and raises if it reaches a pole (`√g = a cosφ → 0`,
[`spherical/grid.py`](../../../src/fridom/spatial/spherical/grid.py):91-96).
Document the **excluded-cap convention** in the plan and the model docs:
the domain is a latitude band; the caps are structural no-normal-flow
walls (interior faces only; the cap value is a boundary condition, not a
DOF — exact impermeability, the `sw2` precedent
[`test_spherical_shallowwater.py`](../../../tests/validation/test_spherical_shallowwater.py):260-277).
Poles-inside-the-domain needs a multi-chart atlas / cubed sphere and is a
non-goal (§6). The vertical extrusion in S0 inherits this guard verbatim.

### SP-D5 — S1 lands per-scheme (centered first) or all-at-once. **Recommend: centered first; biased schemes a fenced follow-up.**

The order-2 centered scheme is the clean minimal metric-aware momentum
advection that unblocks S2/S4. The biased schemes' order-wide
uniform-offset windows need a mapped/metric-aware reconstruction — the
same reason they opt out of mapped grids today via `_supports_mapped`
([`advection.py`](../../../src/fridom/model/modules/advection.py):2564-2572).
Land `CenteredAdvection` on charts in S1; keep `Upwind`/`WENO` a **taught
error** on charts (extend the `_supports_mapped`/`_bind_mapping` gate to
`chart_coords`) until a dedicated follow-up phase generalizes the biased
reconstructions. The metric-blindness stays fenced, never silently wrong.

### SP-D6 — where the 3-D thin-shell metric operators are built. **Recommend: extend the spatial metric operators to a charted-subset ⊕ flat-remainder form; the momentum curvature source rides the advection module.**

The probe proved the 2-D chart `div`/`grad`/`curl`/`laplacian` reject a
3-D operand. Two homes for the 3-D form:

- **(recommended) Spatial-layer extension.** Generalize
  `MetricDivergence`/`MetricGradient`/`RaiseIndex`/`LowerIndex`/
  `MetricLaplacian` and `_require_chart_axes` to accept an operand whose
  axes are `chart_coords ∪ flat_axes`: apply the `√g` weighting /
  `raise_index` on the charted axes and plain staggered differences on
  the flat remainder (`√g` z-independent makes the vertical leg exactly
  the flat one). Single source of truth: the **scalar** continuity /
  tracer divergence, the pressure gradient, and — crucially — the S4
  chart Laplace–Beltrami all resolve through the same seeded kinds, and
  the mimetic `div = −grad*` adjointness stays proven centrally. Keeps
  the house rule "modules resolve kinds, never hand-build metric
  compositions".
- **(fallback) Module-level composition** in `_flux_divergence`: the
  advection module builds `(1/√g)[∂_lon(√g F^lon)+∂_lat(√g F^lat)] +
  ∂_z F^z` directly from `grid.metric(space,"sqrt_g")` and plain
  `.diff()` — consistent with how the mapped-column branch is already
  hand-written (:2976-3008), but S4 would then re-hand-build the
  Laplace–Beltrami instead of reusing the kind.

Either way, the momentum **curvature/Christoffel source** is
momentum-specific (the covariant divergence of the rank-2 momentum-flux
tensor, not of a vector) and is carried by the advection module on top of
the scalar-transport divergence. Recommendation: spatial extension for
the scalar operators, module source for curvature.

### SP-D7 — PHYSICAL-components invariant on charts + seam-conversion home. **Recommend: keep the invariant (ratified); promote the seam helpers to a shared location.**

Stored `u,v,w` are PHYSICAL m/s on every grid, all the way down (pytree,
tendencies, checkpoints, io) — the owner-ratified invariant
([physical_state_components.md](../../decisions/physical_state_components.md),
rulings (a)–(d); `sw2` flip merge `90722cf2`, hydrostatic-`w` flip
`28a5ff3d`). The spherical 3-D formulation converts physical →
contravariant at term entry (sealed divide by `√g_ii` on the component's
own space) and rescales tendencies to physical at exit — the `sw2`
`chart.py` seam pattern, reused verbatim for the 3-D momentum trio.
Chart-native coordinate velocities are exposed read-only via
`state.chart` (the `ChartView` / `nonhydro2/chart.py` hook). Because three
packages now need the seam conversions, promote `to_contravariant` /
`to_physical_tendency` (and the sealed metric divide) from
`shallowwater2/chart.py` to a shared home (`fridom.spatial` or
`fridom.model`), with per-package `state.chart` hooks unchanged.

### SP-D8 — fence the metric-blind flux-form path first. **Recommend: yes, taught error before the metric path.**

The first commit of S1 converts the latent silent curvature-drop into an
**explicit taught error**: `_bind_mapping` / `_flux_divergence` raise on a
chart grid ("metric-blind on charts; use the metric-aware chart path")
for any scheme not yet chart-capable, so no intermediate commit ships a
silently-wrong chart advection. The fence stays for the biased schemes
until SP-D5's follow-up lands them.

### SP-D9 — Coriolis on the 3-D chart. **Recommend: reuse RotationCoriolis; traditional approximation default, non-traditional terms a follow-up.**

`RotationCoriolis` is the chart-capable rotation (derives `f = 2Ω·n̂`
from the chart surface normal; shipped, scoping §3 lists Coriolis as "None
(shipped)" for both models). Verify at S2/S4 that it wires into the 3-D
`(u,v,w)` core state and default to the traditional approximation
(horizontal `f v`, `−f u`); the non-traditional `2Ω cosφ` terms coupling
`w` are a recorded follow-up (a physics refinement, not an assembly
blocker).

## 3. Phases and gates

Each phase keeps the taught-error discipline: the chart path a phase does
**not** yet make correct stays an explicit refusal (never silent). Every
step-path phase ships the house autodiff regression (AGENTS
differentiability policy: a `jax.grad` of a quadratic loss through a short
chart run, FD-matched to rtol 1e-4, via `Model.propagator` or
`_chunk_body`).

**S0 — 3-D spherical (and torus) grid ergonomics.** A vertical-extrusion
convenience so `spherical.Grid` yields a `(lon,lat,z)` grid (the
hand-assembly the scoping probe used already builds); a `torus` chart
preset (SP-D3). No model code. *Size: small* (the `sw2` grid-ergonomics
precedent, E-series).
- Gate S0-1: constructed 3-D sphere/torus grids report the right
  `chart_coords`, and `grid.metric` derives `√g`/`g_ii`/`inv_g` on 3-D
  cell and face spaces (probe-level, now a test).
- Gate S0-2: the polar-cap guard (SP-D4) fires on a pole-reaching band;
  the torus has no wall and `√g > 0` everywhere.
- Gate S0-3: `test_init` re-exports; ruff clean.

**S1 — 3-D metric-aware momentum advection (centered) — the
wide-blast-radius phase.** (a) Fence the metric-blind path (SP-D8). (b)
Build the charted-subset ⊕ flat-remainder metric `div`/`grad` (and
`raise`/`lower`, needed by the pressure gradient and S4) per SP-D6. (c)
Teach `CenteredAdvection`'s `_flux_divergence` the chart branch: `√g`-
weighted horizontal flux divergence + flat vertical + the momentum
curvature source; tracers `√g`-only, no curvature. (d) Reuse the seam
conversions (SP-D7). *Size: module→campaign, the largest phase* (new
spatial operators + shared advection change + curvature derivation;
calibrated against the mapped-advection F5 stage plus the C2 metric-op
work).
- Gate S1-1 (flat-chart bitwise): on the identity chart `X=(x,y,z)` the
  centered metric advection reduces to the flat scheme bitwise unfused
  (`jax.disable_jit`), ≤ few ULP jitted — the load-bearing gate (the
  `sw2` identity-chart precedent).
- Gate S1-2 (`sw2` cross-check, one-layer thin-shell): a single-`z`-level
  advective run's horizontal momentum matches `sw2` solid-body-rotation
  steadiness and TC2 drift to truncation and converges at 2nd order.
- Gate S1-3 (curvature is real): solid-body zonal flow stays steady only
  with the curvature source on; dropping it drives the O(1) `u·v·tanφ/a`
  drift (the `sw2` Sadourny-off discriminator, generalized).
- Gate S1-4 (tracer): a passive tracer under solid-body rotation on the
  sphere/torus conserves `√g`-weighted mass to rounding and stays
  monotone-consistent (no curvature applied to the scalar).
- Gate S1-5 (autodiff): `jax.grad` through a short centered-advection
  chart run, FD-matched; the sealed seam divide keeps the VJP finite at
  the caps.
- Gate S1-6 (forced-4): the centered chart tendency is device-count
  invariant to the honest to-rounding bound (the `sw2` forced-4
  precedent).
- Gate S1-7 (fence intact): `Upwind`/`WENO` on a chart raise the taught
  error (SP-D5); the flat and mapped-column paths stay byte-identical.

**S2 — hydrostatic chart core + explicit free surface (a minimal
spherical hydrostatic ships).** Lift the chart refusal
([`terrain.py`](../../../src/fridom/hydrostatic/modules/terrain.py):98-106)
into a chart-aware arm. Chart `_diagnose_w`: the horizontal continuity
becomes the `√g`-weighted metric divergence `(1/√g)[∂_lon(√g u^lon)+
∂_lat(√g u^lat)]` (contravariant conversion), the `CumulativeIntegral`
over the flat `z` unchanged (shallow-atmosphere `√g` z-independent, so the
vertical FTC telescopes as before). Chart `pressure_gradient`: `grad →
raise_index` on `p_hyd` (the covariant gradient raised to the
contravariant tendency `−g^{ij}∂_j p_hyd`), rescaled to physical — the
`sw2` gravity pattern. `_diagnose_p_hyd` (vertical buoyancy integral) is
unchanged under thin-shell. `ExplicitFreeSurface` runs with no elliptic
solve — the minimal spherical hydrostatic. Coriolis via `RotationCoriolis`
(SP-D9). *Size: module→campaign* (HY-scale core surgery, no solver).
- Gate S2-1 (flat-chart bitwise): identity chart reduces to the Cartesian
  hydrostatic byte-for-byte (the `self._column is None` flat arm).
- Gate S2-2 (`sw2` limit): one-layer thin-shell hydrostatic + explicit
  free surface reproduces `sw2` mass conservation (`√g`-weighted, to
  rounding) and TC2 steadiness.
- Gate S2-3 (rest state): a stratified fluid at rest on the sphere and on
  the torus stays at rest to machine precision (no spurious current from
  the chart pressure gradient).
- Gate S2-4 (analytic convergence): Rossby–Haurwitz wave (or a
  manufactured balanced state) converges at 2nd order; solid-body
  rotation steadiness.
- Gate S2-5 (caps): polar-cap no-normal-flow is structural and exact; the
  mass budget closes to machine zero.
- Gate S2-6 (autodiff, forced-4): the house autodiff regression through a
  short spherical hydrostatic run, FD-matched; forced-4 device-count
  invariance.
- Gate S2-7 (fence intact): `ImplicitFreeSurface`/`SplitExplicit` on a
  chart raise the taught error until S3; nonhydro still refuses charts.

**S3 — barotropic Helmholtz on charts (implicit / split-explicit).** The
variable-coefficient barotropic Helmholtz on a chart:
`(εI − dt'² ∇_h·(H √g^{-1}... ∇_h)) ps = rhs`, the chart Laplace–Beltrami
of the barotropic pressure. Reuse the shipped `BarotropicPressureSolver`
scaffolding
([`hydrostatic/modules/barotropic_pressure.py`](../../../src/fridom/hydrostatic/modules/barotropic_pressure.py),
GM campaign): the per-axis field-arithmetic SPD operator, the flat
spectral-inverse preconditioner (CS-D2), the point-Jacobi multigrid
option, warm start, mean gauge — swapping the terrain-column operator legs
for the chart metric legs (SP-D6 kinds). Lift the
`ImplicitFreeSurface.bind` chart refusal. *Size: module→campaign*
(directly analogous to the GM terrain-barotropic phase, GM-D1..GM-D8).
- Gate S3-1 (exact cancellation): at ε=0 the corrected barotropic
  transport divergence is ≤ 1e-13·scale.
- Gate S3-2 (flat-chart): identity chart matches the flat spectral solve
  to machine precision.
- Gate S3-3 (volume): plain `∫ps` conserved to round-off (the GM-D1
  volume-exact form on a chart).
- Gate S3-4 (h-independent PCG): iteration count flat across resolutions
  on the sphere (the spike-count precedent); multigrid preconditioner
  green.
- Gate S3-5 (autodiff, forced-4): house autodiff regression through the
  implicit chart solve; forced-4 parity (MG-D5 replication on the 2-D
  chart hierarchy).
- Gate S3-6: the taught error flips to asserting the chart solve engages.

**S4 — nonhydro on charts.** (a) Thread `self._coords` into
`field_declarations` (the `sw2` pattern:
[`core.py`](../../../src/fridom/nonhydro2/modules/core.py):509-517 →
`zonal,merid,vert = self._coords`) and expose `coords=` on `nh.Model` —
*small*, closes the internal inconsistency the scoping probe found. (b)
Chart Laplace–Beltrami pressure solver: assemble `(1/√g)∂_i(√g g^{ij}∂_j
p) = div(u*)` (thin-shell: horizontal Laplace–Beltrami ⊕ flat `∂_z²`)
through the SP-D6 `MetricLaplacian` kind, solved SPD-PCG reusing the
`MappedPressureSolver` PCG / preconditioner / warm-start scaffolding — a
new chart operator (or a chart arm in `MappedPressureSolver` + routing at
[`core.py`](../../../src/fridom/nonhydro2/modules/core.py):731) since the
solver rejects charts today (`mapped_pressure.py`:509-520). (c)
Metric-aware projection div/grad: `_project` (:739-758) uses flat
`Divergence()`/`Gradient()`; the chart projection needs the `√g`-weighted
divergence RHS and the `raise_index` gradient correction (the `sw2`
precedent). *Size: campaign* (the chart elliptic operator is the hard
piece; C3/mapped-pressure calibration).
- Gate S4-1 (assembly): `nh.Model` builds on a `(lon,lat,z)` grid; the
  `x/y/z` KeyError is gone; flat/`x,y,z` runs byte-identical (coords
  default unchanged).
- Gate S4-2 (flat-chart): identity chart matches the flat
  `SpectralPressureSolver` projection to machine precision.
- Gate S4-3 (solenoidal): post-projection `√g`-weighted divergence ≤ CG
  residual on the sphere; the operator is the Laplace–Beltrami, not the
  Cartesian Laplacian (a manufactured-solution convergence check).
- Gate S4-4 (rest state / steadiness): rest state on sphere + torus to
  machine precision; solid-body geostrophic balance steady to truncation.
- Gate S4-5 (`sw2` limit): thin-shell one-layer nonhydro reduces to the
  shallow-water balance.
- Gate S4-6 (autodiff, forced-4): house autodiff regression through the
  chart projection (the `custom_vjp`-free CG-through-grad recipe);
  forced-4 parity.
- Gate S4-7 (fence): biased advection on charts still refused until the
  SP-D5 follow-up; chart+immersed and chart+terrain(sigma) remain taught
  errors (§6).

## 4. Gates catalog (cross-cutting)

The recurring gate families, gathered so each phase references rather than
re-specifies them:

- **Flat-chart reduction (bitwise).** Identity chart `X=(x,y,z)` == the
  Cartesian model; unfused bitwise under `jax.disable_jit`, ≤ few ULP
  jitted (the `sw2` precedent — SP-D1's decisive property). Every phase.
- **`sw2` cross-check (thin-shell one-layer limit).** A single-`z`-level
  run reduces to shallow water; matches `sw2` steady states / mass /
  energy to truncation. S1, S2, S4.
- **Solid-body rotation steadiness.** Zonal solid-body flow stays steady
  to truncation (the metric self-advection / curvature balance is O(1) —
  the `sw2` TC2 + Sadourny-off discriminators). S1, S2, S4.
- **Rest states (sphere + torus, machine zero).** No spurious current
  from the chart pressure gradient. S2, S4. The torus (SP-D3) isolates
  metric-term bugs from cap-wall bugs.
- **Analytic convergence for hydrostatic.** Rossby–Haurwitz (or a
  manufactured balanced state) at 2nd order. S2.
- **Autodiff regression (per phase).** `jax.grad` of a quadratic loss
  through a short chart run, finite and FD-matched to rtol 1e-4 (AGENTS
  differentiability policy); the sealed seam divide keeps the cap VJP
  finite. S1–S4.
- **Forced-4 multi-device (per phase).** Device-count invariance to the
  honest to-rounding bound; MG-D5 replication exercised on the chart
  hierarchies in S3/S4 (the `sw2` forced-4 + GM forced-4 precedents).
- **Taught errors that must REMAIN.** Never a silently-wrong chart path.
  The metric-blindness is fenced (SP-D8) from the first S1 commit; biased
  schemes refuse charts until their follow-up (SP-D5); hydrostatic
  implicit/split-explicit refuse charts until S3; nonhydro refuses charts
  until S4; chart+immersed and chart+terrain(sigma) stay refused (§6).

## 5. Sizing (against landed campaigns)

Calibrated on the `done.md` campaign records (terrain/hydrostatic HY-D1..7;
FV F0–F6 / FV-D1..4; multigrid GM-D1..9 phases A–G; mapped+immersed M0–M5;
`sw2` physical flip 1 module; coordinate-systems C0–C4):

| Phase | Scope | Class | Calibration |
|---|---|---|---|
| S0 | 3-D grid + torus ergonomics | small | `sw2` grid ergonomics (E-series, ~1 day) |
| S1 | 3-D metric momentum advection (centered) | module→campaign, **largest** | mapped-advection F5 + C2 metric-op work |
| S2 | hydrostatic chart core + explicit FS | module→campaign | HY core surgery (no solver) |
| S3 | barotropic Helmholtz on charts | module→campaign | GM terrain-barotropic phase (GM-D1..8) |
| S4 | nonhydro chart Laplace–Beltrami + coords | campaign | CS-C3 / mapped-pressure |

Whole feature: a multi-phase **campaign** on the scale of the
coordinate-systems campaign it extends (the largest by stage count),
sequenced S0→S4 with a **shippable minimal spherical hydrostatic at the
end of S2** (before any elliptic work) and a full spherical nonhydro at
S4. S1 is the pole-star risk (shared spatial-layer change); S4 the second
(the chart elliptic operator). SP-D5's staging means biased-scheme charts
are a bounded follow-up, not in the critical path.

## 6. Non-goals (designed-for or deferred, not precluded)

- **Full sphere with the poles inside the domain.** Needs a multi-chart
  atlas / cubed sphere; the excluded-cap band is the scope (SP-D4).
- **Multi-chart atlases** (cubed sphere, genus ≥ 2). Metric derivation is
  per-chart from day one (CS-D1), so this stays additive — but out of
  scope here.
- **Full 3-D / deep-atmosphere chart metric `X(lon,lat,h)`** (SP-D2) —
  thin-shell only; recorded follow-up.
- **Vector-invariant 3-D momentum** (rejected, SP-D1) — `sw2` keeps it in
  2-D.
- **Biased advection (upwind/WENO) on charts** — a bounded S1 follow-up
  (SP-D5), fenced until then.
- **Chart + immersed (cut-cell on a chart)** — `sw2` already refuses it
  (`core.py`:408-421, `sadourny.py`:671-685); the 3-D models inherit the
  refusal. A later composition, like mapped+immersed was.
- **Chart + terrain(sigma) composition** (sigma coordinate on a sphere) —
  the hydrostatic already refuses maps+chart together (`terrain.py`);
  keep it a taught error (a later composition).
- **Non-traditional Coriolis** `2Ω cosφ w` terms — a physics follow-up
  (SP-D9); the traditional approximation is the default.
- **Unstructured grids, dynamic AMR** — out of the framework's structured
  tensor-factor model entirely.
- **ALE / moving charts on the sphere** — the CS-D4 `MeshVelocityCorrection`
  module exists but a time-dependent sphere chart is out of scope here.

## 7. Risks

- **S1 spatial-operator generalization (SP-D6) could reach deeper than
  the operand gate.** The `charted-subset ⊕ flat-remainder` change touches
  `_require_chart_axes`, the component-count check, and the mimetic
  adjointness proof. Mitigation: a throwaway spike (the GM precedent —
  hack the operators on a 3-D grid, check `div = −grad*` and the flat
  reduction) before S1 freezes; the module-composition fallback of SP-D6
  is the escape hatch.
- **Curvature source correctness.** The Christoffel terms are easy to get
  subtly wrong; Gate S1-3 (curvature-off drifts O(1)) and the `sw2`
  one-layer cross-check (S1-2) are the discriminators. Derive the
  Christoffel symbols from `grid.metric` (autodiff of the chart), never
  hand-typed for the sphere only — so the torus (SP-D3) is a second,
  independent metric.
- **Seam-divide VJP at the caps.** `√g_ii → 0` in never-valid padding is
  a masked singularity; reuse the shipped double-`where` sealed divide
  (`chart.py`, `_sealed_metric_divide`) verbatim — do not re-invent it.
- **Forced-4 close-over blind spot.** Single-controller gates cannot catch
  jit close-over of sharded chart-metric constants; the S3/S4 elliptic
  chart operators are the risk. A 2-process CPU harness (the
  mapped+immersed `forced4-closeover` precedent) is the guard if a real
  `srun -n 4` cannot be run — and **agents never submit GPU jobs**
  (owner ruling); any multi-process GPU checkpoint is owner-authorized.
- **`RotationCoriolis` 3-D wiring (SP-D9).** Verify it declares against
  `(u,v,w)` and not just the `sw2` `(u,v)` — a S2 assembly check, cheap
  to falsify early.

## 8. Roadmap tie-in

Closes the open.md "Spherical 3-D models — hydrostatic first, nonhydro
after" item (promoted 2026-07-19). Extends the coordinate-systems
campaign (C2 shipped `sw2` on charts; this brings the 3-D models to
parity) and reuses the multigrid-generalization scaffolding (the chart
barotropic and chart Laplace–Beltrami solves ride the shipped
BarotropicPressureSolver / MappedPressureSolver CG + preconditioner +
warm-start machinery). References, not claims: CS-D1/CS-D2 (chart route +
PCG ancestor), GM-D1 (volume-exact barotropic), the
physical_state_components invariant, the `sw2` `chart.py` seam. Roadmap
3.7's "operator assembly must be written" is confirmed: the chart
Laplace–Beltrami genuinely must be built (the sphere's induced metric is
not a single-base analytic column), while the SPD-PCG structure,
preconditioners, and warm starts are mature reusable infrastructure.
