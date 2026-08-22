---
status: complete
date: 2026-08-13
---

# Gallery expansion — coverage gaps, cost envelope, candidate examples

Answer to the question "what examples should we add, to cover physics
and features the current gallery misses?" Three inputs, all executed
rather than assumed: an audit of the new-stack capability surface
(`spatial/`, `model/`, the three models), an audit of what the 14
shipped examples already demonstrate, and a survey of the example
galleries and verification suites of Oceananigans, MITgcm, Dedalus,
Basilisk, Gusto, ROMS and MOM6. Every configuration proposed below was
**assembled and stepped on this machine** before it was recommended;
the cost numbers are measured, not estimated.

Companion records: [`../plans/active/docs_examples_plan.md`](../plans/active/docs_examples_plan.md)
(how examples are built and shipped), [`../specs/docs/structure.md`](../specs/docs/structure.md)
(the page tree — several gaps below are already claimed by it),
[`../specs/docs/style_guide.md`](../specs/docs/style_guide.md).

## 1. The cost envelope (measured)

Budget rule: an example stays under **~90 s locally** to survive the CI
slowdown factor (`docs_examples_plan.md`). Measured on this machine
(8 cores, `JAX_PLATFORMS=cpu`, AB3, warm), reporting steps/s and the
step count that fits the budget. GHA public runners are 4 vCPU, so
treat these as roughly 2x optimistic against CI.

| Configuration | steps/s | steps in 90 s |
|---|---:|---:|
| nh 1-D column, 1x1x128, +f | 300 | 27 000 |
| sw 2-D 256x256 | 208 | 18 700 |
| hy 96x96x4 (comparison preset) | 183 | 16 500 |
| nh 2-D vertical 256x1x128, WENO5 +b | 151 | 13 600 |
| nh 2-D horizontal 192x192x1, WENO5 | 150 | 13 500 |
| sw sphere 128x64 lat-lon | 101 | 9 100 |
| nh 2-D vertical 384x1x192, WENO5 +b | 85 | 7 700 |
| nh 2-D terrain sigma 256x1x128, centered +b | 66 | 6 000 |
| nh 3-D 64x64x16, WENO5 +b +f | 56 | 5 000 |
| hy 128x128x8 | 55 | 4 900 |
| sw 2-D 512x512 | 56 | 5 000 |
| nh 3-D 48^3, WENO5 +b +f | 42 | 3 800 |
| nh 2-D immersed 256x1x128, WENO5 +b | 35 | 3 200 |
| nh 3-D 64^3, WENO5 +b +f | 17 | 1 500 |
| sw sphere 256x128 lat-lon | 17 | 1 500 |
| nh 3-D immersed 64x64x24 | 9 | 800 |

Reverse mode, measured separately (`Model.propagator(wrt=("b",),
steps=150, remat=True)`, nh 96x1x48): **15 s compile + 1.8 s per
gradient** when the gradient is wrapped in `jax.jit`. Un-jitted it is
an order of magnitude slower — a gotcha worth a line of prose in any
example that uses it.

Three conclusions shape everything below.

- **2-D is the format.** A 256x128 slice buys well over ten thousand
  steps; 64^3 buys fifteen hundred. Any candidate whose physics
  genuinely needs three dimensions (DOME with rotational veering,
  Langmuir cells, rotating convection) is out of budget unless it is
  thin-lidded (64x64x16 is comfortable).
- **Immersed boundaries cost about 4x a flat grid** at equal size, and
  their compile is noticeably longer. Budget an immersed example at
  half the resolution you would give a flat one.
- **The sphere is metric-bound, not point-bound.** 128x64 on a chart
  costs about what 512x512 costs on a flat Cartesian grid.

## 2. What the current gallery already covers

Fourteen scripts, all new-stack, **every one of them a uniform
Cartesian box**. Eleven are `nonhydro2`, two `shallowwater2`, one
`hydrostatic`. Physics covered: barotropic shear instability (twice,
sw and nh), dipole interaction, tracer stirring, internal-wave
eigenmodes/packets/beams, equatorial trapped waves, Rayleigh-Benard,
Rayleigh-Taylor, symmetric instability, geostrophic adjustment, and one
advection/closure sweep.

Over-covered, and worth *not* repeating: WENO5-as-the-only-dissipation
(7 examples, near-identical justifying paragraph in 4 of them), the
Gaussian-jet -> vortical-projection -> vortex-street experiment (2
examples sharing whole sentences), AB3 (the only stepper anywhere in
the gallery), the doubly-periodic box, and the `b`-heatmap animation
(20 of 24 videos).

**Zero coverage** across three categories:

- *Geometry*: immersed boundaries, terrain-following coordinates,
  stretched meshes, partial bottom cells, chart/spherical grids,
  moving geometry. Every example is a flat uniform box.
- *Model surface*: `Model.propagator` and `jax.grad` through a run;
  the hydrostatic model assembled by hand (only the preset appears);
  all three free-surface variants; every stepper other than AB3;
  `VerticalMixing` and the IMEX steppers; surface forcing
  (`WindStress`, `SurfaceBuoyancyFlux`, `BoundaryFlux`); the balance
  transforms (`OptimalBalance`, `BalanceExpansion`,
  `AdiabaticProjection`); random-spectrum initial conditions;
  `fr.io.TimeSeries`; snapshots and restart.
- *Physics*: flow over topography of any kind, Kelvin-Helmholtz,
  gravity currents and overflows, Ekman layers, wind-driven
  circulation, baroclinic/Eady instability, geostrophic turbulence,
  coastal upwelling, and anything on a sphere.

Already claimed by `structure.md` — **do not re-propose as gallery
work**: the `sw.eigenbasis` slow-mode-filtering example, the two-layer
QG "One Model, Three Ways" chapter, Guide chapters 1-8, and the
Verification section (convergence orders, Taylor-Green, eigenmode
orthogonality).

## 3. Candidate examples

Each pairs a physical phenomenon with framework capability that
nothing currently demonstrates. Configurations are the ones I actually
ran; costs follow from §1.

### Tier 1

**T1. Internal tide over a ridge — the same mountain, two coordinate
systems.** Barotropic oscillating flow over a Gaussian ridge in uniform
stratification radiates internal-tide beams at `theta = arccos(omega/N)`,
which the example measures off the field and checks against the
dispersion relation. The framework payoff is that the *same bathymetry*
is discretized twice, side by side: a terrain-following sigma
coordinate (`CoordinateMapping`) and immersed cut cells
(`ImmersedDomain(order=4)`). This is the "different coordinates"
comparison in its most natural physical setting, and it closes the two
largest geometry gaps in one file.
*Config*: 2-D (x,z), 192x1x96 per leg, `CenteredAdvection` on both legs
(forced — biased schemes refuse mapped grids, which makes the
comparison fair by construction), `Relaxation` sponge on the lid.
~1500 steps each, roughly 25 s + 45 s.
*Prior art*: MITgcm `internal_wave` (60x1x20), Basilisk `lee.c`,
Oceananigans `internal_tide`. Khatiwala (2003); Garrett & Kunze (2007).

**T2. Dense overflow down a slope.** Dense water released on a shelf
descends a slope as a bottom gravity current whose head rolls into
Kelvin-Helmholtz billows — the overflow problem, and the classic test of
whether a model's topography handling spuriously entrains. The
framework payoff is cut cells versus staircase: run the identical
setup with `min_fraction=0.0, order=4` (genuine partial cells) and with
`order=1` (collocation staircase) and show what the quadrature buys.
*Config*: 2-D (x,z), 192x1x96 per leg, `BuoyancyTracer`,
`HarmonicFriction` (the only immersed-legal friction), FV family
(auto-selected). ~1500 steps each.
*Prior art*: MITgcm `tutorial_plume_on_slope` (320x1x60 — note how
small the reference configuration is). Legg, Hallberg & Girton (2006).

**T3. The Ekman spiral.** Wind stress on a rotating stratified column;
the velocity turns with depth, the surface flow sits 45 degrees off the
wind, and the depth-integrated transport is 90 degrees to the right.
Checked against the closed-form Ekman solution. The framework payoff is
`nh.WindStress` (zero coverage), vertical-only friction (`nu_v=`), and a
`MappedIntervalMesh` stretched vertical that resolves the surface layer
without wasting cells at depth (zero coverage).
*Config*: **1-D column**, 1x1x128, measured at 300 steps/s — this
example is effectively free, with room for a sweep over viscosity or
resolution.
*Payoff route*: matplotlib hodograph plus profile. No animation, which
matters — see §5. It would also be the gallery's first example under 90
lines.

**T4. The optimal perturbation.** Gradient ascent on the initial
buoyancy field to find the perturbation that maximizes final kinetic
energy of a stratified shear layer — nonmodal growth via the Orr
mechanism, which is real GFD with a tilted-against-the-shear answer
that looks like nothing a modal analysis produces. The framework payoff
is `Model.propagator(wrt=("b",), steps=..., remat=True)` under
`jax.grad`, which AGENTS.md calls a tested invariant of the new stack
and which **no example demonstrates at all**.
*Config*: 96x1x48, 150 steps, ~25 gradient-ascent iterations. Measured
15 s compile + 1.8 s per gradient = ~60 s.
*Strategic note*: nothing in Oceananigans, Dedalus or Basilisk has an
analogue, and MITgcm's adjoint tutorials are global-grid and expensive.
This is the single most differentiating page available.

### Tier 2

**T5. Wind-driven gyre and western intensification.** Steady wind-stress
curl on a beta plane in a closed basin gives the Sverdrup interior and a
western boundary current, with Stommel's closed-form solution to plot
alongside. Animate the spin-up (Rossby waves sweeping westward, the
boundary layer forming) rather than the steady state. Framework payoff:
`shallowwater2` in a **doubly walled** basin, `BetaPlaneCoriolis`,
`Source` with a `Ramp` law as steady forcing, `Relaxation` as Rayleigh
drag. This is the most conspicuous shallow-water omission.
*Config*: 128x128, measured 276 steps/s — very roomy.
*Prior art*: MITgcm `tutorial_barotropic_gyre` (62x62x1); Basilisk
`stommel-ml.c`. Stommel (1948); Munk (1950).

**T6. Coastal upwelling.** Alongshore wind drives offshore Ekman
transport; isopycnals tilt and outcrop at the coast and a coastal jet
spins up. The most-run test case in regional ocean modelling, and the
only candidate here that speaks directly to applied oceanography.
Framework payoff is unusually dense: `hy.Model` assembled by hand
(only the preset appears today), `ImplicitFreeSurface`, `BoundaryFlux`
as the wind stress, and `VerticalMixing` under a `CNAB2` IMEX stepper —
four zero-coverage features, and the whole hydrostatic gap, in one
file.
*Config*: 2-D cross-shore (y,z), 1x128x24, measured 440 steps/s.
*Prior art*: ROMS `upwelling` (41x80x16). Allen et al. (1995).

**T7. Hydrostatic versus nonhydrostatic: the lock exchange.** One
initial condition, two of FRIDOM's three models, side by side: the
nonhydrostatic run makes Kelvin-Helmholtz billows on the interface and
the hydrostatic one structurally cannot. Front speed checked against
`0.5*sqrt(g'H)`. Framework payoff: a cross-model comparison that is
awkward in most frameworks and easy here, plus FV tracer conservation
via `state["b"].integrate()`.
*Config*: 256x1x64, measured 135 steps/s. Both legs cheap.
*Prior art*: Basilisk `kh.c` vs `kh-ns.c`, documented explicitly as
"excellent agreement with NS, large divergence from hydrostatic".
Haertel et al. (2000); Ilicak et al. (2012).

**T8. Barotropic instability on the sphere.** A perturbed midlatitude
jet rolls up into vortices — Galewsky et al. (2004). The only route to
the chart machinery: `fr.spatial.spherical.Grid`, `sw.Core(coords=)`,
`RotationCoriolis`, `CoriolisEnergyCorrection`, and
`sw.diagnostics.etot_full` closing to machine zero, which is a
ready-made second figure.
*Cost risk — the one real budget problem on this list.* 128x64 buys
9 100 steps, but the lat-lon CFL collapses as `cos(lat)` near the
polar caps, and six days of evolution at +/-80 degrees needs roughly
15 000. Mitigations: cap `lat_extent` nearer +/-60 degrees (the jet
sits at 45N and the vortices stay in midlatitudes), or accept 128x64,
which is coarse for the roll-up. **Scope this with a real run before
committing.** Fallback if it will not fit: a Williamson case-2 steady
geostrophic zonal flow as a metric-accuracy check — cheap, but a dull
picture.

### Tier 3 — worth writing, lower priority

- **Flow past an island (von Karman street).** Immersed cylinder plus
  `background={"u": 1.0}` Doppler splitting (zero coverage). Verified
  at 51 steps/s, 256x1x128. Iconic and cheap, but its immersed-boundary
  story duplicates T1/T2.
- **Geostrophic turbulence from a random spectrum.** `nh.random_vortical(
  spectral_energy_density=...)` (zero coverage), inverse cascade,
  energy/enstrophy budgets through `fr.io.TimeSeries` (zero coverage).
  Verified 236 steps/s at 128^2. Visually overlaps `dancing_eddies`.
- **Taylor column over a seamount.** Rotation genuinely needs 3-D;
  measured 9.1 steps/s at 64x64x24 immersed, i.e. ~800 steps — feasible
  but tight.
- **Eady baroclinic instability.** `hy.ThermalWindBackground` has zero
  coverage and is *different physics* from the `nh` module the symmetric
  instability example already uses (zonal mean flow over a meridional
  gradient, not a lateral front). Cheap as a 2-D (y,z) front-relaxation
  run; MITgcm's `front_relax` is 1x32x25.
- **Optimal balance and the slow manifold.** `OptimalBalance` has zero
  coverage and is a distinctive FRIDOM capability with a ready-made
  figure (the leakage law `eta(tau) ~ exp(-c*sqrt(tau))`). **My naive
  first attempt diverged** (`ramp_period=1.0` on a 48x48x16 box);
  it needs parameter scoping before it is proposed as a page. Cost is
  `max_it * 2` full integrations of `ramp_period`.

## 4. Sharp edges an author must know

Found while probing; each of these cost me a failed run.

1. **Biased advection refuses mapped and stretched grids.** `WENO` and
   `Upwind` raise at bind on any `CoordinateMapping` grid *and* on a
   bare `MappedIntervalMesh`. Terrain and stretched-mesh examples are
   `CenteredAdvection` (order 2) only. This is a taught error, not a
   silent demotion.
2. **The immersed indicator must name every grid coordinate.** A 2-D
   (x,z) slice still lives on a 3-D grid, so the callable signature is
   `(x, y, z)` even when `ny == 1`; `(x, z)` raises. It returns the
   **wet** fraction, and its parameters are keyword-matched to
   coordinate names.
3. **`lat_extent` is radians**, and the poles are metric-singular —
   `(-80.0, 80.0)` raises with a taught message pointing at
   `(-1.4, 1.4)`.
4. **Jit the gradient.** An un-jitted `jax.grad` through a run is about
   an order of magnitude slower than the jitted one; `remat=True` also
   *lowered* compile time in both my measurement and the audit's.
5. **One grid, one model.** The grid freezes at first assembly, so
   every comparison example needs a fresh grid per configuration.
6. **No root-package aliases.** `fr.BC`, `fr.Real`, `fr.Profile`,
   `fr.Grid`, `fr.modules.*`, `fr.every` do not exist, despite several
   source docstrings claiming they do (`spatial/bc.py:23`,
   `spatial/scalars.py:9,52`, `model/modules/coriolis.py:35-36`,
   `io/triggers.py:6`, `model/transforms/__init__.py:6`). An example
   copied from those docstrings fails at import. Worth a fix pass
   before docs writing starts.
7. **`nh.Model` on a chart grid raises a bare `KeyError`**, not a
   taught error. Spherical is `shallowwater2` only.
8. **Smagorinsky and biharmonic closures are refused on immersed
   grids**; harmonic, free-slip only.

## 5. Sequencing notes

- **Favour matplotlib payoffs for now.** Four owner-requested animation
  improvements are blocked on an unreleased CDFViewer (`roadmap/open.md`
  §5). T3, T4, T5's analytic comparison and T7's front-speed check all
  land as stills or line plots, so they are unblocked.
- **Examples are docs content**, so each goes on a local `docs/<topic>`
  branch that is never pushed until Silvano approves it in chat
  (AGENTS.md; the `docs-review` skill has the mechanics). That argues
  for writing them one at a time rather than in a batch.
- **`FRIDOM_EXAMPLES_FAST` is retired** (owner ruling 2026-08-12): an
  example that is too slow is made cheaper by choosing a smaller
  problem, not by branching on an environment variable.
- Recommended order: **T3 (nearly free, fills the "no short example"
  gap), then T1 (largest geometry gap), then T4 (largest
  claim-versus-demo gap), then T2**. T8 needs a scoping run before it
  is committed to.
