---
status: done
date: 2026-07-17
---

# Variable boundary forcing — design and implementation plan

**Goal (Silvano, 2026-07-17):** variable boundary conditions — wind
forcing and (potentially time-dependent) buoyancy forcing at the
boundaries. Designed after a survey of Oceananigans.jl (BC taxonomy and
mechanics), MITgcm/EXF, NEMO SBC, MOM6, Veros (forcing-as-data with
traced time interpolation), jax-cfd, Dedalus, and FEM natural BCs
(survey summaries in §6).

## 1. The governing finding

Every surveyed model splits boundary treatment the same way:

- **Prescribed boundary *values*** (Dirichlet/Neumann/Robin data) are
  imposed by ghost/halo filling — in fridom that is the deferred
  boundary-closure stage 2e (`boundary_plan.md`, `("ghost_fill",
  space)` rows).
- **Prescribed boundary *fluxes*** (wind stress, surface heat/buoyancy
  flux — the actual ocean-forcing cases) never touch halos. The halo
  is filled zero-gradient so interior operators produce no spurious
  boundary flux, and the physical flux is injected as a **tendency
  contribution in the wall-adjacent cell**, scaled by face-area over
  cell-volume (Oceananigans `compute_flux_bcs.jl`: `Gc[wall cell] ∓=
  q · A_face / V_cell`; MITgcm/NEMO/Veros apply surface fluxes as
  `flux / dz(top)` source terms in the top level).

fridom's structural-BC design already provides the first half for
free: a walled field's NEUMANN tag *is* the zero-gradient fill. What
is missing is only the flux-injection half — and it is a plain
tendency module, not new grid machinery.

**BF-D1 — Boundary flux forcing is a tendency module (`BoundaryFlux`);
it does not consume the 2e ghost-fill path.** The wall-adjacent-row
weight `A_face/V_cell` is an assembly-materialized AUXILIARY profile
field, so decomposition needs no `shard_map` special-casing (the mask
is ordinary sharded field data; the pointwise product is
halo-neutral). Consequence for the roadmap: stage 2e stays deferred —
a flux-BC module turns out *not* to be its consumer; its consumers
remain inhomogeneous boundary *values* (moving lids, prescribed wall
buoyancy) and dynamic Robin data.

## 2. Decisions

**BF-D2 — `fr.modules.BoundaryFlux(field, coord, side, flux=,
scale=)`, one prognostic field per instance.** Semantics: for the
PROGNOSTIC `field` on the bounded axis `coord` at `side` in
{"left","right"}, add

```
d(field)/dt += sign(side) · scale(t) · F(x_tang) · W(x_coord)
```

with `sign = +1` left / `-1` right (the Oceananigans convention:
positive flux transports the quantity in the +coord direction, so a
positive flux at a right wall is a loss), `F` the spatial flux
pattern, and `W` the wall weight — `A_face/V_cell` in the
wall-adjacent cell row, `0` elsewhere (`1/Δn_wall-cell` on an
unmapped mesh, stretched meshes included via the measure).

- `flux`: number | profile callable of *tangential* coordinate names
  (the `Relaxation` normalization: a number is a one-DOF
  `fr.Profile()`, a callable is sampled on the coordinates its
  signature names). Naming the normal coordinate in the callable is a
  taught error (the flux lives on the wall face).
- `scale`: `float | TimeDependent`, published as the dynamic-leaf
  parameter `boundary_flux.<field>.<coord>_<side>.scale` (the
  `Relaxation.rate` pattern) — Ramp/TimeFunction/TimeSeries-capable,
  swept by `model.update_parameters` without re-assembly, resolved at
  stage time through `ctx.params`. The module never reads the clock
  directly, so the term is pure field arithmetic with no
  `extra_halo`.
- `W` is an AUXILIARY `fr.Profile(coord)` field built by an
  unbound-owner-method default (`(self, grid, space)`, the
  `_f_const_ingredient` precedent): index-based
  `data = zeros(space.shape); data[wall-adjacent entry] = 1/Δn` with
  `Δn` read from the grid measure at assembly — never a
  float-equality coordinate test. `F` is a second AUXILIARY profile.
  Both move onto the forced field with `.to` in the term (pure
  broadcast on shared nodes).
- FV exactness: on `CellAvg` spaces the term is exactly the
  flux-divergence contribution of a prescribed face flux, so the
  global budget `d/dt ∫ field dV = Σ_wall q·A` holds to rounding —
  a gate test.

Taught errors at bind:
(a) `coord` unknown or periodic → name the walled axes;
(b) `field` staggered along `coord` (wall-normal velocity; its wall
    faces are not DOFs) → prescribing normal flow is an open-boundary
    condition, out of scope;
(c) the field's resolved BC tag on `(coord, side)` is DIRICHLET → the
    wall value is pinned; point at the NEUMANN sibling or
    `Relaxation` (best effort: if the resolved tag is not reachable
    at bind, the check moves to the assembly dry run);
(d) chart/mapped grids → iteration 1 computes `A_face/V_cell` from
    1-D measures; the metric-aware weight is designed-for (the
    diffusion walled-grid-rejection precedent).

**BF-D3 — Two new generic `TimeDependent` curves; no new clock
plumbing.** Time dependence rides entirely on the existing
stage-time parameter resolution:

- `fr.TimeFunction(fn, params=())` — arbitrary user law `fn(t,
  *params)`; `fn` STATIC (a law change is different math — exactly
  one recompile, the `Ramp.curve` split), `params` dynamic leaves
  (sweeps never recompile). Unlocks oscillating forcing
  (`sin(ω t)` wind) for *every* declared scalar parameter, not just
  forcing.
- `fr.TimeSeries(times, values)` — tabulated data with branch-free
  `jnp.interp` at the traced stage time, end-clamped; `times`/
  `values` dynamic leaves (the Veros two-record blend, and the
  MITgcm/EXF time-interpolation pattern, as a curve). Data-driven
  forcing without recompiles.

Both live in `model/time_dependent.py` beside `Ramp` and satisfy the
existing contract (pure, branch-free, valid at every scan step,
affine-composable).

**BF-D4 — Model-package wrappers own the physical sign
conventions.** The generic module keeps the axis-direction flux
convention; users get wrappers with the oceanographic signs:

- `nonhydro2.modules.WindStress(tau_x=0.0, tau_y=0.0, coord="z",
  side="right", scale=1.0)` — one module, terms on `u` and `v`;
  positive `tau_x` accelerates the surface flow in +x (internally
  `q = -tau` at a right/top wall). `tau_*` are numbers or tangential
  profile callables; kinematic stress (τ/ρ₀) in model units;
  `scale` published as `wind_stress.scale` (TimeDependent-capable).
- `nonhydro2.modules.SurfaceBuoyancyFlux(q, coord="z", side="right",
  scale=1.0)` — positive `q` adds buoyancy to the wall-adjacent
  water (a surface *gain*; internally `flux = -q` at the top).
  Subclass/thin wrapper of `BoundaryFlux` on `b`.

shallowwater2 wind stress is volumetric over the single layer (no
vertical axis) — it is served by the optional generic `Forcing`
module (V4), not by a boundary mechanism.

**BF-D5 — Out of scope (designed-for, not precluded).**
- Inhomogeneous boundary *values* / dynamic Robin data — stays
  boundary-closure 2e, unchanged and still without a consumer.
- Open/radiation boundaries, prescribed inflow (OBCS/Flather/
  perturbation-advection family).
- Field-dependent fluxes (linear/quadratic bottom drag): the natural
  next consumer of the same wall-weight machinery — the term reads
  `state[...]` like `Relaxation`; add when a model asks.
- Tabulated space-and-time forcing fields (Veros monthly-map gather
  along a leading time axis): `TimeSeries` covers the scalar case;
  the field-valued case waits for the general time-dependent-fields
  design (roadmap entry).
- Immersed-boundary fluxes (the 2026-07-17 immersed-partial-cells
  plan): wall-row weights here are *domain-box* walls only.
- Metric-aware weights on chart grids (BF-D2 d).

## 3. What exists (probed 2026-07-17 on `dev`)

| Needed | Exists |
|---|---|
| Stage-time scalar resolution for `float | TimeDependent` params | `Relaxation.rate` → `leaf()` + `ParameterDeclaration` + `ctx.params` (`modules/relaxation.py:173,321`) |
| Spatial patterns as AUXILIARY profiles from callables | `Relaxation._profile_declaration` (`relaxation.py:237`) |
| Owner-method field defaults building arrays | `_f_const_ingredient` (`modules/coriolis.py`), `FieldDeclaration` D4 form (`declarations.py:78-84`) |
| Zero-gradient halos at walls | structural NEUMANN mirror fills (`spatial/decomposition/tensor.py`) |
| Measures for `1/Δn` | `grid.measure(space, name)` (`grid.py:948`) |
| Term/param/lifecycle validation patterns | `Relaxation.bind`, `GaussianWaveMaker` |

Missing (the new work): `BoundaryFlux` + wall-weight builder;
`TimeFunction`/`TimeSeries`; the two nonhydro2 wrappers; optional
generic `Forcing`.

## 4. Stages

All stages on `feat/boundary-forcing` (one worktree, sequential), per
AGENTS.md gates: mirrored tests (95% branch), `ruff` clean, one model
smoke file where core machinery is touched, forced-4 where
decomposition-relevant.

| Stage | Work | Gate |
|---|---|---|
| **V1** | `TimeFunction`, `TimeSeries` in `model/time_dependent.py`; exports + `test_init` rows. | Unit tests: values, end-clamping, affine composition, compile-once across dynamic-leaf sweeps (compile-counter), `jax.grad` through leaves runs; ruff. |
| **V2** | `fr.modules.BoundaryFlux` (`model/modules/boundary_flux.py`): wall-weight owner-method default, flux profile, published scale param, taught errors (a)–(d). | Oracles on a walled-z nonhydro2 box: buoyancy flux — `b(top row) = q·t/Δz`, others exactly 0, global budget `d/dt ∫b = q·A` to rounding; wind stress — `u(top row) = τ·t/Δz`, projection a no-op for uniform τ; stretched-mesh Δn correct; Ramp and `TimeFunction` scales match stepper-order oracles; taught errors asserted; forced-4 bitwise vs single-device; mirrored tests + `tests/nonhydro2` smoke; ruff. |
| **V3** | `WindStress`, `SurfaceBuoyancyFlux` in nonhydro2; lazypimp exports, `test_init` rows. | Sign-convention tests (+τx accelerates +x; +q gains buoyancy) against the raw `BoundaryFlux` spelling; combined wind+buoyancy smoke run stays finite; ruff. |
| **V4** *(optional)* | Generic volumetric `fr.modules.Forcing(field, profile=, scale=)` — the SW wind-stress route and the Oceananigans-`Forcing` analogue. | Oracle: uniform forcing advances the field linearly; sw2 smoke. |
| **V5** | Hygiene: this plan → outcome recorded; roadmap — new entry moved to `done.md`, 2e entry's "no flux-BC module exists or is planned" corrected to cite `BoundaryFlux` (BF-D1: 2e still has no consumer). | `open.md` holds only open work. |

## 5. Risks / verification notes

- **Wall-weight construction under decomposition**: the owner-method
  default builds true-shape global data at assembly
  (`create_field(space, data=...)`); the forced-4 gate asserts the
  weight has exactly one nonzero row globally and results match
  single-device bitwise.
- **`.to` must not smear the indicator**: the weight profile and the
  forced field share Center-family nodes on `coord` (staggered-normal
  fields are excluded by taught error b), so `.to` is a broadcast,
  not an interpolation — asserted in tests.
- **Term flags**: state-independent source ⇒ mirror the
  `GaussianWaveMaker` term flags (explicit, not `linear=True`), so
  ETDRK4's frozen-`L` snapshot is untouched (no AR-D7 interaction).
- **Parameter-name collisions**: two `BoundaryFlux` on the same
  `(field, coord, side)` collide on the scale name — intended
  (declaration-collision error); different sides/fields coexist.

## 6. Outcome (2026-07-17, merge `24ee6fd0`)

V1–V3 shipped on `feat/boundary-forcing`
(`835f9713`/`a63a8cac`/`43eb564b`/`b3988739`); V4 (generic volumetric
`Forcing`) stays deferred per its optional marking — promote when a
shallowwater wind consumer appears; V5 done in the landing commit.
Every stage gate met: analytic oracles to rtol 1e-11 (wall-row
`q·t/Δz`, global budget `= q·A`, hand-stepped Ramp/`TimeFunction`
scales), forced-4 bitwise vs single-device, 100% branch coverage on
`boundary_flux.py` and `surface_forcing.py`, nonhydro2 suite green,
ruff clean; end-to-end oscillating-wind + buoyancy run matches the
analytic integrals.

Corrections vs the text above, resolved at implementation:
- Canonical spellings are `fr.model.TimeFunction` /
  `fr.model.TimeSeries`, `fr.model.modules.BoundaryFlux`,
  `nh.WindStress` / `nh.SurfaceBuoyancyFlux` — the `fr.modules.*`
  alias used in §2 does not exist.
- Both wrappers publish SIDE-QUALIFIED scale names
  (`wind_stress.<coord>_<side>.scale`,
  `surface_buoyancy_flux.<coord>_<side>.scale`, via an overridable
  `BoundaryFlux._make_scale_name`) so opposite-wall instances coexist
  (Rayleigh–Bénard) while same-side duplicates still collide.
- No `extra_halo` exemption was needed (the term is pure field
  arithmetic; `scale` comes from `ctx.params`, never the clock).
- The DIRICHLET-tag taught error is enforceable at bind
  (`table[field].space.factor(coord).bc`); no dry-run fallback.
- Stretched-mesh wall weights are asserted directly against the
  measure (a walled-z *mapped* nonhydro2 model cannot assemble — the
  spectral pressure solver lacks the transform — so the model-level
  stretched oracle is unreachable today).
- Multi-device validation is single-controller forced-4; real
  multi-process (`srun -n N`) is unexercised (the weight builder
  stays in field space, so it is expected safe).
