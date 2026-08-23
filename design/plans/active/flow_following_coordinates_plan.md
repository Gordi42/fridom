---
status: draft
date: 2026-08-23
---

# Flow-following vertical coordinates — z* and target-following (isopycnal / hybrid) interfaces

> **Status 2026-08-23.** Owner asked (2026-08-23) how z* and MOM6's
> isopycnal-following coordinates work and whether FRIDOM's
> moving-geometry design gives a short path. The survey is
> [`../../research/flow_following_coordinates_survey.md`](../../research/flow_following_coordinates_survey.md);
> this plan is the answer. **Stage Z (z* for the hydrostatic model)
> SHIPPED 2026-08-23** (merges `805b467a` threading, `ad641355`
> geometry; entry in [`../../roadmap/done.md`](../../roadmap/done.md);
> `hy.zstar_mapping`, `hy.ZStarGeometry`, `tests/hydrostatic/test_zstar.py`,
> `test_core_terrain_moving.py`). **Stage I (target-following
> interfaces) is the open remainder**, first rung on nonhydro2.
>
> Two facts the gates established beyond §3: the hydrostatic package
> runs the nodal family only (`Core/diagnose_w` raises on a flat FV
> grid), so the z* tracer budget is the advective-route one —
> `∫ J b` drifts at truncation level (3.9e-6 relative over 6 steps,
> dominated by the non-telescoping column chain, not the GCL); the
> 1e-12 form of gate 6 needs the hydrostatic FV family (a separate
> project, §5.5). And the discrete GCL condition is pinned directly:
> `eta_dot == (d ps/dt)/g` to 1e-14 at a 40 % surface displacement
> because the free surface and the geometry module read the same
> current η.

## 1. The fit — what the moving-geometry design already carries

The coordinate-systems stage C4 (decision CS-D4) was built with this
in mind, and the pieces line up:

- `CoordinateMapping` takes **parameter fields** by keyword in the
  map callable, chain-rules them into every metric (`d<m>_d<x_i>`,
  the column Jacobian and its reciprocal, the sensitivities
  `d<m>_d<p>`), and re-derives everything per query from the
  `params=` overload — traced values, compiled once.
- `MovingGeometry` owns a parameter value field and its `<p>_dot`,
  rewritten every substage by a SELF_UPDATE stage;
  `MeshVelocityCorrection` adds the mesh-velocity transport per
  prognostic field, family-aware (advective form on point values,
  the Reynolds-transport flux form on `CellAvg` columns) with exact
  constancy by construction and semi-discrete conservation of
  `∫ J f`.
- SELF_UPDATE stages may read state fields (the split-explicit
  snapshot precedent), so a **state-driven** geometry — η from `ps`,
  interface heights from `b` — is a stage-vocabulary citizen.
- The hydrostatic model runs on a terrain column `zp = z·H(x, y)`
  (base `z ∈ [−1, 0]`) with J-weighted `w`, `p_hyd` and transport;
  the free-surface family owns `ps = g η` in three variants.
- A stretched base mesh composes with the chart (stretching in
  `grid.measure`, the chart in `grid.metric`,
  `stretched_terrain_combined.md`), and since 2026-08-23 the biased
  advection works on it.

What is missing:

1. the hydrostatic package queries `grid.metric` with the **static**
   parameter defaults (13 sites in `core.py`, `free_surface.py`,
   `barotropic_pressure.py`, `terrain.py`, and
   `jacobian_weight.jacobian_factor` behind
   `Integral`/`CumulativeIntegral(jacobian=)`), so a moving η would
   be invisible to `w`, `p_hyd`, the slope terms and the transport;
2. a state-driven geometry module (the schedule is the only source
   of `<p>_dot` today);
3. `MeshVelocityCorrection` corrects *every* prognostic, including
   the 2-D `ps`/`U`/`V` (no column factor) and, for Stage I, the
   geometry parameter itself;
4. target builders (z*, isopycnal, hybrid) and — for the
   Lagrangian-remap variant only — a conservative column remap.

## 2. The geometric conservation law in this formulation

The survey's one hard rule: the mesh-induced thickness change must
be the **same discrete operator** that moves the geometry parameter;
an independently evaluated `∂_t M` breaks constancy and conservation
(Campin et al. 2004; MITgcm `dEtaHdt`; MPAS `projectedSSH`).

FRIDOM's split — physics modules on the snapshot geometry, the ALE
term adding the node motion — satisfies it under one condition.
Semi-discretely, with `J = ∂M/∂b`, the physical `w` diagnosed from
the snapshot continuity `∇·(J u) + ∂_b(J ω) = 0` (the *true*
continuity, since `∂_t J = ∂_b ż` cancels against the mesh flux), and
the FV ALE term `(1/J)[D_b(f ż) − f D_b(ż)]`, the tracer obeys

    ∂_t(J f) = −∇·(J u f) − ∂_b((J ω − ż) f)

— the thickness-weighted flux form with the relative vertical flux —
**exactly when the realized `∂_t J` equals `D_b(ż)` discretely**.
That holds when the geometry parameter is advanced by the stepper
with the very tendency the ALE term reads:

- **z\***: `M = η + (H + η)·z` is *linear* in η, so
  `J^{n+1} − J^n = (∂J/∂η)(η^{n+1} − η^n)`. With the **explicit**
  free surface, `ps` advances by `−g T*` and the geometry module
  evaluates `η_dot = −T*` by the same formula at the same stages:
  the GCL is exact to round-off. With the **implicit** or
  **split-explicit** variants `ps^{n+1}` comes from a solve /
  subcycle, so the `η_dot` the ALE term used differs from the
  realized `Δη/Δt` by `O(Δt)`: constancy stays exact (by
  construction of the bracket), conservation of `∫ J b` drifts at
  that order — the MITgcm/ROMS "h and η must not evolve
  independently" caveat. Owner call §5.1.
- **Stage I**: the interface-height parameter is itself PROGNOSTIC
  with tendency `Z_dot`; the stepper applies it and the ALE term reads
  it — GCL by construction for any target and any relaxation rate.

## 3. Stage Z — z* for the hydrostatic model (in progress)

Mapping: `zp = eta + (H + eta)·z` on the base column `z ∈ [−1, 0]`
(`H(x, y)` the static depth, `eta(x, y, t)` the dynamic parameter);
a stretched base mesh puts the stretching in z*. Work, on two
branches with disjoint files:

**H — hydrostatic threading** (`feat/zstar-hydrostatic`):
`params = mapping_params(state, grid)` into every metric query of
the core (`_diagnose_w`, the slope-corrected pressure gradient), the
free-surface family (`_terrain_transport_div`, the implicit /
split-explicit depth weights), the barotropic solver and the terrain
helpers; a `with_params` seam on `Integral` / `CumulativeIntegral`
(`jacobian_weight.jacobian_factor`) mirroring
`MappedDerivative.with_params`, so `p_hyd = −∫ b J dz` sees the
current η. `vertical_extent` stays the static scaling read. Gate:
every terrain test bitwise unchanged (`params=None` is the exact
static path); a `MovingGeometry` `H(t)` column in the hydrostatic
model stays consistent (the nonhydro2 validation battery, ported).

**G — geometry** (`feat/zstar-geometry`):
`hydrostatic/modules/zstar.py` with `zstar_mapping(depth, ...)` (the
`CoordinateMapping` above) and `ZStarGeometry` (declares `eta`,
`eta_dot` AUX on `Profile(x, y)`, `time_dependent`; SELF_UPDATE reads
`ps`, `u`, `v`: `eta = ps/g` (dimensional; the nondimensional
variant is a taught error in iteration 1), `eta_dot = −∫ ∇·(J u) dz`
with the *current* params; bind validates the mapping declares
`eta`); `MeshVelocityCorrection` skips fields without a column
factor and the mapping's own parameter fields; `hy.Model` accepts
the pair through `modules_extra` (a `vertical_coordinate=` kwarg is
§5.3).

Gates (G, after H lands): frozen (`η ≡ 0`) bitwise against the sigma
run; constancy exact; `∫ (H + η) dA` exact; tracer content
`∫ J b` conserved to 1e-12 per step with the explicit free surface;
**nonlinear shallow-water oracle** — a barotropic (`b ≡ 0`,
depth-uniform IC) z* run reproduces `fridom.shallowwater2`'s
nonlinear solution to truncation; small-amplitude wave vs the
fixed-domain linear free surface at `O(η/H)`; compile-once sweep of
η; `Model.propagator` autodiff; forced-4 device-count invariance.

## 4. Stage I — target-following interfaces (planned)

**Prerequisite I1 — parameter tangents along a bounded base
coordinate (measured 2026-08-23).** The mapping machinery accepts a
parameter that depends on the column's own base coordinate
(`maps={"zp": lambda z, Z: Z}`, `params={"Z": lambda x, y, z: ...}`
builds, and declares `dzp_dz`, `dzp_dx`, `dz_dzp`, `dzp_dZ`,
`sqrt_g`), but its parameter-tangent closure assumes a
**wall-mirror-even** parameter (`coordinate_mapping.py`,
`param_tangent`: the centred difference vanishes at the wall face,
the face→centre move adopts the Dirichlet sibling). For a height
parameter the slope at the wall is the layer thickness, not zero:
on a uniform column `Z = 2 z` the derived `dzp_dz` at the centres is
`[1, 2, 2, 2, 2, 2, 2, 1]` — **half the true Jacobian in the two
wall-adjacent cells** — and a query on the `Outer` (wall) faces
raises (`('interpolate', Center(z))` lands on `Inner`, not `Outer`).
Stage I therefore starts with a framework change: a parameter-tangent
closure along a bounded base coordinate that extrapolates one-sided
at the walls (the `LinearInterp(boundary="one_sided")` /
`CellAvg -> Outer` closures are the precedent) and reaches the Outer
faces, gated bitwise on every existing terrain / moving-geometry
test (parameters along horizontal coordinates keep the even
closure). Wide blast radius (terrain, moving geometry, the mapped
pressure solve all read these tangents) — owner-reviewed design
before implementation.

**Design.** The map is `zp = Z` with `Z(x, y, z)` a PROGNOSTIC
parameter field (cell-centre heights on `Profile(x, y, z)`; the
machinery derives `J = ∂Z/∂z` by its discrete parameter tangent and
the slopes `∂Z/∂x_i` the same way — after I1), tendency

    Z_dot = (Z_target(state) − Z)/τ   [+ λ · w_interface]

owned by a `TargetGeometry` module (declares `Z` PROGNOSTIC and
`Z_dot` AUX, one term writing the tendency and the `_dot` field;
`MeshVelocityCorrection` excludes `Z`). `τ → Δt` is MPAS's ALE
(`w^t = … − (h^ALE − h^n)/Δt`); `λ = 1, τ → ∞` is MOM6's Lagrangian
step (the regrid then needs a remap, §4 follow-up). GCL exact (§2).

**Target builders** — pure per-column functions of the state,
vectorized (`jnp.interp` / `lax.scan` over levels, no Python
branching), each returning monotone interface positions with a
minimum thickness and the column total preserved:

- `zstar_target(nominal, H, η)`: nominal spacing × `(H + η)/H`
  (state-independent; the frozen-target gate);
- `isopycnal_target(b_targets, z_floor, z_max, h_max, h_min)` —
  HYCOM1's rule (survey §2): monotonize `b` bottom-up
  (`b_k ← max(b_k, b_{k+1})`, buoyancy increasing upward), locate the
  target buoyancies by piecewise-linear interpolation (out-of-range
  → surface / bottom; inside a jump → the jump), then per interface
  `z = min(max(z_iso, z*_floor), bottom, z_max, z_{K−1} + h_max)`,
  then the bottom-up `h_min` sweep. The `max` with the z* floor is
  what keeps a mixed layer and a convecting column z-like; the
  ceilings stop interfaces from piling at the mixed-layer base.
  Optional artificial compressibility (`REGRID_COMPRESSIBILITY_FRACTION`);
- `adaptive_target` (Hofmeister / MOM6 ADAPTIVE) later.

**Hydrostatic composition.** With a free surface the column total is
`H + η` and the top interface must stay at η exactly: declare
`zp = zstar(z, H, η) + δ(x, y, z)` with `δ` the prognostic interior
perturbation (zero at top and bottom, zero column sum) — NEMO's
`e3 = e3* + e3'` split — so the free surface stays exact and the
target relaxation only redistributes the interior. Rigid-lid
nonhydro2 needs no split.

**Safeguards.** The target is monotone with `h_min` by construction;
the relaxation with `Δt/τ ≤ 1` is a convex combination of two
monotone grids (non-tangling for forward Euler, nearly for AB3); a
CONSTRAINT-stage clamp is the non-conservative last resort, counted
in a diagnostic. The relative vertical velocity through fast-moving
interfaces is advected explicitly (Megann et al. 2022's limiter):
recommend `τ ≥ a few Δt` and document.

**Initialization.** ICs are callables of physical coordinates, so:
build `Z` from the target of the IC sampled on the default geometry,
re-sample the IC on the new nodes, iterate — MOM6's
`REGRID_ACCELERATE_INIT` without a remap.

**Validation** (nonhydro2 rigid lid first, the moving-geometry
battery): frozen target = the static stretched mesh bitwise; a
uniform target relaxation keeps a constant field; an isopycnal target
on a stably stratified adiabatic flow keeps each layer's tracer
content constant; a deep-convection column (unstable IC) keeps
monotone interfaces at `h_min` with the z* floor active; compile
once; autodiff.

**Follow-up (designed-for).** A conservative column remap operator
(PLM/PPM, MOM6 `remapping_core_h`'s sub-cell decomposition) for the
Lagrangian + periodic-regrid variant — the only genuinely new
numerics in this plan.

## 5. Owner calls

1. z* with the implicit / split-explicit free surface: accept the
   documented `O(Δt)` GCL residual, feed the constraint's realized
   `Δη/Δt` into the next step's `η_dot` (lagged), or restrict z* to
   the explicit variant in iteration 1 (the gates use the explicit
   variant).
2. Stage I parameter placement: cell-centre heights (the machinery
   as is; `J` at centres is the wide 3-point difference) vs
   face-declared parameters (needs face-space parameter alignment).
3. A `hy.Model(vertical_coordinate=...)` convenience kwarg.
4. Stage I scheduling after Stage Z lands.
5. The hydrostatic FV family (`family="fv"` in `hy.Core`): the
   prerequisite for exact tracer-content conservation under z*
   (the flux-form ALE route) — and for the Stage I budget gates on
   the hydrostatic model.
6. `tests/validation/test_moving_geometry.py`: four gates fail on
   the owner's machine (the two frozen-motion bitwise and the two
   compile-once ones). Bisected 2026-08-23: they fail identically at
   `1a82bea9`, before every merge of that day (WENO, z*), so this is
   either a local jax/XLA drift in bitwise behaviour or a regression
   older than 2026-08-22; needs a look (CI state unknown, dev pushes
   are batched).
7. Stage I's I1 (the wall closure of parameter tangents along a
   bounded base coordinate): design review before the change.
