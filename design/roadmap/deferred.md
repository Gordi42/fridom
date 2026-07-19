---
status: active
date: 2026-07-19
---

# Roadmap — deferred work (the shelf)

Parked work: real, sized, worth keeping — but nothing is waiting on
it. Every entry carries a **trigger**; when the trigger fires, the
entry moves back to [`open.md`](open.md) (same-change hygiene, like
the done.md rule). Entries here are *intended eventually*; things
decided **against** live in [`declined.md`](declined.md) instead.
The deep design detail stays in the cited plan/research records —
this file is the searchable index so the thought is findable later.

## Boundary closures 2e — Robin dynamic `(α, g)` path (+ 2f)

*Deferred — medium (1–2 weeks). Trigger: a model needs inhomogeneous
boundary values (moving lids, prescribed wall buoyancy) or dynamic
Robin data; open-boundaries Tier 2 would be the natural consumer.*
`BC.ROBIN` structure exists; no consumer does. Two recorded design
questions (traced α/g vs assembly seeding under jit; sharded
inhomogeneous fill inside `shard_map`) are why this is not days.
Stage 2f (mirror/halo-claim widening) is pure perf, gated on
profiling nobody has done. **Partial slip** (NEMO `shlat`-style) is
a Robin wall flux and rides on this path.
[`../plans/active/boundary_plan.md`](../plans/active/boundary_plan.md)

## Open boundaries — sponge (small) / through-flow (large)

*Deferred — sponge tier days; genuine through-flow 3–6 weeks in four
stages. Trigger: a consumer needing net inflow/outflow (a sponge-
emulated open boundary is already possible with zero code).* Five
owner calls are listed in the record §4 before Tier 2 starts; Tier 2
gives boundary-closure 2e its consumer. A shallowwater2 Flather
module (~1 wk) is the cheap pilot.
[`../research/open_boundaries_scoping.md`](../research/open_boundaries_scoping.md)

## Eigenmode phase I — `Banded` as a first-class realized map

*Deferred — medium for the FD/nodal vertical route. Trigger: a
stretch-only vertical map in a hot solve path (per-mode Thomas would
replace fixed-iteration CG exactly).* Landmine recorded: `d/dz` in
Chebyshev coefficient space is dense-triangular, not banded — the
literal `Fourier ⊗ Chebyshev` headline needs the Shen/Galerkin basis
and is large.
[`../plans/active/projection_eigenmode_roadmap.md`](../plans/active/projection_eigenmode_roadmap.md)

## High-order mapped stencils — the full lift

*Deferred — medium (1–2 weeks), held on payoff. Trigger: demand for
honest ENO/dispersion behaviour on stretched meshes (the C-grid
biased tendency stays 2nd order regardless; the win is quality, not
order).* The blocker is answered: the same-row discrete Jacobian
divisor (2026-07-16 spike) restores design order and the discrete
metric identity; only the lift remains.
[`../plans/active/high_order_mapped_plan.md`](../plans/active/high_order_mapped_plan.md)
§3, [`../research/mapped_jacobian_spike.md`](../research/mapped_jacobian_spike.md)

## TangentPropagator — the D5 forward-mode surface

*Deferred — small. Trigger: a `jax.jvp` consumer (NNMD descoped).*
The shared name-resolution piece shipped with `Model.propagator`.
[`../plans/active/differentiability_plan.md`](../plans/active/differentiability_plan.md)
§5.4

## Moving-geometry multi-device gates — layout negotiation

*Deferred — small. Trigger: a multi-device ALE consumer.* The two
`tests/validation/test_moving_geometry.py` gates stay
`single_device`: one-sided boundary variants patch physical edges at
static indices, so `z` must stay undistributed, and negotiation
sharded it. Tractable (declare the layout requirement, or reshard at
the module seam); the original XLA `RET_CHECK` blocker is gone.
[`../plans/active/perf_geometry_merge_plan.md`](../plans/active/perf_geometry_merge_plan.md)
§3c

## No-slip immersed side-drag

*Deferred — small/medium, design recorded. Trigger: an immersed
no-slip consumer.*
[`../plans/active/immersed_closures_sadourny_plan.md`](../plans/active/immersed_closures_sadourny_plan.md)
§5

## Smagorinsky — immersed, terrain/mapped, walled FV

*Deferred (owner 2026-07-19). Triggers: an immersed-LES consumer;
terrain LES (needs the metric-tensor strain and the W3 J-weighted
filter width); a walled-FV (`CellAvg`) Smagorinsky consumer (a
narrow taught error added at the W1 landing so the FV promotion
does not run an unvalidated cross-derivative path — the diffusion
campaign's FV walled lift is the precedent for the small fix).*
Layering proven safe: each stage reduces at `Cs=0` to the
corresponding already-ratified walled/immersed/along-σ friction
closure. The walled nodal lift (W1–W3) shipped 2026-07-19
(`c028a84d`).
[`../research/smagorinsky_walls_scoping.md`](../research/smagorinsky_walls_scoping.md)
§(vi)–(vii)

## Stage-5 diffusion — geopotential-correct full-metric tensor

*Deferred — separate plan when taken. Trigger: physics need for
geopotential (isoneutral-style) mixing on terrain; the ratified
along-σ convention (with the documented tracer diapycnal caveat) is
the default until then.*
[`../research/diffusion_walls_terrain_scoping.md`](../research/diffusion_walls_terrain_scoping.md)
§3.6 (options A/C)

## Moving-geometry implicit column

*Deferred — needs the params-through-solve seam (not built).
Trigger: a consumer running moving terrain with implicit vertical
mixing.* `ImplicitOperator.solve` receives no state, so the
measure-aware band reads static geometry; `VerticalMixing.bind`
raises the taught error when a mapping parameter rides the field
table and couples the solve axis.
[`../research/diffusion_walls_terrain_scoping.md`](../research/diffusion_walls_terrain_scoping.md)
§10

## Partial-bottom `p_hyd` on terrain charts (PB-D3)

*Deferred (owner 2026-07-19) — the chart-consistent wet first-moment
quadrature must be derived and proved (computational vs physical
moment under the column map, consistent with the cumint measure).
Trigger: a terrain-chart + partial-bottom consumer needing the
correction; status quo is the honest uncorrected O(dz)-at-cuts
behavior behind a taught error.*
[`../plans/active/partial_bottom_phyd_plan.md`](../plans/active/partial_bottom_phyd_plan.md)

## sw2 mapped+immersed

*Deferred (owner 2026-07-19, re-affirming the 07-19 scoping) — two
genuinely open design pieces: a sqrt_g-weighted embedding-chart
fraction quadrature (MI-D1 serves column corrections only, not
`chart_coords`), and a re-derived semi-discrete energy-antisymmetry
proof with the combined α·sqrt_g corner weight. Trigger: a concrete
curvilinear-with-islands use case.* Taught errors at both bind sites
(`SadournyAdvection`, `DynamicalCore`) keep it honest meanwhile.
[`../plans/active/mapped_immersed_composition_plan.md`](../plans/active/mapped_immersed_composition_plan.md)

## Mapped-solve residual levers — measured, not taken

*Deferred (owner 2026-07-19). All measured; promote one only when
its trigger appears:*

- **Single-precision distributed solve** — `single_precision_solve`
  is a no-op on multi-device walled/mapped grids (full-precision
  distributed solve takes precedence; documented at
  `spectral_solve.py`). Worth ~the single-device −10% if a
  multi-device user asks.
- **Distributed-transform planner size floor** — small problems pay
  unamortized collective latency (32³ walled/mapped +5.5%); the
  lever if toy-size multi-device runs ever matter
  ([`../plans/active/distributed_transform_plan.md`](../plans/active/distributed_transform_plan.md)).
- **Surplus staggered reblock leg (`n = n_cells + 1`)** — stays on
  the global reblock path (needs a `P*(cells+1)` frame plus one
  realigning collective-permute; gate documented in
  `decomposition/tensor.py`). Revisit only if a Neumann-outer field
  enters a hot loop.

## Slip ownership — per-closure kwarg vs per-wall declaration

*Deferred (owner 2026-07-19). Trigger: a third wall-stress consumer
(after the friction closures and Smagorinsky).* Whether `slip=`
stays a per-closure kwarg or becomes a per-wall grid/field
declaration (one source of truth across advection, friction,
Smagorinsky, open boundaries).
[`../research/smagorinsky_walls_scoping.md`](../research/smagorinsky_walls_scoping.md)
