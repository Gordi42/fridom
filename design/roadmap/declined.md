---
status: active
date: 2026-07-19
---

# Roadmap — declined (decided against)

Rulings **against** doing something, with the reason and what would
have to change to reopen. Distinct from [`deferred.md`](deferred.md)
(parked but intended): these were considered and refused. **Check
this file before re-proposing an idea** — several entries exist
precisely because the proposal keeps coming back. Historical declines
migrate in as they are encountered; the deep reasoning lives in the
cited records.

## Multi-process numeric eigenbasis build (host gather)

*Declined 2026-07-19 (owner): no all-gather mechanisms — they
eventually backfire on simulations whose arrays don't fit one
device.* The numeric channel builder host-fetches its probe
responses (`np.asarray`), which raises under a real `srun -n N`
launch; the `process_allgather` spelling was declined. The taught
error stays: numeric-channel eigenbases build single-controller. The
compiled apply path has zero all-gathers (HLO-asserted) and is
unaffected. *Reopen if:* a real multi-process numeric-eigenbasis
consumer appears — then the honest design is a kx-sharded basis (the
replicated-CoefficientSpace contract is the actual scaling ceiling),
not a setup gather.
[`../research/artifacts/gspmd_campaign_gpu4/RESULTS.md`](../research/artifacts/gspmd_campaign_gpu4/RESULTS.md)

## Non-1-D transform meshes

*Declined (standing): unreachable today — the decomposition
negotiates only single-axis layouts; a hand-built 2-axis mesh dies
at decomposition build.* The pencil primitive (per-mesh-axis
`all_to_all` in one 2-D-mesh `shard_map`) is proven composable for
the day a 2-D backend lands. *Reopen if:* a 2-D mesh backend is
built.
[`../research/multidevice_test_faults.md`](../research/multidevice_test_faults.md)

## Halo width-floor knob

*Declined (owner, 2026-07): never re-propose.* The stencil
width-floor tuning knob was refused; derived `extra_halo` shipped
instead. Shape-"luck" is remainder tail + DRAM stride — no free
flag; measure-and-pin only.
[`../research/storage_halo_width.md`](../research/storage_halo_width.md)

## Trace-embed re-measure job

*Declined (owner, 2026-07): never re-propose a re-measure GPU job.*
The H7 `A(1)` embed-vs-slice evaluation ran; the embed route stays
the default where the slice is invalid (slice valid only flat +
collocated tracers).

## Van Driest wall damping + Scotti anisotropy factor (Smagorinsky)

*Declined 2026-07-19 (owner).* No production ocean/idealized-LES
code applies van Driest (it exists only for resolved no-slip walls);
Oceananigans/MOM6 use the bare local filter-width mean without the
Scotti correction. *Reopen if:* a resolved-no-slip-wall LES consumer
appears (then as an optional `Γ_wall` knob, wide blast radius:
needs `u_τ` + wall distance).
[`../research/smagorinsky_walls_scoping.md`](../research/smagorinsky_walls_scoping.md)

## `WindowAccumulator` preset

*Declined 2026-07-19 (owner): the spec promise is struck.* The S6
DIAGNOSTIC accumulation *idiom* stays normative; the packaged preset
had no consumer. *Reopen if:* coupled models (3.2/3.3) are built —
reintroduce it there if wanted.

## `add_prognostic`

*Declined 2026-07-19 (owner): struck from the specs.* The semantics
live in `state.add(**components)` — every stepper family already
uses it; the named sugar is stepper-internal and never user-facing.

## Auto-detect chart orthogonality

*Declined 2026-07-15 (owner): numeric metric → tolerance →
silent-wrong-physics risk.* Explicit `orthogonal=True` plus a
taught-error safety net shipped instead.
[`../plans/done/chart_ergonomics_plan.md`](../plans/done/chart_ergonomics_plan.md)

## Cut cells (FV)

*Declined (owner, FV campaign): out of scope by decision.* The FV
nonhydro is feature-complete against nodal except cut cells;
immersed partial cells serve the geometry need.
[`../plans/active/fv_nonhydro_scoping.md`](../plans/active/fv_nonhydro_scoping.md)
§9

## Pure-trig homogeneous distributed transform

*Declined 2026-07-19: zero consumers.* Machinery verified ready
(1.3e-14) via the walled solve's SlabPlan; not wired for want of a
consumer. *Reopen if:* one appears — the route is recorded.

## Gather-based 2-D channel distributed transform

*Declined 2026-07-19 (owner): the gather was rejected outright;* the
double-transpose pipeline (`Channel2DPlan`, all-to-all only,
HLO-asserted no all-gather) shipped instead — merge `8752170a`.

## Asymmetric halo — storing the two-sided reach instead of its max

*Declined 2026-08-12: measured to save **zero** planes.* The proposal
was to stop collapsing `reach=(below, above)` to its per-side maximum
(`halo.py` ~1298), on the grounds that WENO5's true reach is (2,3)
while storage is 3+3, so every grid pays an extra plane on every
applied axis. Instrumenting the *input* to `HaloSpec.symmetric` at
both collapse sites across nine configurations (nh centered, weno3,
weno5, upwind5, weno5 walled-z, advection=False, sw advective and
linear) records **zero asymmetric specs** — WENO5 arrives at the
collapse as `(3,3)`.

The asymmetry is real per *row* and **already exploited** per row
(`advection.py` ~1479 tightens to `n + 6` per axis for upwind5, not
`n + 8`), but a complete tendency leg symmetrizes: staggered schemes
pair a forward difference with a backward one, and upwinding keeps
both biased reconstructions resident because the flux sign selects at
runtime. Storage is one array, so it holds the per-side max — exactly
`storage_halo_width.md` §1's own derivation, `[-1,0] ⊕ [-2,+3] =
[-3,+3]`. Ceiling had the asymmetry existed: 4.4% / 2.3% / 1.15% of
step bytes at 64³/128³/256³. The original claim ("plausibly worth more
than flat-axis elision") was a per-row fact stated at tendency scope.

*Reopen if:* a configuration is demonstrated whose **accumulated
per-side** demand is asymmetric — a single-bias scheme, or a wall shift
that survives the leg union. Note the adjacent lever that *is* real and
is now tracked separately in [`deferred.md`](deferred.md): the mapped
grid's `extra_halo = 2` floor over a traced demand of 1.
[`../research/thin_axis_halo_investigation.md`](../research/thin_axis_halo_investigation.md) §12.3.

## Extracting `fridom.spatial` into its own repository

*Declined 2026-08-13 (owner): "I don't need an extra package just for
the spatial stuff."* Investigated on the ClimaCore analogy — spatial is
~34% of src, near-independent, and not ocean-specific. The outward edge
is genuinely clean (16 import lines, 4 symbols, all from the old stack's
`framework/utils`), but the **inward** edge is not: consumers bypass the
lazypimp facade and deep-import ~130 symbols across 40 modules, incl. 13
private symbols and `ScalarField._data` at 14 sites, with `__all__` in
0 of 95 modules. A split converts that seam into a permanent published
API at bus factor 1. Decisive numbers: **30% of spatial-touching commits
also touch model/nh2/sw2** (64 of 213 since 2026-02), the package is
**one month old** (`0260d779`, 2026-07-11) with the boundary still
moving, and the stated motivation — a lighter install for non-ocean
users — **arrives free at the cutover**, since 9 of 12 runtime deps are
old-stack-only. Prior art runs the same way: Firedrake re-absorbed PyOP2
and TSFC, and jax-cfd's reusable FVM layer is unmaintained with no
external dependents. The investigation surfaced three bugs, all fixed
the same day (x64 never enabled by bare `import fridom`; `jaxify`'s
equality gate hardcoding `"fridom"`; an inverted test import).
*Reopen if:* a real external consumer appears, or spatial's co-change
rate falls below ~10% over a full quarter. Note the cheaper lever if the
goal is only installability — a second distribution built from this same
repo needs no second repository.
[`../research/spatial_extraction_investigation.md`](../research/spatial_extraction_investigation.md)
