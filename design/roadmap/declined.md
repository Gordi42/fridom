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
