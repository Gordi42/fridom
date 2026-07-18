---
status: active
date: 2026-07-18
---

# Multigrid coarse-grid agglomeration — replicate the deep levels

A plan to stop sharding the coarsest multigrid levels. Below a
per-shard-extent threshold a level's V-cycle work is replicated onto
every device (the hypre coarse-grid-agglomeration / PETSc `redundant`
pattern), reusing the existing local-axis halo path so it issues **no**
collectives. Two independent drivers, one mechanism.

Grounded in the shipped MG stack
([`multigrid_pathway_plan.md`](multigrid_pathway_plan.md), MG-D5 already
names the replicated-below-the-floor fallback) and the two same-day
measurement records that turned it from a designed-for into a driven
lever.

## 1. Motivation (two-fold)

- **Capability.** A P-device sharded axis cannot coarsen below P cells:
  the V-cycle depth is capped by the device count, not by the problem.
  [`../../research/multigrid_depth_scaling.md`](../../research/multigrid_depth_scaling.md)
  showed that capping depth breaks h-independence — iterations climb
  **10 -> 27** at 512^3 when the hierarchy cannot reach its floor. On
  large device counts this bites at ever larger base sizes. Replicating
  the deep levels lets the hierarchy coarsen all the way to the
  4-cell floor regardless of P, restoring flat 10-iteration
  convergence.
- **Latency.** The collective census
  ([`../../research/multigrid_kernel_study.md`](../../research/multigrid_kernel_study.md)
  Addendum 3; full report
  [`../../research/artifacts/multigrid_gspmd_validation/census_collective_report.md`](../../research/artifacts/multigrid_gspmd_validation/census_collective_report.md))
  found that at 128^3 on 4 A100s the two coarsest levels (x = 8 and 4
  cells total -> **2 and 1 planes per shard**) fire **~33% of the halo
  collective-permutes** — ~86 of 262 per V-cycle, ~860 sub-kilobyte
  latency-only permutes per step — each moving <1 KB at the ~10 us
  NVLink small-message latency floor. That is **~9-14 ms of the ~34 ms**
  4-GPU-minus-1-GPU overhead at 128^3, spent sharding levels that have
  almost nothing left to shard. This is the structural reason
  mg-cuSPARSE is 0.37x spectral at 128^3 on 4 GPUs (Addendum 2).

## 2. Design

**Coarse-grid agglomeration by full replication below a threshold.**

- **Switch rule.** Descend the hierarchy; agglomerate at the first level
  where the per-shard extent along **any** sharded axis drops below
  `tau` planes (`tau ~= 4`, to be measured) **AND** the total level size
  is `<= a` bytes (a guard so a huge-P case never full-replicates a
  still-large level). At P = 4 the guarded and unguarded rules coincide,
  because the levels that undershoot `tau` are already tiny; the guard
  only matters at large P, where a level can undershoot `tau` per shard
  while still being globally large — that **middle regime is PARTIAL
  agglomeration onto a mesh sub-group, explicitly DEFERRED** (§5) until a
  measurement demands it.
- **Mechanism.** Levels at or below the switch carry a **replicated
  layout** (every axis local — `Layout({})` over the same device mesh,
  the MG-D5 fallback). Their halo exchange reuses the **existing
  local-axis path** — no new comm pattern, no new kernel. Restriction
  *across* the switch boundary (fine sharded -> coarse replicated)
  inserts **one small all-gather / reshard per V-cycle**; prolongation
  back (replicated -> sharded) is a **local slice**. Below the switch,
  every device does the same redundant compute on the full small level
  and issues zero collectives.
- **Knob.** A model setting in the existing house style (a sibling of
  `multigrid_levels`), fingerprint-static. **Default OFF in the
  prototype**; making it default-on is a later measurement-backed owner
  decision (Phase 4).
- **Differentiability.** This is step-path code, so per AGENTS.md the
  change ships one cheap `jax.grad` regression test through a short run
  (Phase 2). The transfers are linear; the reshard is a pure relayout —
  no new singular divide is introduced, but the gate is required
  regardless.

The replicated-below-the-floor fallback already exists structurally
(MG-D5, `negotiate(..., allow_replicated=True)`, exercised by the
`x12->6` GB-5 shard). This plan **chooses** it deliberately for deep
levels rather than only falling into it when an axis stops dividing P,
and adds the threshold rule + the cross-boundary reshard.

## 3. Phases

- **Phase 0 — probe (cheap, CPU forced-device).** Characterize the
  current behavior when a level's axis extent `< device count` under
  forced 4/8/16 host devices: does floor-depth coarsening stop early,
  does jax pad with empty shards, does the negotiator replicate or
  raise, what breaks. No repo edits — reuses the existing hierarchy
  builder. Establishes the exact status-quo the agglomerated path must
  match and the failure it must remove.
- **Phase 1 — implement.** Replicated-below-threshold in the hierarchy
  builder (`nonhydro2/modules/multigrid_hierarchy.py`) + the switch rule
  + the cross-boundary reshard on the transfers, behind the new knob.
  Reuses `Grid.coarsened` with the replicated device set and the
  existing local-axis halo lowering.
- **Phase 2 — parity gates.** Agglomerated vs status-quo results
  **machine-precision identical** (the V-cycle is the same operator, only
  its layout changed) on walled / mapped / immersed grids and on the
  semicoarsening fallback (`multigrid_coarsen_vertical=False`), under
  forced 4 / 8 / 16 devices. Plus a **capability test**: a small grid on
  16 forced devices whose floor depth is reachable *only* with
  agglomeration (status quo caps depth and the iteration count degrades;
  agglomerated restores flat convergence). Plus the autodiff regression
  test (Phase 1 change is step-path).
- **Phase 3 — 4-GPU GB-2 measurement.** Real 4x A100: 128^3 and 512^3
  (+ immersed), agglomerated vs status-quo mg vs spectral, ms/step under
  the GB-2 protocol. Threshold sweep `tau in {2, 4, 8}` to pin the
  switch. Confirms the ~9-14 ms recovery projected at 128^3 and checks
  it does not regress 512^3 (where the coarse levels are a smaller
  fraction). **Agents never submit GPU jobs unless Silvano asks in
  chat** (AGENTS.md).
- **Phase 4 — default decision (owner gate).** With the sweep in hand,
  the owner decides whether agglomeration becomes the default for
  multi-device mg (and at what `tau` / byte guard).

## 4. Prototype results

*Placeholder — a parallel agent is prototyping on branch
`feat/multigrid-agglomeration`; it fills this section with the Phase 0
probe findings and the Phase 1/2 outcomes.*

## 5. Open questions

- **Partial agglomeration onto a mesh sub-group** (the large-P middle
  regime where a level is per-shard-tiny but globally large): deferred
  until a measurement demands it. Would agglomerate onto a *subset* of
  devices rather than replicating on all — the Fast(er)PM / PETSc
  sub-communicator pattern. Not built first; the byte guard keeps the
  full-replication rule from mis-firing there.
- **2-D multigrid (`ImplicitFreeSurface`)** shares this machinery: its
  2-D V-cycle hits the same coarse-level sharding floor, and the same
  replicate-below-threshold lever applies later with no new design.
- **`tau` and the byte guard** are the two tunables; Phase 3's sweep
  fixes `tau`, and the byte guard only becomes load-bearing at large P
  (Phase 4+ / the partial-agglomeration follow-up).

## 6. Relation to the existing record

MG-D5 ([`multigrid_pathway_plan.md`](multigrid_pathway_plan.md)) already
specifies "one device set for all levels; replicate below the
shardability floor" as the correctness fallback. This plan is the
**perf/capability promotion** of that fallback: choosing replication
deliberately for deep levels, adding the threshold + cross-boundary
reshard, and driving it with the census + depth-scaling evidence. The
sibling roadmap item lives under "Multigrid preconditioner — follow-up
measurements" in [`../../roadmap/open.md`](../../roadmap/open.md).
