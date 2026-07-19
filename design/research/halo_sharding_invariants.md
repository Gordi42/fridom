---
status: frozen
date: 2026-07-18
---

# Halo negotiation under sharding — three invariant fixes

Three related defects in the halo-negotiation/verify machinery, all
found and fixed 2026-07-18 while validating the channel-eigenmode and
GSPMD-illegality work. Common thread: the sharding cap and the halo
claims must respect two invariants — **(1)** capping may squeeze a
multi-application chain (runtime re-sync repairs it) but never a
single application's stencil reach, and **(2)** every path that writes
or checks a width must apply the same cap the negotiation applied.

## 1. Frozen-verify cap asymmetry (merge `8a787452`)

`negotiate()` ran the sync-free demand through `_cap_for_sharding`
(cap = shortest shard − 1, floored) and `freeze()` sealed the *capped*
width — but `_verify_frozen` recomputed the **raw** demand and
compared it to the capped record. Every re-assembly on a cap-engaged
grid (`model.variant()`, `linearize`, AdiabaticRamping, IMEX) faulted
with `GridFrozenError`, violating the ⊆ lemma promised in
`Model.variant`'s docstring. For the nonhydro channel the fault window
is exactly shortest shard == 2, i.e. n ∈ {8, 11, 14, 17} on 4 devices
— which is why the fused-contraction tests at n = 12/16 never saw it.
Fix: the verify path applies the identical cap (same function, same
args, same device count from the frozen decomposition); capped-vs-
capped restores the lemma. Note the cap deliberately also applies to
explicit `halo=` on verify, consistent with negotiate — ratification
of that reading and of the cap's *over-reach* (it caps every ghost
axis, including non-sharded ones) were flagged as open owner items.
*Update 2026-07-19:* both ruled and shipped (dev merge `fc2a3b66`,
`fix/halo-floor-semantics`): explicit `halo=`/`extra_halo` instead
joins the uncappable per-application floor on negotiate and verify
(a wide width on a small axis disqualifies it from sharding), and
the cap is scoped to sharding-candidate axes only.

## 2. Cap floor blind to trace-only wide stencils (in dev via `94786a7c`)

After the interval-accounting landing (`40a24df8`) the registered
per-op reach dropped to 1, so `_cap_for_sharding`'s floor
(`_registry_halo`) fell to 1 — but the wide advection stencils
(`_BiasedFaceReconstruction`, `_CenteredFaceInterpolation`) are
trace-only, invisible to the registry. Small axes that the old wider
floor had disqualified now sharded with a capped halo *below one
application's reach*, and the term evaluation raised loudly
("negotiated halo width 1 … too small for the N-point stencil"): 97
forced-4 advection tests. Real multi-GPU scale was unaffected (the cap
only fires when the shortest shard ≤ stencil reach). Fix, two parts:
the halo tracer records each application's reach and exposes the
per-axis maximum, which both `negotiate` and `_verify_frozen` merge
into the floor (symmetry preserved); and assembly gained a
**pre-validation negotiate** (step 6a) so the construction-time grid
collapses *before* `composer.dry_run` validates on it — dry_run's
provisional sharded view is consumed by nothing (it is pure
`eval_shape` validation), the pre-pass only widens or collapses
(never narrows below the bind halo, preserving the chart-coupled
exemption), and `extra_halo` stays deferred to step 7 (preserving the
WideHalo bypass). All 97 tests flip green unedited; the n=8 linear
channel (chain of reach-1 ops, demand 2, cap 1) still shards 4-way.

## 3. Bounded footprint reach — silent wrong physics (merge `b57e3e78`)

The severe one. A staggered kernel that shrinks the point count
(`Center→Inner`) published its requirements from `exterior_reach`,
which on a bounded axis cancels to `(0, 0)` (the shrinking codomain
offsets the stencil overhang at the wall). `_ensure_valid` and the
negotiation tracer read that as "no ghosts needed" and **skipped the
inter-shard halo sync** whenever the sharded axis was walled: wall
cells correct (BC fill runs), interior shard-boundary cells grossly
wrong, error growing with N (0.59 @ 16³ → 35.2 @ 128³ on 4 devices) —
**silent**, at real multi-GPU scale, in every bounded diffusion/
friction closure (nodal and FV) and, latently, advection. Periodic
was unaffected (`n_out == n_in`, no cancellation). Fix: requirements
publish the new `footprint_reach` (the per-shard stencil footprint);
`exterior_reach` is kept for the R1 boundary-legality check where the
global term is genuinely correct. Sound because bounded intermediate
validity is 0-or-full, so the footprint only decides whether the sync
fires. Also closed a latent hole: a *lone* bounded staggered op
negotiated storage width 0. Periodic storage is bit-identical (the
n+8→n+6 win survives); the only added step cost is the sync that
correctness requires. Fixed transitively: FV closures; verified cured
downstream: the sadourny doubly-walled sharded energy-rate and
stability failures.

## Verification of the combined state (2026-07-18)

On the merged dev: advection walls/fv/base/background/selected 316
passed (3 remaining reds are the known immersed-graded conservation
regression, another campaign's); diffusion walls/fv/mapped 65 passed;
sadourny 37 passed; fv_default periodic and the fallback file green
after the last bare-negotiate pin; freeze-fingerprint/grid/
decomposition 520 passed; ruff clean.
