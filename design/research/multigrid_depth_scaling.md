---
status: done
date: 2026-07-18
---

# Multigrid size scaling — iterations vs per-iteration cost

Question (owner, 2026-07-18): why does the multigrid-preconditioned
mapped solve scale so badly with n — do the iterations grow, or does
each iteration get more expensive? Measured on one A100-SXM4-80GB
(same node/protocol as
[`multigrid_gb2_wallclock.md`](multigrid_gb2_wallclock.md)), dev at
the post-swap lineage (`0ece46b1`), steep mapped a=0.8 GB-2 config,
FV auto family, tol 1e-8, budget 100, float64.

## Verdict: iteration growth — and it is a depth-cap artifact

The widening deficit vs spectral (0.975x at 128^3 -> 0.67x at 512^3,
kernel-study Addendum) is **iteration-count growth, not per-iteration
cost** — and the growth is caused by the fixed `multigrid_levels=5`
default, not by the algorithm. At depth scaled to the coarsening
floor the count is flat 10 at every size, and the in-model 512^3 step
flips from 0.67x to **1.22x faster than spectral**.

## Iterations and per-iteration cost (standalone probe)

Manual PCG loop mirroring the production recurrence exactly
(`krylov._step` arithmetic, measure-weighted dots, mean projection;
achieved counts reproduce the production tolerance path — 10/36 at
64-128^3 match the campaign), random mean-free RHS, production
`MappedPressureSolver` construction (core.py parameters). Per-
iteration cost = steady median of the jitted single CG iteration with
concrete metric memos (the in-solve steady state).

| n | sp iters | mg iters (L=5) | mg iters (deep) | sp ms/iter | mg ms/iter (L=5) | mg/sp |
|-------|----|--------|---------------------|-------|-------|-------|
| 128^3 | 36 | 10 | — (L=5 is at floor+1) | 0.96 | 3.02 | 3.14x |
| 256^3 | 36 | 15 | **10** (L=6) | 6.41 | 15.8 | 2.46x |
| 512^3 | 36 | 27 | **10** (L=7; L=8 same) | 51.9 | 127.0 | 2.45x |

- **Per-iteration cost is healthy.** Over the 64x cell increase
  128^3 -> 512^3 a V-cycle-preconditioned iteration grows 42x
  (sublinear per cell) vs spectral's 54x; the cost ratio *improves*
  with n (3.14x -> 2.45x). Deep hierarchies add nothing measurable:
  125.9 (L=7) vs 127.0 ms (L=5) at 512^3 — the extra coarse levels
  are geometrically small.
- **The iteration count is what degraded.** At L=5 the h-independence
  breaks (10 -> 15 -> 27); the advantage over spectral collapses from
  3.6x to 1.33x, and 1.33x fewer iterations at 2.45x cost each
  reproduces the measured 0.67x step deficit. The campaign's "flat
  10-15" held only at 32-96^3, where 5 levels already reach the
  4-8-cell floor.

## Root cause: the coarsest level outgrows its 8 sweeps

With 5 levels the coarsest grid is `n/16` per horizontal axis at full
n_z: 8x8x128 at 128^3 (eight damped line-Jacobi sweeps effectively
solve it) but **32x32x512 at 512^3** — a ~1000x larger problem whose
smooth *horizontal* modes the vertical-line smoother reduces only at
Jacobi speed. The unsolved coarse correction re-enters every V-cycle,
degrading it as a preconditioner, and the outer CG pays 2.7x the
iterations. Restoring the floor-depth hierarchy (L=7: coarsest
8x8x512) brings back flat 10; L=8 (4x4x512) is identical (10
iterations, same per-iteration cost) — depth to the floor is free.

## V-cycle decomposition (L=5, per level, chained-jit medians, ms)

512^3: whole V-cycle 102.9 (component reconstruction 98.9 — the gap
is the per-level mean projections, unmeasured):

| level | shape | sweep | op apply | tridiag | restrict+prolong |
|---|---|---|---|---|---|
| 0 | 512x512x512 | 28.12 | 12.02 | 15.68 | 4.08 |
| 1 | 256x256x512 | 7.10 | 3.23 | 3.82 | 1.08 |
| 2 | 128x128x512 | 1.91 | 0.87 | 1.02 | 0.37 |
| 3 | 64x64x512 | 0.57 | 0.30 | 0.27 | 0.09 |
| 4 | 32x32x512 | 0.17 | 0.11 | 0.08 | — |

Cost per cycle = 2 sweeps + residual op + transfer round trip per
fine level + 8 coarse sweeps: level 0 alone is 72 of the 103 ms; the
coarsest level is 1.4 ms. Geometric decay is intact — the cycle is
bandwidth-dominated by the finest level (one sweep = one cuSPARSE
batched solve 15.7 + one mapped operator apply 12.0), no latency
pathology remains post-swap. (128^3: whole 2.15, recon 2.22, level 0
= 0.49 sweep / 0.25 op / 0.21 tridiag.)

Raw kernel scaling confirms proportionality: cuSPARSE per solve
0.215 / 1.84 / 15.7 ms at 128/256/512^3 (73x per 64x cells; pcr
0.32 / 2.59 / 29.5), and the 512^3 value matches the level-0 tridiag
entry exactly.

## In-model GB-2 steps at scaled depth (ms/step, median of 6x20)

| n | spectral | multigrid (levels) | mg vs spectral |
|-------|--------|---------------|-------|
| 128^3 | 40.97 | 42.01 (L=5, floor+1) | 0.975x (parity) |
| 256^3 | 291.61 | 237.83 (L=6, floor+1) | **1.23x** |
| 512^3 | 2277.91 | 1873.29 (L=7) | **1.22x** |

Physics: the L=7 512^3 final state matches the spectral reference to
rel 4.7e-11, the L=6 256^3 one to 1.6e-11 (same equivalence class as
the swap record); no panic; peak 43.8 GiB (mg) vs 28.5 (spectral) at
512^3. The 256^3 spectral rerun reproduces the 07-17 record to 0.03%
(291.61 vs 291.69).

**This corrects the kernel-study Addendum's size-trend conclusion**:
"the deficit widens with n" was measured *at the depth cap*; at
floor-scaled depth the trend rises with n (0.975x -> 1.23x -> 1.22x)
and multigrid beats spectral from 256^3 up. The GB-2 gate (>=1.5x)
remains unmet at <=512^3, so spectral remains an excellent default —
but multigrid at proper depth is now the *faster* mapped-GPU option
at production sizes, not a robustness-only fallback.

## Consequence: the `multigrid_levels=5` default (owner decision)

The hierarchy builder already floors every horizontal axis at four
cells and stops there, so `multigrid_levels` only ever *caps* depth —
and the cap is what broke h-independence. Floor depth is measured
free (per-cycle cost unchanged, L=8 == L=7 == L=5 per iteration).
Recommendation: default the model knob to floor-limited depth (e.g.
`multigrid_levels=None` -> coarsen to the floor, int retained as an
explicit cap), which restores flat iterations at every size with no
cost. Src change deliberately **not** made in this investigation;
needs the usual branch + tests if adopted.

## Anomalies

- The standalone probe's mg count at 512^3 L=5 (27, random RHS) is an
  upper bound on the in-model count (~24 implied by the step budget;
  the in-model divergence RHS is smoother) — same relation as the
  campaign noted. Trend conclusions are unaffected.
- `scan` raw-kernel timing at 512^3 OOMs in the chained standalone
  harness (8 unrolled scans' buffers) — not meaningful; the in-model
  scan step at 512^3 (7394 ms) was measured in the swap Addendum.
- Probe scripts live in the (wipeable) session scratchpad:
  `probe_cost.py` / `probe_vcycle.py` / `probe_kernel.py` /
  `probe_common.py` + `run_probes.sh` / `run_depth.sh`; results
  `probe_results.jsonl` / `depth_results.jsonl`.
