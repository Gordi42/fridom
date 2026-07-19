# Multigrid coarse-grid agglomeration — Phase 3 (4×A100 wall-clock)

Owner-authorized single SLURM job (2026-07-19). Job `26355284`, node
l50193, `--partition=gpu`, `--exclusive`, 4× A100-80GB, single-process
GSPMD, ~59 min, exit 0. Source **pinned** to a detached worktree at dev
`8752170a` (`PYTHONPATH` forces the pinned tree ahead of the venv's
editable install; the driver hard-fails unless `fridom` resolves under
it). jax 0.10.2, float64.
`XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion` (jax#39100
silent-miscompile workaround) on **every** run.

**Protocol (GB-2).** ms/step = median of 6×20 steps, compile excluded,
`block_until_ready`. Linear nonhydro2, FV auto, dsqr 0.25, FPlaneCoriolis
f0=1, AB3, dt 0.02, `chunk_size=20`, budget 100, tol 1e-8, floor depth
(`multigrid_levels=None`). CG iterations from `solve_info` on a mean-free
RHS. **Mapped** = steep terrain `H(x)=1+0.8 sin(x)` (ratio 9), x/y
periodic, z walled, full-3D coarsening. **Immersed** = tilted slope
order-4 quadrature (wet frac 0.706), semicoarsening. `mg-off/t2/t4/t8`
all use `method=auto` (= cuSPARSE on GPU); only the
`multigrid_agglomerate` τ knob differs. Matrix: mapped `n∈{128,256,512}`
× {spectral, mg-off, mg-t2, mg-t4, mg-t8}; immersed `n∈{128,256}` ×
{spectral, mg-off, mg-t4}. All configs `finite=true`.

Files: `results.jsonl` (42 records), `census_mg-off_n128.txt` +
`census_mg-t4_n128.txt` (GPU HLO collective censuses of the mapped n=128
step), `driver.py`, `hlo_dump.py`, `census.py` (parser), `gb2_common.py`
(builders, extended with the τ knob), `run_agglom_phase3.sbatch`.

---

## 1. Timing + iterations

### Mapped (median ms/step [min, max]; CG iters; ×spec = spectral/mg; ×off = off/τ)

| n | tag | ms/step | iters | ×spec | ×off |
|---|---|---|---|---|---|
| 128 | spectral | 29.92 [28.94, 31.16] | 36 | 1.00 | — |
| 128 | mg-off | 73.92 [67.75, 75.52] | 10 | 0.405 | 1.000 |
| 128 | mg-t2 | 76.35 [68.66, 80.97] | 10 | 0.392 | 0.968 (−3.2%) |
| 128 | mg-t4 | 73.28 [66.05, 80.71] | 10 | 0.408 | 1.009 (+0.9%) |
| 128 | mg-t8 | 68.13 [61.43, 75.27] | 10 | 0.439 | 1.085 (+8.5%) |
| 256 | spectral | 98.72 [95.14, 102.56] | 36 | 1.00 | — |
| 256 | mg-off | 139.35 [127.12, 141.16] | 10 | 0.708 | 1.000 |
| 256 | mg-t2 | 137.50 [125.79, 138.99] | 10 | 0.718 | 1.013 (+1.3%) |
| 256 | mg-t4 | 135.01 [123.62, 136.35] | 10 | 0.731 | 1.032 (+3.2%) |
| 256 | mg-t8 | 131.82 [120.58, 133.22] | 10 | 0.749 | 1.057 (+5.7%) |
| 512 | spectral | 605.62 [585.41, 625.08] | 36 | 1.00 | — |
| 512 | mg-off | 499.89 [459.85, 502.15] | 10 | 1.211 | 1.000 |
| 512 | mg-t2 | 499.83 [459.81, 502.16] | 10 | 1.212 | 1.000 (+0.0%) |
| 512 | mg-t4 | 497.48 [457.82, 499.71] | 10 | 1.217 | 1.005 (+0.5%) |
| 512 | mg-t8 | 493.12 [453.86, 495.24] | 10 | 1.228 | 1.014 (+1.4%) |

### Immersed (only τ4 measured against off)

| n | tag | ms/step | iters | ×spec | ×off |
|---|---|---|---|---|---|
| 128 | spectral | 57.23 [56.69, 57.43] | 73 | 1.00 | — |
| 128 | mg-off | 168.80 [155.22, 179.17] | 20 | 0.339 | 1.000 |
| 128 | mg-t4 | 170.36 [163.16, 170.40] | 20 | 0.336 | 0.991 (−0.9%) |
| 256 | spectral | 205.42 [203.94, 206.87] | 71 | 1.00 | — |
| 256 | mg-off | 359.65 [341.65, 372.93] | 21 | 0.571 | 1.000 |
| 256 | mg-t4 | 375.77 [357.86, 388.69] | 21 | 0.547 | 0.957 (−4.3%) |

### Decision numbers

- **The 128³ gap is not closed by any τ.** mg-off is 0.405× spectral
  (73.9 vs 29.9 ms). The best τ, t8, reaches only 0.439× (68.1 ms) —
  still 2.3× slower than spectral, and above the 42.0 ms single-GPU mg
  cost cited as the target. τ recovers at most **5.8 ms** off mg-off
  (t8); τ4, the structural default, recovers **0.6 ms**.
- **512³ (mg already wins).** mg-off is 1.211× spectral; τ widens it
  only to 1.228× (t8) — far below the 1.5× GB bar. τ helps by ≤1.4%.
- **256³.** mg-off 0.708× spectral; τ8 lifts mg by 5.7% (to 0.749×),
  never overtaking spectral.
- **No τ is never-worse-than-off.** On mapped, larger τ is monotone-
  better (t8 best at every n). On immersed, τ4 is **worse** than off at
  both sizes (−0.9%, −4.3%); τ2/τ8 on immersed were not measured, so no
  τ can be claimed universally non-regressive.
- **Mapped τ deltas sit within thermal spread.** Every `all_ms` array
  is bimodal (2 cool reps then 4 throttled); the median tracks the
  throttled regime and the ≤8.5% τ effects overlap the min/max ranges.
  The t8 minima are consistently below the off minima, so a small real
  effect is plausible, but not a wall-clock win worth the cost.

### Sanity

- **Iterations IDENTICAL ON vs OFF** in every case (mapped 10 across all
  τ; immersed 128 20=20, 256 21=21). No drift. Residual norms agree ON
  vs OFF to ~13 significant figures.
- **maxu ON vs OFF** agrees to ~1e-15–1e-16 relative in every (case,n) —
  far below the ~1e-8 solve tolerance and the ~1e-8 expected. No maxu
  gap. (Spectral vs mg maxu differ at ~1e-10, the expected
  different-path floor.)
- **Peak memory unchanged** by agglomeration (off ≈ τ to 3 digits;
  mapped 128 ~0.17, 256 ~1.31, 512 ~9.59 GiB/dev; immersed 128 ~0.23,
  256 ~1.57). Replicated coarse levels are tiny.
- **Compile time roughly triples with τ** (mapped 128 mg-off 45 s →
  t4 143 s / t8 134 s; noisy, autotuning-dependent) — the extra
  replicated-level modules are not free at build time.

### Cross-check against this week's baselines

(`../multigrid_gspmd_validation/results.md`, node l50009, dev 0c950a33.
Node-to-node variance expected; verdicts use within-job ratios only.)

| case | this job (l50193) | baseline (l50009) |
|---|---|---|
| 128³ mapped spectral | 29.9 | 32.3 |
| 128³ mapped mg-off | 73.9 (0.40×) | 86.4 (0.37×) |
| 512³ mapped spectral | 605.6 | 600.5 |
| 512³ mapped mg-off | 499.9 (1.21×) | 539.8 (1.11×) |

Spectral matches within ~0.9–7%. mg-off is ~7–14% faster on this node,
so the within-job 512³ mg win (1.21×) reads a touch stronger than the
baseline's 1.11×; the 128³ mg/spectral ratio (0.40×) matches the
baseline 0.37× within variance. **The baseline immersed numbers are
1-GPU** (validation LEG 2: mg 60.8 ms at 128³, 1.09× spectral); this job
is the first **4-GPU** immersed measurement, where mg-off is 168.8 ms
(0.339× spectral) — the small-n multi-GPU latency regression the census
predicts, ~2.8× the 1-GPU mg cost. Not comparable across device counts;
noted so the 60.8-vs-168.8 gap is not misread as a regression.

---

## 2. GPU HLO census (mapped n=128, mg-off vs mg-t4)

The census is the **mapped full-3D** hierarchy (`hlo_dump.py` builds the
mapped model, `coarsen_vertical=True`). Its OFF coarse-permute counts
(L8 33+7, L4 24+22 per V-cycle) match kernel-study Addendum 3 exactly, so
it is directly comparable to that CPU census — but it is **not** the
immersed/semicoarsen hierarchy the CPU "24→0, 6→0" projection came from.

Per V-cycle (real conditional branch = 1 CG iter) and per step
(iters=10 = 1 peel + 9 real trips + outside-while):

| kind | OFF /Vcyc | ON(τ4) /Vcyc | OFF /step | ON /step |
|---|---|---|---|---|
| collective-permute | 262 | 241 | 2930 | 2700 |
| all-reduce | 26 | 29 | 288 | 321 |
| all-to-all | **0** | **0** | 0 | 0 |
| all-gather | 0 | 0 | 1 | 1 |
| **total** | | | **3219** | **3022** |

Per-level CP (real/Vcyc), OFF → ON: L128 44→44, L64 33→33, L64* 5→5,
L32 42→42, L32* 5→5, L16 42→42, L16* 5→5, **L8 33→9, L8* 7→10**,
**L4 24→24, L4* 22→22**. Only the L8 level moved.

### Verdict (a) — did the sub-KB coarse permutes vanish? NO — only a partial dent.

On the mapped full-3D hierarchy τ4 removed **only** the L8 720-B halos
(real/Vcyc 33→9, partly replaced by a handful of 1600-B 2-plane
`f64[2,10,10]` reshard halos), and **left the coarsest L4 permutes
untouched** — the 128-B/576-B `f64[1,4,4]`/`f64[2,6,6]` halos (24+22 per
V-cycle) are identical OFF and ON, despite the switch rule at τ=4
nominally replicating every sub-4-extent level (L8 at 2 planes/shard, L4
at 1). Net coarse-permute removal is 262→241 per V-cycle (−8%),
2930→2700 per step. Addendum 3 flagged **86** of the 262 per-trip CPs
(L8=40, L4=46) for removal; τ4 removed **21**, and specifically not the
tiniest L4 slice, the worst latency offender. The dramatic CPU
immersed-semicoarsen removal (halo CP 24→0, all-to-all 6→0) does **not**
appear here: this hierarchy has **no** vertical-line-smoother all-to-alls
to remove (A2A=0 in both OFF and ON), and τ4 under-fires on its coarse
permutes.

*Why L4 was not replicated at τ4 cannot be determined from the census
alone (the switch nominally fires at L8 and replicates L8 + below). τ8 —
fastest on mapped — presumably switches one level higher and removes
more, but no census was dumped for t8, so that is inference not
measurement. Moot for the default decision given the null wall-clock
result; an HLO-level follow-up only if the lever is ever revisited.*

### Verdict (b) — folding of replicated-level reductions? NO — GPU re-partitions like CPU.

The GPU partitioner did **not** fold the replicated coarse levels'
projection sums to local reductions. all-reduce **rose** 288→321 per
step (+11%; per V-cycle 26→29); all-gather stayed at 1 (payload 8→16 B).
So the total collective count barely moves: **3219→3022 per step
(−6%)** — the same net-flat pattern the CPU forced-4 census showed. The
folding gap the plan flagged is present on GPU too; a
`with_sharding_constraint` hint remains untried but is moot given the
null timing.

---

## 3. Verdict

- **τ recommendation: keep default OFF.** No τ closes the small-n gap
  (spectral stays the right 4-GPU default at ≤256³: mg-off 0.34–0.71×,
  best τ ≤0.75×), no τ meaningfully widens the 512³ mg win (≤1.4%, still
  <1.5× bar), and the one τ tested on immersed (τ4) **regresses** it
  (−0.9%, −4.3%). The mapped τ gains (≤8.5%) sit within thermal spread
  and cost ~3× compile. If ever revisited, τ8 is the only mapped-positive
  setting, but it needs (i) immersed τ2/τ8 shown non-regressive and (ii)
  the L4-not-replicated under-fire fixed so it actually removes the
  coarsest permutes.
- **Correction to the ~9–14 ms projection** (Addendum 3 / plan §4): the
  projected 128³ recovery is contradicted — best measured recovery is
  **5.8 ms** (t8), and τ4 (the structural default) recovers **0.6 ms**.
  The census explains it: τ4 removes only 21 of the 86 flagged coarse
  permutes and adds all-reduces (folding gap), so the collective count —
  and the wall clock — barely move.
</content>
