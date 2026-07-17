---
status: done
date: 2026-07-17
---

# GB-2 wall-clock gate — multigrid vs spectral pressure preconditioner

Measured 2026-07-17 on one A100-SXM4-80GB (Levante GPU node), dev at
the phase-B merge lineage (`87aeabea` + `ee350bda`), `JAX_PLATFORMS=cuda
CUDA_VISIBLE_DEVICES=0`, float64. Steep terrain-following (mapped)
nonhydro2 **linear** step, depth `H(x) = 1 + 0.8 sin(x)` (ratio 9.0),
FV default family. GPU idle at start; single GPU, so the 4-GPU
`multi_output_fusion` XLA flag is not needed. Gate definition and the
shipped iteration numbers:
[`../plans/active/multigrid_pathway_plan.md`](../plans/active/multigrid_pathway_plan.md)
§3.

## Verdict: GATE FAILS at every size (multigrid is 5.5–13.4x SLOWER)

| n     | spectral ms/step (med [min,max]) | multigrid ms/step (med [min,max]) | multigrid speedup | gate ≥1.5x? |
|-------|----------------------------------|-----------------------------------|-------------------|-------------|
| 128^3 | 40.51 [40.27, 40.75]             | 542.70 [539.75, 548.63]           | **0.075x (13.40x slower)** | **NO** |
| 192^3 | 130.65 [130.64, 132.90]          | 976.01 [968.22, 980.60]           | **0.134x (7.47x slower)**  | **NO** |
| 256^3 | 291.69 [291.66, 297.59]          | 1614.75 [1611.02, 1622.08]        | **0.181x (5.54x slower)**  | **NO** |

Spread is sub-1% at every point (6 reps, 20 steps each). The gate asks
for multigrid ≥1.5x faster; it is 5.5–13.4x slower. The slowdown eases
with size (13.4 → 7.5 → 5.5) but never approaches parity, let alone
1.5x. **Spectral stays the correct production default on GPU.**

## Achieved CG iterations & compile (context)

Iteration counts: production `ConjugateGradient.solve`
`info["iterations"]` on a random mean-free RHS at the model tolerance
1e-8 (campaign methodology; worst-case — the in-model divergence RHS is
smoother). The wall-clock verification below confirms both sides
early-exit at these counts in-model.

| n     | spectral iters | multigrid iters | spectral compile s | multigrid compile s |
|-------|----------------|-----------------|--------------------|---------------------|
| 128^3 | 36             | 10              | 5.9                | 65.2                |
| 192^3 | 36             | 13              | 7.9                | 75.4                |
| 256^3 | 36             | 15              | 10.6               | 88.3                |

Matches the campaign (spectral flat ~36–44, multigrid flat ~10–15,
resolution-independent). Multigrid compiles ~10x longer (the 5-level
V-cycle unrolls into the trace).

Break-even (pressure-dominated limit): multigrid needs its
per-iteration cost within `spectral_iters / (1.5 * mg_iters)` of a
spectral iteration = 2.4x (128), 1.85x (192), 1.6x (256). Measured
cost of one V-cycle is ~66x a spectral CG iteration at 128^3 (below).

## Why: the V-cycle is latency-bound on GPU (root cause)

Decomposing the 128^3 measurements (linear model in the budget/skip
controls) gives, per step:

- shared physics (tendency, Coriolis, stratification, AB3): T ≈ 4.3 ms
- one **spectral** CG iteration: ≈ 0.80 ms
- one converged/no-op scan trip: ≈ 0.12 ms
- one **multigrid V-cycle**: ≈ 53 ms (**~66x a spectral CG iteration**)

The V-cycle uses vertical **line** smoothing (a tridiagonal Thomas
solve along z), and semicoarsening (MG-D4) keeps the vertical mesh at
FULL resolution at every level. Each V-cycle runs ~18 line-smoother
sweeps (5 levels x 2 + 8 coarse sweeps), each a **sequential** Thomas
solve of depth n_z (128/192/256). On GPU these sequential vertical
scans are latency-bound, and their sequential depth grows WITH n_z and
never shrinks under semicoarsening. So the iteration-count win
(36 → 10–15, 3.6x fewer) is swamped by a ~66x per-iteration cost. The
slowdown eases at larger n (more parallel horizontal columns hide the
fixed vertical latency) but the absolute V-cycle cost still climbs
(53 → ~75 → ~107 ms).

Follow-up (same day): the idealized kernel study
([`multigrid_kernel_study.md`](multigrid_kernel_study.md)) pinned ~91%
of the V-cycle cost to the scan-Thomas *lowering* (not the algorithm)
and **refuted** the z-parallel smoother lever hypothesized here
(point/Chebyshev fails even the isotropic control — line smoothing is
load-bearing for the semicoarsening). The recovery lever is a batched
tridiagonal kernel swap in `banded.py`: mapped reaches ~parity with
spectral, immersed wins outright. The iteration-count result
(robustness on steep/masked problems) stands on its own and is
CPU-relevant as measured.

## Measurement-fairness verification (why budget=100 is honest here)

The tolerance early-exit is a masked `lax.scan` with `lax.cond`
no-ops, not a `while_loop` (`krylov.py:167-190`). The wall-clock
numbers are only meaningful if the runtime skip actually shortens
compute. Confirmed:

- **Skip works (spectral).** 128^3 tol=1e-8 40.51 ms vs tol=None
  (fixed 100 iters) 84.24 ms — 2.08x faster, so ~36 real iterations
  execute, not 100.
- **Skip works (multigrid).** 128^3 mg budget=30 537.99 ms ≈
  budget=100 542.70 ms (Δ 4.7 ms). Both early-exit at ~10-13 V-cycles;
  the 542.70 is NOT 100 V-cycles. This is the control that proves
  budget=100 is a fair "converged cost", not a 100-iteration penalty.
- **No-op trip overhead is small and symmetric** (~0.12 ms/trip):
  spectral budget=60 35.88 ms vs budget=100 40.51 ms (Δ 4.63 ms over
  40 fewer trips). Penalizes both sides equally and marginally; the
  verdict is robust to it.

## Physics equivalence (timing equivalent work)

Final max|u| after 20 steps, spectral vs multigrid, relative diff:
128^3 5.1e-11 · 192^3 7.4e-11 · 256^3 2.8e-10 — all ≪ 1e-6. Every run
finite, none panicked. Both preconditioners solve the same Poisson
problem to the same tolerance and produce the same trajectory.

## Exact configuration (reproducibility)

- Model: `fridom.nonhydro2.Model`, FV default family (auto), linear
  (`advection=False`), `dsqr=0.25`,
  `coriolis=nh.FPlaneCoriolis(f0=1.0)`, default
  `ConstantStratification(n2=1.0)`, `AdamBashforth(order=3)`,
  `dt=0.02`.
- Grid: `IntervalMesh` x,y periodic (0,2π), z walled (0,1), n cells
  each; `CoordinateMapping(maps={"zp": z*H},
  params={"H": 1 + 0.8 sin(x)})`.
- Pressure: `pressure_iterations=100` (budget never truncates),
  `pressure_tolerance=1e-8`, `multigrid_levels=5`,
  `pressure_preconditioner` in {spectral, multigrid}.
- IC: u=sin(x)cos(y), v=0.3cos(x), b=0.01cos(πz).
- Timing: `chunk_size=STEPS=20`, `model.advance(20)` per rep, first
  advance (compile) excluded, 6 timed reps, sync
  `jax.block_until_ready(tree_leaves(model._carry))`; state fetched
  once after all reps for the sanity check only.
- Env: A100-SXM4-80GB, JAX_PLATFORMS=cuda, CUDA_VISIBLE_DEVICES=0,
  default allocator (75% preallocation ≈ 61 GB); 256^3 fit with no
  OOM.

## Anomalies

- None fatal. No OOM at 256^3, no GPU contention, no panic, all
  finite.
- The gate failure itself is the headline; it is not a measurement
  artifact (skip-works + budget-flatness controls confirm fairness).
- Multigrid compile is ~65-88 s (5-level trace unroll) vs ~6-11 s
  spectral — a one-time cost, excluded from ms/step.
