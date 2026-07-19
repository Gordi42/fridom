---
status: done
date: 2026-07-17
---

# Multigrid V-cycle kernel study — where the 53 ms goes, and the floor

Idealized (bare-jax) follow-up to the GB-2 wall-clock failure
([`multigrid_gb2_wallclock.md`](multigrid_gb2_wallclock.md)), same day,
same A100 (jax 0.10.2, float64). Refutes the z-parallel-smoother lever
recorded at the GB-2 landing; establishes the kernel-swap lever instead.
Scripts lived in the session scratchpad (node-local /tmp, wiped with
the SLURM job); this record is the durable copy.


Read-only GPU study, one A100-SXM4-80GB, jax 0.10.2, float64, `JAX_PLATFORMS=cuda`.
All scripts bare-jax (no fridom) except the in-model calibration in the Synthesis.
Scripts in this directory: `phaseA.py`, `phaseBC.py`, `vcyc_extra.py`,
`spectral_ref.py`, `faithful_twin.py`, `inmodel_measure.py`.

Measured A100 effective bandwidth (add-kernel, read+write, 536 MB): **1618 GB/s**.

## Headline

- **91%+ of the 53 ms V-cycle is the scan-Thomas smoother lowering.** Confirmed
  three ways: 16 sweeps x 2.75 ms/solve = 44 ms; bare-jax scan V-cycle 48.5 ms;
  faithful mapped twin scan V-cycle 48.2 ms — all ~= the in-model 53 ms.
- **The scan solve is latency-bound and batch-INDEPENDENT**: 2.75 ms/solve whether
  the batch is 16384 columns (128^2) or 64 columns (8^2). Cost = 2*n_z sequential
  scan-loop launches (~256 tiny kernels), ~10.7 us launch latency each.
- **A batched/parallel kernel recovers the V-cycle 40-48x**: PCR 1.01 ms,
  tds_batched (cuSPARSE gtsv2StridedBatch) 1.00-1.06 ms, both including the
  faithful mapped operator, per-column-varying bands, guards and projections.
- **All four kernels are reverse-mode differentiable** in jax 0.10.2, including
  `jax.lax.linalg.tridiagonal_solve` (grad wrt rhs AND wrt diag) — this overturns
  the banded.py:28-30 claim that its "autodiff / batching support is
  backend-uneven". No custom_jvp/custom_vjp is needed for any candidate.
- **Keep line smoothing.** It is anisotropy-flat (15 Richardson iters from
  A_ratio 20 to 800). A point/Chebyshev smoother is fundamentally incompatible
  with the horizontal semicoarsening hierarchy and fails even the isotropic
  control. Since PCR/tds_batched are bit-identical to scan-Thomas, swapping the
  kernel is convergence-NEUTRAL: same iteration count, 40-48x cheaper V-cycle.

---

## Phase A — batched tridiagonal kernel shootout

Problem: solve `T x = b` along z for every column; n_z=128, float64, SPD
diagonally-dominant anisotropic-Poisson tridiagonal (diag ~= 2/dz^2 + 4/dx^2,
off-diag = -1/dz^2). Timing = device time, median of 10, k=100 solves chained in
a jitted `fori_loop` (excludes python dispatch and compile), us/solve.

| n_xy^2 | batch | scan-Thomas | tds_single | tds_batched | PCR | assoc-2x2 | floor(5-arr) | floor(2-arr) |
|--------|-------|-------------|------------|-------------|-----|-----------|--------------|--------------|
| 128^2 | 16384 | **2826** | 156034 | 244 | 396 | 222 | 51.8 | 20.7 |
| 64^2  |  4096 | 2746 | 39034 | 69 | 90 | 101 | 13.0 | 5.2 |
| 32^2  |  1024 | 2761 | 9811 | 28 | 21 | 28 | 3.2 | 1.3 |
| 16^2  |   256 | 2768 | 2505 | 21 | 13 | 22 | 0.8 | 0.3 |
| 8^2   |    64 | 2744 | 691 | 18 | 15 | 21 | 0.2 | 0.1 |

Correctness (max abs err vs dense solve): all four kernels 1e-18..1e-19.

Differentiability (jax.grad of sum(x^2)): scan-Thomas, tds_single, PCR, assoc-2x2
all finite, identical gnorm 5.44e-6; tds also differentiable wrt diag (2.85e-9).

Reading of the table:
- **scan-Thomas is FLAT ~2750 us across a 256x range in batch** — the signature
  of a latency-bound sequential loop. At batch 64 it is still 2744 us: the cost is
  the 2*n_z=256 sequential scan-loop iterations, not the arithmetic. ~2744/256 =
  10.7 us/step, consistent with GPU kernel-launch latency.
- **tds_single (cuSPARSE gtsv2, one system, many RHS) is pathological** — 156 ms
  at batch 16384; it does O(nrhs) serial work. Never use the single-system form
  with a wide RHS.
- **tds_batched (gtsv2StridedBatch, one distinct system per column) is the fast
  drop-in for the varying-coefficient (mapped) case**: 244 us = 11.6x faster than
  scan, 4.7x the 5-array bandwidth floor.
- **PCR (parallel cyclic reduction, log-depth, pure jax) 396 us** = 7.1x faster
  than scan; band-value-independent so identical cost for constant OR per-column
  bands; trivially differentiable.
- **assoc-2x2 (Mobius+affine associative_scan) 222 us** — fastest microbench, but
  see Phase B: its lead does NOT survive the V-cycle context.

Bandwidth floor: 5-array (lo,di,up,rhs read + x write, 5x16.78 MB / 1618 GB/s) =
51.8 us; 2-array (constant-coeff bands tiny) = 20.7 us. tds_batched at 4.7x floor
and PCR at 7.6x floor are near-optimal for multi-pass algorithms; the latency-bound
scan-Thomas is at 55x the floor (2826/51.8) — the entire deficit is launch latency.

STRETCH (Pallas) not built: A2b/A3 already land at 4-8x the bandwidth floor, so
the headroom to a hand-fused kernel is <5x and not on the critical path for the
conclusion. Noted as a possible follow-up only.

---

## Phase B — idealized V-cycle twin (128^3)

5 semicoarsened levels (128^2, 64^2, 32^2, 16^2, 8^2, all x n_z=128), V(1,1),
8 coarse sweeps, damped line-Jacobi omega=0.8, linear full-weighting transfer,
mean-removal projection per level, periodic x,y / Neumann z. A_ratio
(inv_dz^2 / inv_dx^2 at level 0) = 39.5 (the true 128^3 geometric value).
ms per V-cycle, median, compile excluded (constant-coeff twin):

| smoother | V-cycle ms | vs scan |
|----------|------------|---------|
| scan-Thomas (shipped mirror) | **48.50** | 1.00x |
| PCR | **1.01** | 48x |
| tds_single | 424.20 | 0.11x (pathological) |
| assoc-2x2 | 6.02 | 8x (microbench lead lost) |
| Chebyshev deg-2 (point) | 0.45 | — |
| Chebyshev deg-3 (point) | 0.68 | — |
| Chebyshev deg-4 (point) | 1.29 | — |

Faithful MAPPED twin (variable-coeff operator, per-column-VARYING vertical bands,
double-where guards, per-level projection — i.e. the real mapped smoother path):

| smoother | V-cycle ms |
|----------|------------|
| scan-Thomas | 48.19 |
| PCR | **1.18** |
| tds_batched | **1.00** |

operator-apply decomposition (faithful twin, all levels): **0.407 ms total**
(finest 60 us x3, then 22/12/12 us, coarsest 11 us x8).

**Calibration verdict (loud):** the twin scan V-cycle (48.5 ms, and 48.2 ms with
the faithful mapped operator) ~= the in-model 53 ms. **So the 53 ms IS the
scan-Thomas lowering, NOT fridom-side assembly overhead.** The idealized replica
reproduces the pathology from first principles.

Component decomposition (PCR, constant twin): smoother sweeps 0.991 ms;
transfer + restriction-residual + projection 0.019 ms. **The V-cycle cost is
essentially 100% the vertical solves; transfers/projections are free.** After the
kernel swap the next bottleneck is the finest-level smoother solve itself
(229 us/sweep), already at ~4-5x the bandwidth floor.

Per-level 1-sweep cost (PCR): lev0 128^2 229 us, lev1 64^2 80 us, lev2 32^2 37 us,
lev3 16^2 30 us, lev4 8^2 30 us. Note coarse levels DON'T shrink much (they hit a
latency floor ~30 us) but the 8 coarse sweeps at 30 us = 0.24 ms is still cheap.

assoc-2x2 note: 222 us/solve microbench but 6.0 ms V-cycle — the associative_scan
2x2 matmul does not fuse across the 5-level unroll as well as PCR/cuSPARSE
(a "micro wins reverse in real-step fusion context" case). Not recommended.

---

## Phase C — does a cheap V-cycle still converge?

Stationary Richardson iteration `x += Vcycle(b - A x)` (mean-projected), to
relative residual 1e-8, cap 200-300. Line smoother = PCR (bit-identical to
scan-Thomas, so this IS the shipped V-cycle's convergence). A_ratio =
inv_dz^2/inv_dx^2.

| A_ratio | line (PCR) iters | Chebyshev-3 point iters |
|---------|------------------|-------------------------|
| 1 (isotropy control) | 34 | 300 (stall, rel 1.9e-6) |
| ~20 (geometric grid)  | 15 | 300 (stall, rel 1.4e-3) |
| ~100 (steep mapped proxy) | 15 | 300 (stall, rel 2.9e-3) |
| ~800 (very steep) | 15 | 300 (stall, rel 1.4e-2) |

Best-tuned Chebyshev sweep (degree 2/4/6 x bounds {0.3,0.5,0.1}xrho, coarse 12):
- isotropy: only deg-6/bounds(0.1,1.1) reaches 1e-8, and needs **173** iters.
- steep~100: NONE converge (best rel 1.3e-3 after 200 iters).

Interpretation:
- **Line smoothing is anisotropy-robust and is what buys the iteration win**: 15
  iters flat from A_ratio 20 to 800 (better than isotropy's 34 — strong vertical
  coupling makes the vertical line solve capture almost the whole operator). The
  15 Richardson iters is consistent with the report's 10-13 in-model CG iters
  (CG accelerates ~sqrt).
- **Point/Chebyshev smoothing is not a viable substitute in this hierarchy.** It
  fails even the isotropic control (173 iters best case). Root cause: the
  semicoarsening (MG-D4: vertical FULL at every level) coarsens ONLY horizontally
  and relies on the line solve to handle the uncoarsened vertical direction. A
  point smoother leaves the vertical error modes neither smoothed nor coarsened,
  so nothing damps them. **Line smoothing is load-bearing for the entire
  semicoarsening design; you cannot swap it out without redesigning the
  hierarchy.**
- **Therefore the right play is: keep the line smoother, swap only the kernel
  lowering.** PCR and tds_batched compute the SAME T^{-1} to machine precision
  (Phase A: 1e-18 agreement), so the swap changes ZERO about convergence (same
  10-15 iters) and makes the V-cycle 40-48x cheaper.

---

## In-model measurement (real fridom, read-only, no repo edit)

The scratchpad already carried the prior campaign's construction helpers, so I
reached fridom's OWN operators/transfers/bands and swapped only the smoother by
subclassing `VerticalLineJacobi` in the script (PCR / tds_batched) and
monkeypatching `solver._build_vcycle` on the instance. `MappedPressureSolver`
(steep mapped a=0.8, n=128, depth=5); `ImmersedPressureSolver` (slope, n=128,
nz=128).

**Isolated components (chained fori_loop, device time):**
- operator apply `A` (mapped FV, ~15-20 array passes, metric tensor + cross
  terms): **0.225 ms** — matches the spectral-decomposition inference
  (0.80 spectral iter - 0.48 FFT - 0.09 CG work = 0.23).
- V-cycle: scan **44.2 ms** (calibrates the report's ~53 ms), PCR **2.10 ms**,
  tds_batched **1.97 ms** (21-22x).

**End-to-end jitted CG SOLVE (the ground truth — real masked-scan CG, all
per-iteration context included):**

| case | budget/iters | spectral | scan (shipped) | PCR | tds_batched |
|------|--------------|----------|----------------|-----|-------------|
| mapped 128^3   | 100 (conv @10)   | (36 it, ~36 ms) | **551 ms** | **59.6 ms** | **42.9 ms** |
| immersed 128^3 | 30 fixed         | 25.96 ms | **1034.9 ms** | **94.96 ms** | **58.13 ms** |

The scan solve (551 ms) reproduces the report's 542.7 ms step -> the monkeypatch
is faithful. **The isolated V-cycle (2.1 ms) was optimistic**: inside the masked
lax.scan CG, per-V-cycle cost is ~5 ms (no cross-iteration fusion, the masked
`lax.cond` blocks it, and the full CG state is carried each trip). The
end-to-end solve is the honest number: **PCR 9.3x, tds_batched 12.9x** faster
than the shipped scan on the mapped case; **11x / 18x** on immersed.

---

## Synthesis

Model (calibrated: scan-multigrid 4.3 + 10x53.9 = 543 ~= measured 542.7):
`step = physics(4.3) + iters x (V-cycle + CG-context)`. I project by
**substitution off the report's measured step** (cancels the fixed context):
`step_new = step_scan - iters x (solve_scan - solve_new)/iters` == `step_scan -
(solve_scan - solve_new)`.

### Mapped (the GB-2 gate case), 128^3

| preconditioner | solve ms | projected STEP ms | vs spectral 40.5 | GB-2 (>=1.5x, <=27ms)? |
|----------------|----------|-------------------|------------------|------------------------|
| spectral (baseline) | ~36 | 40.5 | 1.00x | — |
| scan multigrid (shipped) | 551 | 542.7 | 0.075x (13.4x slower) | NO |
| **PCR multigrid** | 59.6 | **51.1** | **0.79x** | NO (but 10.6x better) |
| **tds_batched multigrid** | 42.9 | **34.5** | **1.17x** | NO (beats spectral) |

**GB-2 verdict (mapped, 128^3): the kernel swap does NOT reach the 1.5x bar
(<=27 ms). It turns a 13.4x LOSS into rough PARITY** — PCR 0.79x, tds_batched
1.17x. Reaching <=27 ms needs per-real-iter <= 2.27 ms; tds_batched is at
~3.7 ms/iter (V-cycle ~2 ms + 16 operator applies + CG context). The next
bottleneck, once the scan is gone, is the ~20 heavy mapped-FV operator applies
(0.225 ms each) per V-cycle plus the masked-scan context — NOT the tridiagonal
kernel. Closing the last ~1.5x would take fewer coarse sweeps / a cheaper
operator / a tighter budget, not a better kernel.
Note the size trend: the report's multigrid slowdown eases 13.4->7.5->5.5x at
128->192->256, and spectral's per-iter FFT grows with N while multigrid iters
stay flat (10-15), so GB-2 is materially more likely at 192^3/256^3 than at
128^3 (not measured here — a recommended follow-up).

### Immersed (the robustness case), 128^3

Iteration counts (report/campaign): **multigrid converges in 15-18 iters;
spectral needs ~80** and so does NOT converge within the production budget=30.
Per-iter wall-clock from my fixed-30 solve: spectral 0.87 ms/iter, PCR
3.17 ms/iter, tds_batched 1.94 ms/iter. Converged-solve estimate:

| preconditioner | iters to converge | converged solve ms | converged STEP ms | vs spectral |
|----------------|-------------------|--------------------|-------------------|-------------|
| spectral | ~80 (busts budget 30) | ~69 (cannot, budget-capped) | ~73 | 1.00x (or FAILS) |
| **PCR multigrid** | 15-18 | ~54 | ~58 | **1.26x faster** |
| **tds_batched multigrid** | 15-18 | ~33 | ~37 | **2.0x faster** |

**Immersed verdict: a cheap V-cycle WINS OUTRIGHT.** Spectral cannot converge
within the production budget (needs ~80 iters); multigrid converges in 15-18.
With the kernel swap the multigrid solve is also 1.3-2.0x faster in wall-clock
than even a (budget-permitting) converged spectral. This is exactly the case the
task anticipated: multigrid's iteration robustness plus a cheap V-cycle makes it
the correct immersed default regardless of the mapped-case parity.

### Levers ranked by measured impact

1. **Swap scan-Thomas -> PCR / tds_batched in the line smoother** (i.e. in
   `banded.tridiagonal_solve_along_axis`, which all callers share): mapped solve
   551 -> 43-60 ms (9-13x), immersed 1035 -> 58-95 ms (11-18x). THE lever.
2. **tds_batched over PCR**: a further ~1.4x (mapped 59.6->42.9, immersed
   95->58). GPU-only (cuSPARSE gtsv2StridedBatch).
3. **Cut per-V-cycle operator work / coarse sweeps** to chase the last ~1.5x for
   mapped GB-2 (operator applies now dominate the cheap V-cycle). Not measured.
4. **Tighten the pressure budget** (100 wastes ~90 no-op trips ~6 ms mapped).
   Minor.

### Engineering cost / risks

- **PCR** (recommended default): ~30 lines pure jax, log-depth (7 passes for
  n_z=128), reverse-mode differentiable (verified, gnorm matches scan to
  machine precision), band-value-independent cost so it handles the mapped
  per-column-varying bands at the same speed as constant coeff. Portable (any
  backend). Bit-identical solution to the shipped scan-Thomas (Phase A: 1e-18),
  so convergence is UNCHANGED (same 10-18 iters) — a pure drop-in kernel swap.
- **tds_batched** (`jax.lax.linalg.tridiagonal_solve` on transposed bands,
  cuSPARSE gtsv2StridedBatch): fastest, and `jax.grad`+`vmap` both work in
  jax 0.10.2 — **this contradicts the banded.py:28-30 docstring** ("autodiff /
  batching support is backend-uneven"), which is stale and should be re-verified
  per backend before relying on it; CUDA-only, so it needs the PCR fallback on
  CPU/TPU. The single-system form (`tds_single`) is a TRAP: 156 ms at
  batch 16384 — always use the batched (per-column) form.
- No pytree/jaxify/treedef changes: `MultigridVCycle` is a trace-time object,
  not a leaf; the swap is confined to the smoother's `sweep` (or, better, to the
  shared `banded.tridiagonal_solve_along_axis`). `custom_jvp`/`custom_vjp` are
  NOT needed for any candidate — all are natively differentiable.
- The same swap also speeds the IMEX implicit vertical-diffusion solve (the
  other `banded.tridiagonal_solve_along_axis` caller), a free side benefit.

## Bottom line

The hypothesis is CONFIRMED: the 53 ms V-cycle is ~91% the sequential
scan-Thomas lowering (latency-bound, 2.75 ms/solve, batch-independent), its
bandwidth floor is ~1 ms, and a batched/parallel kernel recovers the V-cycle
21-22x and the full solve 9-13x (mapped) / 11-18x (immersed). BUT the recovered
step-time lands at PARITY with spectral on the mapped GB-2 case (PCR 0.79x,
tds_batched 1.17x at 128^3 — NOT the 1.5x bar), because once the scan is gone the
heavy mapped-FV operator applies and masked-scan CG context dominate. The
unambiguous win is the IMMERSED case, where spectral cannot converge in budget
and a cheap-V-cycle multigrid wins 1.3-2.0x outright. Keep the line smoother
(anisotropy-robust, load-bearing for the semicoarsening); swap only the kernel.


---

## Addendum (2026-07-18) — swap SHIPPED; in-model + 512^3 measurements

The kernel swap is on dev (merge `0ece46b1`):
`banded.tridiagonal_solve_along_axis(..., method=...)` with
`{"auto", "cusparse", "pcr", "scan"}`, auto = cuSPARSE on a GPU backend /
PCR elsewhere, explicit `"cusparse"` off-GPU raises a taught ValueError;
threaded to the model API as `multigrid_tridiagonal_method`. Microbench
on the shipped code reproduces Phase A: scan 2.80 / pcr 0.37 /
cusparse 0.20 ms/solve (n_z=128, batch 128^2).

In-model GB-2-protocol measurements on the shipped code (same A100, steep
mapped a=0.8, FV auto, budget 100, tol 1e-8, ms/step, median of 6 x 20
steps, compile excluded; scan\@512^3 median of 3):

| n | spectral | mg cusparse (auto) | mg scan (old) | swap gain | vs spectral |
|-------|--------|----------|----------|-------|-------|
| 128^3 | 40.97 | **42.01** | 542.7 (07-17 record) | 12.9x | 0.975x (parity) |
| 512^3 | 2277.9 | **3401.7** | 7394.3 | 2.2x | 0.67x |

Corrections to this record's synthesis, from the measured runs:

- **The 128^3 substitution projection (1.17x, "beats spectral") was
  ~18% optimistic**: measured in-model parity, 0.975x. The projection
  substituted the standalone-solve delta; the in-model CG context does
  not cancel exactly.
- **The size trend refutes "GB-2 likelier at 192^3+"**: post-swap the
  deficit vs spectral *widens* with n (0.975x at 128^3 -> 0.67x at
  512^3). The old "slowdown eases with size" trend was a property of the
  scan kernel's batch-independent latency being amortized, not of the
  algorithm. GB-2 (>=1.5x) stays unmet at every measured size; spectral
  stays the mapped GPU production default.
  *Corrected same day
  ([`multigrid_depth_scaling.md`](multigrid_depth_scaling.md)): the
  widening was the `multigrid_levels=5` depth cap breaking
  h-independence (10 -> 27 iterations at 512^3), not the algorithm or
  the kernels; at floor-scaled depth (L=7 at 512^3) iterations are
  flat 10 and the in-model step beats spectral 1.22x-1.23x from 256^3
  up. GB-2 (>=1.5x) still unmet; the "spectral stays default at every
  size" conclusion no longer holds at scaled depth.*
- **PCR does not fit 512^3 mapped on one A100-80GB**: the XLA live set
  after rematerialization is >= 76 GiB (the 9 host-unrolled passes'
  shifted temporaries stay live inside the CG scan) -> RESOURCE_EXHAUSTED
  under the default 0.75 pool, and 76 GiB exceeds even a 0.9 pool. On
  GPU the cuSPARSE default is also the memory-viable kernel
  (peak 43.8 GiB vs spectral 28.5 at 512^3); PCR remains the portable
  CPU/TPU + multi-device-safe kernel.
  *Scoped single-GPU-only (Addendum 2, 2026-07-18): PCR **does** fit
  512^3 on 4 A100s — under 4-way sharding its live set shrinks ~4x to
  peak 12.1 GiB/dev (vs cuSPARSE 9.6, spectral 7.3), so it is a viable
  multi-device 512^3 kernel; the >= 76 GiB wall is one-GPU-only.*
- **The "free IMEX side benefit" claim above is WRONG**:
  `model/implicit.py` uses the dense `solve_along_axis`, not
  `tridiagonal_solve_along_axis`; switching IMEX to the tridiagonal
  kernel would be a separate (unclaimed) change.

Physics equivalence spectral vs mg-cusparse at equal step counts:
rel diff ~5e-11 at both 128^3 and 512^3. The scan\@512^3 run's final
state is not comparable (fewer reps = fewer total steps), and its
correctness is unit-covered instead. Immersed post-swap in-model
standing was not re-measured (the 1.3-2.0x win above remains a
substitution projection). Multi-device cuSPARSE-under-GSPMD validation
is an open residue (roadmap).
*Both now measured (Addendum 2, 2026-07-18): multi-device
cuSPARSE-under-GSPMD is validated on 4 A100s (partitions cleanly, no
all-gather; 1.11x at 512^3 / 0.37x at 128^3), and the immersed
post-swap standing is measured at ~1.1x (not 1.3-2.0x — that was a
budget=30 artifact; at production budget=100 spectral converges).*

---

## Addendum 2 (2026-07-18) — cuSPARSE under GSPMD on 4xA100; immersed in-model standing

Closes the two residues the first addendum left open: the multi-device
cuSPARSE-under-GSPMD HLO/perf leg (the `banded.py` caveat's worry) and
the immersed post-swap in-model standing (previously a substitution
projection).

Environment: node l50009, 4x A100-SXM4-80GB, jax 0.10.2, float64,
dev @ `0c950a33` (clean). All 4-device runs used
`XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion` (jax#39100);
single-GPU legs pinned `CUDA_VISIBLE_DEVICES=0`; default allocator.
GB-2 mapped case: steep terrain `H(x)=1+0.8 sin(x)` (ratio 9), linear
nonhydro2, FV auto, dsqr=0.25, FPlane f0=1, AB3, dt=0.02, budget 100,
tol 1e-8, `multigrid_levels=None` (floor depth). ms/step = median of
6x20, compile excluded. Full data + scripts:
[`artifacts/multigrid_gspmd_validation/`](artifacts/multigrid_gspmd_validation/)
(`results.md` authoritative).

### Sharding layout

The production 4-GPU model shards the **x axis** 4-way; y and z (the
tridiagonal solve axis) stay local:
`u f64[128,128,128] NamedSharding P('devices',None,None) shard=[32,128,128]`.
The z line solve therefore batches over the sharded x and the local y —
exactly the custom-call partitioning question.

### HLO verdict — cuSPARSE is PARTITIONED, no all-gather

Minimal standalone jit (`minimal_hlo.py`, n=128, solve axis z): the
cuSPARSE custom call receives the **per-shard** batch
`f64[4096,128,1]` — `4096 = (128/4)*128`, not the full `16384` — and
the module has **zero collectives** (`grep -E 'all-gather|all-reduce|
all-to-all|collective-permute|reduce-scatter'` -> none):

```
%custom-call = f64[4096,128,1]{1,2,0} custom-call(
    %loop_dynamic_update_slice_fusion, %bitcast.18.0,
    %loop_dynamic_update_slice_fusion.1, %bitcast.59.0),
    custom_call_target="cusparse_gtsv2_ffi",
    operand_layout_constraints={f64[4096,128]{1,0}, f64[4096,128]{1,0},
        f64[4096,128]{1,0}, f64[4096,128,1]{1,2,0}},
    frontend_attributes={num_batch_dims="1"}, ...
ROOT %bitcast.31.0 = f64[32,128,128]{2,1,0} bitcast(%custom-call)
```

In-model 4-GPU 128^3 step (`jit__chunk_body`): 54 `cusparse_gtsv2_ffi`
custom calls, **all per-shard** across the 6 full-3D-coarsening levels
(`inmodel_hlo_excerpt.txt`):

| level shape | cuSPARSE batch call | = (x/4)*y , z |
|---|---|---|
| 128x128x128 | `f64[4096,128,1]`  | 32*128 , 128 |
| 64x64x64    | `f64[1024,64,1]`   | 16*64 , 64 |
| 32x32x32    | `f64[256,32,1]`    | 8*32 , 32 |
| 16x16x16    | `f64[64,16,1]`     | 4*16 , 16 |
| 8x8x8       | `f64[16,8,1]`      | 2*8 , 8 |
| 4x4x4       | `f64[4,4,1]` (x24) | 1*4 , 4 |

The module's collective all-gathers (an `f64[4]` projection global-mean
reduction and an `s32[8]` index-gather for a `jnp.take` in the
projection while-body) feed **no** cuSPARSE operand — every cuSPARSE
operand is a `loop_dynamic_update_slice_fusion` / `bitcast`.
*Correction (Addendum 3, same day): there is only **one** collective
all-gather — the `f64[4]` global mean. The `s32[8]` "index-gather" is a
**local** `gather` op, not a collective; the census parser confirms a
single all-gather in the whole step.* The
remaining collectives (all-reduce, collective-permute) are the CG
measure-weighted inner products and the halo exchanges, inherent to the
sharded elliptic solve, not the tridiagonal kernel. pcr partitions
cleanly by construction (756-line unroll, zero collectives).

**Verdict: XLA partitions the batched cuSPARSE custom call cleanly along
the sharded batch axis at every multigrid level; the `banded.py` caveat's
worry does not materialize on jax 0.10.2 / this XLA.** This is observed
lowering behaviour, not an API contract — pcr stays the portable kernel.

### Parity and iterations (4 GPU vs 1 GPU)

Max relative difference over {u,v,w,b} after 20 steps:

| pair | 128^3 | 512^3 |
|---|---|---|
| 4GPU cuSPARSE vs 4GPU pcr      | 1.93e-14 | 9.02e-14 |
| 4GPU cuSPARSE vs 1GPU cuSPARSE | 2.17e-14 | 9.38e-14 |
| 4GPU cuSPARSE vs 4GPU spectral | 2.36e-10 | 4.67e-10 |

cuSPARSE/pcr agree ~1e-14 (kernel-identical); device-count invariant
~1e-13; physics vs spectral ~1e-10 (CG dot reorder across shards). CG
iterations flat **10** at both sizes, kernel- and device-count-
independent.

### Timing, 4 GPU (ms/step median of 6x20; compile s; peak GiB/dev)

| n | preconditioner | ms/step | vs spectral | compile s | peak GiB/dev |
|---|---|---|---|---|---|
| 128^3 | spectral    | 32.31 | 1.00x | 5.6  | 0.14 |
| 128^3 | mg-cuSPARSE | 86.39 | **0.37x** | 50.8 | 0.17 |
| 128^3 | mg-pcr      | 87.17 | 0.37x | 57.2 | 0.20 |
| 512^3 | spectral    | 600.5 | 1.00x | 27   | 7.3  |
| 512^3 | mg-cuSPARSE | 539.8 | **1.11x** | 251  | 9.6  |
| 512^3 | mg-pcr      | 774.5 | 0.78x | 280  | 12.1 |

- **128^3: spectral wins decisively on 4 GPUs (mg 0.37x).** The V-cycle's
  per-level halo/collective latency dominates and does not amortize over
  only 32-cell x-shards — mg-cuSPARSE at 4 GPUs (86.4 ms) is even slower
  than its own 1-GPU 42.0 ms; interconnect latency under GSPMD swamps the
  small problem.
- **512^3: mg-cuSPARSE BEATS spectral 1.11x on 4 GPUs** (single-GPU was
  1.22x; the collective overhead narrows but does not erase the win).
  mg-pcr is ~1.43x slower than mg-cuSPARSE (cuSPARSE > pcr, as the kernel
  shootout found).
- **pcr FITS at 512^3 on 4 GPUs** (12.1 GiB/dev): the 1-GPU >= 76 GiB live
  set shrinks ~4x under sharding, so pcr is a viable multi-device 512^3
  kernel (it OOMs only on one GPU — the first addendum's "PCR does not fit
  512^3" is single-GPU-only, scoped there).

GB-2 (>= 1.5x) is unmet at every measured size/device count: spectral
stays an excellent 4-GPU default at <= 128^3; mg-cuSPARSE is the faster
mapped option at 512^3, but by 1.11x, not the 1.5x bar.

### Immersed post-swap in-model standing (1 GPU)

Canonical tilted-slope geometry (mirrors
`tests/nonhydro2/test_immersed_pressure.py` `slope`, order-4 quadrature,
genuine partials, wet frac 0.706). GB-2 protocol, budget 100, tol 1e-8.

| n | preconditioner | ms/step | iters | rel residual | speedup |
|---|---|---|---|---|---|
| 128^3 | spectral    | 66.47 | 73 | 9.13e-09 (converged) | 1.00x |
| 128^3 | mg-cuSPARSE | 60.81 | 20 | 6.28e-09 | **1.09x** |
| 256^3 | spectral    | 482.15 | 71 | 9.85e-09 (converged) | 1.00x |
| 256^3 | mg-cuSPARSE | 430.15 | 21 | 6.88e-09 | **1.12x** |

**Correction to the Synthesis' "immersed wins outright 1.3-2.0x".** That
projection was relative to the study's **budget=30**, at which spectral
(needing 71-73 iters) cannot converge and mg wins categorically. At the
production **budget=100** spectral **does** converge (71-73 iters, relres
~9e-9 < 1e-8), so the win shrinks to measured ~1.1x: mg converges in
~3.5x fewer CG iters (20-21 vs 71-73) but each immersed mg V-cycle
(semicoarsening + line smoother + per-level wet-mean) is ~3x costlier
than a spectral CG iteration, netting ~1.1x (flat/slightly rising with n).
Budget-sensitive verdict: below budget ~70, spectral fails and mg is the
**only** converged option (categorical win); at budget=100 both converge
and mg is ~1.1x faster. In-model trajectories match ~1e-10.

Caveat: the HLO/parity/iteration runs (correctness, not timing) were
taken under light contention from two other pytest sessions on the node;
every ms/step timing was taken on `nvidia-smi`-verified-idle GPUs.

---

## Addendum 3 (2026-07-18) — collective census of the 4-GPU step; why small-n multi-GPU mg is slow

Addendum 2 measured mg-cuSPARSE at **0.37x** spectral on 4 A100s at
128^3 (86.4 vs 32.3 ms/step) but only *asserted* per-level collective
latency as the cause. This census counts it. A parser
(`census.py`) walks each optimized-HLO step module, builds its
call graph, and attributes every collective op **definition** (not
operand reference) to the computation it lives in and to a structural
region (outside the CG while / inside the conditional's real branch /
inside the skip branch). Full report + tables + the trimmed module
excerpts:
[`artifacts/multigrid_gspmd_validation/census_collective_report.md`](artifacts/multigrid_gspmd_validation/census_collective_report.md).

Environment: node l50009, 4x A100-SXM4-80GB, jax 0.10.2, float64, dev
checkout (no repo edits), `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`
(jax#39100). GB-2 steep-mapped nonhydro2 (`H(x)=1+0.8 sin(x)`), n=128,
FV auto, dsqr=0.25, budget 100, tol 1e-8, `multigrid_levels=None` (floor
depth = 6 levels, full-3D coarsening 128->4). x sharded 4-way; y, z
local. Step HLO dumped with `model.advance(1)` so the outer time-scan
collapses and the module is exactly one step.

### Module structure

Both mg and spectral modules are **exactly one `while`** (the CG masked
scan, `known_trip_count=99` = budget 100 minus the peeled first
iteration) and **one `conditional`** (the tolerance real/skip branch).
The 6-level V-cycle is **fully unrolled** straight-line code inside the
conditional's real branch — it is not a loop. The skip branch (taken
once convergence is reached) is a **bare tuple pass-through: 0
collectives, 0 FLOPs** (verified directly for both modules). Achieved
production CG iterations (random mean-free RHS, tol 1e-8): mg **10**,
spectral **36**.

Regions: *outside-while* = executed once/step (setup, tendency, the
peeled first CG iteration, final projection); *real/trip* = the
conditional's collective-bearing branch = one CG iteration = one V-cycle
+ one operator apply + the dot products.

### Collectives per V-cycle (= one real mg CG trip), by kind and level

262 collective-permute + 26 all-reduce = **288 collectives per real
trip**. No all-gather, no all-to-all in-loop. CPs are attributed to a
level by the Y-extent of the moved x-plane payload `f64[hx,Y,Z]`
(storage Y = true + 2 ghosts: 130/66/34/18/10/6 for levels
128/64/32/16/8/4). `L*` rows are the true-size inter-level
restriction/prolongation transfer exchanges. Left and right x-halos are
**separate** collective-permute ops, never fused.

| level | CP per V-cycle | bytes each |
|---|---|---|
| L128  | 44 | 134160 |
| L64   | 33 |  34320 |
| L64*  |  5 |  32768 |
| L32   | 42 |   8976 |
| L32*  |  5 |   8192 |
| L16   | 42 |   2448 |
| L16*  |  5 |   2048 |
| L8    | 33 |    720 |
| L8*   |  7 |    512 |
| L4    | 24 |    576 |
| L4*   | 22 |    128 |
| **TOTAL CP** | **262** | |

The 26 all-reduces per trip are the CG dot products (`<p,Ap>`, `<r,z>`)
and the per-level wet/mean projection reductions, all global 4-shard
reductions carrying ~9.7 MB/step total — not the bottleneck.

### Collectives executed per STEP — mg vs spectral

mg-cuSPARSE (10 iters = 1 peel + 9 real trips + 90 zero-cost skips):

| kind | outside/step | per real trip | executed/step | bytes/step |
|---|---|---|---|---|
| collective-permute | 572 | 262 | **2930** | 89.8 MB |
| all-reduce         |  54 |  26 |  **288** |  9.7 MB |
| all-gather         |   1 |   0 |    **1** |  32 B |
| all-to-all         |   0 |   0 |    0 | 0 |
| **TOTAL**          | 627 | 288 | **3219** | **99.4 MB** |

spectral (36 iters = 1 peel + 35 real trips + 64 skips; per real trip
11 CP + 3 all-reduce + 2 all-to-all = 16):

| kind | outside/step | per real trip | executed/step | bytes/step |
|---|---|---|---|---|
| collective-permute | 53 | 11 | **438** |  58.6 MB |
| all-reduce         |  8 |  3 | **113** |  19.8 MB |
| all-to-all         |  4 |  2 |  **74** | 315.2 MB |
| all-gather         |  0 |  0 |   0 | 0 |
| **TOTAL**          | 65 | 16 | **625** | **393.6 MB** |

**mg issues 5.1x MORE collectives than spectral (3219 vs 625) while
moving 4x FEWER bytes (99 vs 394 MB).** Per CG iteration mg is 288 vs
spectral's 16 — an **18x** higher per-iteration collective count that
mg's 3.6x iteration advantage (10 vs 36) comes nowhere near offsetting.
mg = hundreds-to-thousands of tiny latency-bound collectives; spectral =
tens of large bandwidth-bound ones (its 74 all-to-all FFT transposes,
~4.26 MB each, carry 80% of its bytes). **Exactly one** all-gather
exists in the whole mg step: the `f64[4]` (32 B) global-mean projection
gather; it touches no cuSPARSE operand and no mg transfer. (Corrects
Addendum 2 / `results.md`: the "second all-gather, s32[8] index gather"
is a **local** `gather` op, not a collective.) The 1-GPU mg module has
**zero** collectives — every collective above is sharding overhead, none
intrinsic to the kernels.

### The per-collective cost: ~10 microseconds, latency-floored

The 1-GPU mg module has zero collectives and runs 42.0 ms/step
(Addendum 2). Census-day 4-GPU mg is 75.7 ms (probe below; a lighter
node than Addendum 2's 86.4). The **~34 ms 4-GPU-minus-1-GPU gap** is
therefore pure collective overhead, spread across the **3219**
executed collectives -> **~10 us per collective**. That is the
NVLink/NCCL small-message latency floor (a bidirectional NVLink hop is
sub-microsecond, but the fixed per-collective launch + rendezvous
dominates once each message is under a kilobyte), and it is
**count-dominated, not launch-bloat**: the payloads are tiny (99.4 MB /
3219 = ~31 KB average, and the coarse-level CPs move <1 KB). The mg
step pays for issuing thousands of collectives, not for the bytes they
carry.

### No-op trips exonerated (the budget=100 masked scan)

The tolerance masked scan computes `converged = rr_c <= threshold` from
the residual **already carried** in the scan state — the predicate needs
no collective, and the `<r,r>` reduction lives inside the real branch.
`lax.cond` lowers to a real stablehlo `conditional` (not a compute-both
`select`), and its skip branch is a bare tuple pass-through, so the 90
post-convergence trips each execute **0 collectives, 0 FLOPs**. A timing
probe (GPUs verified idle) prices what the budget=100 setting *does*
cost — pure `while`-loop plumbing:

| budget | scan trips (skip) | median ms/step |
|---|---|---|
| 100 | 99 (90 skip) | 75.72 |
|  15 | 14 (5 skip)  | 72.75 |

Delta **2.97 ms** for 85 extra no-op trips ~= **35 us/trip** (scalar
predicate + `get-tuple-element`/`copy` of the multi-level carry tuple),
**not collectives**. Dropping 85 no-op trips recovers only ~3 of the
~34 ms overhead — the budget is not the source of the slowdown.

### Conviction: the two coarsest levels fire ~33% of the halo permutes

At floor depth the V-cycle coarsens x to 8 (L8) and 4 (L4) cells total —
**2 and 1 planes per shard** across 4 devices. Each such level still
pays a full ring halo exchange per smoother sweep: per V-cycle
L8 = 40 CP (33 + 7 `L8*`), L4 = 46 CP (24 + 22 `L4*`) = **~86 of the 262
per-trip CPs (33%)**, each moving **<1 KB**. Over 10 V-cycles/step that
is **~860 sub-kilobyte latency-only collective-permutes**. This is
structural, not a bug: the coarse grids have almost nothing left to
shard 4 ways, but the halo machinery fires regardless.

### Conclusion

The 128^3 4-GPU mg deficit is **count x latency of a deeply-coarsened
sharded V-cycle**: 3219 collectives/step, ~10 us each latency-floored,
~34 ms of overhead that the 1-GPU kernel (zero collectives) never pays.
It is not the cuSPARSE kernel (Addendum 2: cleanly partitioned, no
all-gather), not the no-op trips (35 us/trip plumbing, ~3 ms total), and
not bandwidth (mg moves 4x fewer bytes than the winning spectral). The
single largest recoverable slice is the ~33% of halo permutes fired by
the two coarsest levels on 1-2 per-shard planes — **~9-14 ms of the
~34 ms recoverable at 128^3** by not sharding levels below a per-shard
extent threshold (agglomerate them to replicated layout, where the halo
runs local and issues no collective). This motivates the coarse-level
agglomeration plan:
[`../plans/active/multigrid_agglomeration_plan.md`](../plans/active/multigrid_agglomeration_plan.md).
*Correction (Phase 3 4-GPU wall-clock, 2026-07-19, job `26355284`;
[`artifacts/multigrid_agglomeration_phase3/`](artifacts/multigrid_agglomeration_phase3/)):
the projected ~9–14 ms recovery did **not** reproduce. Agglomeration at
the best `tau` (t8) recovers only 5.8 ms of the 128³ mg-off cost, and
`tau=4` recovers 0.6 ms — the GPU census shows `tau=4` removes only 21
of the 86 flagged coarse permutes (it leaves the tiniest L4 halos in
place) and the GPU re-partitions the replicated-level reductions into
all-reduces rather than folding them, so the collective count moves only
−6% and the wall clock barely at all. No `tau` is a net win and the
immersed case regresses; the knob stays default OFF.*
This addendum also hypothesized a **capability** driver — a P-device
sharded axis cannot coarsen below P cells, so large device counts cap
V-cycle depth and break h-independence. *That did NOT reproduce: the
prototype (plan §4, shipped `9e08493f`) found floor depth is already
reached at forced 16/32 devices, because layout negotiation (MG-D5
`allow_replicated`) already replicates below the shardability floor —
no crash, no empty shards, no depth cap. The earlier 10 -> 27
h-independence break
([`multigrid_depth_scaling.md`](multigrid_depth_scaling.md)) was the
fixed `multigrid_levels=5` cap, not a device-count effect. The
reproduced driver is **latency only**: in semicoarsen/immersed
hierarchies the sharded axis flips x -> z as the horizontals coarsen,
so the coarse levels stay z-sharded at 2 planes/shard — the sub-KB
regime above, reproduced at P = 16. Agglomeration's value is making that
replication deliberate and threshold-driven, not curing a crash.*
