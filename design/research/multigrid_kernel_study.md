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
