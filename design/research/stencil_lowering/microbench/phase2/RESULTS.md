# Phase 2 — WENO divide-reduction spellings (A100-SXM4-80GB, 1 GPU)

Reuses the parent `microbench/` harness (`common.py`, `e3.py`);
identical methodology (block_until_ready, 5 warmups, median of 40,
fori_loop T=50 loop probe, memory/cost/HLO capture). Scripts:
`weno_variants.py`, `p2a_micro.py`, `p2b_model.py`. Raw JSON in
`results/`, per-variant optimized HLO in `results/hlo/`.

## P0 — raw op-cost reference (256^3, f64/f32)

| probe | f64 | f32 |
|---|---|---|
| single elementwise `x/y` (bandwidth-bound) | 0.321 ms, 1263 GB/s | 0.200 ms |
| single elementwise `x*y` | 0.322 ms, 1258 GB/s | — |
| compute-bound chain, divide (ns / op-layer, K=32) | 43600 | 16690 |
| compute-bound chain, multiply | 15283 | 9036 |
| **divide / multiply** | **2.85x** | **1.85x** |

- A lone divide is **bandwidth-bound** (`div == mul`): the divide cost
  only surfaces at high arithmetic intensity (the WENO case).
- Compute-bound headroom: **f64 divide = 2.85x an f64 multiply**;
  f64 divide (43.6 us/op-layer, ~383 Gdiv/s) = **2.6x an f32 divide**
  (16.7 us, ~1006 Gdiv/s).
- Caveat: the f32 IEEE (Newton-Raphson) divide **spills registers at
  K >= 64** (~4x inflation) — a real hazard for divide-heavy f32 kernels.

## P2a — micro E3 weno5 composed tendency (256^3, f64 flux)

| variant | ms single | loop per-iter ms | f64 div | f32 div | GB/s | flops G | max_abs vs base |
|---|---|---|---|---|---|---|---|
| baseline (4 div/recon) | 3.255 | 4.122 | 24 | 0 | 222 | 6.57 | — |
| singlediv (1 div/recon) | 2.027 (-37.7%) | 2.280 (-44.7%) | 6 | 0 | 357 | 6.87 | 5.3e-15 |
| f32w (std spelling, f32) | 1.685 (-48.2%) | 1.725 (-58.2%) | 0 | 24 | 429 | 7.30 | 1.8e-6 |
| combined (single-div, f32) | 1.803 (-44.6%) | 1.353 (-67.2%) | 0 | 6 | 401 | 7.60 | 1.8e-6 |
| full_f32 (ceiling) | 1.189 (-63.5%) | — | 0 | 24 | — | — | — |

- temp bytes identical (404 MB) for every f64 variant: the isolated
  tendency fuses fully regardless of spelling, so the micro **is**
  divide-bound and single-divide wins.
- `singlediv` is exact algebra modulo FP (5.3e-15). f32-weight variants
  drift ~1.8e-6 abs (f32 weight precision on rough random data).

## P2b — real nonhydro2 step (periodic, 10000x10000x100, f0=1e-4,
##       N^2=(50 f0)^2, dt=20, AB3, chunk_size=50, WENOAdvection(5))

Harness validated: `centered` = 7.486 ms/step **matches the repo
baseline** `nh_flat_advective` 256^3 (7.493 ms/step). Linear spectral
step (no advection) = 5.16 ms/step (repo `nh_flat_periodic`).

### 256^3, median ms/step (min ~= median, stable)

| variant | ms/step | vs baseline WENO | step f64 div | step f32 div | temp MB |
|---|---|---|---|---|---|
| centered (ref) | 7.486 | — | 13 | 0 | 2103 |
| **baseline WENO** | **25.640** | — | 493 | 0 | 2932 |
| **singlediv** | **36.634** | **+42.9% (SLOWER)** | 133 | 0 | 6553 |
| **f32w** | **19.742** | **-23.0%** | 13 | 480 | 3947 |

### 512^3, median ms/step

| variant | ms/step | vs baseline | temp MB |
|---|---|---|---|
| baseline WENO | 239.62 | — | 22450 |
| **f32w** | **181.76** | **-24.1%** | 30266 |
| singlediv | **OOM** (needs +46.9 GiB) | — | ~50000 |

### Correctness (max_abs state diff after 20 steps, identical IC)

| n | variant | u | v | w | b |
|---|---|---|---|---|---|
| 256 | singlediv | 4.0e-13 | 6.9e-13 | 4.2e-15 | 8.4e-16 |
| 256 | f32w | 5.9e-9 | 1.3e-8 | 9.6e-10 | 1.2e-11 |
| 32 | singlediv | 1.3e-13 | 1.7e-13 | 2.2e-15 | 4.1e-16 |
| 32 | f32w | 1.2e-8 | 3.4e-8 | 9.2e-10 | 3.4e-11 |

All runs finite. Patch liveness confirmed by trace-call counter (48
`weno_reconstruct` calls/step) and the step divide-count delta.

## Verdict

The micro and the real model **disagree on single-divide**, and the
real model is decisive:

- **single-divide (f64):** wins -38% in the isolated micro tendency
  (divide-bound, fully fused), but **LOSES in the real step**: +43% at
  256^3 (25.6 -> 36.6 ms/step) and **OOM at 512^3**. Fewer divides
  (493 -> 133) yet slower, because XLA materializes the extra product
  intermediates (s_m, n_m) when the advection is fused into the full
  step: temp memory **doubles** (2.9 -> 6.6 GB at 256; ~50 GB / OOM at
  512). The full WENO step is **memory/fusion-bound**, not
  divide-bound, so the divide-count lever backfires. Numerically exact
  (1e-13), but not shippable as spelled.

- **f32-weights (standard per-candidate spelling):** the robust win —
  **-23% at 256^3, -24% at 512^3** — because it halves the
  weight-subgraph **bytes** (the real bottleneck) and cheapens the
  divides, at ~1e-8 state drift after 20 steps (owner judges; well
  below discretization error). Must use the **standard** spelling: the
  single-divide product form (n_m ~ beta^4) **overflows f32** for the
  real eps=1e-10 on smooth data (s ~ 1e-20, prod ~ 1e-40, reciprocal
  overflows -> NaN at step 1). This landmine was invisible in the micro
  (eps=1e-6 + rough data).

**Bottom line:** ship f32-weights (standard spelling), not
single-divide. Roughly -6 ms/step at 256^3 and -58 ms/step at 512^3 on
the WENO step, mapping directly onto the Oceananigans-comparison gap.
