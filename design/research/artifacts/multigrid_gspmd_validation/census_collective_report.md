# Collective census — GB-2 steep-mapped nonhydro2, n=128, 4x A100 (GSPMD)

Node l50009, jax 0.10.2, float64, dev @ dev checkout (no repo edits).
All 4-GPU runs: `XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`
(jax#39100). GB-2 mapped case: steep terrain `H(x)=1+0.8 sin(x)`, linear
nonhydro2, FV auto, dsqr=0.25, budget 100, tol 1e-8, floor-depth
multigrid (full-3D coarsening 128→4, 6 levels). x-axis sharded 4-way;
y,z local. Optimized step HLO = `jit__chunk_body` module, dumped with
`model.advance(1)` so the outer time-scan collapses (`n=1`) and the
module is exactly **one time step**.

Source modules copied into this dir:
`step_mg4_n128.txt` (101849 lines), `step_sp4_n128.txt` (13516),
`step_mg1_n128.txt` (71830, 1-GPU sanity). Parser: `census.py`
(counts op DEFINITIONS, not operand references; attributes each
collective to the computation it lives in and to a structural region
via the call graph).

## Module structure (both mg and spectral)

Exactly **one `while`** op (the CG masked scan, `known_trip_count=99` =
budget 100 minus the peeled first iteration) and **one `conditional`**
(the tolerance real/skip branch) per step. The 6-level multigrid
V-cycle is **fully unrolled** straight-line code inside the CG real
branch — it is *not* a loop.

- CG scan: peel = iteration 1 (unrolled, outside the while); scan body
  runs 99 trips. Each trip's `conditional(converged, skip, real_step)`
  runs the real branch until convergence, then the skip branch.
- **The skip branch is a pure tuple pass-through: 0 collectives, 0
  compute** (verified directly — it only forwards/copies the carry
  tuple). Confirmed for both mg (`region_283`) and spectral
  (`region_46`).
- Achieved CG iterations (measured, production solver, random mean-free
  RHS, tol 1e-8, 4-GPU): **multigrid = 10, spectral = 36** (peel + 9
  real trips + 90 skip; peel + 35 real trips + 64 skip).

Regions used below: **outside-while** = executed once/step (setup,
tendency/divergence, the peeled 1st CG iteration, final projection +
velocity correction); **real/trip** = the conditional's
collective-bearing branch = one CG iteration = one V-cycle + one
operator apply + the dot products.

---

## (i) mg-cusparse: collectives per V-cycle (= one real CG trip) by kind & level

One real CG trip (`region_284`): **262 collective-permute + 26
all-reduce = 288 collectives**. No all-gather, no all-to-all in-loop.

Collective-permute halo exchanges attributed to multigrid level by the
Y-extent of the moved x-plane payload `f64[hx, Y, Z]` (storage Y = true
+ 2 ghosts: 130/66/34/18/10/6 for levels 128/64/32/16/8/4). `L*` rows
are the exact-power-of-two payloads (`f64[1,64,64]` etc.) = the
restriction/prolongation inter-level transfer exchanges (true-size, no
ghost pad).

| level | CP per V-cycle (real trip) | CP outside-while | bytes each |
|---|---|---|---|
| L128  | 44 | 116 | 134160 |
| L64   | 33 |  66 |  34320 |
| L64*  |  5 |  10 |  32768 |
| L32   | 42 |  84 |   8976 |
| L32*  |  5 |  10 |   8192 |
| L16   | 42 |  84 |   2448 |
| L16*  |  5 |  10 |   2048 |
| L8    | 33 |  66 |    720 |
| L8*   |  7 |  14 |    512 |
| L4    | 24 |  48 |    576 |
| L4*   | 22 |  44 |    128 |
| 1-D scalars | 0 | 20 | 8-16 |
| **TOTAL CP** | **262** | **572** | |

All-reduce per real trip: 26 (dot products `<p,Ap>`, `<r,z>` and the
per-level wet/mean projection reductions), all global 4-shard
reductions (`replica_groups=mesh['axis_0'=4]` on 68/80, explicit
`{{0,1,2,3}}` on 12/80).

Halo direction: left and right x-halos are **separate**
collective-permute ops, never fused. Static ring counts:
`{{0,1},{1,2},{2,3},{3,0}}` (forward periodic) x238 and
`{{0,3},{1,0},{2,1},{3,2}}` (reverse periodic) x238; plus 3-pair
non-wraparound variants `{{0,1},{1,2},{2,3}}` x114 /
`{{1,0},{2,1},{3,2}}` x194 (the restriction/prolongation transfers).

**No all-gather feeds any level.** The whole step has exactly **one**
all-gather: `(f64[1],f64[4]) all-gather-start` over `axis_0=4` (32 B
gathered) in ENTRY — the mean-free projection's global-mean gather. It
does not touch cuSPARSE or any multigrid transfer. (The morning's
`s32[8]` "index gather" is a local `gather`, not a collective — it
does not appear as an all-gather here.)

---

## (ii) collectives executed per STEP — mg vs spectral

mg-cusparse (10 iters = 1 peel + 9 real trips + 90 zero-cost skips):

| kind | outside/step | per real trip | executed/step | payload bytes/step |
|---|---|---|---|---|
| collective-permute | 572 | 262 | **2930** | 89.8 MB |
| all-reduce         |  54 |  26 |  **288** |  9.7 MB |
| all-gather         |   1 |   0 |    **1** |  32 B |
| all-to-all         |   0 |   0 |    0 | 0 |
| **TOTAL**          | 627 | 288 | **3219** | **99.4 MB** |

spectral (36 iters = 1 peel + 35 real trips + 64 zero-cost skips). Per
real trip: **11 CP + 3 all-reduce + 2 all-to-all = 16 collectives**.
The 2 all-to-alls are the distributed-FFT transposes
`c128[128,65,32]` / `c128[32,65,128]` = 4.26 MB each.

| kind | outside/step | per real trip | executed/step | payload bytes/step |
|---|---|---|---|---|
| collective-permute | 53 | 11 | **438** |  58.6 MB |
| all-reduce         |  8 |  3 | **113** |  19.8 MB |
| all-to-all         |  4 |  2 |  **74** | 315.2 MB |
| all-gather         |  0 |  0 |   0 | 0 |
| **TOTAL**          | 65 | 16 | **625** | **393.6 MB** |

**mg issues 5.1x MORE collectives than spectral (3219 vs 625) while
moving 4x FEWER bytes (99 vs 394 MB).** Per CG iteration mg issues 288
collectives vs spectral's 16 — an **18x** higher per-iteration
collective count. mg's 3.6x iteration advantage (10 vs 36) comes
nowhere near offsetting it. The hypothesis holds: **mg = hundreds→
thousands of tiny latency-bound collectives; spectral = tens of large
bandwidth-bound ones** (74 A2A carry 80% of spectral's bytes).

1-GPU mg-cusparse sanity: **0 collectives** in the whole step module
(baseline confirmed — every collective above is sharding overhead, none
intrinsic to the kernels).

---

## (iii) The no-op-trip question (Task 3)

**Converged no-op trips launch ZERO collectives.** The tolerance masked
scan (`krylov.py` `tol_body`) computes `converged = rr_c <= threshold`
from the residual `rr_c` **already carried** in the scan state; the
predicate needs no collective. The `<r,r>` reduction that would need an
all-reduce lives inside `real_step` (`self._dot(n_r, n_r)`), i.e. inside
the conditional's real branch. `lax.cond` lowers to a real stablehlo
`conditional` (not a compute-both `select`) — verified: the module has a
`conditional` op with two branch computations, and the skip branch is a
bare tuple pass-through. So the 90 mg (64 spectral) post-convergence
trips each execute the identity branch: **0 collectives, 0 FLOPs**. The
budget=100 setting adds no per-trip latency collectives; the only
residual cost is the 90 scalar predicate evaluations + tuple plumbing of
the while loop itself (priced in Task 5).

---

## (iv) Timing probe (Task 5) — pricing the ~85 no-op trips

GPUs verified idle (0%/0 MiB) before timing. mg-cusparse 128^3 4-GPU,
GB-2 protocol (median of 6x20, compile excluded, block_until_ready):

| budget | scan trips (skip) | median ms/step | [min, max] |
|---|---|---|---|
| 100 | 99 (90 skip) | 75.72 | [67.86, 84.40] |
|  15 | 14 (5 skip)  | 72.75 | [64.92, 81.62] |

**Delta = 2.97 ms/step (3.9%)** for the 85 extra no-op trips ≈ **35 µs
per no-op trip**. This is pure `while`-loop plumbing (scalar predicate +
`get-tuple-element`/`copy` of the multi-level carry tuple), **not
collectives** — exactly as the HLO predicted (the skip branch is
collective-free). The no-op trips are NOT the source of the mg slowdown:
dropping 85 of them recovers only ~3 of the ~44 ms mg-vs-spectral gap.
The gap is the **3219 real collectives of the 10 executed V-cycles**
(2930 tiny latency-bound halo CPs), not the masked-scan budget.

(Absolute medians here are ~10 ms below the morning's 86.4 ms — a
lighter node — but the budget delta is the load-bearing measurement.)

---

## (v) Anything genuinely redundant

- **Coarse-level halo exchanges where the per-shard x-extent is 1-2
  planes.** At floor depth the V-cycle coarsens x to 4 (L4) and 8 (L8)
  cells, i.e. **1 and 2 planes per shard** across 4 devices. Each such
  level still pays a full ring halo exchange per smoother sweep: per
  V-cycle L4 = 46 CP (24 + 22 `L4*`), L8 = 40 CP (33 + 7 `L8*`) — ~86
  of the 262 per-trip CPs (33%) move <1 KB at the two coarsest levels.
  Over 10 V-cycles/step that is ~860 sub-kilobyte latency-only
  collective-permutes. This is the structural driver of the 4-GPU mg
  slowdown at n=128: the coarse grids have almost nothing to shard, but
  the halo machinery fires regardless. Not a bug — it is the cost of
  sharding a deeply-coarsened V-cycle 4 ways at a small base size.
- **All halos are directional pairs (left CP + right CP), never
  fused** — inherent to `collective-permute`, not redundant, but it
  doubles the op count vs a single bidirectional exchange.
- **No duplicate CPs for the same halo, no unexplained all-gathers.**
  The single all-gather is the global-mean projection and is accounted
  for. Restriction/prolongation add their own transfer CPs (the 3-pair
  non-wraparound variants) but no extra all-gather feeds them.
- The 288 all-reduces/step (mg) are the CG + per-level projection dot
  products; they carry only ~9.7 MB and are not the bottleneck.
