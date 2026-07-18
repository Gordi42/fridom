# Multigrid cuSPARSE-under-GSPMD (4xA100) + immersed post-swap standing

Node l50009, 4x A100-SXM4-80GB, jax 0.10.2, float64, dev @ 0c950a33 (clean,
no repo edits). All 4-device runs used
`XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion` (jax#39100).
Single-GPU legs pinned `CUDA_VISIBLE_DEVICES=0`. Default allocator (no
`XLA_PYTHON_CLIENT_*`). GB-2 mapped case = steep terrain
`H(x)=1+0.8 sin(x)` (ratio 9), linear nonhydro2, FV auto, dsqr=0.25,
FPlaneCoriolis f0=1, AB3, dt=0.02, budget 100, tol 1e-8,
`multigrid_levels=None` (floor depth). ms/step = median of 6x20 steps,
compile excluded, block_until_ready.

Scripts: `gb2_common.py` (builders/timing), `minimal_hlo.py`,
`inmodel_hlo.py`, `parity_iters.py`, `timing.py`, `immersed_leg2.py`,
`inspect_shard.py`, `sanity.py`. Raw: `results_timing.jsonl`,
`results_immersed.jsonl`, `log_*.txt`, `hlo_dump/`,
`minimal_hlo_{cusparse,pcr}_n128.txt`.

---

## Sharding of the production model (4 GPU, n=128, mg-auto)

Prognostic fields shard the **x axis** 4-way; y and z (the tridiagonal
solve axis) stay local:

```
u: shape=(128,128,128) NamedSharding P('devices',None,None) shard=(32,128,128)
w: shape=(128,128,127) P('devices',None,None) shard=(32,128,127)
is_local(x)=False  is_local(y)=True  is_local(z)=True
```

The z line solve therefore batches over the sharded x and the local y —
exactly the custom-call partitioning question.

---

## LEG 1(a) — HLO verdict: cuSPARSE is PARTITIONED (no all-gather)

### MINIMAL standalone (`minimal_hlo.py`, n=128, P('devices',None,None), solve axis z)

cuSPARSE optimized HLO — the ONE decisive excerpt (per-shard batch
`4096 = (128/4) x 128`, NOT the full `16384`; output bitcast to the
per-shard `[32,128,128]`; **zero collectives in the module**):

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
`grep -E 'all-gather|all-reduce|all-to-all|collective-permute|reduce-scatter'`
-> NONE. Correctness vs replicated pcr: max|.| = 5.33e-15.

pcr control: 756-line log-depth unroll, zero collectives, err 0.0 —
partitions cleanly along the batch as expected.

### IN-MODEL 4-GPU 128^3 step (`inmodel_hlo.py`, module `jit__chunk_body`)

54 `cusparse_gtsv2_ffi` custom-calls, **all per-shard** — the 6 mg levels
(full-3D coarsening default 128^3 -> 4^3):

| level shape | cusparse batch call | = (x/4)*y , z |
|---|---|---|
| 128x128x128 | `f64[4096,128,1]` | 32*128 , 128 |
| 64x64x64    | `f64[1024,64,1]`  | 16*64 , 64 |
| 32x32x32    | `f64[256,32,1]`   | 8*32 , 32 |
| 16x16x16    | `f64[64,16,1]`    | 4*16 , 16 |
| 8x8x8       | `f64[16,8,1]`     | 2*8 , 8 |
| 4x4x4       | `f64[4,4,1]` (x24) | 1*4 , 4 |

Full batch would be 16384/4096/... ; every call is n/4 on x. The whole
`chunk_body` module has exactly **one** all-gather (`f64[4]{0}` — the
global mean), and it does **not** feed any cusparse operand (operand
set ∩ all-gather-result set = ∅; every cusparse operand is a
`loop_dynamic_update_slice_fusion`/`bitcast`). The module's other
collectives (all-reduce, collective-permute) are the CG measure-weighted
inner products and the halo/transfer exchanges — inherent to the sharded
elliptic solve, not the tridiagonal kernel.

**Verdict: XLA partitions the cuSPARSE batched custom-call cleanly along
the sharded batch axis at every multigrid level; it does NOT all-gather.
The `banded.py` multi-device caveat's worry does not materialize on jax
0.10.2 / this XLA.** (pcr also partitions cleanly and is the memory-viable
512^3 alternative, below.)

---

## LEG 1(b) — Parity & iterations (4 GPU vs 1 GPU)

Max relative difference over {u,v,w,b} after 20 steps from the record IC:

| pair | 128^3 | 512^3 |
|---|---|---|
| 4GPU cusparse vs 4GPU pcr      | 1.93e-14 | 9.02e-14 |
| 4GPU cusparse vs 1GPU cusparse | 2.17e-14 | 9.38e-14 |
| 4GPU cusparse vs 4GPU spectral | 2.36e-10 | 4.67e-10 |

cusparse/pcr agree to ~1e-14 (kernel-identical); device-count invariant
to ~1e-13; physics equivalence vs spectral ~1e-10 (the CG dot-product
reduction reorders across shards) — all << 1e-6.

Achieved CG iterations (production `ConjugateGradient.solve`, random
mean-free RHS, tol 1e-8):

| n | cusparse 4GPU | cusparse 1GPU | pcr 4GPU | pcr 1GPU |
|---|---|---|---|---|
| 128^3 | 10 (r 4.24e-08) | 10 | 10 | 10 |
| 512^3 | 10 (r 3.32e-08) | 10 | 10 | 10 |

Flat 10 at both sizes, kernel- and device-count-independent (matches the
depth-scaling record's floor-depth counts).

---

## LEG 1(c) — Timing, 4 GPU (ms/step, median[min,max] of 6x20; compile s; peak GiB/dev)

| n | preconditioner | ms/step | vs spectral | compile s | peak GiB/dev |
|---|---|---|---|---|---|
| 128^3 | spectral        | 32.31 [30.38, 39.86] | 1.00x | 5.6  | 0.14 |
| 128^3 | mg-cusparse     | 86.39 [77.67, 92.56] | **0.37x** | 50.8 | 0.17 |
| 128^3 | mg-pcr          | 87.17 [78.25, 92.41] | 0.37x | 57.2 | 0.20 |
| 512^3 | spectral        | 600.5 [—]            | 1.00x | 27   | 7.3  |
| 512^3 | mg-cusparse     | 539.8 [—]            | **1.11x** | 251 | 9.6  |
| 512^3 | mg-pcr          | 774.5 [703.97, 784.70] | 0.78x | 280 | 12.1 |

- 128^3: spectral wins decisively on 4 GPUs (mg 0.37x). At this small size
  the mg V-cycle's per-level halo/collective latency dominates and does
  not amortize across only 32-cell x-shards; the single-GPU parity
  (42 vs 41 ms, addendum) is lost to interconnect latency under GSPMD.
- 512^3: mg-cusparse BEATS spectral 1.11x on 4 GPUs (single-GPU was 1.22x;
  the 4-GPU collective overhead narrows but does not erase the win).
- mg-pcr is ~1.43x slower than mg-cusparse (539.8 vs 774.5) — consistent
  with the kernel study's cusparse>pcr ~1.4x.
- **pcr FITS at 512^3 on 4 GPUs (12.1 GiB/dev, no OOM)** — the 1-GPU
  >=76 GiB live set shrinks ~4x under sharding, so pcr is a viable
  multi-device kernel at 512^3 (it OOMs only on one GPU). cusparse peak
  9.6 GiB/dev, spectral 7.3.

GB-2 gate (>=1.5x) unmet at every measured size/device count — spectral
stays an excellent 4-GPU default at <=128^3; mg-cusparse is the faster
mapped option at 512^3 (1.11x), not by the 1.5x bar.

---

## LEG 2 — Immersed post-swap in-model standing (1 GPU)

Canonical tilted-slope geometry (mirrors
tests/nonhydro2/test_immersed_pressure.py `slope`, order-4 quadrature,
genuine partials, wet frac 0.706). GB-2 protocol, budget 100, tol 1e-8.
Iterations from `solve_info` on the wet-supported RHS = masked divergence
of a random face velocity (the compatible in-model RHS form).

| n | preconditioner | ms/step | iters | rel residual | speedup |
|---|---|---|---|---|---|
| 128^3 | spectral    | 66.47 [66.01, 66.63] | 73 | 9.13e-09 (converged) | 1.00x |
| 128^3 | mg-cusparse | 60.81 [58.70, 60.86] | 20 | 6.28e-09 | **1.09x** |
| 256^3 | spectral    | 482.15 [482.12, 482.19] | 71 | 9.85e-09 (converged) | 1.00x |
| 256^3 | mg-cusparse | 430.15 [430.08, 440.92] | 21 | 6.88e-09 | **1.12x** |

**Immersed verdict: mg-cusparse wins, but only ~1.1x — well BELOW the
study's 1.3-2.0x projection.** mg converges in ~3.5x fewer CG iterations
(20-21 vs 71-73) but each immersed mg V-cycle iteration (semicoarsening +
line smoother + per-level wet-mean) is ~3x costlier than a spectral CG
iteration, netting ~1.1x. Trend is flat/slightly rising with n
(1.09x -> 1.12x).

Crucially, **spectral does NOT bust budget=100** on this slope — it
converges at 71-73 iters (relres ~9e-9 < 1e-8). The study's "spectral
cannot converge, multigrid wins outright 1.3-2.0x" was relative to the
tighter budget=30 it used; at the production budget=100 spectral
converges (slowly), so the win shrinks. The verdict is budget-sensitive:
at any budget < ~70 spectral fails and mg (20-21 iters) is the only
converged option (categorical win); at budget=100 both converge and mg is
~1.1x faster. Physics: mg and spectral in-model trajectories match to
~1e-10 (max|u| identical to 9 digits).

---

## Anomalies / caveats

- **Concurrent sessions early on.** For the first ~25 min two other
  Claude sessions ran on the node (one forced-4 GPU pytest holding
  ~1.9 GiB/dev, one CPU pytest). HLO dumps and parity/iteration runs
  (correctness, not timing) were taken under that light contention;
  **every ms/step timing run was taken with `nvidia-smi` verified idle
  (0 MiB, 0% util, no other compute-apps)** and one python process at a
  time.
- First immersed run used an invalid (non-wet-supported) probe RHS
  (`projection(random_cell_field)`), which made the mg standalone probe
  diverge (relres 9e31) — a probe artifact, not a solver bug: the
  in-model steps were finite and matched spectral. Corrected to the
  masked-divergence RHS (numbers above).
- 256^3 immersed compile logged a benign XLA "constant folding > 1s"
  slow-operation alarm on a `pad` — compile-time only, excluded from
  ms/step.
- Peak memory is `device.memory_stats()` peak_bytes_in_use (live
  high-water), not the 75%-preallocated pool.
