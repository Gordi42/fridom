---
status: frozen
date: 2026-07-17
---

# Upwind-5 advection revisited — RTX 3060 re-baseline, byte-level attribution, and the ceiling

Charge (owner, 2026-07-16): the stencil-lowering study concluded linear
upwind5 "cannot do better" — hard to believe, given fridom beats
Oceananigans clearly on centered and WENO. Re-research on new hardware
(the owner's laptop, a consumer GPU) and explore every remaining option.

Answer up front: **the conclusion replicates, and is now *proven* at the
byte level rather than asserted.** Every remaining lever was measured —
one-path spellings (again, on opposite hardware), an XLA flag sweep, and
the deferred Pallas hand kernel — and every one is a clean negative with
a quantified mechanism. fridom's upwind5 is at the practical ceiling of
the XLA lowering on BOTH tested architectures, and it already meets or
beats Oceananigans in absolute terms (1.05x A100 512^3; 1.01–1.06x RTX
3060). What it cannot reach is the 1.26–1.6x edge fridom enjoys on
centered/WENO — and the residual is now attributed: ~1/3 XLA's flux/
reconstruction materialization, ~2/3 wide-stencil fusion efficiency,
~0 arithmetic. Neither component is addressable by spelling, flags, or
Pallas-on-Ampere.

Benchmark commit: dev `597223b1` (nodal family forced; see side
finding 1). All raw data, scripts, and harnesses: subdirectories of
[`upwind5_revisit/`](upwind5_revisit/). Prior art (read first):
[`stencil_lowering.md`](stencil_lowering.md) §5–§8.

## 1. Machine and protocol

RTX 3060 Laptop (GA106, sm_86), 6 GB GDDR6, 55 W cap, driver 595.71 /
CUDA 13.2; jax/jaxlib 0.10.2 (`jax[cuda12]` wheels) in an isolated venv,
f64 throughout. Measured rooflines ([`fridom/roofline.json`](upwind5_revisit/fridom/roofline.json)):
f64 FMA **87.6 GFLOP/s** (f32 5551 — the 1/64 consumer-Ampere rate),
f64 divide 10.7 Gdiv/s (= 4.1x an FMA), triad ~107 GB/s measured
(hw peak ~330). f64 machine balance 0.82 flop/byte vs the A100's ~7.6:
**~9x more f64-compute-starved** — the opposite regime from the A100,
and exactly the hardware on which arithmetic-bound spellings would flip
if they were ever going to.

fridom side: the phase-3 harness
([`fridom/p3b_model_rtx3060.py`](upwind5_revisit/fridom/p3b_model_rtx3060.py);
matched config, periodic n^3, dt=20, AB3, chunk 50; fresh process per
variant, 1 warmup + 6 timed chunks). Oceananigans side rebuilt from the
matched protocol (the comparison suite is untracked and machine-local):
Julia 1.12.6, **Oceananigans 0.105.3 pinned** (same as the July A100
comparison), CUDA.jl 5.11.3, `QuasiAdamsBashforth2` (1 RHS/step, matches
AB3), matched grid/ICs, no Simulation wrapper
([`ocean/bench.jl`](upwind5_revisit/ocean/bench.jl); one delta: no
stratification tendency term — one cheap linear term, below noise).

Thermal discipline matters on this laptop (55 W, never saturates): drift
+2–4% over a matrix, and an unbracketed flag comparison produced fake
±5% results. Cross-variant tables use contemporaneous pass-1 runs;
the flag sweep uses full baseline bracketing (§4).

## 2. Fresh matched comparison (median ms/step, f64)

| ms/step | 96^3 | 128^3 | 160^3 | 192^3 |
|---|---:|---:|---:|---:|
| fridom centered | 5.25 | 12.17 | 24.84 | 42.70 |
| Oceananigans Centered(2) | 6.63 | 15.36 | 31.84 | 53.93 |
| fridom upwind5 | 9.90 | 22.70 | 46.00 | 77.85 |
| Oceananigans UpwindBiased(5) | 9.83 | 23.00 | 47.33 | 82.29 |
| fridom weno5 (selected) | 20.32 | 46.95 | 91.42 | 157.90 |
| Oceananigans WENO(5) | 31.99 | 73.27 | 145.43 | — |

Edges (Ocean/fridom): **centered 1.26x, upwind5 1.01–1.06x, weno5
1.56–1.59x** — the same shape as the A100 (1.38x / 1.05x / ~1.8x
projected). Internal ratios: fridom upwind5/centered = **1.82–1.89,
identical to the A100's 1.82** despite the 63x f64-FMA gap between the
machines — the single strongest evidence that upwind5's overhead is not
arithmetic. Oceananigans' internal ratio is 1.48–1.53 (they also
compute both biased sides for linear upwind; their increment is
in-register arithmetic, fridom's carries extra HBM structure, §3).
Side note: Oceananigans' WENO/centered ratio here is 4.6–4.8x (A100
~2.7x) — the shipped weno5 selected-input is worth even more on
f64-starved consumer GPUs, as the divide attribution predicted.

One-path replication (fresh A/B, machine-precision-gated): selected
−7.1% @96^3, −2.5% @128^3, **+2.4% @160^3, +5.2% @192^3** (A100 256^3:
+5.8%); dissipation +4–10% everywhere. The A100 verdict replicates at
production sizes on opposite hardware; the small-n win (crossover
~128–160^3, cache-influenced) is real but not shippable — it reverses
exactly where the sizes matter.

## 3. Byte-level attribution — what the +82% over centered actually is

Method: XLA buffer-assignment + live-range dumps with `op_name`
metadata, per-variant recompiles reproducing the benchmarked temp bytes
exactly, plus `cost_analysis()` on a chunk_size=1 module
([`attribution/`](upwind5_revisit/attribution/); SUMMARY.json,
DECOMPOSITION.json). Trap for future measurements: the 50-step chunk
loop is **unrolled by 3** (`known_trip_count=16`), so static byte sums
over the chunk module overcount per-step traffic ~3.7x — use a
chunk_size=1 recompile.

Where the temp goes (peak-liveness): centered's 275 MB peak (128^3) is
the **time-integration** phase (9 accumulation `add` buffers + 3 flux
products). upwind5's 419 MB peak is a **new advection-phase peak** that
overtakes integration: 9 flux-product full fields `[n+8]^3` plus **6
materialized face-reconstruction arrays** `[n+5]x[n+8]^2` live at once —
centered materializes zero reconstruction arrays (its 2-point interp
fuses into the flux multiply; the 6-tap window is too expensive for XLA
to duplicate into the flux-difference's two face consumers, the same
cost-model behavior as the biharmonic case in
[`stencil_lowering.md`](stencil_lowering.md) §2). Wider halo storage
(`n+8` vs centered's `n+4` — 2x the nominal reach on both) contributes
another ~25%/13% of the delta at 128/192^3.

**It is NOT the both-then-select duplication:** u5_selected's compiled
step is byte-identical to baseline in temp (418.6 MB), bytes accessed
(3286.9 MB/step) and peak — the second biased reconstruction is fused/
recomputed in registers, never materialized. This is the mechanism
behind every failed spelling probe on both machines: Python-level
spellings do not change what XLA materializes, and the arithmetic they
save is hidden under the memory traffic.

Decomposition of the +35.15 ms increment at 192^3 (n128 in brackets):

- **(a) extra HBM bytes, ~35%** [+30%]: upwind5 moves +2446 MB/step
  (+29%) — the materialized reconstructions, more live flux products,
  wider-halo reads. At centered's efficiency: +12.2 ms [+3.2].
- **(b) kernel efficiency, ~65%** [+70%]: upwind5's bytes run at
  **141 GB/s vs centered's 201** — the wide-halo reconstruction fusions
  are register-heavier and strided, and XLA's loop emitter does no
  shared-memory tiling for stencils. +23.0 ms [+7.35].
- **(c) f64 arithmetic, ~0**: selected-input has fewer flops, identical
  bytes, and is slower at 192^3; the 1.82 ratio is machine-invariant
  across a 63x f64 gap. (Confidence: (c)≈0 and bandwidth-boundedness
  high; the (a)/(b) split medium — `bytes_accessed` is a static count.)

## 4. XLA flag sweep — null

Nine flagsets (double-buffering, latency-hiding scheduler, autotune 4,
command-buffer off, alias-scope metadata, multi_output_fusion off, ...)
under full baseline bracketing: **every candidate within ±0.2% of a
thermally-matched baseline** — no mover, nothing near the 3% gate
([`flags/summary.json`](upwind5_revisit/flags/summary.json); six more
candidate flags rejected by jaxlib 0.10.2, list in the JSON). The one
real effect is `--xla_disable_hlo_passes=multi_output_fusion`: a pure
**memory knob** for upwind5 (executable scratch 790 -> 266 MB at ~zero
u5 speed cost; centered +4.7% slower) — back-pocket headroom for
larger n on small-VRAM GPUs, never a default. Methodology note that
generalizes: on a laptop GPU that never thermally saturates, any flag
comparison that is not baseline-bracketed is worthless (run-order drift
manufactured ±5% "results" in the naive pass).

## 5. Pallas/Triton hand kernel — decisive negative on sm_86

The one structural trick XLA cannot express (shared-memory halo staging,
one read per input) was the last live lever
([`stencil_lowering.md`](stencil_lowering.md) §8, deferred until
selected-input landed). Spike ([`pallas/`](upwind5_revisit/pallas/)):
the per-axis upwind5 flux-divergence kernel, both stencil axes, n=128
and 192 padded shapes, vs the XLA composed micro.

Result: best Pallas config is **+34–35% slower at 192^3 on both axes**
(+43–54% at 128^3), while bitwise-identical in output. Mechanism, fully
quantified: on this GPU the isolated kernel is **~70%
f64-arithmetic-bound** (XLA's 2.53 ms vs a 1.76 ms 10-FMA/face floor at
192^3) — and Pallas-Triton's power-of-2 array constraint forbids the
fat-window-plus-slice idiom, forcing the flux to be reconstructed **at
both faces of every output cell**. That doubles the reconstruction
arithmetic (floor 3.51 ms), and the measured Pallas wall (3.41 ms) sits
exactly on that doubled floor. The pow2 tax IS the loss; SMEM staging
never gets a chance to matter. Triton-f64 itself was numerically clean
(bitwise vs XLA; no miscompile), and two sharp edges are recorded for
posterity: jaxlib 0.10.2 defaults `jax_pallas_use_mosaic_gpu=True`,
which cannot lower on sm_86 (force Triton via
`pltriton.CompilerParams()`), and only pow2 output blocks that divide
the grid are legal.

Engineering risk (had it won): a `pallas_call` is an opaque custom call
— it does not compose with GSPMD, so the distributed path would need a
hand-written `shard_map` + manual halo exchange; plus backend-flag churn
and pow2 tile coupling. **Pallas for linear upwind is closed** on
consumer-Ampere f64 by measurement, and remains unattractive on the
A100 (fused stencils already at 60–90% of triad there). The only Pallas
question still arguably alive is the *WENO* per-point kernel on the
A100 (real per-path divide savings) — unchanged from
[`stencil_lowering.md`](stencil_lowering.md) §8.

## 6. Verdict and do-not-revisit

fridom's upwind5 both-then-select spelling is at the practical ceiling
of the XLA:GPU lowering on both tested architectures. It ties/beats
Oceananigans absolutely; the missing relative edge vs fridom's own
centered/WENO wins is XLA's materialization of wide face
reconstructions plus untiled wide-stencil fusion efficiency — a code
structure only a hand kernel could change, and the hand kernel loses on
this hardware class for quantified structural reasons.

Extended do-not-revisit list (with mechanisms, superseding "measured,
didn't win"): one-path/selected/dissipation spellings for linear upwind
(identical bytes, arithmetic hidden — both machines); XLA:GPU flags for
upwind speed (null sweep, consumer Ampere); Pallas/Triton linear-upwind
kernels on sm_8x (pow2 => 2x arithmetic, arithmetic-bound regime).

Residual open leads, all small or roadmap-level: the storage-halo width
(`n+8` per axis vs nominal `n+6`; ~6% on every upwind5 buffer, est.
2–3 ms/step @192^3 — core staggering policy, parity-sensitive); the
`multi_output_fusion` memory knob for large-n on small VRAM; the axis-0
layout lever ([`stencil_lowering.md`](stencil_lowering.md) §7 lever 3,
unchanged); Pallas-for-WENO on A100.

## 7. Side findings

1. **FV biased-advection assembly blocker (live on dev).** The
   benchmarks had to force `family="nodal"`: since the F3 default flip
   (`cb752b85`), a default-constructed periodic (now: any unmapped,
   unimmersed) `nh.Model` with `UpwindAdvection`/`WENOAdvection` fails
   at dry-run with `SpaceMismatchError` (`Center` vs `CellAvg` retag in
   `_face_value`) — velocity self-advection: `u`'s x-factor is nodal
   `Right`, so `_flux_space` derives `CellAvg` via `diff` while
   `_biased_pair` picks the nodal reconstruction (codomain `Center`),
   bridged only by a BC-only `retag`; `CenteredAdvection` survives
   because `_velocity_face` does a real `.to` conversion. Coverage gap:
   the FV advection tests drive raw `fr.model.Model` (nodal default
   grid) with a `CellAvg` tracer only, so FV velocity self-advection
   through the biased path is never assembled; `test_fv_default.py`
   uses centered only. Re-verified at `cf633de1`. Repro:
   [`upwind5_revisit/fv_repro.py`](upwind5_revisit/fv_repro.py).
   Tracked in the FV roadmap entry.
2. **Laptop-GPU benchmarking methodology**: thermal bracketing is
   mandatory (§4); fresh process per variant remains mandatory (chunk
   executables cache on the static AssemblyRecord).
3. The A100 numbers in [`stencil_lowering.md`](stencil_lowering.md)
   remain the reference for production hardware; nothing here contradicts
   them — this study strengthens them with an independent replication on
   opposite-regime hardware.
