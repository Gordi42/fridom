---
status: frozen
date: 2026-07-18
---

# The 192³ upwind5 storage-shape regression — the mechanism behind the "shape luck"

Charge: replace the working label "XLA:GPU kernel-selection shape
luck" from [`storage_halo_gpu_ab.md`](storage_halo_gpu_ab.md) §3 with a
verified mechanism. The `n+8 → n+6` biased-halo narrowing made upwind5
at 192³ ~12% *slower* on the A100 (200³ storage → 198³), while only the
uniform 200³ shape was fast across a width scan. Which kernel(s) get
slower, why, why the reported instability, and is there a safe
mitigation? (Single A100-SXM4-80GB, Levante, jax/jaxlib 0.10.2 cuda12,
float64, dedicated GPU. Predecessor: `storage_halo_gpu_ab.md`.)

Answer up front: **no kernel is mis-selected; the label was an
artifact of incomplete profiling.** The 192³ upwind5 step is
GPU-bound (nvidia-smi util 99%), dominated ~86% by the advection loop
fusions, and the entire +12% lives in those fusions — the pressure
solve's cuFFT and cublas kernels run on the invariant *true* 192³
spectral shape and are byte-identical between the two storage widths.
The gap is **two microarchitectural effects, both functions of the
exact padded storage shape**: (1) the register-heavy multi-output flux
fusions (block 128 × unroll 4) get a **remainder-tail / bounds-checked
launch** when the padded element count is not divisible by 512 — 200³
= 2⁹·5⁶ gives grid 15625 exactly, 198³ gives 15161 with a partial last
block (+5–18%); (2) the light elementwise fusions suffer a
**stride-dependent DRAM/L2 efficiency loss** — ncu shows identical
access pattern (32 sectors/request, maximally uncoalesced) and
identical occupancy, but achieved DRAM throughput 39% on the 198-row
stride vs 54% on 200, a 2.2× kernel slowdown that is the single
largest contributor. A width scan (w3..w8) and a scheme swap prove
there is **no exploitable rule**: only 200³ is fast for upwind5 (208³,
also ÷512, is slow), and centered's sweet spot is a *different* shape
(196³). No XLA flag fixes it for free (`multi_output_fusion` off
equalizes the two shapes only by making the fast one 22% slower). The
"instability" did not reproduce on a dedicated cool GPU — it was
concurrent-tenant contention, to which the DRAM-inefficient 198³ shape
is the more sensitive of the pair. Net: the narrowing verdict is
unchanged (keep it; memory −3%, GPU direction-mixed, per-config width
pinning an optional knob), but "shape luck" now has a mechanism, and
the biased/centered halo widths sit on a real XLA:GPU loop-emitter
knife edge that a byte count cannot predict.

## 1. Reproduction and the profiling correction

192³ upwind5, fresh process per width, `advance(50)`×5, compile
excluded (`rep_step.py`). Reproduced cleanly and **stably**:

| width | storage | med (min) ms/step (3 procs) | vs 200³ |
|---|---|---|---|
| w3 natural | 198³ | 6.216 / 6.223 / 6.219 (6.214) | **+11.7%** |
| w4 wide | 200³ | 5.568 / 5.571 / 5.572 (5.564) | — |

Compiled `memory_analysis` tracks (198/200)³ exactly (args 807.3 vs
832.0 MB) — the storage claim holds; the wall-clock does not.

**Methodology correction (load-bearing).** A first nsys pass
(`--cuda-graph-trace=graph`, the default) reported the GPU 85% *idle*
with only cuFFT/cublas kernels visible, suggesting a host-bound step.
That was wrong: XLA:GPU captures most of the step into **CUDA graphs
(command buffers)**, and graph-level tracing collapses each graph
execution into a single opaque event, hiding the graphed kernels
inside the apparent "idle" gaps. `nvidia-smi` under sustained load
reports **99% util**. Re-profiling with `--cuda-graph-trace=node`
restores the true picture: GPU-active fraction 99.2% (w3) / 99.1%
(w4), loop fusions ≈86% of GPU time, cuFFT pressure solve ≈13%. The
step is GPU-bound and advection-dominated. *Any nsys profile of this
codebase must use node-level graph tracing.*

## 2. Localisation — the delta is entirely in the loop fusions

Per-kernel matched diff (node trace, 200 steps; `diff_kern.py` /
`trace_analyze.py`). The true-shape pressure-solve kernels are
identical; the padded-shape loop fusions carry all of it:

| kernel (registers) | w3 ms | w4 ms | Δ ms | note |
|---|---|---|---|---|
| `loop_add_fusion_10` (32) | 63.7 | 26.8 | **+36.9** | 2.4× — reg-32, sec 4 |
| `loop_add_fusion_11` (32) | 36.3 | 15.9 | +20.4 | |
| `loop_add_fusion`×3 (254) | ~96 ea | ~83 ea | +11–15 ea | flux, remainder |
| `loop_multiply_fusion`×3 (255) | ~134 ea | ~125 ea | +8 ea | flux, remainder |
| `regular_fft` (cuFFT) | 61.5 | 61.5 | +0.09 | **true 192³, identical** |
| `scal_kernel_val` (cublas) | 27.9 | 27.8 | +0.11 | **true, identical** |
| `wrapped_transpose_12` | 17.3 | 17.3 | 0.00 | **identical** |

Total Δ = 134.3 ms / 200 = 671.7 µs/step = the measured 0.65 ms/step
gap. (Some `loop_add_fusion_N` differ in name/instance count between
the two compilations because XLA renumbers fusions per schedule — a
naming artifact, not distinct kernels.) The FFT operates on
`fft_length={192,192,192}` regardless of halo width, so it is
structurally immune; only the buffers stored at `n+2w` are exposed.

## 3. Sub-mechanism A — the remainder-tail on the flux fusions

Every loop fusion launches with **block = 128**; XLA's loop emitter
uses unroll 4, so the launch granularity is 512 threads×elements and
grid = ⌈flat / 512⌉:

| kernel | 198³ grid | 200³ grid | 208³ grid |
|---|---|---|---|
| `loop_multiply_fusion` (reg 255) | 15161 (rem) | **15625** = 200³/512 | 17576 = 208³/512 |
| `loop_add_fusion` (reg 254) | 15161 (rem) | **15625** | 17576 |

200³ = 8,000,000 = 2⁹·5⁶ is divisible by 512 → grid 15625 with **no
partial last block**; 198³ = 7,762,392 = 2³·3⁶·11³ is not → a partial
block that forces the emitter's bounds-checked path (+5–18% on these
register-capped fusions). ncu on `loop_multiply_fusion_1`: w3 2.24 ms
vs w4 2.14 ms at grid 15161 vs 15625, and w3 pushes *higher* DRAM%
(52.5 vs 47.7) while running longer — i.e. the extra time is the tail,
not a bandwidth deficit. These fusions are occupancy-capped (reg 255 →
2 blocks/SM), which is why the tail matters.

## 4. Sub-mechanism B — stride-dependent DRAM efficiency (the bigger one)

The light elementwise fusions (`loop_add_fusion_10/11`, reg 32) are
**2.4× slower on 198³** and dominate the delta. ncu (`loop_add_fusion_10`):

| metric | 198³ | 200³ |
|---|---|---|
| duration | ~498 µs | ~227 µs |
| `dram_throughput` %peak | **39.0** | **53.7** |
| `sm__throughput` %peak | 6.8 | 16.7 |
| `warps_active` %peak | 92.0 | 88.1 |
| sectors / global-ld request | 31.99 | 32.00 |
| grid | 14932 | 15391 |

Identical (maximally uncoalesced) access pattern, identical occupancy
— **only the achieved DRAM throughput differs**. This is a
leading-dimension / DRAM-partition (partition-camping / L2 set
conflict) effect: the 198-element inner row stride (1584 B) distributes
its scattered accesses across DRAM partitions worse than 200 (1600 B).
It is the same family as the classic "pad the leading dimension" GEMM
lore, except here 200 is the accidentally-friendly stride, not a
padded one. The reg-32 kernel is fast on **both** 200³ and 208³ and
slow only on 198³, so it is not simply a size effect.

## 5. Why there is no rule — the width scan and the scheme swap

Uniform width scan, 192³ upwind5 (2 procs each, median ms/step):

| w | storage | factorisation | med | |
|---|---|---|---|---|
| 3 | 198³ | 2·3²·11 | 6.22 | slow |
| 4 | 200³ | 2³·5² | **5.57** | **fast** |
| 5 | 202³ | 2·101 | 6.46 | slow |
| 6 | 204³ | 2²·3·17 | 6.33 | slow |
| 7 | 206³ | 2·103 | 6.79 | slowest |
| 8 | 208³ | 2⁴·13 | 6.24 | slow |

208³ (flat 2¹²·13³, **also** divisible by 512, remainder-free) is
slow — so "divisible by 512" is necessary but **not sufficient**; 200³
is a genuine joint sweet spot (remainder-free flux launches *and* a
friendly DRAM stride) and it is the *smallest* such shape, so anything
larger just moves more bytes. Bytes rise monotonically w3→w8, killing
any byte/alignment story.

The sweet spot is also **scheme-dependent**. Centered advection at
192³: 196³ (its natural w2) is fastest (3.06 ms), 200³ is only middling
(3.22), 198³/202³/208³ slow (3.39–3.53). upwind5 wants 200³, centered
wants 196³ — because the two schemes compile to different fusion
populations with different launch/stride sensitivities. There is no
storage-rounding rule that serves both.

## 6. Mitigation — no free flag

192³ upwind5, w3 vs w4, fresh compile:

| flag | w3 | w4 | gap | verdict |
|---|---|---|---|---|
| default | 6.22 | 5.57 | +11.7% | — |
| `xla_gpu_autotune_level=0` | 6.222 | 5.570 | +11.7% | **not autotune** |
| `xla_gpu_enable_command_buffer=` (off) | 6.291 | 5.664 | +11.1% | both ~+1.5%, no help |
| `xla_disable_hlo_passes=multi_output_fusion` | 6.868 | 6.800 | +1.0% | **net loss** |

Disabling `multi_output_fusion` (the pass that builds the reg-254/255
flux fusions; temp 1297→435 MB) *equalises* the two shapes but only by
making the fast 200³ 22% slower — it removes the win, not the gap. The
only real lever is **pinning storage to the lucky shape** via a
per-model width floor (the optional knob already noted in
`storage_halo_gpu_ab.md` §4.2): memory +3%, fragile across jaxlib
upgrades and array sizes. Nothing upstream-reportable — this is
expected loop-emitter / DRAM-partition behaviour, not a miscompile
(contrast jax#39100, a correctness bug). The honest posture is
measure-and-pin per flagship config, exactly as the predecessor
concluded, now with the mechanism.

## 7. The instability was environmental

The predecessor reported the slow shape wobbling up to 16% across
processes. It did **not** reproduce here: all runs are stable to
±0.02 ms across processes and reps, SM clock pinned at 1410 MHz
throughout (logged around every process), and `autotune_level=0` gives
byte-identical medians (ruling out autotune nondeterminism). The
predecessor ran while a foreign ~65 GiB `spike.py` shared the device;
the DRAM-inefficient 198³ shape (§4) is the more memory-contention-
sensitive of the pair, so a shared GPU amplifies its variance. On a
dedicated cool GPU the slow shape is *deterministically* slow.

## 8. Implication for the pending centered n+4 → n+2 gate

The centered narrowing changes 192³ storage from 196³ (current) to
194³. §5 measured 196³ as centered's **fast** shape, and 194³ = 2·97
has a hostile factorisation of exactly the 198³/202³ family. **Expect
the centered narrowing to regress at 192³**, mirroring upwind5's
n+8→n+6, and expect the sign to flip at other sizes (as upwind5 did:
faster at 512³ per the predecessor). Bindings for the gate: (i) always
compare both widths in fresh processes with node-level graph tracing —
never trust the byte delta; (ii) profile at ≥2 flagship sizes because
the sweet spot is size- and scheme-dependent; (iii) a per-size
regression is not a bug and not a reason to withhold the memory win —
it is the loop-emitter knife edge, and the mitigation (if wanted) is a
width-pin, not a flag.

## 8b. The centered gate, executed (same day)

The §8 gate ran immediately after this record (same A100, fresh
processes, harness [`ab_centered.py`](upwind5_shape_regression/), raw
[`RESULTS_RAW_centered_gate.txt`](upwind5_shape_regression/); in every
narrow run the negotiated width was verified to drop to {1,1,1} and
storage to (n+2)³ before timing — the exact seam the derived
declaration will use). Median (min) ms/step per fresh process:

| size | stock (n+4) | narrow (n+2) | narrow vs stock |
|---|---|---|---|
| 128³ (132³→130³) | 1.110 (1.109) / 1.114 (1.103) | 1.150 (1.135) / 1.149 (1.137) | +2–3% (near noise) |
| 192³ (196³→194³) | 3.069 (3.065) / 3.099 (3.085) | 3.218 (3.215) / 3.203 (3.198) | **+4.3%** |
| 256³ (260³→258³) | 7.518 (7.515) / 7.521 (7.510) | 7.688 (7.682) / 7.693 (7.690) | +2.3% |
| 512³ (516³→514³) | 61.829 (61.801) / 61.809 (61.793) | 63.874 (63.869) / 63.873 (63.860) | +3.3% |

Compiled args bytes track the (n+2)³/(n+4)³ ghost-shell ratio to
5 s.f. at every size (−1.2% to −4.5% of args). The §8 prediction
**held**: 194³ is hostile and 192³ is the worst size — but the
regression is mild and *uniformly signed* for centered (+2.3–4.3% at
every production size), unlike upwind5's mixed-sign ±12-18%. The
sharpest knife-edge datapoint yet: a linear-advection pair at 192³ on
the **identical** 194³ storage is −1.2% *faster* (stock 2.328/2.326,
narrow 2.300/2.298 ms/step) — the sign is a property of the scheme's
fusion population, not of the shape alone. Gate consequence: the
derived-declaration width drop cannot be justified on GPU wall-clock
for the centered default (it prices in ~2–4% there); it ships, if it
ships, on memory/tightness/CPU grounds with that cost recorded, and
the recovery for a flagship centered config is the same width-pin
knob as §6.

## 9. What was tried and refuted

- **"A kernel gets mis-selected / different tile" (the working
  label).** Refuted: node-level tracing shows identical kernel
  *set* and identical launch configs for the true-shape FFT/cublas/
  transpose kernels; the loop fusions differ only in grid *size*
  (remainder) and achieved DRAM %, not in selection.
- **Host/dispatch-bound.** Refuted: graph-tracing artifact; nvidia-smi
  99% util, node trace 99% GPU-active.
- **"Flat count ÷ 512 → fast" divisibility rule.** Refuted by 208³
  (÷512 yet slow) and by the reg-255 vs reg-32 split (only the reg-255
  flux fusions see the remainder; the reg-32 kernels see DRAM stride).
- **Coalescing / occupancy difference on the reg-32 kernel.** Refuted:
  ncu shows identical 32 sectors/request and ~90% warps-active on both
  shapes; only DRAM efficiency differs.
- **Autotune nondeterminism (for the instability).** Refuted:
  `autotune_level=0` is byte-identical and equally stable.
- **A free XLA flag.** Refuted: autotune/command-buffer no effect;
  `multi_output_fusion` off removes the win, not the gap.
