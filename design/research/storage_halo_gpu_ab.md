---
status: frozen
date: 2026-07-18
---

# Storage-halo narrowing on the GPU — the A/B, and what it actually found

Charge: the manual GPU wall-clock A/B left open by
[`storage_halo_width.md`](storage_halo_width.md) §5 (roadmap
remainder (b)), run on owner request (2026-07-18, single A100-80GB,
Levante, jax 0.10.2). Expectation going in: the `n+8 → n+6` biased
narrowing saves the ghost-shell bytes (−3.0% @192³) and therefore
~2-3 ms/step @192³ on RTX-3060-class hardware.

Answer up front: **the bytes are exactly as predicted; the wall-clock
is not.** Compiled step time on the A100 is shape-sensitive far beyond
byte volume: per (scheme, size) the narrowing swings **±10-20% in
either direction** — upwind5 is ~12-18% *slower* at 192³ and ~3-14%
*faster* at 512³; weno5 is ~5-8% slower at 256³/512³; 128³ and
256³-upwind5 are neutral. The old width-4 shape at 192³ (200³ storage)
is a lucky XLA:GPU shape: uniquely fast among widths 3-6, and the
advantage needs *all* axes at 200 — so this is kernel-selection luck,
not alignment or bytes, and there is no padding rule to chase. Net
verdict: keep the narrowing (memory −3%, CPU faster, GPU
direction-mixed), treat per-config width pinning as an optional tuning
knob, and close the blind spot that let this class of change go
unpriced: `bench_step.py` has no biased-advection case.

## 1. Harness

[`storage_halo_gpu_ab/ab_step.py`](storage_halo_gpu_ab/): methodology
copied from `benchmarks/model/bench_step.py` (triperiodic n³ f-plane
`nh.Model`, `family="nodal"` (biased FV assembly still blocked), jet
ICs, CFL dt, `chunk_size=50`, `advance(50)` per rep, compile
excluded, 5 reps) — one variant per fresh process. `natural` = dev
after the narrowing (biased width 3, centered 2); `wide` = the
pre-merge widths forced back via a merge-max floor on the two
negotiation seams (biased 4; centered 2, i.e. an A/A control).
Numbers below: median (min) ms/step.

## 2. Results

| case | natural (n+6) | wide (n+8) | narrow vs wide |
|---|---|---|---|
| 128³ upwind5 | 1.842 (1.834) | 1.859 (1.825) | neutral |
| 128³ weno5 | 2.297 (2.289) | 2.270 (2.254) | neutral (+1%) |
| 128³ centered A/A | 1.119 (1.118) | 1.118 (1.111) | noise floor ~0.3% |
| 192³ upwind5 ×3 runs | 6.28 / 6.65 / 7.28 (best 6.27) | 5.62 / 5.62 / 5.88 (best 5.62) | **+12-18% slower** |
| 192³ weno5 | 6.983 (6.974) | 7.423 (6.782) | ~neutral (noisy) |
| 192³ centered A/A | 3.081 (3.078) | 3.068 (3.064) | noise floor |
| 256³ upwind5 | 14.650 (14.606) | 14.489 (14.340) | neutral (+1%) |
| 256³ weno5 | 18.436 (16.325) | 15.623 (15.618) | **+5-18% slower** |
| 512³ upwind5 | 123.707 (123.705) | 143.204 (127.445) | **−3-14% faster** |
| 512³ weno5 | 152.617 (140.125) | 130.001 (129.986) | **+8-17% slower** |

Compiled `memory_analysis` bytes track the `(n+6)³/(n+8)³` ghost-shell
ratio to 4 s.f. at every size (e.g. args @192³ 807.3 vs 832.0 MB,
−3.0%) — the storage claim of the narrowing holds exactly.

Cross-check against *real* pre-merge dev (throwaway detached worktree
at `4c287c12`, 192³ upwind5): 5.632 (5.624) / 5.634 (5.631) ms/step —
and the persistent compile cache served the same-code forced-wide
executable for it (first advance 1.0 s, byte-identical
`memory_analysis`), i.e. **forced-wide new code ≡ old dev, compiled
byte for byte**. The 192³ regression is real, not a forcing artifact.

Reliability notes: the slower variant of each pair is also the
unstable one (medians wobble up to 16% across processes; the fast
variants repeat to ±0.01 ms) — consistent with the slow shapes landing
on an occupancy-marginal kernel. Two 512³ processes died silently on
their first batch attempt (no output; clean on retry) — transient,
not reproduced.

## 3. The width scan — why there is no fix to chase

192³ upwind5, forced uniform widths (storage row = 192 + 2w), best of
runs: w3 6.27, **w4 5.62**, w5 6.49, w6 6.37. Not monotone in bytes
(w5, w6 beat w3 while carrying more bytes) and not alignment (204 is
as 32-byte-sector-aligned as 200, yet slow). Per-axis probes: z-only
wide (x,y 198, z 200) 6.54 (5.89), x,y-only wide (200, z 198) 6.77
(6.55) — **only the uniform 200³ shape is fast**, so padding a single
(e.g. innermost) axis recovers nothing. Conclusion: XLA:GPU
kernel-selection luck on the exact shape tuple. Any rule-based
storage rounding would be tuning against a compiler heuristic that
moves with every jaxlib upgrade; the honest posture is measure-and-pin
per flagship config, not a global width policy. (Same family as the
micro-vs-real fusion reversal lesson,
[`stencil_lowering.md`](stencil_lowering.md).)

## 4. Follow-ups (roadmap)

1. **Blind spot:** `benchmarks/model/bench_step.py` prices advection
   only via the centered default — the biased/WENO family, where all
   recent stencil work landed, has no case, which is why neither this
   change nor the −39/−46% weno5 win was guard-visible. Add
   `nh_flat_advective_upwind5` / `_weno5` cases (append-only; baseline
   recorded at the owner's next batched guard run).
2. **Optional knob:** a supported per-model storage-width floor
   (the negotiation already accepts a widening `halo=`; today only
   the probe monkeypatch reaches it) would let a flagship config pin
   a lucky shape, e.g. 192³ upwind5 back to width 4. Owner call on
   whether the ~12% at one size is worth a public knob.
3. The RTX-3060-class estimate of the original roadmap item is
   untested (no such device here); given the A100 verdict, treat the
   2-3 ms/step figure as withdrawn rather than pending.
