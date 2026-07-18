# The FV-vs-nodal step-time gap: attribution and fix

Investigation record, 2026-07-18 (4× A100-80GB node l50054, jax
0.10.2, dev `96767550` → fix branch `perf/fv-walled-storage-frame`).
Closes the roadmap item "Close the FV-vs-nodal step-time gap where it
exists" (entry now in [`../roadmap/done.md`](../roadmap/done.md)).

FV and nodal are bitwise-identical (flat) / ≤1.2e-14 (walled, jitted)
trajectories, so any step-time gap between them is a lowering/fusion
artifact by construction. The nodal sibling cases of
`benchmarks/model/bench_step.py` are the measuring stick throughout.

## 1. Re-measured gap (pre-fix)

Per the roadmap guard ("compiler-artifact class — re-measure before
investing"), the gap was re-measured on clean dev before any work.
Fresh FV/nodal step-time ratios (min over 5 reps of 50-step chunks):

| geometry | 1 GPU | 4 GPUs |
|---|---|---|
| flat periodic n=256/512 (control) | 1.00 | 1.00 |
| walled-z n=256 | 1.007 | **1.039** |
| walled-x n=256 | 1.003 | **1.017** |
| mapped n=128 (i30/i12) | 1.006/1.002 | **1.083/1.087** |
| mapped n=256 (i30/i12) | 1.007/1.009 | **1.157/1.182** |

Two corrections to the T7 folklore: (a) the **1-GPU gap does not
exist** — the "+1…+10% walled on 1 GPU" of the T7 re-record was
FV-vs-*old-FD-baseline*, not FV-vs-nodal; (b) the gap is purely a
**multi-device** phenomenon. Iteration-differencing the mapped rows
shows the per-CG-iteration cost is *equal* (0.244 vs 0.246 ms/it at
256³) — the whole mapped gap sits in the iteration-independent part
of the step, outside the pressure-solve loop.

## 2. Mechanism (HLO-attributed, causally confirmed)

Nodal stencils run one windowed kernel over the halo-extended
**storage frame** (`staggering.apply_staggered`). The FV family
shares that body (`reconstruct.apply_fv_staggered`) on uniform rows —
hence flat parity — but the walled step path ran two special branches
that left the storage frame entirely (`f.data` unpad → `jnp.pad` the
exact-zero walls → kernel → `store()` repad):

1. `FluxDifference._apply_factor`'s homogeneous `Inner` arm — the
   projection-RHS divergence of the Dirichlet-tagged wall-normal
   velocity, on every walled axis;
2. `LinearReconstruction._reconstruct_walled_face` — the
   claim-consuming `Inner(DIRICHLET) → CellAvg` row (the stratified
   `w.to(b)` seam).

On one device XLA absorbs the excursion at zero cost (measured
parity, +45 data-movement instructions). On several devices the SPMD
partitioner **materializes the true-frame tensors in a transposed
`{2,1,0}` layout and reroutes the periodic-axis halo
collective-permutes through it**: mapped n=128/4-GPU whole-module
counts FV vs nodal — collective-permutes on the transposed layout
**58 vs 26**, full-shard transposes 31 vs 20, plus 9 true-frame
(unpadded-shape) transposes that exist only in FV (the fingerprint of
the `f.data` excursion). Total collective *count* is essentially equal
(140 vs 146), which **refutes** the competing hypothesis that the
`store()`-built results' all-axis halo-claim loss forces extra
refill exchanges — the cost is the *layout* of the exchanges, not
their number.

Causal confirmation: monkeypatching the two branches to the
storage-frame spelling (below) made the compiled step structurally
identical to nodal (transpose/copy histogram byte-identical) and
closed the measured gap; the patched HLO has 0 true-frame transposes
and 26 transposed collectives.

## 3. The fix (landed with this record)

Both branches now have a **storage-frame windowed fast path**, gated
by `wall_slots_addressable(f, axis)` (applied axis device-local AND
negotiated halo ≥ 1): write an exact 0 into the two wall-adjacent
ghost slots of the operand storage at static indices
(`wall_zeroed_operand` — the `Inner` column becomes the `Outer`-like
n+1-face column), then run the ordinary `apply_fv_staggered` window
(alignment m0 = 1, reach 1); mapped meshes divide by the VJP-sealed
`divide_by_codomain_measure`. Byte-for-byte the true-frame arithmetic
on every output cell; keeps the operand's periodic-axis halo claims.
The prior true-frame code remains as the fallback for a distributed
walled axis or an un-negotiated halo (correct on any layout, slower
spelling).

Measured after the fix (same node, 4 GPUs, fusion workaround set),
FV/nodal ratio: walled-z n=256 **1.039 → 1.003**, walled-x
**1.022 → 0.995**, mapped n=128 **1.083/1.093 → 1.003/1.002**,
mapped n=256 **1.151/1.177 → 1.008/1.006**. Flat periodic and all
1-GPU cases unchanged within noise. Physics: 10-step state
**bitwise-identical** to the true-frame spelling on 1 and 4 GPUs
(walled-z and mapped, n=64); `jax.grad` finite and matching central
FD (mirrored autodiff test).

## 4. Residuals and notes

- **1-GPU mapped n=256 pays +1.6%** (83.3 → 84.6 ms/step): the fast
  path's storage-frame sealed divide (measure sync + double-`where`)
  replaces the true-frame divide and scales with n; walled and
  mapped-128 are neutral-to-faster. Accepted as a favorable trade
  against the −12.5% 4-GPU mapped win (mapped at that size is
  primarily a multi-GPU concern). If it ever matters: a `custom_jvp`
  whose primal is the raw divide (untouched forward, sealed reverse)
  is the sanctioned lever (AGENTS.md differentiability policy).
- **Distributed walled axis keeps the slow spelling** (the fallback).
  Reachable only when the decomposition must shard a walled axis
  (e.g. fully-walled domains at high device counts); the shard-axis
  selection avoids it wherever a periodic axis is available.
- **The FV fusion-guard ratchet counts rose** (cpu|1 op-count ratio
  mapped 1.012 → 1.030, walled_x 1.015 → 1.022) while wall-clock
  reached parity: the windowed spelling adds cheap instructions
  (wall-zero plane writes, full-storage windows, sealed divide).
  Baseline re-recorded via `FRIDOM_REGEN_FV_RATCHET` — the ratchet
  gates *counts*, and count ≠ cost.
- **Stale gpu1 mapped baseline found** during the re-measure: the
  committed `step-gpu1.json` mapped rows predate the mapped-solver
  optimization (~2.4× slower than current dev). Both baselines are
  re-recorded on the post-fix dev in the follow-up baseline commit.
- The remaining intrinsic FV-vs-nodal difference (the two-stage
  `reconstruct → flux_diff` vs one-stage `FiniteDifference`, ~+45
  data-movement instructions) is benign on every device count — the
  E1 uniform-parity guard pins the periodic case opcode-for-opcode.

## 5. Method notes (for the next perf hunt)

- Iteration-differencing the committed mapped rows (i30 vs i12)
  located the gap outside the CG loop before any profiling.
- The 1-GPU compile of the same program is the cheapest control for
  separating "structural but benign" from "multi-device-costly".
- A monkeypatched A/B of the candidate respelling (scratchpad scripts,
  no repo edits) proved causality in one afternoon; HLO category
  counts alone would have left the layout mechanism circumstantial.
- Full evidence (scripts, HLO dumps, A/B tables) in the session
  scratchpad `hlo_attrib/`; the committed artifacts are this record,
  the fix, its tests, and the re-recorded baselines.
