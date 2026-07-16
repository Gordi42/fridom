# Implementation A/B — WENO selected-input (shipped code)

Productionization of the phase-3 winner (§6 / RESULTS.md): the
`WENOAdvection` face value now runs ONE left `weno_reconstruct` of the
sign-selected order+1 union window (`_SelectedFaceReconstruction`,
`nonhydro2/modules/advection.py`) instead of reconstructing both biases
and `Where`-selecting. This measures the **shipped, unpatched** code —
no monkeypatch — by checking the parent commit and the branch into the
worktree and running the same harness on each.

- Harness: a self-contained copy of the phase-3 `build_model` (matched
  Oceananigans config: periodic, 10000×10000×100, f0=1e-4, N²=(50 f0)²,
  dt=20 s, AB3, smooth jet IC, `chunk_size=50`). Methodology identical
  to `p3b_model.py`: `block_until_ready`, one warmup chunk discarded,
  median of 6 timed 50-step chunks.
- Hardware: A100-SXM4-80GB, 1 GPU, `JAX_PLATFORMS=cuda`, float64, fresh
  process per point.
- Parent = `78ccc786` (both-then-select). Branch =
  `perf/weno-selected-input` (selected-input).

## Timing (ms/step, median of 6×50-step chunks)

| variant                          | 256³ ms/step | 512³ ms/step |
|----------------------------------|-------------:|-------------:|
| weno5 parent (both-then-select)  |       25.65  |      239.80  |
| **weno5 branch (selected-input)**|   **15.57**  |   **129.78** |
| weno5 delta                      |  **−39.3%**  |  **−45.9%**  |
| upwind5 parent (linear)          |       13.69  |        —     |
| upwind5 branch (linear)          |       13.69  |        —     |

- weno5: −39.3% @256³ / −45.9% @512³ — reproduces the research
  monkeypatch (−39.4% / −46.4%, RESULTS.md M2) with the shipped
  operator. Parent 25.65 / 239.80 match the recorded baselines
  (25.54 / 239.83).
- linear upwind5: unchanged within noise (13.687 vs 13.687), and its
  **compiled chunk HLO is byte-identical** parent-vs-branch (9471 lines,
  empty `diff`) — the linear path is provably untouched.

## Correctness (20-step branch-vs-parent, matched IC, 128³)

| scheme | max\|Δu\|  | max\|Δv\|  | max\|Δw\|  | max\|Δb\|  | finite |
|--------|-----------:|-----------:|-----------:|-----------:|--------|
| weno5  |  9.9e-14   |  1.9e-13   |  1.3e-15   |  2.1e-16   |  yes   |
| weno3  |   0.0      |   0.0      |   0.0      |   0.0      |  yes   |

- weno5 ≤ 1.9e-13 (well within the 1e-12 gate) — reversed-summation
  ulps on the v≤0 faces, exactly as §6 predicted.
- weno3 is **bitwise** identical (right row == exact reversal of the
  left row; the mirror trick is exact for the order-3 tables).
- Field scales for reference: |u|≈0.10, |v|≈0.16, |b|≈1e-4 (so the
  weno5 relative drift is ~1e-12).

## Notes

- The mirrored parity shard `tests/nonhydro2/test_advection_selected.py`
  gates the face value directly against `Where(left, right)` built from
  the retained biased pair, on both C-grid directions, both orders,
  periodic + z-walled grids, and the v=0 tie (right-biased side).
- **Knife-edge, for the owner:** the pre-existing forced-4/4-thread
  divergence test `test_walled_biased_projected_tendency_stays_
  divergence_free[channel-and-lid-*]` sits at the 1e-13 threshold. Its
  `upwind5` case already fails identically on the parent (linear code,
  untouched). The selected-input kernel's different FP reassociation
  now also tips the `weno5` case (1.137e-13 vs 1e-13 — machine
  precision, not a physics regression). Reported, not retuned, per the
  standing instruction.
