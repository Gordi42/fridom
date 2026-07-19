---
status: frozen
date: 2026-07-18
---

# Naive GSPMD transform path — illegality study and campaign record

Owner directive (2026-07-18 chat): "make it illegal in general to use
the naive GSPMD transform path." This record holds the doability/impact
study and what shipped the same day (phases 0–1 plus the fallout
waves). Remaining phases:
[`../plans/done/gspmd_transform_illegality_plan.md`](../plans/done/gspmd_transform_illegality_plan.md).

## The study (verified findings)

- **Architecture pivot:** `Transform.forward`/`backward`
  (`spatial/operators/transform.py`) always ran the single-device plan;
  the `distributed_forward_plan`/`distributed_backward_plan` seams
  existed but were consumed only by the spectral solver's `SlabSolve`.
  The safe distributed lowerings (`SlabPlan` in `distributed_solve.py`,
  `ContractPlan` in `distributed_contract.py`) run their FFTs inside
  their own `jax.shard_map` and **bypass** `Transform.forward/backward`
  entirely — so a guard at that ABC seam fires exactly for naive
  consumers and never for the safe paths (verified empirically with an
  instrumented guard: zero trips across the distributed test files,
  every naive consumer caught). One bypass exists: the direct
  `jnp.fft.fftn` in the `numeric_eigenpairs` probe (`model/eigen.py`).
- **Two tiers.** Tier 1: a *transform* axis is sharded — on XLA:GPU
  (jax/jaxlib 0.10.2) this crashes **today** in the HLO verifier (the
  distributed-Cooley-Tukey c64-twiddle fault, jax#39291); on CPU it
  silently all-gathers, so "passes forced-4 CPU" never meant
  multi-device-safe. Tier 2: only non-transform axes sharded — a
  silent all-gather perf trap, numerically correct.
- **Impact:** ≥140 forced-4 CPU tests exercised the Tier-1 naive path
  (they passed only by all-gather); **Tier-2-only breakage was empty**
  across the whole suite. On real multi-GPU the Tier-1 paths were
  already broken, so illegality converts silent crashes into taught
  errors; the genuine day-one cost is test-suite churn.

## Shipped (all 2026-07-18)

- **Phase 0 — Tier-1 guard** (merge `83fbc56c`): taught
  `NotImplementedError` in `Transform.forward/backward`, predicate =
  device_count > 1 AND the *operand's own* layout shards one of this
  transform's stage axes (never fires on `device_ids=(0,)`, collapsed
  layouts, or replicated operands; Tier 2 stays legal). Plus the
  `numeric_eigenpairs` probe gather (single-device bit-identical) and
  the suite conversion (`device_ids=(0,)` pins preferred, taught-error
  counterparts for the sharded cases).
- **Phase 1 — channel synthesis reroute** (merge `e316987d`):
  `ContractPlan.synthesize` — a backward-only `shard_map` region
  reusing `_backward_component`; `_eigenbasis._backward_synthesis`
  routes `mode()`/`channel_random_state` through it whenever the
  layout shards a periodic axis. Coefficients are built directly in
  the region's internal frame (the forward `all_to_all` is tiled and
  moves no axes, so store frame and internal frame differ only by the
  half-axis pad and sharding). Parity sharded-vs-`device_ids=(0,)`:
  bit-identical (absmax 0.0); reverse-mode VJP finite. The 2-D channel
  stays on the taught error (phase 2, pending ratification).
- **CI**: `test_eigen.py` + `test_freeze_fingerprint.py` added to the
  forced-4 leg (`867537e1`); forced-4 expectation repairs
  (`3f2c637f`).
- **Fallout wave 1 — fixture pins** (merge `a6bc4b39`): the
  interval-halo floor change (2→1) made many small fixture grids newly
  shard under forced-4, exposing ~233 unmarked tests to the guard
  beyond the study's estimate. Fixed at the *fixture* level
  (`device_ids=(0,)`), never pinning `multi_device` tests; two
  device-count-invariance tests that asserted invariance of
  GPU-broken paths became taught-error assertions.
- **Fallout wave 2 — two genuine interval-accounting regressions**
  found by the residual sweep (one loud, one **silent wrong
  physics**), fixed the same day — record:
  [`halo_sharding_invariants.md`](halo_sharding_invariants.md).

## Residual test debt (marked tests) — resolution

The four marked multi-device tests flagged at phase 0–1, revisited by
the phase-3 consumer wave (`feat/distributed-transform-consumers`):

- `test_krylov::test_solution_is_device_count_invariant` (the CG
  apply's naive `SpectralDerivative` — the real phase-3 consumer):
  **converted** to a real device-count-invariance gate, the spectral
  apply routed through `Transform.apply_diagonal` (the fused
  forward → diagonal → backward). Was a taught-error skip; now the
  whole 5-iteration Poisson solve matches single-device to 1e-11.
- `test_cumulative::test_decomposed_axis_matches_single_device`:
  **passes** on the forced-4 leg — never real transform debt.
  `CumulativeIntegral` is a `layout="local"` reshard operator (the
  integration axis is resharded local and back), not a
  change-of-representation, so the Tier-1 guard never applied; it was
  mis-listed.
- `test_eigenbasis::test_function_application_is_device_count_invariant`:
  **passes** on the forced-4 leg — the `f(L)` application on the 2-D
  channel is served by `Channel2DPlan` (phase 2), bit-identical
  (absmax 0.0).
- `test_reblock_step_collectives::test_x_walled_step_collectives_equal_periodic`
  (and `..._match_off_axis_wall`): still **red** on the forced-4 leg,
  but this is **not** transform-path debt — it asserts a step's
  collective *count*, a decomposition/reblock property shifted by the
  interval-halo floor change, owned by the halo-floor-semantics
  campaign (`fc2a3b66` on dev advanced that machinery further; the
  expectations need re-baselining there). Left blocked, out of the
  transform consumer wave's scope.
