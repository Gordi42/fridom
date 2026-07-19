# GSPMD illegality campaign — GPU checkpoint results

**Job:** 26365157 (Levante `gpu`, 1 node, 4x A100-80GB, 2026-07-19,
COMPLETED 0:0; owner authorized this run in chat the same day).
**Tree:** dev `3b7eea58` (campaign merges `dec698e2` / `8462be11` /
`4367d2f9` / `48cfc25c` all included).

## Verdicts

- **Leg A (1-GPU references): PASS** (rc=0, 38 refs).
- **Leg B (4-GPU single-controller pytest, 20 campaign files):
  526 passed, 3 failed, 3 skipped** — all three failures interpreted
  below; none indicts the campaign machinery.
- **Leg C (real `srun -n 4` multi-process invariance): PASS** —
  items a–d (Wave B walled projections + balance, no-gather
  random/mode synthesis periodic + walled fallback, mixed
  `apply_diagonal`, Tier-2 escape solve + unescaped guard raising
  the taught error) all device-invariant vs the 1-GPU references;
  item e (ETDRK4) is the documented builder-gap skip (below). This
  is the leg the forced-4 single-controller suite cannot substitute
  (jit close-over of sharded constants) — it is clean.

## Leg B failure interpretation

1. `test_eigenbasis_synthesis_distributed.py::test_real_mode_matches_one_device`
   (absmax 0.128) and `::test_real_random_state_matches_one_device`
   (absmax 3.49). **Cross-build eigenvector-gauge rotation, not a
   synthesis defect.** The fixture builds *two independent* real
   `eigh` eigenbases (sharded grid vs `device_ids=(0,)` twin) and
   compares synthesized fields at 1e-10. The eigenfrequency
   comparison in the same test **passed at rel 1e-11** — eigenvalues
   agree, fields rotate: the gauge (phase / degenerate-subspace
   orientation) of `eigh` eigenvectors is not stable across builds
   whose probe inputs differ by sharded-reduction FP noise (or whose
   cusolver batch shapes differ). Synthesis from prescribed
   coefficients is gauge-dependent, so the cross-build premise fails.
   These two tests are GPU-only by construction (the fixture skips on
   CPU, jax#39292 heap corruption) — this was their **first
   execution**; a latent test-premise issue, not a regression. The
   campaign's fused machinery is acquitted by the same-build parity
   files, all green on GPU in this run
   (`test_distributed_contract*.py`, `test_eigenbasis.py`). Gauge is
   unobservable within a single run (projections pair `p·(...)·q`);
   no wrong-physics exposure. **Follow-up (roadmap):** canonicalize
   the eigenvector gauge in the numeric basis build (e.g. pin the
   largest-magnitude component real-positive per column), which also
   makes cross-device-count synthesis reproducible; then these tests
   assert the real invariant.

2. `test_wave3_integration.py::test_fv_pipeline_is_device_count_invariant`:
   differences at the 1e-16 scale against an exact-invariance pin —
   the known **backend-specific bitwise class**
   (GPU fusion/reassociation vs the CPU-tuned pin). Green on the CPU
   legs CI runs; no action.

## Known limitation carried by the harness

**ETDRK4 under real multi-process (Leg C item e):** the numeric
channel eigenbasis builder host-gathers its probe responses
(`fridom.model.eigen_channel._gathered_responses`, `np.asarray`),
which raises "non-addressable devices" under `srun -n N`. The ETDRK4
*application* halves are validated 4-GPU single-controller in Leg B
(`test_exponential.py` passed). **Follow-up (roadmap):** switch the
builder's probe gather to
`jax.experimental.multihost_utils.process_allgather` to serve real
multi-process setup.

## Logs

`legA_1gpu.txt`, `legB_4gpu_pytest.txt`, `legC_srun4.txt`,
`gspmd-26365157.out`, `RESULT.txt` (generated per job) in this
directory.
