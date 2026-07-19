# GSPMD illegality campaign — GPU validation checkpoint

Manually-submitted GPU checkpoint for the *naive GSPMD transform
illegality* campaign (plan record
[`../../../plans/done/gspmd_transform_illegality_plan.md`](../../../plans/done/gspmd_transform_illegality_plan.md);
open item in
[`../../../roadmap/open.md`](../../../roadmap/open.md) §"Naive GSPMD
transform path — GPU checkpoint"). It exercises the campaign's
multi-device paths on real A100s — the fused-synthesis parity, the
ETDRK4 end-to-end run, and every shipped consumer — including the paths
that need real `eigh` bases (CPU-unsafe, jax#39292) and a real
multi-process launch (which the forced-4 single-controller test suite
cannot fully cover).

## What the job validates

One SLURM job, three sequential legs (`gspmd_campaign_check.sbatch`):

- **Leg A — 1 GPU, 1 process.** `smoke_refs.py --write-ref` computes the
  five campaign smoke items on a single device and saves each produced
  field as a `.npy` reference for Leg C.
- **Leg B — 4 GPUs, 1 process (single-controller GSPMD).** `pytest`
  (serial, no xdist) over the campaign's test files with real `eigh`
  bases on real 4 GPUs. `jax.device_count() == 4`, so `multi_device`
  tests run and `single_device` ones skip. Files:
  the fused-synthesis parity set
  (`test_distributed_contract*.py`, `test_eigenbasis.py`,
  `test_eigenbasis_synthesis_distributed.py`), the ETDRK4 run + grad
  (`test_exponential.py`), and this session's campaign consumers
  (`test_transform`, `test_spectral_solve`, `test_mixed`,
  `test_distributed_solve`, `test_distributed_transform`,
  `test_krylov`, `test_wave3_integration`, `test_analytic_distributed`,
  `test_eigenstates`, `test_balance_expansion`, `test_walled_eigenmodes`,
  `nonhydro2/test_transforms`, `nonhydro2/test_initial_conditions`,
  `shallowwater2/test_initial_conditions`).
  **Backend-specific bitwise/HLO pins tuned on CPU may legitimately
  differ on GPU** — the leg records every failure (`-rf`) for manual
  interpretation and does **not** gate the job red on the pytest rc.
- **Leg C — `srun -n 4`, real multi-process.** `srun_campaign.py`
  re-runs the smoke items sharded across 4 processes (one GPU each),
  gathers with `process_allgather(..., tiled=True)`, and rank 0
  compares vs the Leg-A references (device-count invariance to float64).
  This is where a jit close-over of a sharded constant, or a host-fetch
  of a sharded array, is caught (the forced-4 single-controller suite
  cannot see it).

### The five smoke items (Legs A and C)

Each mirrors an existing green test's public-surface setup; the public
consumer auto-routes through the fused `shard_map` regions when the grid
shards a transform axis and stays bit-identical eager on one device.

| item | path | mirrors |
|------|------|---------|
| a | Wave B walled-vertical projections + order-1 balance | `test_analytic_distributed.py`, `test_walled_eigenmodes.py`, `test_balance_expansion.py` |
| b | no-gather `random_state`/`mode` synthesis (periodic `hermitian_reframe` route + walled replicated fallback) | `test_eigenbasis_synthesis_distributed.py` |
| c | trig/mixed `ComposedTransform.apply_diagonal` Helmholtz apply | `test_mixed.py` |
| d | Tier-2 `SpectralSolve(allow_replicated=True)` escape **and** the taught-error guard on the unescaped path | `test_spectral_solve.py` |
| e | sw2 ETDRK4 run + scalar-amplitude grad (real `eigh` basis) | `test_exponential.py` |

## KNOWN LIMITATION — item e under real multi-process

Item **e** (ETDRK4) is validated single-process (Leg A writes its refs)
and single-controller (Leg B runs `test_exponential.py`'s sharded test
on 4 real GPUs). It is **skipped in Leg C**: the numeric channel
eigenbasis *builder* host-gathers its probe responses
(`fridom.model.eigen_channel._gathered_responses` uses `np.asarray` for
the host `eigh`), which is fine single-controller (the one process
addresses every device) but raises *"Fetching value for jax.Array that
spans non-addressable devices"* under a real `srun -n N` launch. So an
ETDRK4 model cannot be **built** under real multi-process today; only
its *application* (project/synthesize step halves) is device-invariant,
and that is what Leg B validates. Leg C detects and reports this as a
documented skip (it does not turn the leg red). Fixing the builder to
use `process_allgather` would let item e run in Leg C unchanged.

## How to submit (MANUAL ONLY)

**Owner ruling: agents never submit GPU jobs unless Silvano names the
run in chat.** Do not wire this into scrontab / cron / CI / any
auto-submitter. From the repo root:

```bash
sbatch design/research/artifacts/gspmd_campaign_gpu4/gspmd_campaign_check.sbatch
```

Outputs land beside this README: `legA_1gpu.txt`, `legB_4gpu_pytest.txt`,
`legC_srun4.txt`, the SLURM `gspmd-<jobid>.out`, and a per-leg verdict
in `RESULT.txt`. The `.npy` references are gitignored.

## Pre-flight (already run on the login node, CPU only)

- `bash -n gspmd_campaign_check.sbatch` — syntax OK.
- `smoke_refs.py --write-ref` on CPU (`JAX_PLATFORMS=cpu`,
  `OPENBLAS_NUM_THREADS=8`) — all five items compute, 38 refs written,
  exit 0.
- The multi-process logic on CPU with **2 processes** (explicit
  `jax.distributed.initialize`, `JAX_PLATFORMS=cpu`) — items a–d PASS
  vs the CPU refs (absmax ≤ 1.2e-14; the item-d unescaped guard raises
  the taught `NotImplementedError`), item e is the documented Leg-C
  skip. This is the strongest pre-flight the login node allows; the GPU
  legs (real 4-GPU `eigh`, real multi-process) are what this job adds.
