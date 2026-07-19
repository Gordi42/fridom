#!/usr/bin/env bash
# =====================================================================
#  GSPMD illegality campaign: GPU validation checkpoint (owner-batched)
# =====================================================================
#  A single, MANUALLY-SUBMITTED SLURM batch job for one DKRZ Levante
#  A100-80GB node. It validates the naive-GSPMD-transform illegality
#  campaign's multi-device paths (design/plans/done/
#  gspmd_transform_illegality_plan.md; roadmap/open.md "Naive GSPMD
#  transform path — GPU checkpoint") in three sequential legs:
#
#    Leg A (1 GPU, 1 process): write the 1-device .npy references for
#      the five campaign smoke items (smoke_refs.py --write-ref).
#    Leg B (4 GPUs, 1 process, single-controller GSPMD): pytest over the
#      campaign's fused-synthesis / ETDRK4 / consumer test files with
#      real eigh bases (CPU-unsafe, jax#39292) on real 4 GPUs. Records
#      results — backend-specific bitwise/HLO pins tuned on CPU may
#      legitimately differ on GPU; the leg RECORDS, it does not mask.
#    Leg C (srun -n 4, real multi-process): re-run the smoke items
#      sharded across 4 processes, gather with process_allgather, and
#      compare vs the Leg-A references (the forced-4 single-controller
#      suite cannot catch jit close-over of sharded constants).
#
#  Each leg writes its own log in the artifacts dir and appends a
#  per-leg verdict line to RESULT.txt.
#
#  MANUAL SUBMISSION ONLY (owner ruling — agents never submit GPU jobs
#  unless Silvano names the run in chat):
#      sbatch design/research/artifacts/gspmd_campaign_gpu4/gspmd_campaign_check.sbatch
#  Do NOT wire this into scrontab / cron / CI / any auto-submitter.
# =====================================================================

# ---------------------------------------------------------------------
#  SLURM headers  (1 node, 4x A100-80GB, four tasks for srun -n 4)
# ---------------------------------------------------------------------
#SBATCH --job-name=fridom-gspmd-chk
#SBATCH --partition=gpu
#SBATCH --account=uo0780_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:a100_80:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
# Relative to the submit dir — submit from the repo root so the SLURM
# log lands in the retained artifacts dir.
#SBATCH --output=design/research/artifacts/gspmd_campaign_gpu4/gspmd-%j.out

set -uo pipefail

REPO="/work/uo0780/u301533/fridom/fridom-dev"
ART="${REPO}/design/research/artifacts/gspmd_campaign_gpu4"
PY="${REPO}/.venv/bin/python"   # absolute venv python; NEVER `uv run`
                                # (uv can strip the cuda extra mid-run)

cd "${REPO}"

# Mandatory on every 4-GPU new-stack run until jax-ml/jax#39100 is
# fixed: without it XLA:GPU miscompiles the multi-device step (silently
# wrong physics, times fine). The flag is ~free.
export XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion
# Cheap safety for any host-side eigh (jax#39292); harmless on GPU.
export OPENBLAS_NUM_THREADS=8

log() { printf '[gspmd-chk %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "repo        : ${REPO}"
log "artifacts   : ${ART}"
: > "${ART}/RESULT.txt"   # fresh verdict file for this job

# ---------------------------------------------------------------------
#  Leg A — single GPU: write the .npy references
# ---------------------------------------------------------------------
#  Do NOT set FRIDOM_JAX_CACHE_DIR: fridom auto-disables its persistent
#  compile cache under a real multi-process launch (Leg C); enabling one
#  would require --xla_gpu_shard_autotuning=false — leave all that alone.
log "=== leg A: gpu1 reference (--write-ref) ==="
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 timeout 1200 \
    "${PY}" "${ART}/smoke_refs.py" --write-ref \
    |& tee "${ART}/legA_1gpu.txt"
legA_rc=${PIPESTATUS[0]}
log "leg A rc=${legA_rc}"
if [[ ${legA_rc} -eq 0 ]]; then
    echo "leg A (1-GPU references): PASS (rc=0)" >> "${ART}/RESULT.txt"
else
    echo "leg A (1-GPU references): FAIL (rc=${legA_rc})" >> "${ART}/RESULT.txt"
fi

# ---------------------------------------------------------------------
#  Leg B — 4 GPUs, single process: the campaign test files on real GPUs
# ---------------------------------------------------------------------
#  Single-controller GSPMD: one process owns all 4 GPUs, so device_count
#  == 4 (multi_device tests run, single_device skip) and host fetches
#  just work. Serial (NO xdist — one process, all 4 GPUs). `-rf` lists
#  every failure; backend-specific bitwise/HLO pins tuned on CPU may
#  legitimately differ on GPU — this leg RECORDS them for manual
#  interpretation and does NOT gate the job red on the pytest rc.
log "=== leg B: 4-GPU single-controller pytest (campaign files) ==="
LEGB_FILES=(
    tests/spatial/operators/test_distributed_contract.py
    tests/spatial/operators/test_distributed_contract_synthesis.py
    tests/spatial/operators/test_distributed_contract_channel2d.py
    tests/model/test_eigenbasis.py
    tests/model/test_eigenbasis_synthesis_distributed.py
    tests/model/time_steppers/test_exponential.py
    tests/spatial/operators/test_transform.py
    tests/spatial/operators/test_spectral_solve.py
    tests/spatial/operators/test_mixed.py
    tests/spatial/operators/test_distributed_solve.py
    tests/spatial/operators/test_distributed_transform.py
    tests/spatial/operators/test_krylov.py
    tests/spatial/test_wave3_integration.py
    tests/model/test_analytic_distributed.py
    tests/model/test_eigenstates.py
    tests/model/transforms/test_balance_expansion.py
    tests/nonhydro2/test_walled_eigenmodes.py
    tests/nonhydro2/test_transforms.py
    tests/nonhydro2/test_initial_conditions.py
    tests/shallowwater2/test_initial_conditions.py
)
JAX_PLATFORMS=cuda timeout 2400 \
    "${PY}" -m pytest "${LEGB_FILES[@]}" -rf --tb=line -q \
    |& tee "${ART}/legB_4gpu_pytest.txt"
legB_rc=${PIPESTATUS[0]}
log "leg B pytest rc=${legB_rc}"
# Summarize (never mask): record the rc and the pytest summary tail.
legB_summary=$(grep -E \
    "passed|failed|error|skipped" "${ART}/legB_4gpu_pytest.txt" | tail -1)
echo "leg B (4-GPU pytest): rc=${legB_rc} — ${legB_summary:-no summary} " \
     "(backend-specific bitwise/HLO diffs expected; interpret the log)" \
     >> "${ART}/RESULT.txt"

# ---------------------------------------------------------------------
#  Leg C — real multi-process (srun -n 4): sharded vs the Leg-A ref
# ---------------------------------------------------------------------
#  `timeout` is mandatory: a dead rank leaves the others blocked at the
#  coordination barrier until the SLURM wall limit. `--gpu-bind=none`
#  (NOT --gpus-per-task=1) so every task sees all GPUs and
#  jax.distributed.initialize() assigns one GPU per local rank.
log "=== leg C: srun -n 4 (real multi-process invariance) ==="
JAX_PLATFORMS=cuda timeout 1800 \
    srun -n 4 --gpu-bind=none \
    "${PY}" "${ART}/srun_campaign.py" \
    |& tee "${ART}/legC_srun4.txt"
legC_rc=${PIPESTATUS[0]}
log "leg C rc=${legC_rc}"
if [[ ${legC_rc} -eq 0 ]] && grep -qx "PASS" "${ART}/legC_srun4.txt"; then
    legC_skips=$(grep -E "^SKIPPED" "${ART}/legC_srun4.txt" | tail -1)
    echo "leg C (srun -n 4 invariance): PASS — ${legC_skips:-no skips}" \
        >> "${ART}/RESULT.txt"
else
    echo "leg C (srun -n 4 invariance): FAIL (rc=${legC_rc}; no PASS line)" \
        >> "${ART}/RESULT.txt"
fi

# ---------------------------------------------------------------------
#  Verdicts
# ---------------------------------------------------------------------
log "=== RESULT.txt ==="
cat "${ART}/RESULT.txt"
log "leg A log   : ${ART}/legA_1gpu.txt"
log "leg B log   : ${ART}/legB_4gpu_pytest.txt"
log "leg C log   : ${ART}/legC_srun4.txt"
