#!/usr/bin/env bash
# =====================================================================
#  Mapped + immersed composition: real-multi-process (srun -n 4) check
# =====================================================================
#  A single, MANUALLY-SUBMITTED SLURM batch job for one DKRZ Levante
#  A100-80GB node. It validates the mapped + immersed composed-grid
#  model legs under a REAL multi-process launch (N processes, one GPU
#  each, arrays sharded across processes) against a 1-GPU reference:
#
#    Leg A (gpu1): write the .npy reference on a single GPU.
#    Leg B (srun -n 4): re-run both legs sharded across 4 processes,
#      gather the finals, compare vs the Leg-A reference, print PASS.
#
#  MANUAL SUBMISSION ONLY (owner ruling — agents never submit GPU jobs
#  unless Silvano names the run in chat):
#      sbatch design/research/artifacts/mapped_immersed_gpu4/mapped_immersed_check.sbatch
#  Do NOT wire this into scrontab / cron / CI / any auto-submitter.
# =====================================================================

# ---------------------------------------------------------------------
#  SLURM headers  (1 node, 4x A100-80GB, four tasks for srun -n 4)
# ---------------------------------------------------------------------
#  Partition / GRES per benchmarks/ci/step_guard.sbatch: partition `gpu`
#  carries `gpu:a100_80:4`; account uo0780_gpu.
#SBATCH --job-name=fridom-mi-srun
#SBATCH --partition=gpu
#SBATCH --account=uo0780_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:a100_80:4
#SBATCH --exclusive
#SBATCH --time=00:45:00
# Relative to the submit dir — submit from the repo root so the SLURM
# log lands in the retained artifacts dir.
#SBATCH --output=design/research/artifacts/mapped_immersed_gpu4/mi-srun-%j.out

set -uo pipefail

REPO="/work/uo0780/u301533/fridom/fridom-dev"
ART="${REPO}/design/research/artifacts/mapped_immersed_gpu4"
PY="${REPO}/.venv/bin/python"   # absolute venv python; NEVER `uv run`
                                # (uv can strip the cuda extra mid-run)

cd "${REPO}"

# Mandatory on every 4-GPU new-stack run until jax-ml/jax#39100 is
# fixed: without it XLA:GPU miscompiles the multi-device step (silently
# wrong physics, times fine). The flag is ~free.
export XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion

log() { printf '[mi-srun %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "repo        : ${REPO}"
log "artifacts   : ${ART}"

# ---------------------------------------------------------------------
#  Leg A — single GPU: write the .npy reference
# ---------------------------------------------------------------------
#  Do NOT set FRIDOM_JAX_CACHE_DIR: fridom auto-disables its persistent
#  compile cache under a real multi-process launch (Leg B); enabling one
#  would require --xla_gpu_shard_autotuning=false — leave all that alone.
log "=== leg A: gpu1 reference (--write-ref) ==="
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 timeout 1200 \
    "${PY}" "${ART}/smoke_mapped_immersed.py" --write-ref \
    |& tee "${ART}/legA_1gpu.txt"
legA_rc=${PIPESTATUS[0]}
log "leg A rc=${legA_rc}"

# ---------------------------------------------------------------------
#  Leg B — real multi-process (srun -n 4): compare vs the Leg-A ref
# ---------------------------------------------------------------------
#  `timeout` is mandatory: a dead rank leaves the others blocked at the
#  coordination barrier until the SLURM wall limit. `--gpu-bind=none`
#  (NOT --gpus-per-task=1) so every task sees all GPUs and
#  jax.distributed.initialize() assigns one GPU per local rank.
log "=== leg B: srun -n 4 (sharded invariance check) ==="
JAX_PLATFORMS=cuda timeout 1200 \
    srun -n 4 --gpu-bind=none \
    "${PY}" "${ART}/srun_mapped_immersed.py" \
    |& tee "${ART}/legB_srun4.txt"
legB_rc=${PIPESTATUS[0]}
log "leg B rc=${legB_rc}"

# ---------------------------------------------------------------------
#  Verdict marker
# ---------------------------------------------------------------------
#  GREEN only if Leg B ran AND printed a PASS line; RED otherwise.
if [[ ${legB_rc} -eq 0 ]] && grep -qx "PASS" "${ART}/legB_srun4.txt"; then
    echo "GREEN" > "${ART}/RESULT.txt"
    log "RESULT: GREEN — sharded run matches the 1-GPU reference"
else
    echo "RED" > "${ART}/RESULT.txt"
    log "RESULT: RED — leg B failed, timed out, or did not print PASS"
fi

log "leg A log   : ${ART}/legA_1gpu.txt"
log "leg B log   : ${ART}/legB_srun4.txt"
log "result      : ${ART}/RESULT.txt"
