#!/usr/bin/env bash
# =====================================================================
#  FRIDOM step-benchmark performance guard  (G3, perf_guard_plan.md)
# =====================================================================
#  A single, MANUALLY-SUBMITTED SLURM batch job for one DKRZ Levante
#  A100-80GB node. It runs the model step suite (benchmarks/model) on
#  1 GPU and on 4 GPUs, then compares each leg against the committed
#  baseline in benchmarks/baselines/ with --fail-on-regression, and
#  writes a GREEN/RED marker plus per-leg JSON and markdown reports
#  into the (gitignored) results dir.
#
#  MANUAL SUBMISSION ONLY (owner ruling, perf_guard_plan.md §5.1):
#      sbatch benchmarks/ci/step_guard.sbatch
#  Do NOT wire this into scrontab / cron / CI / any auto-submitter.
#  Unattended GPU submissions silently consume the project's cluster
#  usage limits; every run here is a deliberate human action. See
#  benchmarks/ci/README.md.
#
#  It is the pre-merge protocol for perf-sensitive merges (anything
#  touching step-path lowering); see the AGENTS.md merge-gate line.
# =====================================================================

# ---------------------------------------------------------------------
#  SLURM headers  (1 node, 4x A100-80GB, single process)
# ---------------------------------------------------------------------
#  Partition / GRES are confirmed from `sinfo` on Levante: partition
#  `gpu` carries `gpu:a100_80:4` (the 80GB A100 nodes; a100_40 also
#  lives in `gpu`, hence the explicit 80GB GRES so we land on the same
#  hardware the baselines were recorded on — device_kind
#  "NVIDIA A100-SXM4-80GB", node l50109.lvt.dkrz.de).
#SBATCH --job-name=fridom-step-guard
#SBATCH --partition=gpu
# Account confirmed 2026-07-18: jobs 26346485/26346504 ran and were
# accounted under uo0780_gpu.
#SBATCH --account=uo0780_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100_80:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
# Relative to the submit dir — submit from the repo root (see README)
# so the SLURM log lands in the retained results dir.
#SBATCH --output=benchmarks/results/step-guard-%j.out

set -euo pipefail

# ---------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------
#  REPO: repository root. Under sbatch this is SLURM_SUBMIT_DIR (the dir
#  the job was submitted from — submit from the repo root). For a local
#  DRY_RUN=1 invocation, fall back to the script's own location
#  (benchmarks/ci/ -> ../.. is the repo root).
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="${SLURM_SUBMIT_DIR:-$(cd "${_SCRIPT_DIR}/../.." && pwd)}"

# Where fresh JSON, markdown reports and the GREEN/RED marker land.
# Default benchmarks/results/ is gitignored; nothing here is committed.
RESULTS_DIR="${RESULTS_DIR:-${REPO}/benchmarks/results}"

BASELINE_DIR="${REPO}/benchmarks/baselines"
SUITE="${REPO}/benchmarks/model"

# Per-leg wall-clock ceiling (seconds) enforced with `timeout`, so a
# hung run cannot sit on the node until the SLURM wall limit.
LEG_TIMEOUT="${LEG_TIMEOUT:-7200}"

# 4-GPU runs MUST carry this flag until jax-ml/jax#39100 is fixed:
# without it XLA:GPU miscompiles the 4-device step (silently wrong
# physics, times fine). The flag is ~free. (benchmarks/README.md trap 3.)
XLA_FUSION_WORKAROUND="--xla_disable_hlo_passes=multi_output_fusion"

# Set DRY_RUN=1 to echo each command instead of running it (local
# sanity check of the script logic without touching SLURM / GPUs).
DRY_RUN="${DRY_RUN:-0}"

# One shared timestamp so every artifact from this run is grouped and
# NEVER collides with or overwrites a previous run's output (retention
# ruling, perf_guard_plan.md §5.4 — the script only ever creates files).
TIMESTAMP="$(date +%Y%m%dT%H%M%S)"

# ---------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------
log() { printf '[step-guard %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

# run_step "<env assignments>" <cmd...> : run one benchmark leg under a
# `timeout`, honouring DRY_RUN. Returns the command's exit status.
run_step() {
    local envs="$1"; shift
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY_RUN: ${envs} timeout ${LEG_TIMEOUT} $*"
        return 0
    fi
    # shellcheck disable=SC2086  # envs is a deliberate word list
    env ${envs} timeout "${LEG_TIMEOUT}" "$@"
}

cd "${REPO}"
mkdir -p "${RESULTS_DIR}"

log "repo         : ${REPO}"
log "results dir  : ${RESULTS_DIR}"
log "baselines    : ${BASELINE_DIR}"
log "timestamp    : ${TIMESTAMP}"
log "dry run      : ${DRY_RUN}"

# Aggregate status: stays 0 (GREEN) only if every leg ran AND every
# compare passed. Captured leg-by-leg so one failure never aborts the
# other leg or the marker write.
overall_rc=0

# ---------------------------------------------------------------------
#  Leg 1 — single GPU (gpu1)
# ---------------------------------------------------------------------
#  Mirrors the baseline-record invocation EXACTLY (benchmarks/README.md):
#      CUDA_VISIBLE_DEVICES=0 uv run python -m fridom.benchmarking \
#          run benchmarks/model -o <out>
#  Plain `run` with default reps/warmup — no CI smoke flags — so the
#  comparison is apples-to-apples with the committed baseline.
GPU1_JSON="${RESULTS_DIR}/step-gpu1-${TIMESTAMP}.json"
GPU1_REPORT="${RESULTS_DIR}/report-gpu1-${TIMESTAMP}.md"

log "=== leg 1/2: gpu1 (single device) ==="
set +e
run_step "JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0" \
    uv run python -m fridom.benchmarking run "${SUITE}" -o "${GPU1_JSON}"
gpu1_run_rc=$?
set -e
if [[ ${gpu1_run_rc} -ne 0 ]]; then
    log "gpu1 run FAILED (rc=${gpu1_run_rc}) — skipping its compare"
    overall_rc=1
else
    # Compare vs committed baseline. `--fail-on-regression` is the only
    # gate flag used, and it works identically before and after the
    # compare hardening landing on branch feat/bench-compare-hardening
    # (new env-guard / min / per-case-tolerance flags are opt-in; this
    # script deliberately does not depend on them).
    log "gpu1 compare vs ${BASELINE_DIR}/step-gpu1.json"
    set +e
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY_RUN: compare step-gpu1.json ${GPU1_JSON} --fail-on-regression --markdown | tee ${GPU1_REPORT}"
        gpu1_cmp_rc=0
    else
        uv run python -m fridom.benchmarking compare \
            "${BASELINE_DIR}/step-gpu1.json" "${GPU1_JSON}" \
            --fail-on-regression --markdown | tee "${GPU1_REPORT}"
        gpu1_cmp_rc=${PIPESTATUS[0]}
    fi
    set -e
    log "gpu1 compare rc=${gpu1_cmp_rc}"
    [[ ${gpu1_cmp_rc} -ne 0 ]] && overall_rc=1
fi

# ---------------------------------------------------------------------
#  Leg 2 — four GPUs, single process (gpu4)
# ---------------------------------------------------------------------
#  Mirrors the baseline-record invocation EXACTLY (benchmarks/README.md):
#      XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion \
#          uv run python -m fridom.benchmarking run benchmarks/model -o <out>
#  All four devices visible, single controller (GSPMD) — NOT srun -n 4.
GPU4_JSON="${RESULTS_DIR}/step-gpu4-${TIMESTAMP}.json"
GPU4_REPORT="${RESULTS_DIR}/report-gpu4-${TIMESTAMP}.md"

log "=== leg 2/2: gpu4 (four devices, single process) ==="
set +e
run_step "JAX_PLATFORMS=cuda XLA_FLAGS=${XLA_FUSION_WORKAROUND}" \
    uv run python -m fridom.benchmarking run "${SUITE}" -o "${GPU4_JSON}"
gpu4_run_rc=$?
set -e
if [[ ${gpu4_run_rc} -ne 0 ]]; then
    log "gpu4 run FAILED (rc=${gpu4_run_rc}) — skipping its compare"
    overall_rc=1
else
    log "gpu4 compare vs ${BASELINE_DIR}/step-gpu4.json"
    set +e
    if [[ "${DRY_RUN}" == "1" ]]; then
        log "DRY_RUN: compare step-gpu4.json ${GPU4_JSON} --fail-on-regression --markdown | tee ${GPU4_REPORT}"
        gpu4_cmp_rc=0
    else
        uv run python -m fridom.benchmarking compare \
            "${BASELINE_DIR}/step-gpu4.json" "${GPU4_JSON}" \
            --fail-on-regression --markdown | tee "${GPU4_REPORT}"
        gpu4_cmp_rc=${PIPESTATUS[0]}
    fi
    set -e
    log "gpu4 compare rc=${gpu4_cmp_rc}"
    [[ ${gpu4_cmp_rc} -ne 0 ]] && overall_rc=1
fi

# ---------------------------------------------------------------------
#  Verdict marker  (never deletes / overwrites prior runs)
# ---------------------------------------------------------------------
if [[ ${overall_rc} -eq 0 ]]; then
    MARKER="${RESULTS_DIR}/step-guard-${TIMESTAMP}.GREEN"
    log "RESULT: GREEN — no regression against committed baselines"
else
    MARKER="${RESULTS_DIR}/step-guard-${TIMESTAMP}.RED"
    log "RESULT: RED — a leg failed or a case regressed; investigate"
fi

if [[ "${DRY_RUN}" == "1" ]]; then
    log "DRY_RUN: would write marker ${MARKER}"
else
    {
        echo "timestamp      : ${TIMESTAMP}"
        echo "overall_rc     : ${overall_rc}"
        echo "gpu1 json      : ${GPU1_JSON}"
        echo "gpu1 report    : ${GPU1_REPORT}"
        echo "gpu4 json      : ${GPU4_JSON}"
        echo "gpu4 report    : ${GPU4_REPORT}"
    } > "${MARKER}"
    log "marker written : ${MARKER}"
fi

exit "${overall_rc}"
