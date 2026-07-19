#!/usr/bin/env bash
# =====================================================================
#  Multigrid coarse-grid AGGLOMERATION — Phase 3 (4x A100 wall-clock)
# =====================================================================
#  ONE manually-submitted SLURM job (owner-authorized in chat,
#  2026-07-19). Single-process GSPMD over 4 A100-80GB GPUs. Measures
#  the GB-2 steep-mapped + immersed cases: spectral vs mg-off vs
#  mg-agglomerate(tau in {2,4,8}) ms/step + CG iterations, plus a GPU
#  HLO collective census (mg-off vs mg-t4) at n=128.
#
#  Source is PINNED to a detached worktree (src-pin) so a concurrent
#  edit of the main checkout cannot change what runs. The shared .venv
#  is editable-installed against the MAIN checkout, so PYTHONPATH forces
#  the pinned tree to the front of sys.path; the driver hard-fails if
#  fridom does not resolve under src-pin.
# =====================================================================
#SBATCH --job-name=agglom-phase3
#SBATCH --partition=gpu
#SBATCH --account=uo0780_gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100_80:4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --output=/work/uo0780/u301533/fridom/agglom_phase3/agglom-phase3-%j.out
#SBATCH --error=/work/uo0780/u301533/fridom/agglom_phase3/agglom-phase3-%j.err

set -uo pipefail

JOBDIR=/work/uo0780/u301533/fridom/agglom_phase3
SRC_PIN="${JOBDIR}/src-pin"
PY=/work/uo0780/u301533/fridom/fridom-dev/.venv/bin/python

# Pinned source to the FRONT of sys.path (venv editable install points at
# the main checkout; PYTHONPATH wins). Driver asserts fridom under src-pin.
export PYTHONPATH="${SRC_PIN}/src"
export JAX_PLATFORMS=cuda
# MANDATORY on every 4-GPU run (jax#39100 silent miscompile). ~free.
FUSION="--xla_disable_hlo_passes=multi_output_fusion"
export XLA_FLAGS="${FUSION}"

# Per-config wall ceiling so one hung/OOM config cannot sit on the node.
CFG_TIMEOUT="${CFG_TIMEOUT:-1800}"

log() { printf '[agglom %s] %s\n' "$(date +%H:%M:%S)" "$*"; }

log "jobdir     : ${JOBDIR}"
log "src-pin    : ${SRC_PIN}"
log "pinned SHA : $(cat "${JOBDIR}/PINNED_SHA.txt")"
log "python     : ${PY}"
log "PYTHONPATH : ${PYTHONPATH}"
log "XLA_FLAGS  : ${XLA_FLAGS}"
log "SLURM job  : ${SLURM_JOB_ID:-<none>} on $(hostname)"
nvidia-smi -L || true

cd "${JOBDIR}"

# run_cfg <case> <n> <tag> : one config, its own process, never aborts
# the loop. Timing survives even if the iters build OOMs (driver order).
run_cfg() {
    local case="$1" n="$2" tag="$3"
    log ">>> config case=${case} n=${n} tag=${tag}"
    local t0=$SECONDS
    timeout "${CFG_TIMEOUT}" "${PY}" "${JOBDIR}/driver.py" "${case}" "${n}" "${tag}"
    local rc=$?
    log "<<< config case=${case} n=${n} tag=${tag} rc=${rc} dt=$((SECONDS-t0))s"
    return 0
}

# ---------------------------------------------------------------------
#  (1) n=128 mapped — cheapest first
# ---------------------------------------------------------------------
for tag in spectral mg-off mg-t2 mg-t4 mg-t8; do
    run_cfg mapped 128 "${tag}"
done

# ---------------------------------------------------------------------
#  (2) GPU HLO collective census at n=128 (mg-off AND mg-t4)
# ---------------------------------------------------------------------
census_one() {
    local tag="$1"
    local ddir="${JOBDIR}/hlo_${tag}_n128"
    rm -rf "${ddir}"; mkdir -p "${ddir}"
    log ">>> HLO dump case=mapped n=128 tag=${tag} -> ${ddir}"
    XLA_FLAGS="${FUSION} --xla_dump_to=${ddir} --xla_dump_hlo_as_text" \
        timeout "${CFG_TIMEOUT}" "${PY}" "${JOBDIR}/hlo_dump.py" 128 "${tag}" \
        > "${JOBDIR}/hlo_dump_${tag}_n128.log" 2>&1
    log "<<< HLO dump tag=${tag} rc=$?"
    # biggest after-optimizations module text
    local biggest
    biggest="$(ls -S "${ddir}"/*after_optimizations*.txt 2>/dev/null | head -1)"
    if [[ -z "${biggest}" ]]; then
        log "!!! no after_optimizations HLO for tag=${tag} (see dir listing)"
        ls -la "${ddir}" | head -40 >> "${JOBDIR}/census_${tag}_n128.txt"
        return 0
    fi
    log "census: parsing ${biggest}"
    {
        echo "# census tag=${tag} n=128 module=${biggest}"
        echo "# $(ls -la "${biggest}")"
        "${PY}" "${JOBDIR}/census.py" "${biggest}"
    } > "${JOBDIR}/census_${tag}_n128.txt" 2>&1
    log "census: wrote ${JOBDIR}/census_${tag}_n128.txt"
}
census_one mg-off
census_one mg-t4

# ---------------------------------------------------------------------
#  (3) n=256 mapped
# ---------------------------------------------------------------------
for tag in spectral mg-off mg-t2 mg-t4 mg-t8; do
    run_cfg mapped 256 "${tag}"
done

# ---------------------------------------------------------------------
#  (4) n=512 mapped
# ---------------------------------------------------------------------
for tag in spectral mg-off mg-t2 mg-t4 mg-t8; do
    run_cfg mapped 512 "${tag}"
done

# ---------------------------------------------------------------------
#  (5) immersed case — n=128 and n=256, {spectral, mg-off, mg-t4}
# ---------------------------------------------------------------------
for n in 128 256; do
    for tag in spectral mg-off mg-t4; do
        run_cfg immersed "${n}" "${tag}"
    done
done

log "ALL CONFIGS SUBMITTED — job complete"
log "results.jsonl: ${JOBDIR}/results.jsonl"
