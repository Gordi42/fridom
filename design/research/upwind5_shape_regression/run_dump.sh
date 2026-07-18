#!/bin/bash
# Dump optimized HLO (+ LLVM/PTX metadata) for the step executable at a
# given storage width. Large; scratch only. GPU-touching (autotune runs
# on device) so it needs the GPU free.
# Usage: run_dump.sh <n> <scheme> <variant> <tag>
set -u
source "$(dirname "$0")/env.sh"
N=$1; SCHEME=$2; VAR=$3; TAG=$4
DUMP="$HERE/dump_${TAG}"
LOG="$HERE/log_dump_${TAG}.txt"
rm -rf "$DUMP"; mkdir -p "$DUMP"

fg=$(gpu_foreign)
if [ -n "$fg" ]; then
  echo "GPU BUSY (pids: $fg) — refusing dump $TAG" | tee -a "$LOG"
  exit 3
fi

# Fresh compile: bypass the persistent cache so XLA re-runs optimization
# and dumps. Keep n_prof tiny (1) — we only need the compiled module.
echo "=== $(date +%H:%M:%S) dump $TAG n=$N scheme=$SCHEME var=$VAR ===" | tee "$LOG"
JAX_COMPILATION_CACHE_DIR= \
XLA_FLAGS="--xla_dump_to=$DUMP --xla_dump_hlo_as_text --xla_dump_hlo_as_long_text" \
  "$PY" "$HERE/prof_step.py" "$N" "$SCHEME" "$VAR" 1 >>"$LOG" 2>&1
rc=$?
echo "dump rc=$rc" | tee -a "$LOG"
echo "modules dumped:" | tee -a "$LOG"
ls -la "$DUMP" | grep after_optimizations | tee -a "$LOG"
