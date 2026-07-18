#!/bin/bash
# Run N fresh-process reps of rep_step.py for a (n,scheme,variant).
# Guards on a free GPU up-front; logs SM clocks around each process.
# Usage: run_repro.sh <n> <scheme> <variant> <reps> <nproc> <logtag>
set -u
source "$(dirname "$0")/env.sh"
N=$1; SCHEME=$2; VAR=$3; REPS=$4; NPROC=$5; TAG=$6
LOG="$HERE/log_repro_${TAG}.txt"

fg=$(gpu_foreign)
if [ -n "$fg" ]; then
  echo "GPU BUSY (pids: $fg) — refusing to run $TAG" | tee -a "$LOG"
  exit 3
fi

echo "=== $(date +%H:%M:%S) repro n=$N scheme=$SCHEME var=$VAR reps=$REPS nproc=$NPROC ===" | tee -a "$LOG"
for i in $(seq 1 "$NPROC"); do
  fg=$(gpu_foreign)
  if [ -n "$fg" ]; then
    echo "GPU BUSY mid-batch (pids: $fg) — aborting after proc $((i-1))" | tee -a "$LOG"
    exit 3
  fi
  echo "--- proc $i clocks_before: $(clocks) ---" | tee -a "$LOG"
  "$PY" "$HERE/rep_step.py" "$N" "$SCHEME" "$VAR" "$REPS" >>"$LOG" 2>&1
  rc=$?
  echo "--- proc $i rc=$rc clocks_after: $(clocks) ---" | tee -a "$LOG"
  if [ $rc -ne 0 ]; then echo "PROC $i FAILED rc=$rc" | tee -a "$LOG"; fi
done
echo "=== done $TAG ===" | tee -a "$LOG"
