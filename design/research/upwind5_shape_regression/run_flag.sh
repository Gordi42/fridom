#!/bin/bash
# Flag-sensitivity: run w3(natural) vs w4(wide) under a given XLA_FLAGS,
# NPROC fresh processes each, fresh compile (cache disabled). A flag that
# closes the gap fingerprints the responsible pass.
# Usage: run_flag.sh <n> <scheme> <flagtag> <nproc> "<xla_flags>"
set -u
source "$(dirname "$0")/env.sh"
N=$1; SCHEME=$2; FLAGTAG=$3; NPROC=$4; XF="$5"
LOG="$HERE/log_flag_${FLAGTAG}.txt"

fg=$(gpu_foreign)
if [ -n "$fg" ]; then
  echo "GPU BUSY (pids: $fg) — refusing flag $FLAGTAG" | tee -a "$LOG"; exit 3
fi
echo "=== $(date +%H:%M:%S) flag=$FLAGTAG XLA_FLAGS='$XF' ===" | tee -a "$LOG"
for VAR in natural wide; do
  for i in $(seq 1 "$NPROC"); do
    fg=$(gpu_foreign)
    if [ -n "$fg" ]; then echo "BUSY mid (pids:$fg) abort" | tee -a "$LOG"; exit 3; fi
    JAX_COMPILATION_CACHE_DIR= XLA_FLAGS="$XF" \
      "$PY" "$HERE/rep_step.py" "$N" "$SCHEME" "$VAR" 5 >>"$LOG" 2>&1
    echo "  [$FLAGTAG $VAR proc$i clk:$(clocks)]" >>"$LOG"
  done
done
echo "=== done flag $FLAGTAG ===" | tee -a "$LOG"
grep REPRESULT "$LOG" | tail -n $((NPROC*2))
