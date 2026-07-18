#!/bin/bash
# Test the launch-granularity divisibility mechanism at 192^3 upwind5.
# Prediction: storage divisible by 8 per axis (flat divisible by 2^9) is
# fast; otherwise slow. Widths -> storage: w3=198 w4=200(fast) w5=202
# w6=204 w7=206 w8=208. 200 and 208 are the div-by-8 shapes -> predict
# BOTH fast; the rest slow. Bytes rise monotonically, so a fast 208
# (more bytes than slow 204) refutes any byte/alignment story.
set -u
source "$(dirname "$0")/env.sh"
LOG="$HERE/log_divtest.txt"
fg=$(gpu_foreign)
if [ -n "$fg" ]; then echo "GPU BUSY ($fg)"|tee -a "$LOG"; exit 3; fi
echo "=== $(date +%H:%M:%S) divtest 192^3 upwind5 ===" | tee -a "$LOG"
# each width via uniform per-axis floor wX,X,X (2 procs)
for W in 3 4 5 6 7 8; do
  for i in 1 2; do
    fg=$(gpu_foreign)
    if [ -n "$fg" ]; then echo "BUSY mid ($fg)"|tee -a "$LOG"; exit 3; fi
    "$PY" "$HERE/rep_step.py" 192 upwind5 "w$W,$W,$W" 5 >>"$LOG" 2>&1
  done
done
echo "=== divtest done ===" | tee -a "$LOG"
grep REPRESULT "$LOG" | tail -12
