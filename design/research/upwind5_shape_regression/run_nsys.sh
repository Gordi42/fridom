#!/bin/bash
# nsys-profile one advance() region of prof_step.py (compile excluded via
# cudaProfilerApi capture range), then emit a per-kernel GPU summary.
# Usage: run_nsys.sh <n> <scheme> <variant> <n_prof_steps> <tag>
set -u
source "$(dirname "$0")/env.sh"
N=$1; SCHEME=$2; VAR=$3; NPROF=$4; TAG=$5
OUT="$HERE/nsys_${TAG}"
LOG="$HERE/log_nsys_${TAG}.txt"

fg=$(gpu_foreign)
if [ -n "$fg" ]; then
  echo "GPU BUSY (pids: $fg) — refusing nsys $TAG" | tee -a "$LOG"
  exit 3
fi

echo "=== $(date +%H:%M:%S) nsys $TAG n=$N scheme=$SCHEME var=$VAR nprof=$NPROF ===" | tee "$LOG"
echo "clocks_before: $(clocks)" | tee -a "$LOG"
"$NSYS" profile \
  --trace=cuda \
  --cuda-graph-trace=node \
  --capture-range=cudaProfilerApi \
  --capture-range-end=stop \
  --cuda-memory-usage=false \
  --force-overwrite=true \
  -o "$OUT" \
  "$PY" "$HERE/prof_step.py" "$N" "$SCHEME" "$VAR" "$NPROF" >>"$LOG" 2>&1
rc=$?
echo "nsys rc=$rc clocks_after: $(clocks)" | tee -a "$LOG"
if [ $rc -ne 0 ]; then echo "NSYS FAILED rc=$rc"; exit $rc; fi

echo "=== kernel summary $TAG (table) ===" | tee -a "$LOG"
"$NSYS" stats --report cuda_gpu_kern_sum --format table "$OUT.nsys-rep" 2>>"$LOG" \
  | tee "$HERE/kern_${TAG}.txt"
echo "=== kernel summary $TAG (csv) ===" >>"$LOG"
"$NSYS" stats --report cuda_gpu_kern_sum --format csv --output - "$OUT.nsys-rep" \
  > "$HERE/kern_${TAG}.csv" 2>>"$LOG"
echo "=== also cuda_gpu_trace csv (per-launch grid/block) ===" >>"$LOG"
"$NSYS" stats --report cuda_gpu_trace --format csv --output - "$OUT.nsys-rep" \
  > "$HERE/trace_${TAG}.csv" 2>>"$LOG"
echo "n_prof_steps=$NPROF total_steps=$((NPROF*50))" > "$HERE/meta_${TAG}.txt"
