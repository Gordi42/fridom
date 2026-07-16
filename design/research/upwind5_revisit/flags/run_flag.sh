#!/bin/bash
# run_flag.sh <variant> <n> <label> <XLA_FLAGS-string>
# One fresh-process timed run (1 warmup + 4 timed chunks of 50) under a
# given XLA_FLAGS string, thermal-logged, JSON tagged by label.
set -u
SP=/tmp/claude-1000/-home-silvano-Projects-fridom/84ef5624-971a-4abb-a5b0-6cbecbfad680/scratchpad
PY=$SP/venv-gpu/bin/python
HARNESS=$SP/fridom_bench/p3b_model.py
WT=$SP/wt-flags/src
OUT=$SP/flag_sweep
RES=$SP/fridom_bench/results
mkdir -p "$OUT"

variant=$1; n=$2; label=$3; flags=$4
smi() { nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw,utilization.gpu \
        --format=csv,noheader | tr '\n' ' '; }

echo "=== [$label] $variant n=$n  FLAGS='$flags'"
echo "    BEFORE :: $(smi)"
PYTHONPATH=$WT JAX_PLATFORMS=cuda JAX_ENABLE_X64=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="$flags" \
  timeout 900 "$PY" "$HARNESS" time --variant "$variant" --n "$n" --chunks 4 \
  2>&1 | grep -Ev "UserWarning|will be truncated|jax_enable_x64|github.com/jax-ml"
rc=${PIPESTATUS[0]}
echo "    AFTER  :: $(smi)  [rc=$rc]"
if [ -f "$RES/p3b_time_${variant}_n${n}.json" ]; then
  cp "$RES/p3b_time_${variant}_n${n}.json" \
     "$OUT/${variant}_n${n}_${label}.json"
  echo "    -> $OUT/${variant}_n${n}_${label}.json"
fi
echo "RC=$rc"
exit $rc
