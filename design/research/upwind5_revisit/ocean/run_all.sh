#!/usr/bin/env bash
# Driver: wait for GPU to be free, then run each (scheme,n) in a fresh
# julia process, logging nvidia-smi clocks/temp/power before and after.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
LOG="$HERE/results/thermal_log.txt"
: > "$LOG"

gpu_state() {
    nvidia-smi --query-gpu=clocks.sm,temperature.gpu,power.draw --format=csv,noheader
}

wait_for_free_gpu() {
    # require 2 consecutive minutes empty; poll every 30s; bail after ~50 min
    local empty=0 waited=0
    while true; do
        apps=$(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader)
        if [ -z "$apps" ]; then
            empty=$((empty+1))
            if [ "$empty" -ge 5 ]; then return 0; fi  # 5*30s=2.5min buffer
        else
            empty=0
        fi
        if [ "$waited" -ge 3000 ]; then
            echo "WARN: waited >50min, proceeding anyway" | tee -a "$LOG"
            return 0
        fi
        sleep 30
        waited=$((waited+30))
    done
}

run_case() {
    local scheme="$1" n="$2" label="${3:-}"
    wait_for_free_gpu
    echo "=== $scheme n$n ${label} BEFORE: $(gpu_state)" | tee -a "$LOG"
    BENCH_LABEL="$label" julia --project="$HERE/ocean" "$HERE/bench.jl" "$scheme" "$n" 2>&1 \
        | tee -a "$HERE/results/run_${scheme}_n${n}${label:+_$label}.log"
    echo "=== $scheme n$n ${label} AFTER:  $(gpu_state)" | tee -a "$LOG"
    echo "" | tee -a "$LOG"
}

for scheme in centered upwind5 weno5; do
    for n in 96 128 160; do
        run_case "$scheme" "$n"
    done
done
# n=192 only for centered & upwind5 (memory)
for scheme in centered upwind5; do
    run_case "$scheme" 192
done
# drift check: re-run n=128 upwind5
run_case upwind5 128 drift

echo "ALL DONE" | tee -a "$LOG"
