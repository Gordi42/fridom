#!/bin/bash
# sweep.sh — u5_baseline (primary) + centered (control) across flagsets.
set -u
SP=/tmp/claude-1000/-home-silvano-Projects-fridom/84ef5624-971a-4abb-a5b0-6cbecbfad680/scratchpad
R=$SP/flag_sweep/run_flag.sh
LOG=$SP/flag_sweep/sweep_log.txt
: > "$LOG"

# label|flags
declare -a SETS=(
  "baseline|"
  "f1_dblbuf|--xla_gpu_enable_while_loop_double_buffering=true"
  "f2_nomof|--xla_disable_hlo_passes=multi_output_fusion"
  "f3_lhs|--xla_gpu_enable_latency_hiding_scheduler=true"
  "f4_slop95|--xla_gpu_memory_limit_slop_factor=95"
  "f5_autotune4|--xla_gpu_autotune_level=4"
  "f6_notriton|--xla_gpu_enable_triton_gemm=false"
  "x1_nocmdbuf|--xla_gpu_enable_command_buffer="
  "x2_aliasscope|--xla_llvm_enable_alias_scope_metadata=true"
)

for variant in u5_baseline centered; do
  for entry in "${SETS[@]}"; do
    label=${entry%%|*}
    flags=${entry#*|}
    # skip u5 baseline (already ran) unless missing
    if [ "$variant" = "u5_baseline" ] && [ "$label" = "baseline" ] \
       && [ -f "$SP/flag_sweep/u5_baseline_n160_baseline.json" ]; then
      echo "skip u5 baseline (already have it)" | tee -a "$LOG"; continue
    fi
    bash "$R" "$variant" 160 "$label" "$flags" 2>&1 | tee -a "$LOG"
    echo "" | tee -a "$LOG"
  done
done
echo "###### SWEEP_DONE ######" | tee -a "$LOG"
