#!/bin/bash
set -e
SCRATCH=/tmp/claude-1000/-home-silvano-Projects-fridom/84ef5624-971a-4abb-a5b0-6cbecbfad680/scratchpad
WT=$SCRATCH/wt-attrib/src
VENV=$SCRATCH/venv-gpu/bin/python
cd $SCRATCH/fridom_bench
prune () {
  D=$1
  find $D -type f ! -name '*chunk_body.sm_8.6_gpu_after_optimizations-buffer-assignment.txt' \
    ! -name '*chunk_body.sm_8.6_gpu_after_optimizations-memory-usage-report.txt' \
    ! -name '*chunk_body.sm_8.6_gpu_after_optimizations-live-range.txt' \
    ! -name '*chunk_body.sm_8.6_gpu_after_optimizations.txt' \
    ! -name '*chunk_body.thunk_sequence.txt' -delete
}
run () {
  V=$1; N=$2
  D=$SCRATCH/fridom_bench/attribution/dump/${V}_n${N}
  mkdir -p $D
  echo ">>> $V n$N  $(date +%T)"
  JAX_PLATFORMS=cuda WT_SRC=$WT PYTHONPATH=$WT \
    XLA_FLAGS="--xla_dump_to=$D --xla_dump_hlo_as_text" \
    $VENV attribution/dump_variant.py $V $N 2>&1 | grep -E 'OK fridom|temp=|ABORT|Error|Traceback' || true
  prune $D
  echo "    pruned -> $(du -sh $D | cut -f1)"
}
run centered 192
run u5_baseline 128
run u5_baseline 192
run u5_selected 128
echo ">>> ALL DONE  $(date +%T)"
