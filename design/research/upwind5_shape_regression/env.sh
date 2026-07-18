# Common env for the GPU A/B runs in this investigation. Source me.
# Portable: derives paths from this file's location. Build the venv with
# `uv sync --extra dev --extra cuda` so jax has the CUDA plugin.
export CUDA_VISIBLE_DEVICES=0
export JAX_PLATFORMS=cuda
export XLA_PYTHON_CLIENT_PREALLOCATE=false

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# repo root = three levels up from design/research/upwind5_shape_regression
WT="$(cd "$HERE/../../.." && pwd)"
PY="$WT/.venv/bin/python"
# Persistent HLO-keyed compile cache so repeated fresh processes of the
# same shape skip recompilation (does not affect step timing).
export JAX_COMPILATION_CACHE_DIR="${FRIDOM_AB_CACHE:-/tmp/upwind5_shape_jaxcache}"
# nsys 2026.1 (cuda-13.3 module on Levante); override NSYS as needed.
NSYS="${NSYS:-/sw/spack-levante/cuda-13.3.0-mem3z7/bin/nsys}"

gpu_foreign() {
  # Print any compute-app PIDs. Empty => free.
  nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '
}

clocks() {
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv,noheader 2>/dev/null
}
