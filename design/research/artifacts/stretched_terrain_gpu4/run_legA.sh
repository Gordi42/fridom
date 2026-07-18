#!/bin/bash
# Sequential runner: attribution (4-GPU, 1-GPU) + Leg A files 2-7.
# Must be run from the MAIN CHECKOUT. Fusion workaround set throughout.
set -u
ART=/work/uo0780/u301533/fridom/fridom-dev/.claude/worktrees/agent-a35d2cf0d179c137f/design/research/artifacts/stretched_terrain_gpu4
MAIN=/work/uo0780/u301533/fridom/fridom-dev
cd "$MAIN" || exit 1
export XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion

run_py() {  # label extra_env
  local label="$1"; shift
  echo "### START $label $(date)"
  ( eval "$@" JAX_PLATFORMS=cuda "$MAIN/.venv/bin/python" \
      "$ART/attribute_multigrid_cusparse.py" ) \
      > "$ART/attr_${label}.txt" 2>&1
  echo "EXIT=$?" >> "$ART/attr_${label}.txt"
  echo "### END $label $(date)"
}

run_pytest() {  # idx file
  local idx="$1"; local file="$2"
  local base=$(basename "$file" .py)
  echo "### START legA_${idx}_${base} $(date)"
  JAX_PLATFORMS=cuda FRIDOM_TEST_FORCED_DEVICES=4 \
    uv run pytest "$file" -q \
    > "$ART/legA_${idx}_${base#test_}.txt" 2>&1
  echo "EXIT=$?" >> "$ART/legA_${idx}_${base#test_}.txt"
  echo "### END legA_${idx}_${base} $(date)"
}

run_py 4gpu ""
run_py 1gpu "CUDA_VISIBLE_DEVICES=0"

run_pytest 2 tests/validation/test_stretched_mesh.py
run_pytest 3 tests/hydrostatic/test_terrain.py
run_pytest 4 tests/hydrostatic/test_core_terrain.py
run_pytest 5 tests/hydrostatic/test_free_surface_terrain.py
run_pytest 6 tests/nonhydro2/test_mapped_pressure_multigrid.py
run_pytest 7 tests/spatial/operators/test_multigrid.py
echo "### ALL DONE $(date)"
