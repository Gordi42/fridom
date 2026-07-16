#!/usr/bin/env bash
# Reproduce the full stencil-lowering study. Single A100, fresh process per
# XLA_FLAGS variant. Never imports fridom (pure jax; f64 set in driver.py).
set -euo pipefail
PY=/work/uo0780/u301533/fridom/fridom-dev/.venv/bin/python
cd "$(dirname "$0")"

JAX_PLATFORMS=cuda $PY validate.py
JAX_PLATFORMS=cuda $PY driver.py roofline
JAX_PLATFORMS=cuda $PY driver.py e1
JAX_PLATFORMS=cuda $PY driver.py e2
JAX_PLATFORMS=cuda $PY driver.py e3
JAX_PLATFORMS=cuda $PY driver.py e4
# E3-mof: fridom's production multi-GPU miscompile workaround (jax#39100)
XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion \
  JAX_PLATFORMS=cuda $PY driver.py e3mof
# 512^3 spot checks
JAX_PLATFORMS=cuda $PY driver.py e1 --spot --n 512
JAX_PLATFORMS=cuda $PY driver.py e3 --spot --n 512
echo "done"
