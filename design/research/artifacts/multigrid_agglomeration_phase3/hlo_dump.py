"""Dump optimized step HLO for a mapped GB-2 config (agglomeration census).

Usage: hlo_dump.py <n> <tag>   (tag: mg-off | mg-t4 | ...)
The caller sets JAX_PLATFORMS, and XLA_FLAGS carrying
--xla_dump_to=<dir> --xla_dump_hlo_as_text plus the fusion workaround.
Builds the mapped model, advances twice (capturing the step module).
"""
from __future__ import annotations

import os
import sys

JOBDIR = "/work/uo0780/u301533/fridom/agglom_phase3"
SRC_PIN = os.path.join(JOBDIR, "src-pin")
sys.path.insert(0, JOBDIR)

import jax  # noqa: E402

import fridom as fr  # noqa: E402,F401
import gb2_common as G  # noqa: E402

FRIDOM_FILE = os.path.realpath(fr.__file__)
print(f"fridom.__file__ = {FRIDOM_FILE}", flush=True)
if os.path.realpath(SRC_PIN) not in FRIDOM_FILE:
    raise SystemExit(f"FATAL: fridom not under src-pin: {FRIDOM_FILE}")

n = int(sys.argv[1])
tag = sys.argv[2]
TAG_MAP = {"mg-off": (None,), "mg-t2": (2,), "mg-t4": (4,), "mg-t8": (8,)}
(agglom,) = TAG_MAP[tag]
print(f"devices={jax.device_count()} n={n} tag={tag} agglomerate={agglom}",
      flush=True)

model = G.build_mapped_model(
    n, preconditioner="multigrid", method="auto", agglomerate=agglom,
    device_ids=tuple(range(jax.device_count())))
G.set_mapped_ic(model)
model.advance(1)
G.sync(model)
model.advance(1)
G.sync(model)
print("advanced; HLO dumped", flush=True)
