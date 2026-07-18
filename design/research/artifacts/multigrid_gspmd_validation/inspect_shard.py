"""Leg 1: inspect the real 4-GPU sharding of a prognostic field."""
from __future__ import annotations

import sys

import jax

sys.path.insert(0, "/tmp/claude-100698/-work-uo0780-u301533-fridom-fridom-dev/4f23fac6-d4e9-4088-a855-5ccf5ca6a06b/scratchpad")
import gb2_common as G  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
print("backend:", jax.default_backend(), "devices:", jax.device_count())

model = G.build_mapped_model(n, preconditioner="multigrid", method="auto")
G.set_mapped_ic(model)
for name in ("u", "v", "w", "p", "b"):
    f = model.state[name].data
    print(f"{name}: shape={f.shape} sharding={f.sharding}")
    try:
        print("   shard_shape:", f.sharding.shard_shape(f.shape))
    except Exception as e:  # noqa: BLE001
        print("   shard_shape err:", e)

dl = model.grid.decomposition.default_layout
for ax in ("x", "y", "z"):
    print(f"is_local({ax}): {dl.is_local(ax)}")
print("mesh:", model.grid.decomposition)
