"""Tiny sanity: model builds, sharding inspection, iterations, ic set."""
from __future__ import annotations

import sys

import jax
import numpy as np

sys.path.insert(0, "/tmp/claude-100698/-work-uo0780-u301533-fridom-fridom-dev/4f23fac6-d4e9-4088-a855-5ccf5ca6a06b/scratchpad")
import gb2_common as G  # noqa: E402

print("backend:", jax.default_backend(), "devices:", jax.device_count())

n = 16
model = G.build_mapped_model(n, preconditioner="multigrid", method="auto")
G.set_mapped_ic(model)
u = model.state["u"].data
print("u shape:", u.shape)
print("u sharding:", u.sharding)
try:
    print("addressable shards:",
          [(s.index) for s in u.sharding.shard_shape(u.shape)] if False else
          u.sharding.shard_shape(u.shape))
except Exception as e:
    print("shard_shape err:", e)
print("mesh/devices on grid decomposition default layout:")
try:
    dl = model.grid.decomposition.default_layout
    for ax in ("x", "y", "z"):
        print(f"  is_local({ax}):", dl.is_local(ax))
except Exception as e:
    print("  layout err:", e)

# standalone solver iterations
solver, grid, space, _m = G.build_mapped_solver(
    n, preconditioner="multigrid", method="cusparse")
rhs = G.mapped_mean_free_rhs(grid, space)
_p, info = jax.jit(lambda d: solver.krylov().solve(rhs.with_data(d)))(rhs.data)
print("mapped cusparse iters:", int(info["iterations"]),
      "resid:", float(info["residual_norm"]))

# one step to confirm advance works
model.advance(1)
G.sync(model)
print("advanced 1 step OK; max|u|:", float(np.abs(np.asarray(model.state["u"].data)).max()))
print("SANITY OK")
