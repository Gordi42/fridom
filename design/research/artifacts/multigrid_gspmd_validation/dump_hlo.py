"""Dump optimized step HLO for a GB-2 mapped config. Caller sets
JAX_PLATFORMS, CUDA_VISIBLE_DEVICES, and XLA_FLAGS (--xla_dump_to=DIR)."""
from __future__ import annotations

import sys

sys.path.insert(0, "/work/uo0780/u301533/fridom/census_scratch")
import jax  # noqa: E402
import gb2_common as G  # noqa: E402

n = int(sys.argv[1])
precond = sys.argv[2]        # "multigrid" | "spectral"
method = sys.argv[3] if len(sys.argv) > 3 else "auto"
budget = int(sys.argv[4]) if len(sys.argv) > 4 else 100
print("devices:", jax.device_count(), "precond:", precond,
      "method:", method, "budget:", budget, flush=True)
model = G.build_mapped_model(
    n, preconditioner=precond, method=method, budget=budget)
G.set_mapped_ic(model)
model.advance(1)
G.sync(model)
model.advance(1)
G.sync(model)
print("advanced; done", flush=True)
