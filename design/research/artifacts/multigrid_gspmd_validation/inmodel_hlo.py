"""Leg 1(a).2 IN-MODEL: dump compiled step HLO of the 4-GPU 128^3
mg-cusparse run; the caller sets XLA_FLAGS=...--xla_dump_to=DIR."""
from __future__ import annotations

import sys

import jax

sys.path.insert(0, "/tmp/claude-100698/-work-uo0780-u301533-fridom-fridom-dev/4f23fac6-d4e9-4088-a855-5ccf5ca6a06b/scratchpad")
import gb2_common as G  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
print("devices:", jax.device_count())
model = G.build_mapped_model(
    n, preconditioner="multigrid", method="cusparse")
G.set_mapped_ic(model)
model.advance(1)
G.sync(model)
model.advance(1)
G.sync(model)
print("advanced; done")
