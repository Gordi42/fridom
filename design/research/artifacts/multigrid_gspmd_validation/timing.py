"""Leg 1(c): GB-2 timing of one mapped config (own process for clean peak mem).

Usage: timing.py <n> <preconditioner> <method> [device_ids_csv]
Appends one JSON line to results_timing.jsonl.
"""
from __future__ import annotations

import json
import sys

import jax
import numpy as np

sys.path.insert(0, "/tmp/claude-100698/-work-uo0780-u301533-fridom-fridom-dev/4f23fac6-d4e9-4088-a855-5ccf5ca6a06b/scratchpad")
import gb2_common as G  # noqa: E402

OUT = "/tmp/claude-100698/-work-uo0780-u301533-fridom-fridom-dev/4f23fac6-d4e9-4088-a855-5ccf5ca6a06b/scratchpad/results_timing.jsonl"

n = int(sys.argv[1])
precond = sys.argv[2]
method = sys.argv[3]
device_ids = None
if len(sys.argv) > 4 and sys.argv[4]:
    device_ids = tuple(int(x) for x in sys.argv[4].split(","))

print(f"n={n} precond={precond} method={method} devices={jax.device_count()} "
      f"device_ids={device_ids}")
model = G.build_mapped_model(
    n, preconditioner=precond, method=method, device_ids=device_ids)
G.set_mapped_ic(model)
t = G.time_step_ms(model, reps=6, steps=20)
mem = G.peak_mem_gib()
maxu = float(np.abs(np.asarray(model.state["u"].data)).max())
rec = {"n": n, "precond": precond, "method": method,
       "device_ids": device_ids, "ndev": jax.device_count(),
       "median_ms": t["median"], "min_ms": t["min"], "max_ms": t["max"],
       "compile_s": t["compile_s"], "all_ms": t["all"],
       "peak_gib": mem, "maxu": maxu}
print(json.dumps(rec, indent=2))
with open(OUT, "a") as fh:
    fh.write(json.dumps(rec) + "\n")
print("WROTE", OUT)
