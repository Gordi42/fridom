"""Phase-3 agglomeration measurement driver — ONE config per process.

Usage:  driver.py <case> <n> <tag>
  case: mapped | immersed
  n   : grid size (per axis)
  tag : spectral | mg-off | mg-t2 | mg-t4 | mg-t8

Maps tag -> (preconditioner, method, agglomerate):
  spectral -> (spectral,  auto, None)
  mg-off   -> (multigrid, auto, None)     # method auto = cuSPARSE on GPU
  mg-t2/4/8-> (multigrid, auto, 2/4/8)

Appends JSON lines to results.jsonl in the job dir (flushed):
  a phase="timing" record (median/min/max ms/step, maxu, finite, peak mem)
  written FIRST so it survives even if the iteration build OOMs, then a
  phase="iters" record (CG iterations + residual on a mean-free RHS).

Runs single-process GSPMD over ALL visible devices (must be 4 cuda).
Hard-fails unless fridom resolves under the pinned src-pin worktree.
"""
from __future__ import annotations

import json
import os
import sys
import time

JOBDIR = "/work/uo0780/u301533/fridom/agglom_phase3"
SRC_PIN = os.path.join(JOBDIR, "src-pin")
OUT = os.path.join(JOBDIR, "results.jsonl")

sys.path.insert(0, JOBDIR)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import fridom as fr  # noqa: E402,F401
import gb2_common as G  # noqa: E402

# ---- source-pin assertion (hard fail) ----------------------------
FRIDOM_FILE = os.path.realpath(fr.__file__)
print(f"fridom.__file__ = {FRIDOM_FILE}", flush=True)
if os.path.realpath(SRC_PIN) not in FRIDOM_FILE:
    raise SystemExit(
        f"FATAL: fridom does not resolve under the pinned worktree\n"
        f"  expected prefix: {os.path.realpath(SRC_PIN)}\n"
        f"  got            : {FRIDOM_FILE}")

NDEV = jax.device_count()
PLATFORM = jax.devices()[0].platform
print(f"devices={NDEV} platform={PLATFORM} "
      f"kinds={[d.device_kind for d in jax.devices()]}", flush=True)

case = sys.argv[1]
n = int(sys.argv[2])
tag = sys.argv[3]

TAG_MAP = {
    "spectral": ("spectral", "auto", None),
    "mg-off": ("multigrid", "auto", None),
    "mg-t2": ("multigrid", "auto", 2),
    "mg-t4": ("multigrid", "auto", 4),
    "mg-t8": ("multigrid", "auto", 8),
}
precond, method, agglom = TAG_MAP[tag]
ids = tuple(range(NDEV))
print(f"CONFIG case={case} n={n} tag={tag} precond={precond} "
      f"method={method} agglomerate={agglom} device_ids={ids}", flush=True)


def append(rec):
    rec = {"case": case, "n": n, "tag": tag, "precond": precond,
           "method": method, "agglomerate": agglom, "ndev": NDEV,
           "platform": PLATFORM, "fridom_file": FRIDOM_FILE, **rec}
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    print("WROTE:", json.dumps(rec), flush=True)


# ================================================================
#  Phase 1 — timing (written first so it survives an iters OOM)
# ================================================================
t_build0 = time.perf_counter()
if case == "mapped":
    model = G.build_mapped_model(
        n, preconditioner=precond, method=method,
        agglomerate=agglom, device_ids=ids)
    G.set_mapped_ic(model)
elif case == "immersed":
    model = G.build_immersed_model(
        n, preconditioner=precond, method=method,
        agglomerate=agglom, device_ids=ids)
    G.set_immersed_ic(model)
else:
    raise SystemExit(f"unknown case {case!r}")
build_s = time.perf_counter() - t_build0

t = G.time_step_ms(model, reps=6, steps=20)
u = np.asarray(model.state["u"].data)
maxu = float(np.abs(u).max())
finite = bool(np.all(np.isfinite(u)))
mem = G.peak_mem_gib()
append({"phase": "timing", "median_ms": t["median"], "min_ms": t["min"],
        "max_ms": t["max"], "compile_s": t["compile_s"],
        "all_ms": t["all"], "build_s": build_s, "maxu": maxu,
        "finite": finite, "peak_gib": mem})
del model


# ================================================================
#  Phase 2 — CG iteration count on a mean-free RHS (guarded)
# ================================================================
try:
    t_i0 = time.perf_counter()
    if case == "mapped":
        solver, grid, space, _m = G.build_mapped_solver(
            n, preconditioner=precond, method=method,
            agglomerate=agglom, device_ids=ids)
        key = jax.random.PRNGKey(0)
        rand = grid.create_field(space)
        rand = rand.with_data(
            jax.random.normal(key, rand.data.shape, dtype=rand.data.dtype))
        rhs = rand - rand.mean()
    else:
        solver, grid, space, _m = G.build_immersed_solver(
            n, preconditioner=precond, method=method,
            agglomerate=agglom, device_ids=ids)
        rhs = G.immersed_rhs_from_velocity(solver, seed=0)

    @jax.jit
    def solve(d):
        _p, info = solver.krylov().solve(rhs.with_data(d))
        return info["iterations"], info["residual_norm"]

    it, rn = solve(rhs.data)
    jax.block_until_ready((it, rn))
    iters_s = time.perf_counter() - t_i0
    append({"phase": "iters", "iterations": int(it),
            "residual_norm": float(rn), "iters_build_s": iters_s})
except Exception as e:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    append({"phase": "iters", "error": repr(e)})

print("DONE", flush=True)
