"""Leg 2: immersed post-swap in-model standing (1 GPU).

spectral vs mg-cusparse, GB-2 protocol (median 6x20 ms/step), plus achieved
iteration counts (spectral expected to bust the budget=100 unconverged).
Usage: immersed_leg2.py <n>   (pin CUDA_VISIBLE_DEVICES=0)
"""
from __future__ import annotations

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/tmp/claude-100698/-work-uo0780-u301533-fridom-fridom-dev/4f23fac6-d4e9-4088-a855-5ccf5ca6a06b/scratchpad")
import gb2_common as G  # noqa: E402

OUT = "/tmp/claude-100698/-work-uo0780-u301533-fridom-fridom-dev/4f23fac6-d4e9-4088-a855-5ccf5ca6a06b/scratchpad/results_immersed.jsonl"
n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
print(f"immersed n={n} devices={jax.device_count()}")

# document the geometry is genuinely partial
gm = G.build_immersed_model(n, preconditioner="spectral")
theta = np.asarray(gm.grid.immersed.fraction(
    gm.state["b"].function_space).data)
interior = np.argwhere((theta > 1e-6) & (theta < 1.0 - 1e-6))
print(f"partial cells: {interior.size} entries; wet frac "
      f"{(theta > 0).mean():.3f}")
del gm


def iters(preconditioner, method):
    solver, grid, space, _m = G.build_immersed_solver(
        n, preconditioner=preconditioner, method=method)
    # wet-supported, compatible rhs = masked divergence of a random vel
    rhs = G.immersed_rhs_from_velocity(solver, seed=0)

    @jax.jit
    def solve(d):
        _p, info = solver.solve_info(rhs.with_data(d))
        return info["iterations"], info["residual_norm"]

    it, rn = solve(rhs.data)
    # relative residual vs rhs norm
    bb = float(jnp.sqrt(jnp.sum((rhs * rhs).integrate().data)))
    return int(it), float(rn), float(rn) / bb


def timing(preconditioner, method):
    model = G.build_immersed_model(
        n, preconditioner=preconditioner, method=method)
    G.set_immersed_ic(model)
    t = G.time_step_ms(model, reps=6, steps=20)
    maxu = float(np.abs(np.asarray(model.state["u"].data)).max())
    return t, maxu


results = {}
for precond, method, tag in (("spectral", "auto", "spectral"),
                             ("multigrid", "cusparse", "mg-cusparse")):
    it, rn, relres = iters(precond, method)
    t, maxu = timing(precond, method)
    rec = {"tag": tag, "n": n, "iters": it, "residual_norm": rn,
           "rel_residual": relres, "median_ms": t["median"],
           "min_ms": t["min"], "max_ms": t["max"],
           "compile_s": t["compile_s"], "all_ms": t["all"], "maxu": maxu}
    results[tag] = rec
    print(json.dumps(rec, indent=2))
    with open(OUT, "a") as fh:
        fh.write(json.dumps(rec) + "\n")

sp = results["spectral"]["median_ms"]
mg = results["mg-cusparse"]["median_ms"]
print(f"\nSUMMARY n={n}: spectral {sp:.2f} ms/step (iters "
      f"{results['spectral']['iters']}, relres "
      f"{results['spectral']['rel_residual']:.2e}) | "
      f"mg-cusparse {mg:.2f} ms/step (iters "
      f"{results['mg-cusparse']['iters']}, relres "
      f"{results['mg-cusparse']['rel_residual']:.2e}) | "
      f"mg speedup {sp / mg:.2f}x")
