"""Leg 1(b): parity (cusparse vs pcr, 4-GPU vs 1-GPU) + iteration counts.

Usage: parity_iters.py <n> [parity|iters|both]
Runs inside a 4-device process; the "1-GPU" reference uses device_ids=(0,).
"""
from __future__ import annotations

import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, "/tmp/claude-100698/-work-uo0780-u301533-fridom-fridom-dev/4f23fac6-d4e9-4088-a855-5ccf5ca6a06b/scratchpad")
import gb2_common as G  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 128
mode = sys.argv[2] if len(sys.argv) > 2 else "both"
FIELDS = ("u", "v", "w", "b")
print(f"n={n} mode={mode} devices={jax.device_count()}")


def run_steps(preconditioner, method, device_ids, steps=20):
    model = G.build_mapped_model(
        n, preconditioner=preconditioner, method=method,
        device_ids=device_ids)
    G.set_mapped_ic(model)
    model.advance(steps)
    G.sync(model)
    return {f: np.asarray(model.state[f].data) for f in FIELDS}


def reldiff(a, b):
    num = max(np.max(np.abs(a[f] - b[f])) for f in FIELDS)
    den = max(np.max(np.abs(b[f])) for f in FIELDS)
    return num / den


ids4 = tuple(range(jax.device_count()))

if mode in ("parity", "both"):
    print("--- parity: 20 steps from the record IC ---")
    cus4 = run_steps("multigrid", "cusparse", ids4)
    pcr4 = run_steps("multigrid", "pcr", ids4)
    cus1 = run_steps("multigrid", "cusparse", (0,))
    spec4 = run_steps("spectral", "auto", ids4)
    print(f"  4GPU cusparse vs 4GPU pcr   : {reldiff(cus4, pcr4):.3e}")
    print(f"  4GPU cusparse vs 1GPU cusparse: {reldiff(cus4, cus1):.3e}")
    print(f"  4GPU cusparse vs 4GPU spectral: {reldiff(cus4, spec4):.3e}")

if mode in ("iters", "both"):
    print("--- iterations: production CG on random mean-free RHS, tol 1e-8 ---")

    def iters(method, device_ids):
        solver, grid, space, _m = G.build_mapped_solver(
            n, preconditioner="multigrid", method=method,
            device_ids=device_ids)
        key = jax.random.PRNGKey(0)
        rand = grid.create_field(space)
        rand = rand.with_data(
            jax.random.normal(key, rand.data.shape, dtype=rand.data.dtype))
        rhs = rand - rand.mean()

        @jax.jit
        def solve(d):
            _p, info = solver.krylov().solve(rhs.with_data(d))
            return info["iterations"], info["residual_norm"]

        it, rn = solve(rhs.data)
        return int(it), float(rn)

    for method in ("cusparse", "pcr"):
        it4, rn4 = iters(method, ids4)
        it1, rn1 = iters(method, (0,))
        print(f"  {method:9s}: 4GPU iters={it4} (resid {rn4:.2e}) | "
              f"1GPU iters={it1} (resid {rn1:.2e})")
