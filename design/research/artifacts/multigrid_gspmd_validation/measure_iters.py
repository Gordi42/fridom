"""Measure production CG iterations for spectral vs multigrid on the
mapped GB-2 case (random mean-free RHS, tol 1e-8), 4-GPU."""
from __future__ import annotations
import sys
sys.path.insert(0, "/work/uo0780/u301533/fridom/census_scratch")
import jax  # noqa: E402
import gb2_common as G  # noqa: E402

n = 128
ids4 = tuple(range(jax.device_count()))
print("devices:", jax.device_count(), flush=True)


def iters(precond, method="auto"):
    solver, grid, space, _m = G.build_mapped_solver(
        n, preconditioner=precond, method=method, device_ids=ids4)
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


for pc, m in [("spectral", "auto"), ("multigrid", "cusparse")]:
    it, rn = iters(pc, m)
    print(f"{pc:10s} iters={it} resid={rn:.3e}", flush=True)
