"""B0 spike — exploratory stage 1: reproduce the §4b baseline."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.nonhydro2.modules.mapped_pressure import MappedPressureSolver

TWO_PI = 2.0 * np.pi


def make_depth(a):
    return lambda x: 1.0 + a * jnp.sin(x)


def build_grid(n, depth_fn):
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI), periodic=True,
                                        name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI), periodic=True,
                                        name="y")
    mz = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), periodic=False,
                                        name="z")
    mapping = fr.spatial.CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": depth_fn})
    return fr.spatial.Grid((mx, my, mz), mapping=mapping)


def dot(a, b):
    return jnp.sum((a * b).integrate().data)


def proj(f):
    return f - f.mean()


def hand_pcg(A, Minv, rhs, iters, tol=1e-10):
    b = proj(rhs)
    bnorm = float(jnp.sqrt(dot(b, b)))
    x = 0.0 * b
    r = b
    z = proj(Minv(r))
    p = z
    rz = dot(r, z)
    hist = [1.0]
    true_hist = [1.0]
    n_conv = None
    for k in range(iters):
        ap = A(p)
        pap = dot(p, ap)
        alpha = rz / pap
        x = x + alpha * p
        r = r - alpha * ap
        rel = float(jnp.sqrt(dot(r, r))) / bnorm
        true_r = b - A(x)
        trel = float(jnp.sqrt(dot(true_r, true_r))) / bnorm
        hist.append(rel)
        true_hist.append(trel)
        if n_conv is None and rel <= tol:
            n_conv = k + 1
        z = proj(Minv(r))
        rz_new = dot(r, z)
        beta = rz_new / rz
        p = z + beta * p
        rz = rz_new
    return hist, true_hist, n_conv


def run(n, a, wz, iters=60, seed=7):
    grid = build_grid(n, make_depth(a))
    space = (grid.factors[0].center * grid.factors[1].center
             * grid.factors[2].center)
    solver = MappedPressureSolver(grid, space, iterations=1,
                                  weights={"z": wz})
    A = jax.jit(lambda p: solver.apply(p))
    Minv = jax.jit(lambda r: solver._preconditioner()(r))
    rhs = grid.random.normal(solver._space, seed=seed)
    hist, true_hist, n_conv = hand_pcg(A, Minv, rhs, iters=iters)
    ratio = (1.0 + a) / (1.0 - a)
    print(f"a={a:.3f} ratio={ratio:.2f} wz={wz} n={n}: "
          f"iters_to_1e-10={n_conv}")
    tbl = {k: hist[k] for k in (8, 12, 20, 30) if k < len(hist)}
    print("   recur resid @k:",
          {k: f"{v:.2e}" for k, v in tbl.items()})
    tbl2 = {k: true_hist[k] for k in (8, 12, 20, 30) if k < len(hist)}
    print("   true  resid @k:",
          {k: f"{v:.2e}" for k, v in tbl2.items()})
    return n_conv


def main():
    print("=== mild (a=0.2) should be ~11 ===")
    for wz in (1.0, 4.0):
        run(64, 0.2, wz)
    print("=== amplitude scan at wz=4.0 (dsqr=0.25), n=64 ===")
    for a in (0.2, 0.4, 7.0 / 11.0, 0.8, 0.9):
        run(64, a, 4.0)
    print("=== amplitude scan at wz=1.0 (dsqr=1.0), n=64 ===")
    for a in (7.0 / 11.0, 0.8, 0.9):
        run(64, a, 1.0)


if __name__ == "__main__":
    main()
