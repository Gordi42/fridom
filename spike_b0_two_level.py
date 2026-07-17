"""B0 two-level multigrid spike — GB-0 kill-criterion measurement.

Throwaway diagnostics for design/plans/active/multigrid_pathway_plan.md
§3 (GB-0). Does a two-grid cycle (one horizontal semicoarsening step,
damped vertical-line-Jacobi V(1,1), near-exact coarse solve) used as a
PCG preconditioner drop the steep-mapped pressure solve below ~20
iterations (from the ~45 spectral-PCG baseline)?

Run: env JAX_PLATFORMS=cpu FRIDOM_TEST_JAX_CACHE_DIR=.../.jax_cache_local
     uv run python spike_b0_two_level.py

CALIBRATION NOTE (see the agent report): the §4b "steep 4.5x" 45-iter
baseline is reproduced by amplitude a=0.8 (depth ratio 9.0), NOT the
a=7/11 (ratio 4.5) the plan text names — the doc mislabels the ratio.
a=0.8 matches §4b's convergence table (3.7e-2/3.9e-3/5.7e-5/2.6e-7 at
k=8/12/20/30) and its resolution table (44/45 at n=32/64); the harness
is independently anchored by the mild a=0.2 case reproducing 11 exactly.
STEEP_A below is the GB-0 anchor; A45_LITERAL is the as-named 4.5x case.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.nonhydro2.modules.mapped_pressure import MappedPressureSolver
from fridom.spatial.operators.transfer import GridTransfer

TWO_PI = 2.0 * np.pi
WZ = 4.0            # weights={"z": 1/dsqr}, dsqr=0.25 (§4b, model default)
STEEP_A = 0.8       # GB-0 anchor: reproduces §4b's 45-iter "steep" case
MILD_A = 0.2        # §4b mild 1.5x -> 11 iters
A45_LITERAL = 7.0 / 11.0   # the as-named ratio-4.5 profile (27 iters)
SEED = 7


# ================================================================
#  Grid / solver construction
# ================================================================
def make_depth(a):
    return lambda x: 1.0 + a * jnp.sin(x)


def build_grid(n, a):
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI), periodic=True,
                                        name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI), periodic=True,
                                        name="y")
    mz = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), periodic=False,
                                        name="z")
    mapping = fr.spatial.CoordinateMapping(
        maps={"zp": lambda z, H: z * H}, params={"H": make_depth(a)})
    grid = fr.spatial.Grid((mx, my, mz), mapping=mapping)
    space = (grid.factors[0].center * grid.factors[1].center
             * grid.factors[2].center)
    return grid, space


def build_solver(grid, space, iterations):
    return MappedPressureSolver(grid, space, iterations=iterations,
                               weights={"z": WZ})


# ================================================================
#  Measure-weighted inner product (krylov._dot) and mean projection
# ================================================================
def dot(a, b):
    return jnp.sum((a * b).integrate().data)


def proj(f):
    return f - f.mean()


# ================================================================
#  Hand-rolled PCG with residual history (mirrors krylov)
# ================================================================
def hand_pcg(apply, minv, rhs, iters, tol=1e-10):
    b = proj(rhs)
    bnorm = float(jnp.sqrt(dot(b, b)))
    x = 0.0 * b
    r = b
    z = proj(minv(r))
    p = z
    rz = dot(r, z)
    hist = [1.0]
    n_conv = None
    for k in range(iters):
        ap = apply(p)
        pap = dot(p, ap)
        alpha = jnp.where(pap == 0.0, 0.0, rz / jnp.where(
            pap == 0.0, 1.0, pap))
        x = x + alpha * p
        r = r - alpha * ap
        rel = float(jnp.sqrt(dot(r, r))) / bnorm
        hist.append(rel)
        if n_conv is None and rel <= tol:
            n_conv = k + 1
        z = proj(minv(r))
        rz_new = dot(r, z)
        beta = jnp.where(rz == 0.0, 0.0, rz_new / jnp.where(
            rz == 0.0, 1.0, rz))
        p = z + beta * p
        rz = rz_new
    return hist, n_conv


# ================================================================
#  27-coloring band extraction (T = z-tridiagonal of A, full diag)
# ================================================================
def a_on_data(solver, grid, space):
    """Jitted A: raw true-shape data -> raw true-shape data."""
    @jax.jit
    def _a(d):
        return solver.apply(grid.create_field(space, data=d)).data
    return _a


def extract_bands(a_data, shape, period=4):
    """Three z-bands (lower, diag, upper) of A via p-coloring.

    period=4 (not 3): the horizontal axes are periodic and the cell
    counts (16,32,64,..) are not divisible by 3, so a 3-coloring puts
    distance-1 wrap neighbors in the same color (aliasing). p=4 divides
    every cell count here and keeps the 3 consecutive residues of any
    +-1 stencil distinct on the torus. 4^3 = 64 operator applications.
    """
    p = period
    ii, jj, kk = np.indices(shape)
    ystack = np.zeros((p, p, p, *shape))
    for cx in range(p):
        for cy in range(p):
            for cz in range(p):
                mask = ((ii % p == cx) & (jj % p == cy)
                        & (kk % p == cz)).astype(np.float64)
                y = np.asarray(a_data(jnp.asarray(mask)))
                ystack[cx, cy, cz] = y
    ci, cj, ck = ii % p, jj % p, kk % p
    diag = ystack[ci, cj, ck, ii, jj, kk]
    lower = ystack[ci, cj, (kk - 1) % p, ii, jj, kk]
    upper = ystack[ci, cj, (kk + 1) % p, ii, jj, kk]
    return lower, diag, upper


def dense_A(a_data, shape):
    """Full dense operator matrix by probing every unit vector."""
    ncells = int(np.prod(shape))
    dense = np.zeros((ncells, ncells))
    for n in range(ncells):
        e = np.zeros(ncells)
        e[n] = 1.0
        col = np.asarray(a_data(jnp.asarray(e.reshape(shape))))
        dense[:, n] = col.reshape(ncells)
    return dense


def verify_extraction(n, a):
    """Dense-A cross-check of the band extraction + symmetry at n^3."""
    grid, space = build_grid(n, a)
    shape = tuple(space.shape)
    solver = build_solver(grid, space, iterations=1)
    a_data = a_on_data(solver, grid, space)

    dense = dense_A(a_data, shape)
    ncells = dense.shape[0]

    # stencil width: max PERIODIC offset per axis over all nonzeros
    # (x,y are periodic: a distance-1 wrap coupling has linear |offset|
    # n-1, so report min(|d|, n-|d|) to see the true torus half-width)
    tol = 1e-9 * np.abs(dense).max()
    rows, cols = np.nonzero(np.abs(dense) > tol)
    ri = np.array(np.unravel_index(rows, shape))
    ci = np.array(np.unravel_index(cols, shape))
    lin = np.abs(ri - ci)
    per = np.array([[True], [True], [False]])  # x,y periodic; z bounded
    ns = np.array(shape).reshape(3, 1)
    off = np.where(per, np.minimum(lin, ns - lin), lin)
    max_off = off.max(axis=1)
    print(f"  stencil half-width per axis (x,y,z), torus metric: "
          f"{tuple(int(v) for v in max_off)}")

    # symmetry (plain + measure-weighted)
    asym = np.abs(dense - dense.T).max() / np.abs(dense).max()
    vol = (grid.measure(space, "x") * grid.measure(space, "y")
           * grid.measure(space, "z"))
    wvec = np.asarray(vol.data).reshape(ncells)
    print(f"  measure per axis dx,dy,dz: "
          f"{float(grid.measure(space,'x').data.ravel()[0]):.4f}, "
          f"{float(grid.measure(space,'y').data.ravel()[0]):.4f}, "
          f"{float(grid.measure(space,'z').data.ravel()[0]):.4f} "
          f"(uniform={np.allclose(wvec, wvec[0])})")
    wa = wvec[:, None] * dense
    wsym = np.abs(wa - wa.T).max() / np.abs(wa).max()
    print(f"  ||A-A^T||/||A|| = {asym:.2e}   "
          f"||WA-(WA)^T||/||WA|| = {wsym:.2e}")

    # band extraction vs dense
    lower, diag, upper = extract_bands(a_data, shape)
    nz = shape[2]
    dd = np.zeros(shape)
    dl = np.zeros(shape)
    du = np.zeros(shape)
    idx = np.arange(ncells).reshape(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(nz):
                p = idx[i, j, k]
                dd[i, j, k] = dense[p, p]
                if k > 0:
                    dl[i, j, k] = dense[p, idx[i, j, k - 1]]
                if k < nz - 1:
                    du[i, j, k] = dense[p, idx[i, j, k + 1]]
    scale = np.abs(dd).max()
    ed = np.abs(diag - dd).max() / scale
    el = np.abs(lower[:, :, 1:] - dl[:, :, 1:]).max() / scale
    eu = np.abs(upper[:, :, :-1] - du[:, :, :-1]).max() / scale
    print(f"  band vs dense (rel): diag={ed:.2e} lower={el:.2e} "
          f"upper={eu:.2e}")
    return asym, wsym, (ed, el, eu), tuple(max_off)


# ================================================================
#  Vertical-line-Jacobi smoother (batched dense tridiagonal solve)
# ================================================================
def build_tinv(bands, shape):
    """T^{-1} per column: (nx*ny, nz, nz)."""
    lower, diag, upper = bands
    nx, ny, nz = shape
    nc = nx * ny
    d_ = jnp.asarray(diag).reshape(nc, nz)
    l_ = jnp.asarray(lower).reshape(nc, nz)
    u_ = jnp.asarray(upper).reshape(nc, nz)
    idx = jnp.arange(nz)
    t = jnp.zeros((nc, nz, nz))
    t = t.at[:, idx, idx].set(d_)
    t = t.at[:, idx[1:], idx[:-1]].set(l_[:, 1:])
    t = t.at[:, idx[:-1], idx[1:]].set(u_[:, :-1])
    return jnp.linalg.inv(t)


def make_tinv_apply(tinv, shape):
    nx, ny, nz = shape
    nc = nx * ny

    @jax.jit
    def _apply(d):
        e = jnp.einsum("cij,cj->ci", tinv, d.reshape(nc, nz))
        return e.reshape(nx, ny, nz)
    return _apply


# ================================================================
#  Two-grid V-cycle preconditioner
# ================================================================
class TwoGrid:
    def __init__(self, n, a, coarse_iterations=200):
        self.grid, self.space = build_grid(n, a)
        self.shape = tuple(self.space.shape)
        self.solver = build_solver(self.grid, self.space, iterations=1)
        self.fine_A = a_on_data_field(self.solver)
        self.fine_tinv = make_tinv_apply(
            build_tinv(extract_bands(
                a_on_data(self.solver, self.grid, self.space),
                self.shape), self.shape), self.shape)

        # coarse level: horizontal semicoarsening x,y by 2 (z stays)
        self.coarse = self.grid.coarsened({"x": 2, "y": 2})
        self.transfer = GridTransfer(self.grid, self.coarse, order=2)
        probe = self.grid.create_field(self.space)
        self.coarse_space = self.transfer.restrict(probe).function_space
        self.cshape = tuple(self.coarse_space.shape)
        self.coarse_solver = build_solver(
            self.coarse, self.coarse_space, iterations=coarse_iterations)
        self.coarse_A = a_on_data_field(self.coarse_solver)
        self.coarse_tinv = make_tinv_apply(
            build_tinv(extract_bands(
                a_on_data(self.coarse_solver, self.coarse,
                          self.coarse_space), self.cshape),
                self.cshape), self.cshape)

        # near-exact coarse solve: inner spectral-PCG. This inner CG is
        # forbidden in the PRODUCTION cycle (a fixed-iteration CG is a
        # nonlinear map that breaks the outer CG) but legitimate HERE:
        # at 1e-12 the map is linear to roundoff and the spike measures
        # the two-grid method's upper bound (plan §3, B0).
        self._coarse_cg = jax.jit(
            lambda d: self.coarse_solver.krylov()(d))

        # jitted transfers (field -> field)
        self._restrict = jax.jit(self.transfer.restrict)
        self._prolong = jax.jit(self.transfer.prolong)

    # -- restrict / prolong with defensive bare retag --------------
    def restrict(self, d):
        return self._restrict(d.retag(self.space.bare))

    def prolong(self, e):
        return self._prolong(e.retag(self.coarse_space.bare)).retag(
            self.space)

    def coarse_solve_exact(self, d_c):
        return self._coarse_cg(d_c.retag(self.coarse_solver._space))

    def coarse_solve_ksweeps(self, d_c, k, omega):
        d_c = d_c.retag(self.coarse_solver._space)
        x = d_c.with_data(0.0 * d_c.data)
        for _ in range(k):
            resid = d_c - self.coarse_A(x)
            x = x.with_data(x.data + omega * self.coarse_tinv(resid.data))
        return x - x.mean()

    # -- the V(pre,post) cycle -------------------------------------
    def cycle(self, r, omega, coarse_solve, npre=1, npost=1):
        x = r.with_data(0.0 * r.data)
        for _ in range(npre):
            resid = r - self.fine_A(x)
            x = x.with_data(x.data + omega * self.fine_tinv(resid.data))
        d = r - self.fine_A(x)
        d_c = self.restrict(d)
        d_c = d_c - d_c.mean()
        e_c = coarse_solve(d_c)
        x = x + self.prolong(e_c)
        for _ in range(npost):
            resid = r - self.fine_A(x)
            x = x.with_data(x.data + omega * self.fine_tinv(resid.data))
        return x - x.mean()


def a_on_data_field(solver):
    """Jitted field->field A (no metric memo; re-derived in trace)."""
    return jax.jit(lambda p: solver.apply(p))


# ================================================================
#  Measurements
# ================================================================
def baseline(n, a, label):
    grid, space = build_grid(n, a)
    solver = build_solver(grid, space, iterations=1)
    apply = jax.jit(lambda p: solver.apply(p))
    minv = jax.jit(lambda r: solver._preconditioner()(r))
    rhs = grid.random.normal(solver._space, seed=SEED)
    _hist, n_conv = hand_pcg(apply, minv, rhs, iters=70)
    print(f"[baseline spectral-PCG] {label}: iters_to_1e-10 = {n_conv}")
    return n_conv


def verify_coarse_solve(tg):
    d = tg.restrict(proj(tg.grid.random.normal(tg.solver._space, seed=3)))
    d = d - d.mean()
    e = tg.coarse_solve_exact(d)
    resid = d - tg.coarse_A(e)
    rel = float(jnp.sqrt(dot(resid, resid)) / jnp.sqrt(dot(d, d)))
    print(f"[coarse near-exact solve] rel residual = {rel:.2e} "
          f"(need <= 1e-12)")
    return rel


def symmetry_gate(tg, omega, coarse_solve, npre=1, npost=1):
    u = proj(tg.grid.random.normal(tg.solver._space, seed=21))
    v = proj(tg.grid.random.normal(tg.solver._space, seed=22))
    mu = tg.cycle(u, omega, coarse_solve, npre, npost)
    mv = tg.cycle(v, omega, coarse_solve, npre, npost)
    left = float(dot(mu, v))
    right = float(dot(u, mv))
    rel = abs(left - right) / abs(left)
    print(f"[symmetry gate] <M^-1 u,v>={left:.6e} <u,M^-1 v>={right:.6e}"
          f"  rel={rel:.2e}")
    return rel


def two_grid_iters(tg, omega, coarse_solve, npre=1, npost=1, label=""):
    rhs = tg.grid.random.normal(tg.solver._space, seed=SEED)
    minv = lambda r: tg.cycle(r, omega, coarse_solve, npre, npost)
    _hist, n_conv = hand_pcg(tg.fine_A, minv, rhs, iters=40)
    print(f"[two-grid PCG] {label} omega={omega}: "
          f"iters_to_1e-10 = {n_conv}")
    return n_conv


def main():
    t0 = time.time()
    print("=" * 60)
    print("STEP 1 — baselines")
    print("=" * 60)
    baseline(64, STEEP_A, "steep a=0.8 (ratio 9.0), n=64 [GB-0 anchor]")
    baseline(64, MILD_A, "mild a=0.2 (ratio 1.5), n=64")
    baseline(64, A45_LITERAL, "as-named 4.5x a=7/11, n=64")

    print("=" * 60)
    print("STEP 2 — band extraction verification at 16^3 (steep)")
    print("=" * 60)
    verify_extraction(16, STEEP_A)

    print("=" * 60)
    print("STEP 3 — build two-grid (steep, n=64) + gates")
    print("=" * 60)
    tg = TwoGrid(64, STEEP_A)
    print(f"  fine shape={tg.shape}  coarse shape={tg.cshape}")
    verify_coarse_solve(tg)
    symmetry_gate(tg, 0.8, tg.coarse_solve_exact)

    print("=" * 60)
    print("STEP 4.1 — omega scan, V(1,1), near-exact coarse (n=64)")
    print("=" * 60)
    results = {}
    for omega in (0.6, 0.7, 0.8, 0.9, 1.0):
        results[omega] = two_grid_iters(
            tg, omega, tg.coarse_solve_exact, 1, 1,
            label="V(1,1) near-exact")
    best = min(results, key=lambda w: (results[w] or 999))
    print(f"  BEST omega={best} -> {results[best]} iters")
    print(f"  GB-0 verdict (best < 20): "
          f"{'PASS' if (results[best] or 999) < 20 else 'FAIL'}")

    print("=" * 60)
    print(f"STEP 4.2 — V(2,2) at best omega={best}")
    print("=" * 60)
    two_grid_iters(tg, best, tg.coarse_solve_exact, 2, 2,
                   label="V(2,2) near-exact")

    print("=" * 60)
    print("STEP 4.3 — production-shaped coarse (k line-Jacobi sweeps)")
    print("=" * 60)
    for k in (8, 16, 32):
        cs = lambda d, k=k: tg.coarse_solve_ksweeps(d, k, best)
        two_grid_iters(tg, best, cs, 1, 1,
                       label=f"V(1,1) coarse_sweeps={k}")

    print("=" * 60)
    print("STEP 4.4 — resolution spot-check at 32^3 (near-exact V(1,1))")
    print("=" * 60)
    tg32 = TwoGrid(32, STEEP_A)
    verify_coarse_solve(tg32)
    symmetry_gate(tg32, best, tg32.coarse_solve_exact)
    two_grid_iters(tg32, best, tg32.coarse_solve_exact, 1, 1,
                   label="32^3 V(1,1) near-exact")

    print("=" * 60)
    print("STEP 4.5 — mild 1.5x (a=0.2) at n=64")
    print("=" * 60)
    tgm = TwoGrid(64, MILD_A)
    symmetry_gate(tgm, best, tgm.coarse_solve_exact)
    two_grid_iters(tgm, best, tgm.coarse_solve_exact, 1, 1,
                   label="mild V(1,1) near-exact")

    print(f"\nTOTAL wall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
