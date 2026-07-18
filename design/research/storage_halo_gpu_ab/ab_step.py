"""GPU wall-clock A/B: natural (narrow) vs forced-wide storage halo.

One variant per process (fresh jit cache). Methodology copied from
benchmarks/model/bench_step.py: triperiodic n^3, jet ICs, CFL dt,
chunk_size=STEPS, first advance (compile) excluded, then REPS timed
advance(STEPS) calls; reports per-step ms median/min/max plus the
compiled chunk's memory_analysis bytes.

Usage: python ab_step.py <n> <scheme> <variant>
  scheme  in {upwind5, weno5, centered}
  variant in {natural, wide}   (wide = old scalar-accounting width:
                                biased -> 4, centered -> 2 [= A/A])
          or "w<X>,<Y>,<Z>" per-axis floors, e.g. w3,3,4
"""
from __future__ import annotations

import contextlib
import statistics
import sys
from time import perf_counter

import jax
import numpy as np

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.model import _CHUNK_COMPILE_LOG
from fridom.model.modules.advection import (
    CenteredAdvection,
    UpwindAdvection,
    WENOAdvection,
)

STEPS = 50
REPS = 5
TWO_PI = 2.0 * np.pi

SCHEMES = {
    "upwind5": lambda: UpwindAdvection(5),
    "weno5": lambda: WENOAdvection(5),
    "centered": CenteredAdvection,
}
OLD_WIDTH = {"upwind5": 4, "weno5": 4, "centered": 2}


@contextlib.contextmanager
def floor_halo(floors):
    """Raise negotiated/verified halo widths to per-name floors."""
    import fridom.spatial.decomposition.decomposition as _decomp
    from fridom.spatial.decomposition.halo import HaloSpec
    from fridom.spatial.grid import Grid

    def _raise(spec):
        return HaloSpec({name: max(width, floors.get(name, 0))
                         for name, width in spec.widths})

    orig_neg = _decomp._negotiated_halo
    orig_dem = Grid._demanded_halo
    _decomp._negotiated_halo = lambda *a, **k: _raise(orig_neg(*a, **k))
    Grid._demanded_halo = lambda self, *a, **k: _raise(
        orig_dem(self, *a, **k))
    try:
        yield
    finally:
        _decomp._negotiated_halo = orig_neg
        Grid._demanded_halo = orig_dem


def build(n, scheme):
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI), periodic=True,
                                        name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI), periodic=True,
                                        name="y")
    mz = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), periodic=True,
                                        name="z")
    grid = fr.spatial.Grid((mx, my, mz))
    dt = 0.25 * TWO_PI / n
    model = nh.Model(grid=grid, dt=dt, advection=SCHEMES[scheme](),
                     coriolis=nh.FPlaneCoriolis(f0=1.0), dsqr=0.25,
                     chunk_size=STEPS, family="nodal")
    hor = (np.arange(n) + 0.5) * (TWO_PI / n)
    ver = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(hor, hor, ver, indexing="ij")
    model.set_fields(u=np.sin(x) * np.cos(y), v=0.3 * np.cos(x),
                     b=0.01 * np.cos(np.pi * z))
    return model


def main(n, scheme, variant):
    if variant == "natural":
        ctx = contextlib.nullcontext()
    elif variant == "wide":
        ctx = floor_halo(dict.fromkeys("xyz", OLD_WIDTH[scheme]))
    else:
        fx, fy, fz = (int(t) for t in variant[1:].split(","))
        ctx = floor_halo({"x": fx, "y": fy, "z": fz})
    with ctx:
        model = build(n, scheme)
        d = model.state["u"].grid.decomposition
        w = {name: d.halo[name] for name in ("x", "y", "z")}
        leaves = jax.tree_util.tree_leaves(model.state)
        jax.block_until_ready(leaves)
        tic = perf_counter()
        model.advance(STEPS)
        jax.block_until_ready(jax.tree_util.tree_leaves(model.state))
        first = perf_counter() - tic
        times = []
        for _ in range(REPS):
            tic = perf_counter()
            model.advance(STEPS)
            jax.block_until_ready(jax.tree_util.tree_leaves(model.state))
            times.append((perf_counter() - tic) / STEPS * 1e3)
    mem = {}
    for key, (_, m) in _CHUNK_COMPILE_LOG.items():
        if m is not None:
            mem = {"temp_MB": m.temp_size_in_bytes / 1e6,
                   "arg_MB": m.argument_size_in_bytes / 1e6,
                   "out_MB": m.output_size_in_bytes / 1e6}
    med = statistics.median(times)
    print(f"RESULT n={n} scheme={scheme} variant={variant} "
          f"widths={w} med={med:.3f} ms/step "
          f"min={min(times):.3f} max={max(times):.3f} "
          f"first_advance={first:.1f}s mem={mem}",
          flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3])
