"""Reproduce harness: per-rep step times, one variant per process.

Extends design/research/storage_halo_gpu_ab/ab_step.py to print the
full per-rep vector (so within-process stability is separable from
cross-process wobble) and the raw halo widths / storage shape.

Usage: python rep_step.py <n> <scheme> <variant> [reps]
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
TWO_PI = 2.0 * np.pi

SCHEMES = {
    "upwind5": lambda: UpwindAdvection(5),
    "weno5": lambda: WENOAdvection(5),
    "centered": CenteredAdvection,
}
OLD_WIDTH = {"upwind5": 4, "weno5": 4, "centered": 2}


@contextlib.contextmanager
def floor_halo(floors):
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


def ctx_for(scheme, variant):
    if variant == "natural":
        return contextlib.nullcontext()
    if variant == "wide":
        return floor_halo(dict.fromkeys("xyz", OLD_WIDTH[scheme]))
    fx, fy, fz = (int(t) for t in variant[1:].split(","))
    return floor_halo({"x": fx, "y": fy, "z": fz})


def main(n, scheme, variant, reps):
    with ctx_for(scheme, variant):
        model = build(n, scheme)
        d = model.state["u"].grid.decomposition
        w = {name: d.halo[name] for name in ("x", "y", "z")}
        store = tuple(n + 2 * w[a] for a in ("x", "y", "z"))
        jax.block_until_ready(jax.tree_util.tree_leaves(model.state))
        tic = perf_counter()
        model.advance(STEPS)
        jax.block_until_ready(jax.tree_util.tree_leaves(model.state))
        first = perf_counter() - tic
        times = []
        for _ in range(reps):
            tic = perf_counter()
            model.advance(STEPS)
            jax.block_until_ready(jax.tree_util.tree_leaves(model.state))
            times.append((perf_counter() - tic) / STEPS * 1e3)
    mem = {}
    for _key, (_, m) in _CHUNK_COMPILE_LOG.items():
        if m is not None:
            mem = {"temp_MB": round(m.temp_size_in_bytes / 1e6, 1),
                   "arg_MB": round(m.argument_size_in_bytes / 1e6, 1)}
    med = statistics.median(times)
    reps_str = ",".join(f"{t:.3f}" for t in times)
    print(f"REPRESULT n={n} scheme={scheme} variant={variant} "
          f"widths={w} store={store} med={med:.3f} min={min(times):.3f} "
          f"max={max(times):.3f} first={first:.1f}s "
          f"reps=[{reps_str}] mem={mem}", flush=True)


if __name__ == "__main__":
    reps = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], reps)
