"""nsys/dump harness: profile one advance() region per storage width.

Mirrors design/research/storage_halo_gpu_ab/ab_step.py build path, but
wraps a cudaProfilerStart/Stop region so `nsys profile
--capture-range=cudaProfilerApi` records only the timed steps (compile
excluded). Also supports plain timing (no nsys) for reproduce runs.

Usage: python prof_step.py <n> <scheme> <variant> [n_prof_steps]
  scheme  in {upwind5, weno5, centered}
  variant in {natural, wide, w<X>,<Y>,<Z>}
"""
from __future__ import annotations

import contextlib
import ctypes
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

def _load_cudart():
    import glob
    import os
    cands = ["libcudart.so", "libcudart.so.12", "libcudart.so.13"]
    for base in (os.path.dirname(os.path.dirname(__file__)),):
        pass
    # search installed nvidia pip package
    for root in sys.path:
        cands += glob.glob(
            os.path.join(root, "nvidia", "cuda_runtime", "lib",
                         "libcudart.so*"))
    for c in cands:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    return None


_CUDART = _load_cudart()
print(f"[prof] cudart loaded: {_CUDART is not None}", flush=True)


def cuda_profiler_start():
    if _CUDART is not None:
        _CUDART.cudaProfilerStart()


def cuda_profiler_stop():
    if _CUDART is not None:
        _CUDART.cudaProfilerStop()


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


def main(n, scheme, variant, n_prof):
    with ctx_for(scheme, variant):
        model = build(n, scheme)
        d = model.state["u"].grid.decomposition
        w = {name: d.halo[name] for name in ("x", "y", "z")}
        jax.block_until_ready(jax.tree_util.tree_leaves(model.state))
        # compile / warmup (excluded)
        tic = perf_counter()
        model.advance(STEPS)
        jax.block_until_ready(jax.tree_util.tree_leaves(model.state))
        first = perf_counter() - tic
        # profiled region
        cuda_profiler_start()
        tic = perf_counter()
        for _ in range(n_prof):
            model.advance(STEPS)
        jax.block_until_ready(jax.tree_util.tree_leaves(model.state))
        prof_wall = perf_counter() - tic
        cuda_profiler_stop()
    per_step = prof_wall / (n_prof * STEPS) * 1e3
    mem = {}
    for _key, (_, m) in _CHUNK_COMPILE_LOG.items():
        if m is not None:
            mem = {"temp_MB": round(m.temp_size_in_bytes / 1e6, 1),
                   "arg_MB": round(m.argument_size_in_bytes / 1e6, 1)}
    print(f"PROFRESULT n={n} scheme={scheme} variant={variant} "
          f"widths={w} per_step~={per_step:.3f} ms "
          f"prof_wall={prof_wall:.2f}s first={first:.1f}s mem={mem}",
          flush=True)


if __name__ == "__main__":
    n_prof = int(sys.argv[4]) if len(sys.argv) > 4 else 20
    main(int(sys.argv[1]), sys.argv[2], sys.argv[3], n_prof)
