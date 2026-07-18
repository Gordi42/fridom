"""GPU wall-clock A/B: stock (width 2, storage n+4) vs narrow
(DynamicalCore.extra_halo forced to 1 -> width 1, storage n+2).

Centered advection, triperiodic flat nonhydro2, family="nodal".
One variant per fresh process. Methodology copied from
storage_halo_gpu_ab/ab_step.py: jet ICs, CFL dt, chunk_size=STEPS,
first advance (compile) excluded, then REPS timed advance(STEPS) calls;
reports per-step ms median/min/max plus the compiled chunk's
memory_analysis bytes.

The narrow technique adapts pressure_solver_halo/probe.py: patch
DynamicalCore.extra_halo to a property returning HaloSpec of the given
width. This FORCES the declaration DOWN (the opposite of ab_step.py's
floor_halo which can only widen).

Usage: python ab_centered.py <n> <variant> <scheme>
  variant in {stock, narrow}
  scheme  in {centered, linear}   (default centered)
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
from fridom.model.modules.advection import CenteredAdvection
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.spatial.decomposition.halo import HaloSpec

STEPS = 50
REPS = 5
TWO_PI = 2.0 * np.pi


@contextlib.contextmanager
def forced_extra_halo(cls, width):
    """Force cls.extra_halo DOWN to a fixed per-coord width."""
    orig = cls.__dict__.get("extra_halo", None)

    def prop(self):
        return HaloSpec(dict.fromkeys(self._coords, width))

    cls.extra_halo = property(prop)
    try:
        yield
    finally:
        if orig is None:
            with contextlib.suppress(AttributeError):
                del cls.extra_halo
        else:
            cls.extra_halo = orig


def build(n, scheme):
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI), periodic=True,
                                        name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI), periodic=True,
                                        name="y")
    mz = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), periodic=True,
                                        name="z")
    grid = fr.spatial.Grid((mx, my, mz))
    dt = 0.25 * TWO_PI / n
    advection = False if scheme == "linear" else CenteredAdvection()
    model = nh.Model(grid=grid, dt=dt, advection=advection,
                     coriolis=nh.FPlaneCoriolis(f0=1.0), dsqr=0.25,
                     chunk_size=STEPS, family="nodal")
    hor = (np.arange(n) + 0.5) * (TWO_PI / n)
    ver = (np.arange(n) + 0.5) / n
    x, y, z = np.meshgrid(hor, hor, ver, indexing="ij")
    model.set_fields(u=np.sin(x) * np.cos(y), v=0.3 * np.cos(x),
                     b=0.01 * np.cos(np.pi * z))
    return model


def main(n, variant, scheme):
    if variant == "stock":
        ctx = contextlib.nullcontext()
    elif variant == "narrow":
        ctx = forced_extra_halo(DynamicalCore, 1)
    else:
        raise SystemExit(f"unknown variant {variant!r}")

    with ctx:
        model = build(n, scheme)
        d = model.state["u"].grid.decomposition
        w = {name: d.halo[name] for name in ("x", "y", "z")}
        # Verify the seam did what we expect before timing. The
        # negotiated width is authoritative: the n+2w storage array is
        # materialized only inside the compiled chunk (host state is
        # always compute-shape), so storage is confirmed via width here
        # and via memory_analysis bytes below.
        expect = 1 if variant == "narrow" else 2
        store = tuple(n + 2 * w[a] for a in ("x", "y", "z"))
        print(f"CHECK n={n} variant={variant} scheme={scheme} "
              f"widths={w} derived_storage={store}", flush=True)
        if w != {"x": expect, "y": expect, "z": expect}:
            raise SystemExit(
                f"ABORT: negotiated width {w} != expected "
                f"{expect} for variant {variant}; patch missed the seam")

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
    print(f"RESULT n={n} variant={variant} scheme={scheme} "
          f"widths={w} storage={store} "
          f"med={med:.3f} ms/step min={min(times):.3f} "
          f"max={max(times):.3f} first_advance={first:.1f}s "
          f"reps={[round(t, 3) for t in times]} mem={mem}",
          flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1])
    variant = sys.argv[2]
    scheme = sys.argv[3] if len(sys.argv) > 3 else "centered"
    main(n, variant, scheme)
