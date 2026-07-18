"""Empirical probe: force extra_halo down on the pressure/constraint
solver paths and compare final states."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import contextlib
import traceback

import numpy as np
import jax.numpy as jnp

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.nonhydro2.modules.core import DynamicalCore
from fridom.model.modules.advection import (
    CenteredAdvection, UpwindAdvection)

N = 16
TWO_PI = 2 * np.pi
COMPS = ("u", "v", "w", "b")


@contextlib.contextmanager
def forced_extra_halo(cls, width, coord_attr="_coords"):
    """Temporarily override cls.extra_halo to a fixed width (or restore)."""
    orig = cls.__dict__.get("extra_halo", None)

    def prop(self):
        if width is None:
            return None
        coords = getattr(self, coord_attr)
        return HaloSpec(dict.fromkeys(coords, width))

    cls.extra_halo = property(prop)
    try:
        yield
    finally:
        if orig is None:
            del cls.extra_halo
        else:
            cls.extra_halo = orig


def make_grid(*, periodic=("x", "y", "z"), mapped=False):
    meshes = []
    for name in ("x", "y", "z"):
        length = TWO_PI if name != "z" else 1.0
        meshes.append(IntervalMesh(
            N, (0.0, length), periodic=(name in periodic), name=name))
    mapping = None
    if mapped:
        mapping = CoordinateMapping(
            maps={"zp": lambda z, H: z * H},
            params={"H": lambda x: 1.0 + 0.2 * jnp.sin(x)})
    return Grid(tuple(meshes), mapping=mapping)


def build(config, adv_factory, family):
    grid = make_grid(periodic=config["periodic"], mapped=config["mapped"])
    kw = dict(grid=grid, dt=0.01, dsqr=0.5,
              coriolis=nh.FPlaneCoriolis(f0=1.0), chunk_size=1)
    if family is not None:
        kw["family"] = family
    if config["mapped"]:
        kw["pressure_iterations"] = 12
    return nh.Model(advection=adv_factory(), **kw)


def run_config(name, config, adv_factory, family, seed=0):
    """Baseline (width 2) vs forced 1 vs forced 0. Return dict of results."""
    results = {}
    # baseline: no override, the hardcoded extra_halo=2
    base_state = None
    for label, width in (("base", None), ("w1", 1), ("w0", 0)):
        try:
            ctx = (contextlib.nullcontext() if width is None
                   else forced_extra_halo(DynamicalCore, width))
            with ctx:
                model = build(config, adv_factory, family)
                rng = np.random.default_rng(seed)
                model.set_fields(**{
                    c: 0.1 * rng.standard_normal(model.state[c].data.shape)
                    for c in COMPS})
                # record negotiated storage halo width along x
                halo = model._carry.state["u"].grid.decomposition.halo
                hx = halo["x"]
                model.run(steps=10, progress=False)
                st = {c: np.asarray(model.state[c].data) for c in COMPS}
                finite = all(np.isfinite(v).all() for v in st.values())
                if label == "base":
                    base_state = st
                    results[label] = dict(halo=hx, finite=finite,
                                          maxabs=max(float(np.max(np.abs(v)))
                                                     for v in st.values()))
                else:
                    md = max(float(np.max(np.abs(st[c] - base_state[c])))
                             for c in COMPS) if base_state else float("nan")
                    results[label] = dict(halo=hx, finite=finite, maxdiff=md)
        except Exception as exc:  # noqa: BLE001
            results[label] = dict(error=f"{type(exc).__name__}: {exc}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()
    return results


def fmt(name, adv_name, res):
    print(f"\n### {name} | advection={adv_name}")
    for label in ("base", "w1", "w0"):
        r = res.get(label, {})
        if "error" in r:
            print(f"  {label:4s}: ERROR {r['error'][:120]}")
        elif label == "base":
            print(f"  {label:4s}: halo_x={r['halo']} finite={r['finite']} "
                  f"maxabs={r['maxabs']:.4e}")
        else:
            print(f"  {label:4s}: halo_x={r['halo']} finite={r['finite']} "
                  f"maxdiff_vs_base={r['maxdiff']:.3e}")


CONFIGS = {
    "triperiodic-spectral": dict(periodic=("x", "y", "z"), mapped=False),
    "walled-y-spectral":    dict(periodic=("x", "z"), mapped=False),
    "mapped-terrain-CG":    dict(periodic=("x", "y"), mapped=True),
}

# advection factories (fresh module per build) + required family
ADV = {
    "linear": (lambda: False, None),
    "centered": (lambda: CenteredAdvection(), None),
    "upwind5": (lambda: UpwindAdvection(order=5), "nodal"),
}


def main():
    for cname, cfg in CONFIGS.items():
        for aname, (fac, family) in ADV.items():
            res = run_config(cname, cfg, fac, family)
            fmt(cname, aname, res)


if __name__ == "__main__":
    main()
