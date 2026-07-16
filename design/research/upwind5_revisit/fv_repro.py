"""Repro: FV default + biased (Upwind/WENO) advection on a periodic grid.

Constructs nonhydro2.Model with several advection/family combos on a
triply-periodic IntervalMesh grid, attempts construction + 1-step advance,
and reports success/failure with the last few fridom traceback frames.
"""
import traceback

import numpy as np

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

N = 8
DT = 0.02
LENGTH = 2 * np.pi


def periodic_grid(n=N):
    return Grid(tuple(
        IntervalMesh(n, (0.0, LENGTH), periodic=True, name=name)
        for name in ("x", "y", "z")))


def coords(n=N, length=LENGTH):
    ax = (np.arange(n) + 0.5) * (length / n)
    return np.meshgrid(ax, ax, ax, indexing="ij")


def last_fridom_frames(tb_text, k=6):
    lines = tb_text.splitlines()
    frames = [ln for ln in lines if "fridom" in ln and "File \"" in ln]
    return "\n".join(frames[-k:])


def run_case(label, advection, family):
    try:
        model = nh.Model(
            coriolis=FPlaneCoriolis(f0=1.0),
            grid=periodic_grid(), dt=DT, advection=advection, family=family)
        resolved = model.grid.default_family
        x, y, z = coords()
        model.set_fields(
            u=np.sin(x) * np.cos(y), v=0.3 * np.cos(x) * np.sin(z),
            w=0.2 * np.sin(z) * np.cos(y), b=0.1 * np.cos(x) * np.cos(z))
        model.advance(1)
        print(f"[OK] {label}: resolved family={resolved!r}, advance ok")
    except Exception as exc:  # noqa: BLE001
        tb = traceback.format_exc()
        print(f"[FAIL] {label}: {type(exc).__name__}: {exc}")
        print("  --- last fridom frames ---")
        print(last_fridom_frames(tb))
        print("  --------------------------")


def main():
    cases = [
        ("(a) default advection, family=None",
         True, None),
        ("(b) UpwindAdvection(order=5), family=None",
         nh.UpwindAdvection(order=5), None),
        ("(c) WENOAdvection(order=5), family=None",
         nh.WENOAdvection(order=5), None),
        ("(d-upwind) UpwindAdvection(order=5), family='nodal'",
         nh.UpwindAdvection(order=5), "nodal"),
        ("(d-weno) WENOAdvection(order=5), family='nodal'",
         nh.WENOAdvection(order=5), "nodal"),
        ("(e-upwind) UpwindAdvection(order=5), family='fv'",
         nh.UpwindAdvection(order=5), "fv"),
        ("(e-weno) WENOAdvection(order=5), family='fv'",
         nh.WENOAdvection(order=5), "fv"),
    ]
    for label, adv, fam in cases:
        run_case(label, adv, fam)


if __name__ == "__main__":
    main()
