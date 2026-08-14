"""Minimal reproduction of A0: advection blows up on a mapped grid.

Grid is the one from tests/model/modules/test_advection_mapped.py:
``zp = z * H``, ``H = 1 + 0.2 sin x`` on a 2-pi box.  Uniform u = 0.4,
n2 = 0, no friction.  Exact solution: steady uniform flow (the mapped
metric is the only thing that can break it).
"""
from __future__ import annotations

import sys

import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.model.model import Model as FrModel
from fridom.model.modules.advection import CenteredAdvection
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import Core
from fridom.nonhydro2.modules.stratification import ConstantStratification
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh

L = 2 * np.pi


def depth(x, amp=0.2):
    return 1.0 + amp * jnp.sin(x)


def make_mapped_model(n, dt, *, ny=4, amp=0.2, periodic_column=False):
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": lambda x: depth(x, amp)})
    grid = Grid((
        IntervalMesh(n, (0.0, L), name="x"),
        IntervalMesh(ny, (0.0, L), name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=periodic_column, name="z"),
    ), mapping=mapping)
    return FrModel(grid=grid,
                   modules=(Core(), ConstantStratification(n2=0.0),
                            CenteredAdvection()),
                   time_stepper=AdamBashforth(dt, order=3))


def make_flat_model(n, dt, *, ny=4):
    grid = Grid((
        IntervalMesh(n, (0.0, L), name="x"),
        IntervalMesh(ny, (0.0, L), name="y"),
        IntervalMesh(n, (0.0, 1.0), periodic=False, name="z"),
    ))
    return FrModel(grid=grid,
                   modules=(Core(), ConstantStratification(n2=0.0),
                            CenteredAdvection()),
                   time_stepper=AdamBashforth(dt, order=3))


def run(model, u0, steps, report):
    model.set_fields(u=u0)
    out = []
    for k in range(0, steps + 1, report):
        if k:
            model.advance(steps=report)
        u = np.asarray(model.state["u"].data)
        amax = np.abs(u).max()
        iz = int(np.unravel_index(np.abs(u).argmax(), u.shape)[2])
        out.append((k, amax, iz))
        print(f"  step {k:4d}  |u|max = {amax:.6e}  iz = {iz}")
        if not np.isfinite(amax):
            break
    return out


def main():
    U = 0.4
    for n, dt in ((32, 0.02), (32, 0.005), (64, 0.01)):
        print(f"mapped n={n} dt={dt}  (u dt/dx = {U * dt / (L / n):.3f})")
        m = make_mapped_model(n, dt)
        run(m, U, 50, 25 if dt > 0.01 else 50)
    print("flat n=32 dt=0.02")
    m = make_flat_model(32, 0.02)
    run(m, U, 100, 50)


if __name__ == "__main__":
    sys.exit(main())
