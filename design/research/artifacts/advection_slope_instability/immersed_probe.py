"""The immersed (cut-cell) half of A0 — status probe.

The mapped fix (`_mapped_nodal_cross`) touches only the
``CoordinateMapping`` column path.  An immersed grid carries no mapping:
its geometry rides the open-area / volume fractions
(``_immersed_flux`` / ``_immersed_scale``).  This probe records whether
the immersed sloping-wall blow-up survives the mapped fix, and at which
slopes.
"""
from __future__ import annotations

import sys

import numpy as np

from fridom.model.model import Model as FrModel
from fridom.model.modules.advection import CenteredAdvection
from fridom.model.time_steppers.adam_bashforth import AdamBashforth
from fridom.nonhydro2.modules.core import Core
from fridom.nonhydro2.modules.stratification import ConstantStratification
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

L = 2 * np.pi


def ramp_grid(n, slope, ny=4):
    """A wedge floor of the given slope (wet above, dry below)."""
    def solid(x, y, z):  # noqa: ARG001
        return (z < slope * x).astype(float)

    return Grid((
        IntervalMesh(n, (0.0, L), name="x"),
        IntervalMesh(ny, (0.0, L), name="y"),
        IntervalMesh(n, (0.0, L), periodic=False, name="z"),
    ), immersed=ImmersedDomain(solid))


def model(n, dt, slope, *, ny=4):
    return FrModel(
        grid=ramp_grid(n, slope, ny=ny),
        modules=(Core(family="fv"),
                 ConstantStratification(n2=0.0, family="fv"),
                 CenteredAdvection()),
        time_stepper=AdamBashforth(dt, order=3))


def run(m, u0, steps, report):
    m.set_fields(u=u0)
    for k in range(0, steps + 1, report):
        if k:
            m.advance(steps=report)
        u = np.asarray(m.state["u"].data)
        amax = np.abs(u).max()
        print(f"  step {k:4d}  |u|max = {amax:.6e}", flush=True)
        if not np.isfinite(amax):
            return


def main():
    for slope, tag in ((0.34, "34%"), (0.04, "4%")):
        print(f"immersed wedge slope {tag}, n=32 dt=0.01", flush=True)
        try:
            run(model(32, 0.01, slope), 0.4, 120, 30)
        except Exception as e:  # noqa: BLE001
            print("  ", type(e).__name__, str(e)[:120], flush=True)


if __name__ == "__main__":
    sys.exit(main())
