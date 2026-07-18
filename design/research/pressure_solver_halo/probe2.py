"""Probe 2: (a) can width drop below 1 and break the projection?
(b) hydrostatic ImplicitFreeSurface barotropic solve at forced halo."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import contextlib
import traceback
import numpy as np

import fridom as fr
import fridom.nonhydro2 as nh
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.nonhydro2.modules.core import DynamicalCore

N = 16
TWO_PI = 2 * np.pi


@contextlib.contextmanager
def forced(cls, width):
    orig = cls.__dict__.get("extra_halo", None)

    def prop(self):
        return None if width is None else HaloSpec(
            dict.fromkeys(self._coords, width))
    cls.extra_halo = property(prop)
    try:
        yield
    finally:
        if orig is None:
            with contextlib.suppress(AttributeError):
                del cls.extra_halo
        else:
            cls.extra_halo = orig


def triperiodic():
    return Grid(tuple(IntervalMesh(N, (0.0, TWO_PI if n != "z" else 1.0),
                                   periodic=True, name=n)
                      for n in ("x", "y", "z")))


def probe_no_coriolis():
    """No Coriolis, no advection: does the projection alone floor width?
    Stratification stays (b is required); its buoyancy coupling reaches
    z only. So x should be free to drop toward 0 -> projection break."""
    print("### nonhydro2, coriolis=None, advection=False "
          "(isolate projection reach)")
    for width in (None, 1, 0):
        try:
            with (contextlib.nullcontext() if width is None
                  else forced(DynamicalCore, width)):
                m = nh.Model(grid=triperiodic(), dt=0.01, dsqr=0.5,
                             coriolis=None, advection=False, chunk_size=1)
                rng = np.random.default_rng(0)
                m.set_fields(**{c: 0.1 * rng.standard_normal(
                    m.state[c].data.shape) for c in ("u", "v", "w", "b")})
                halo = m._carry.state["u"].grid.decomposition.halo
                m.run(steps=10, progress=False)
                u = np.asarray(m.state["u"].data)
                print(f"  force={width}: halo(x,y,z)="
                      f"({halo['x']},{halo['y']},{halo['z']}) "
                      f"finite={np.isfinite(u).all()} "
                      f"|u|max={float(np.max(np.abs(u))):.4e}")
        except Exception as exc:  # noqa: BLE001
            print(f"  force={width}: ERROR {type(exc).__name__}: "
                  f"{str(exc)[:150]}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()


if __name__ == "__main__":
    probe_no_coriolis()
