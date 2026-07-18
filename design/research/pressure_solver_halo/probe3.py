"""Probe 3: hydrostatic ImplicitFreeSurface barotropic 2D solve.
It ALREADY declares extra_halo=1 (the tight value). Confirm it runs and
that forcing to 0 vs 1 behaves like the nonhydro spectral path."""
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import contextlib
import traceback
import numpy as np

import fridom as fr
import fridom.hydrostatic as hy
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.hydrostatic.modules.free_surface import ImplicitFreeSurface

N = 16
TWO_PI = 2 * np.pi
COMPS = ("u", "v", "b")


@contextlib.contextmanager
def forced(cls, width, coord_attr):
    orig = cls.__dict__.get("extra_halo", None)

    def prop(self):
        return None if width is None else HaloSpec(
            dict.fromkeys(getattr(self, coord_attr), width))
    cls.extra_halo = property(prop)
    try:
        yield
    finally:
        if orig is None:
            with contextlib.suppress(AttributeError):
                del cls.extra_halo
        else:
            cls.extra_halo = orig


def make_grid():
    return Grid(tuple(IntervalMesh(
        N, (0.0, TWO_PI if n != "z" else 1.0),
        periodic=(n != "z"), name=n) for n in ("x", "y", "z")))


def run(width):
    m = hy.Model(grid=make_grid(), dt=2e-3, csqr=1.0,
                 free_surface=ImplicitFreeSurface(epsilon=1.0),
                 stratification=hy.ConstantStratification(n2=1.0),
                 advection=False, chunk_size=1)
    rng = np.random.default_rng(0)
    m.set_fields(**{c: 0.05 * rng.standard_normal(m.state[c].data.shape)
                    for c in COMPS})
    halo = m._carry.state["u"].grid.decomposition.halo
    m.run(steps=10, progress=False)
    st = {c: np.asarray(m.state[c].data) for c in COMPS}
    ps = np.asarray(m.state["ps"].data)
    return halo, st, ps


def main():
    print("### hydrostatic ImplicitFreeSurface (declares extra_halo=1)")
    base = None
    for label, width in (("base", None), ("w1", 1), ("w0", 0)):
        try:
            ctx = (contextlib.nullcontext() if width is None
                   else forced(ImplicitFreeSurface, width, "_horizontal"))
            with ctx:
                halo, st, ps = run(width)
                fin = all(np.isfinite(v).all() for v in st.values())
                if label == "base":
                    base = st
                    print(f"  {label}: halo(x,z)=({halo['x']},{halo['z']}) "
                          f"finite={fin} |ps|max={float(np.max(np.abs(ps))):.4e}")
                else:
                    md = max(float(np.max(np.abs(st[c] - base[c])))
                             for c in COMPS)
                    print(f"  {label}: halo(x,z)=({halo['x']},{halo['z']}) "
                          f"finite={fin} maxdiff_vs_base={md:.3e}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {label}: ERROR {type(exc).__name__}: {str(exc)[:150]}")
            if os.environ.get("VERBOSE"):
                traceback.print_exc()


if __name__ == "__main__":
    main()
