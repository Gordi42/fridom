r"""Attribution repro for the 4 multigrid failures in
tests/nonhydro2/test_mapped_pressure_stretched.py under forced-4-GPU.

Hypothesis: the failures are the DOCUMENTED cuSPARSE-GSPMD limitation
(banded.py:498-502 "the cusparse path lowers to a custom call whose
GSPMD partitioning is unvalidated ... prefer method='pcr'"), NOT a
stretched+terrain defect. The multigrid preconditioner defaults to
multigrid_tridiagonal_method="auto" -> cusparse on GPU; under 4-GPU
GSPMD (x-axis sharded) the cusparse custom call mis-partitions.

This script reproduces the vcycle-symmetry check and the build+solve
check with the DEFAULT (auto/cusparse) vs method="pcr", on whatever
device count jax sees. Run with JAX_PLATFORMS=cuda and the fusion
workaround, once with all 4 GPUs and once with CUDA_VISIBLE_DEVICES=0.
"""
import jax
import jax.numpy as jnp
import numpy as np

from fridom.nonhydro2.modules.core import fv_cgrid_overrides
from fridom.nonhydro2.modules.mapped_pressure import MappedPressureSolver
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

DSQR = 0.25


def depth(x):
    return 1.0 + 0.2 * jnp.sin(x)


def stretch(sigma):
    return sigma + 0.15 * jnp.sin(2 * np.pi * sigma) / (2 * np.pi)


def build_mg_grid(nx, nz):
    mx = IntervalMesh(nx, (0.0, 2 * np.pi), periodic=True, name="x")
    ms = MappedIntervalMesh(nz, (0.0, 1.0), stretch, periodic=False,
                            name="sigma")
    mapping = CoordinateMapping(maps={"zp": lambda sigma, H: sigma * H},
                                params={"H": depth})
    grid = Grid((mx, ms), mapping=mapping)
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    return grid, mx, ms


def cell_space(mx, ms):
    return mx.cell_avg * ms.cell_avg


def dot(a, b):
    return float(jnp.sum((a * b).integrate().data))


def check(method):
    grid, mx, ms = build_mg_grid(16, 8)
    space = cell_space(mx, ms)
    kw = {"preconditioner": "multigrid", "multigrid_levels": 5,
          "weights": {"sigma": 1.0 / DSQR}}
    if method is not None:
        kw["multigrid_tridiagonal_method"] = method
    # vcycle symmetry
    solver = MappedPressureSolver(grid, space, iterations=5, **kw)
    vcycle = solver._build_vcycle({})
    u = grid.random.normal(space, seed=1)
    u = u - u.mean()
    v = grid.random.normal(space, seed=2)
    v = v - v.mean()
    left = dot(vcycle(u), v)
    right = dot(u, vcycle(v))
    asym = abs(left - right)
    # build + solve residual reduction
    solver2 = MappedPressureSolver(grid, space, iterations=20,
                                   tolerance=None, **kw)
    rhs = grid.random.normal(space, seed=7)
    rhs = rhs - rhs.mean()
    p = solver2.solve(rhs)
    r0 = float(jnp.abs(rhs.data).max())
    r_end = float(jnp.abs((solver2.apply(p) - rhs).data).max())
    return asym, abs(left), r_end / r0


if __name__ == "__main__":
    nd = jax.device_count()
    print(f"jax.device_count() = {nd}; devices = {jax.devices()}")
    for method in [None, "cusparse", "pcr", "scan"]:
        label = method if method is not None else "auto(default)"
        asym, mag, rel = check(method)
        sym_ok = asym <= 1e-12 * mag
        solve_ok = rel < 1e-8
        print(f"method={label:16s} vcycle_asym={asym:.3e} "
              f"(<= {1e-12 * mag:.2e}? {sym_ok})  "
              f"solve_rel_resid={rel:.3e} (<1e-8? {solve_ok})")
