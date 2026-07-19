r"""Mapped + immersed composition: single-process validation (Legs N, H).

Builds and steps the two shipped composed-grid model paths and, with
``--write-ref``, saves their final states as ``.npy`` references for the
real-multi-process sibling ``srun_mapped_immersed.py`` to compare against
under sharding.

* **Leg N (nonhydro2 composed)** — a 3-D FV nonhydro2 model on a grid
  carrying BOTH a terrain-following ``zp = z H(x)`` chart AND an immersed
  domain whose sloped analytic bottom cuts genuinely through cells
  (partial fractions, not a staircase). The pressure preconditioner is
  left at the model DEFAULT (``None`` = auto), which on a composed grid
  resolves to the composed multigrid V-cycle — validating the default
  path is the point. ``n2 = 0`` makes ``b`` a pure advected tracer so
  its ``theta``-``J``-weighted mass is a genuine conservation number.

* **Leg H (hydrostatic M5)** — a hydrostatic model on a terrain chart +
  immersed wet columns with the ``ImplicitFreeSurface`` (the wet-column
  barotropic solve, default masked spectral preconditioner). Steps under
  ``advection = False`` / ``n2 = 0`` (the validated M5 configuration), so
  ``b`` is inert and its masked physical integral is a machine-zero
  drift; the invariance content is ``u``, ``v``, ``ps``.

The ``build_model_N`` / ``build_model_H`` builders and their helper
functions are imported VERBATIM by ``srun_mapped_immersed.py`` (which
runs each leg under a real ``srun -n 4`` launch), so the two harnesses
step byte-identical models. The iteration-count probes are single-process
diagnostics and live here only.

Usage:
  1 GPU ref:  CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda \
                python smoke_mapped_immersed.py --write-ref
  4 GPU inv:  JAX_PLATFORMS=cuda python smoke_mapped_immersed.py
Always with XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion on GPU.
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr
import fridom.hydrostatic as hy
import fridom.nonhydro2 as nh
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi
ART = os.path.dirname(os.path.abspath(__file__))

COMPS_N = ("u", "v", "w", "b")
COMPS_H = ("u", "v", "ps")

# committed run parameters (the REAL defaults — the local CPU gate may
# shrink --steps for speed, but the reference is written at these).
STEPS = 12
DT = 0.01
DSQR_N = 0.5
CSQR_H = 3.0
ITERS_N, TOL_N = 60, 1e-11
ITERS_H, TOL_H = 60, 1e-10
A_H = 0.4


# ================================================================
#  Geometry (shared verbatim with srun_mapped_immersed.py)
# ================================================================
def depth_N(x):
    """Wavy terrain bottom H(x) for Leg N (25% amplitude)."""
    return 1.0 + 0.25 * jnp.sin(x)


def cut_N(x, y, z):  # noqa: ARG001
    """Sloped immersed bottom carving partial cells (Leg N)."""
    return jnp.clip((z - 0.15 - 0.1 * jnp.sin(x)) * 6.0 + 0.5, 0.0, 1.0)


def depth_H(x, y):
    """Doubly-wavy terrain depth H(x, y) for Leg H (40% amplitude)."""
    return 1.0 + A_H * jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


def cut_H(x, y, z):  # noqa: ARG001
    """Sloped partial bottom in computational z on (-1, 0) (Leg H)."""
    return jnp.clip((z - (-0.5 + 0.1 * jnp.sin(2 * jnp.pi * x))) / (1.0 / 8)
                    + 0.5, 0.0, 1.0)


# ================================================================
#  Model builders (shared verbatim with srun_mapped_immersed.py)
# ================================================================
def build_model_N(nx=32, ny=32, nz=16, *, iters=ITERS_N, tol=TOL_N):
    """Leg N: FV nonhydro2 on a terrain chart + immersed cut grid."""
    mx = IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(nz, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                params={"H": depth_N})
    grid = Grid((mx, my, mz), mapping=mapping,
                immersed=ImmersedDomain(cut_N, order=4, min_fraction=0.1))
    model = nh.Model(
        grid=grid, dt=DT, advection=True, family="fv",
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=0.0),
        dsqr=DSQR_N,
        # pressure_preconditioner left at default (None = auto -> the
        # composed multigrid V-cycle): validating the default path.
        pressure_iterations=iters, pressure_tolerance=tol)
    model.set_fields(
        u=lambda x, y, z: jnp.sin(x) * jnp.cos(y),
        v=lambda x, y, z: 0.3 * jnp.cos(x) * jnp.sin(y),
        w=lambda x, y, z: 0.2 * jnp.sin(jnp.pi * z),
        b=lambda x, y, z: 0.5 * jnp.cos(x) * jnp.sin(jnp.pi * z))
    return model


def build_model_H(nx=64, ny=64, nz=16, *, iters=ITERS_H, tol=TOL_H):
    """Leg H: hydrostatic terrain + immersed wet columns, implicit FS."""
    mx = IntervalMesh(nx, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, 1.0), periodic=True, name="y")
    mz = IntervalMesh(nz, (-1.0, 0.0), periodic=False, name="z")
    mapping = CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                params={"H": depth_H})
    grid = Grid((mx, my, mz), mapping=mapping,
                immersed=ImmersedDomain(cut_H, order=4, min_fraction=0.1))
    model = hy.Model(
        grid=grid, dt=DT, csqr=CSQR_H,
        stratification=hy.ConstantStratification(n2=0.0),
        coriolis=hy.FPlaneCoriolis(f0=0.5), advection=False,
        # default masked spectral preconditioner (ImplicitFreeSurface
        # default pressure_preconditioner="spectral").
        free_surface=hy.ImplicitFreeSurface(
            epsilon=1.0, pressure_iterations=iters, pressure_tolerance=tol),
        time_stepper=fr.model.time_steppers.AdamBashforth(DT, order=2))
    model.set_fields(
        u=lambda x, y, z: 0.2 * jnp.sin(2 * jnp.pi * x)
        * jnp.cos(2 * jnp.pi * y),
        v=lambda x, y, z: 0.06 * jnp.cos(2 * jnp.pi * x)
        * jnp.sin(2 * jnp.pi * y),
        b=lambda x, y, z: 0.1 * jnp.cos(2 * jnp.pi * x)
        * jnp.sin(jnp.pi * (z + 1.0)))
    return model


def theta_mass(model):
    r"""Return the immersed-``theta``-weighted physical mass of ``b``.

    On a chart ``.integrate()`` is ``J``-weighted, so this is exactly
    ``sum_c theta_c J_c V_c b_c`` — the conserved tracer content (Leg N,
    advection on / n2 = 0) or an inert machine-zero-drift baseline
    (Leg H). The integral is a global reduction (a replicated scalar
    under sharding), so ``float(...)`` is addressable multi-process.
    """
    theta = model.grid.immersed.fraction(model.state["b"].function_space)
    return float(jnp.sum((theta * model.state["b"]).integrate().data))


# ================================================================
#  Iteration-count probes (single-process diagnostics only)
# ================================================================
def _cell_space(grid):
    """Return the FV cell-average pressure space (grid.factors meshes)."""
    factors = [mesh.cell_avg for mesh in grid.factors]
    space = factors[0]
    for factor in factors[1:]:
        space = space * factor
    return grid._laid_out(space)


def probe_iters_N(nx=32, ny=32, nz=16, *, dsqr=DSQR_N, budget=200,
                  tol=TOL_N):
    """Report the composed multigrid PCG achieved iterations (Leg N)."""
    from fridom.nonhydro2.modules.composed_pressure import (  # noqa: PLC0415
        ComposedPressureSolver,
    )
    from fridom.nonhydro2.modules.core import (  # noqa: PLC0415
        fv_cgrid_overrides,
    )
    mx = IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(nz, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                params={"H": depth_N})
    grid = Grid((mx, my, mz), mapping=mapping,
                immersed=ImmersedDomain(cut_N, order=4, min_fraction=0.1))
    grid.merge_overrides(fv_cgrid_overrides(grid.factors))
    space = _cell_space(grid)
    solver = ComposedPressureSolver(
        grid, space, weights={"z": 1.0 / dsqr}, iterations=budget,
        tolerance=tol, preconditioner="multigrid")
    rand = grid.random.normal(space, seed=0)
    rhs = solver.apply(rand)               # a range-compatible RHS
    _p, info = solver.krylov().solve(rhs)
    return int(info["iterations"]), float(info["residual_norm"])


def probe_iters_H(nx=64, ny=64, nz=16, *, budget=60, tol=TOL_H,
                  csqr=CSQR_H, dt=DT):
    """Report the wet-masked spectral PCG achieved iterations (Leg H)."""
    from fridom.hydrostatic.modules.barotropic_pressure import (  # noqa: PLC0415
        BarotropicPressureSolver,
    )
    mx = IntervalMesh(nx, (0.0, 1.0), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, 1.0), periodic=True, name="y")
    mz = IntervalMesh(nz, (-1.0, 0.0), periodic=False, name="z")
    mapping = CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                params={"H": depth_H})
    grid = Grid((mx, my, mz), mapping=mapping,
                immersed=ImmersedDomain(cut_H, order=4, min_fraction=0.1))
    ps_space = fr.spatial.Profile("x", "y").resolve(grid).bare
    solver = BarotropicPressureSolver(
        grid, ps_space, ("zp", "z"), "z", epsilon=1.0, inv_depth=1.0,
        iterations=budget, tolerance=tol, preconditioner="spectral")
    template = grid.create_field(ps_space)
    rng = np.random.default_rng(4)
    rhs = template.with_data(jnp.asarray(
        rng.standard_normal(template.data.shape)))
    _ps, info = solver.krylov(csqr=jnp.asarray(csqr),
                              dt=jnp.asarray(dt)).solve(rhs)
    return int(info["iterations"]), float(info["residual_norm"])


# ================================================================
#  Driver
# ================================================================
def _run_leg(name, model, comps, steps, *, write_ref, tag):
    """Advance one leg; save or compare its final state; report drift."""
    mass_before = theta_mass(model)
    model.advance(steps)
    mass_after = theta_mass(model)
    drift = mass_after - mass_before
    print(f"[{name}] panicked after {steps} steps: {model.panicked}")
    print(f"[{name}] theta-mass: before={mass_before:.12e} "
          f"after={mass_after:.12e} drift={drift:.3e}")

    final = {c: np.asarray(model.state[c].data) for c in comps}
    for c in comps:
        print(f"[{name}]   final {c}: shape={final[c].shape} "
              f"max_abs={np.abs(final[c]).max():.6e}")

    if write_ref:
        for c in comps:
            np.save(os.path.join(ART, f"{tag}_{c}.npy"), final[c])
        print(f"[{name}] saved reference .npy ({tag}) to {ART}")
    else:
        maxdiff = 0.0
        for c in comps:
            ref = np.load(os.path.join(ART, f"{tag}_{c}.npy"))
            d = float(np.abs(final[c] - ref).max())
            print(f"[{name}]   max_abs_diff {c}: {d:.3e}")
            maxdiff = max(maxdiff, d)
        print(f"[{name}] OVERALL max_abs_diff vs {tag}: {maxdiff:.3e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-ref", action="store_true",
                    help="save final fields as .npy references (1-GPU run)")
    ap.add_argument("--steps", type=int, default=STEPS,
                    help="steps per leg (default: the committed STEPS)")
    ap.add_argument("--skip-probe", action="store_true",
                    help="skip the iteration-count probes")
    args = ap.parse_args()

    print(f"jax.device_count() = {jax.device_count()}; "
          f"devices = {jax.devices()}")
    print(f"config: steps={args.steps} write_ref={args.write_ref}")

    # ---- Leg N: nonhydro2 composed --------------------------------
    print("=== Leg N: nonhydro2 composed (terrain chart + immersed cut)")
    model_n = build_model_N()
    print(f"[N] state['u'] sharding: {model_n.state['u'].data.sharding}")
    _run_leg("N", model_n, COMPS_N, args.steps,
             write_ref=args.write_ref, tag="refN")
    if not args.skip_probe:
        it, res = probe_iters_N()
        print(f"[N] composed-multigrid PCG: iterations={it} "
              f"residual_norm={res:.3e}")

    # ---- Leg H: hydrostatic M5 ------------------------------------
    print("=== Leg H: hydrostatic terrain + immersed (implicit free surf)")
    model_h = build_model_H()
    print(f"[H] state['u'] sharding: {model_h.state['u'].data.sharding}")
    _run_leg("H", model_h, COMPS_H, args.steps,
             write_ref=args.write_ref, tag="refH")
    if not args.skip_probe:
        it, res = probe_iters_H()
        print(f"[H] wet-masked spectral PCG: iterations={it} "
              f"residual_norm={res:.3e}")


if __name__ == "__main__":
    main()
