r"""Leg B smoke: combined stretched-column + terrain-following nonhydro2
model, 1-GPU vs 4-GPU device-count invariance.

Builds a full FV nonhydro2 model on a 3D grid whose vertical base mesh
``z`` is a stretched ``MappedIntervalMesh`` and whose ``zp = z * H(x)``
terrain mapping follows a wavy bottom -- the combined stretch+terrain
grid. Horizontal x/y are periodic (they shard first under GSPMD); the
mapped column stays undistributed by design.

Preconditioner = "none" (plain-CG stopgap): the spectral preconditioner
is rejected on a stretched base column, and the multigrid default
(cuSPARSE) is unvalidated under GSPMD (banded.py:498-502). Plain CG is
pure-jax, partitions cleanly, and exercises the new N2 measure-adjoint
base-axis down-hop that is the actual stretched+terrain pressure path.

Usage:
  1 GPU: CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda python smoke_... --save
  4 GPU: JAX_PLATFORMS=cuda python smoke_...
Always with XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion.
"""
import argparse
import os

import jax
import jax.numpy as jnp
import numpy as np

import fridom.nonhydro2 as nh
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.meshes.mapped_interval import MappedIntervalMesh

TWO_PI = 2 * np.pi
ART = os.path.dirname(os.path.abspath(__file__))
COMPS = ("u", "v", "w", "b")


def depth(x):
    """Wavy terrain bottom H(x) (20% amplitude)."""
    return 1.0 + 0.2 * jnp.sin(x)


def stretch(z):
    """Monotone sigma clustering (dS/dz > 0)."""
    return z + 0.15 * jnp.sin(2 * np.pi * z) / (2 * np.pi)


def build_model(nx=32, ny=32, nz=16, *, iters=80, tol=1e-10,
                advection=True):
    mx = IntervalMesh(nx, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(ny, (0.0, TWO_PI), periodic=True, name="y")
    mz = MappedIntervalMesh(nz, (0.0, 1.0), stretch, periodic=False,
                            name="z")
    mapping = CoordinateMapping(maps={"zp": lambda z, H: z * H},
                                params={"H": depth})
    grid = Grid((mx, my, mz), mapping=mapping)
    model = nh.Model(
        grid=grid, dt=0.02,
        coriolis=nh.FPlaneCoriolis(f0=1.0),
        stratification=nh.ConstantStratification(n2=1.0),
        advection=advection,
        dsqr=0.25,
        family="fv",
        pressure_preconditioner="none",
        pressure_iterations=iters,
        pressure_tolerance=tol,
    )
    model.set_fields(
        u=lambda x, y, z: jnp.sin(x) * jnp.cos(y),
        v=lambda x, y, z: 0.3 * jnp.cos(x) * jnp.sin(y),
        w=lambda x, y, z: 0.2 * jnp.sin(jnp.pi * z),
        b=lambda x, y, z: 0.5 * jnp.cos(x) * jnp.sin(jnp.pi * z))
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--linear", action="store_true")
    ap.add_argument("--save", action="store_true",
                    help="save final fields as .npy reference (1-GPU run)")
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()

    nd = jax.device_count()
    print(f"jax.device_count() = {nd}; devices = {jax.devices()}")
    print(f"config: iters={args.iters} tol={args.tol} "
          f"advection={not args.linear}")

    model = build_model(iters=args.iters, tol=args.tol,
                        advection=not args.linear)
    # report sharding of a prognostic field
    shard = model.state["u"].data.sharding
    print(f"state['u'] sharding: {shard}")
    try:
        print(f"  addressable shards: "
              f"{len(model.state['u'].data.addressable_shards)} "
              f"of {nd} devices")
    except Exception as exc:  # noqa: BLE001
        print(f"  (shard introspection: {exc})")

    model.advance(args.steps)
    print(f"panicked after {args.steps} steps: {model.panicked}")

    final = {c: np.asarray(model.state[c].data) for c in COMPS}
    for c in COMPS:
        print(f"  final {c}: shape={final[c].shape} "
              f"max_abs={np.abs(final[c]).max():.6e}")

    if args.save:
        for c in COMPS:
            np.save(os.path.join(ART, f"smoke_{args.tag}_{c}.npy"), final[c])
        print(f"saved reference .npy ({args.tag}) to {ART}")
    else:
        # compare against saved reference
        maxdiff = 0.0
        for c in COMPS:
            ref = np.load(os.path.join(ART, f"smoke_{args.tag}_{c}.npy"))
            d = float(np.abs(final[c] - ref).max())
            print(f"  max_abs_diff {c}: {d:.3e}")
            maxdiff = max(maxdiff, d)
        print(f"OVERALL max_abs_diff vs {args.tag}: {maxdiff:.3e}")


if __name__ == "__main__":
    main()
