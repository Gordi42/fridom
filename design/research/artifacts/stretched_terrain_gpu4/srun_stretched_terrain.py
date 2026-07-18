r"""Leg C: real multi-process (srun -n 4) stretched+terrain nonhydro2.

Same combined stretched-column + terrain model as smoke_stretched_terrain.py
but under a REAL multi-process launch: N OS processes, one GPU each, every
array sharded across processes. jax.distributed.initialize() is called
BEFORE importing fridom (fridom touches the backend at import). Bare
initialize() SLURM auto-detect segfaults on this node (IPv6 [::] bind), so
we initialize explicitly per AGENTS.md.

Final global fields are gathered with process_allgather(tiled=True) -- never
np.asarray a global sharded array. Rank 0 loads the Leg-B 1-GPU .npy
reference and prints the max-abs difference.

Launch (from the main checkout, inside the allocation):
  XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion JAX_PLATFORMS=cuda \
    timeout 900 srun -n 4 --gpu-bind=none .venv/bin/python <this>
"""
import os

import jax

# --- explicit distributed init BEFORE importing fridom ----------------
ntasks = int(os.environ["SLURM_NTASKS"])
procid = int(os.environ["SLURM_PROCID"])
localid = int(os.environ["SLURM_LOCALID"])

# deterministic coordinator port from the job id (all ranks agree, no
# file race). SLURM_JOB_ID is identical across tasks of one step.
jobid = int(os.environ["SLURM_JOB_ID"])
stepid = int(os.environ.get("SLURM_STEPID", os.environ.get("SLURM_STEP_ID", 0)))
port = 29500 + ((jobid + stepid * 7919) % 3000)
ART = os.path.dirname(os.path.abspath(__file__))

jax.distributed.initialize(
    coordinator_address=f"localhost:{port}",
    num_processes=ntasks,
    process_id=procid,
    local_device_ids=[localid],
)

# --- now import fridom -------------------------------------------------
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax.experimental import multihost_utils  # noqa: E402

import fridom.nonhydro2 as nh  # noqa: E402
from fridom.spatial.coordinate_mapping import CoordinateMapping  # noqa: E402
from fridom.spatial.grid import Grid  # noqa: E402
from fridom.spatial.meshes.interval import IntervalMesh  # noqa: E402
from fridom.spatial.meshes.mapped_interval import (  # noqa: E402
    MappedIntervalMesh,
)

TWO_PI = 2 * np.pi
COMPS = ("u", "v", "w", "b")


def depth(x):
    return 1.0 + 0.2 * jnp.sin(x)


def stretch(z):
    return z + 0.15 * jnp.sin(2 * np.pi * z) / (2 * np.pi)


def build_model(nx=32, ny=32, nz=16, *, iters=80, tol=1e-10):
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
        advection=True, dsqr=0.25, family="fv",
        pressure_preconditioner="none",
        pressure_iterations=iters, pressure_tolerance=tol,
    )
    model.set_fields(
        u=lambda x, y, z: jnp.sin(x) * jnp.cos(y),
        v=lambda x, y, z: 0.3 * jnp.cos(x) * jnp.sin(y),
        w=lambda x, y, z: 0.2 * jnp.sin(jnp.pi * z),
        b=lambda x, y, z: 0.5 * jnp.cos(x) * jnp.sin(jnp.pi * z))
    return model


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--tol", type=float, default=1e-10)
    ap.add_argument("--tag", default="ref")
    args = ap.parse_args()

    pidx = jax.process_index()
    if pidx == 0:
        print(f"process_count={jax.process_count()} "
              f"device_count={jax.device_count()} "
              f"local_devices={jax.local_devices()}")
        print(f"config: steps={args.steps} iters={args.iters} "
              f"tol={args.tol} tag={args.tag}")
    model = build_model(iters=args.iters, tol=args.tol)
    model.advance(args.steps)
    if pidx == 0:
        print(f"panicked: {model.panicked}")

    maxdiff = 0.0
    for c in COMPS:
        local = model.state[c].data
        glob = multihost_utils.process_allgather(local, tiled=True)
        if pidx == 0:
            ref = np.load(os.path.join(ART, f"smoke_{args.tag}_{c}.npy"))
            g = np.asarray(glob)
            if g.shape != ref.shape:
                print(f"  {c}: SHAPE MISMATCH gathered={g.shape} "
                      f"ref={ref.shape}")
                continue
            d = float(np.abs(g - ref).max())
            print(f"  max_abs_diff {c}: {d:.3e} (shape {g.shape})")
            maxdiff = max(maxdiff, d)
    if pidx == 0:
        print(f"OVERALL srun max_abs_diff vs 1-GPU ref: {maxdiff:.3e}")
        if os.path.exists(os.path.join(ART, "srun_coord_port.txt")):
            os.remove(os.path.join(ART, "srun_coord_port.txt"))


if __name__ == "__main__":
    main()
