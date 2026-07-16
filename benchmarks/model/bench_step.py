"""Model-step timing cases: the performance guard of the new stack.

Description
-----------
Times ``model.advance`` for the production model configurations on
flat and coordinate-mapped grids. These cases are the reference for
"did a change cost us the step time?": run the suite, compare against
the checked-in baseline for the device configuration, and investigate
anything the comparison flags as slower.

.. code-block:: bash

    # 1 GPU (pin one device explicitly)
    CUDA_VISIBLE_DEVICES=0 uv run python -m fridom.benchmarking \
        run benchmarks/model -o /tmp/step-gpu1.json
    uv run python -m fridom.benchmarking compare \
        benchmarks/baselines/step-gpu1.json /tmp/step-gpu1.json \
        --fail-on-regression

    # 4 GPUs (the XLA flag below is REQUIRED, see the trap list)
    XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion \
        uv run python -m fridom.benchmarking \
        run benchmarks/model -o /tmp/step-gpu4.json
    uv run python -m fridom.benchmarking compare \
        benchmarks/baselines/step-gpu4.json /tmp/step-gpu4.json \
        --fail-on-regression

Baselines live in ``benchmarks/baselines/`` and are re-recorded (same
commands, ``-o benchmarks/baselines/...``) whenever a change
*intentionally* moves the numbers; the commit message states the
before/after. Case names and parameter grids are part of the baseline
key — extend grids by appending, never by renaming.

Measurement traps this file guards against (do not "simplify" them
away; benchmarks/README.md tells the full story):

- **Chunk size**: ``model.advance(n)`` with ``n < chunk_size`` runs n
  chunks of length 1, so the scan unroll never engages and per-step
  time reads 2-3x worse. Every case passes ``chunk_size=STEPS`` and
  advances exactly ``STEPS`` steps per repetition.
- **Rotation is opt-in** (2026-07 geometry merge): ``coriolis=None``
  installs no module at all. Every case passes an explicit
  ``coriolis=`` so the workload is stable across default changes.
- **4-GPU runs need**
  ``XLA_FLAGS=--xla_disable_hlo_passes=multi_output_fusion`` —
  without it XLA:GPU miscompiles the 4-device step (silently wrong
  physics, jax-ml/jax#39100). The flag is ~free.
- The first ``advance`` (jit compile + chunk build) happens in the
  setup, is reported through the ``first_advance_s`` extra, and is
  excluded from the timed repetitions.
"""
from __future__ import annotations

from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr
import fridom.nonhydro2 as nh
import fridom.shallowwater2 as sw
from fridom.benchmarking import benchmark_case

#: steps per timed repetition == the chunk size (see the trap list).
STEPS = 50

#: the large sizes are gpu-only; cpu (and the ci smoke job, which
#: runs --first-only) uses the smallest size of each grid.
ON_GPU = jax.default_backend() == "gpu"
SIZES_NH_FLAT = [32] + ([256, 512] if ON_GPU else [])
SIZES_NH_WALLED = [32] + ([256] if ON_GPU else [])
SIZES_NH_MAPPED = [32] + ([128, 256] if ON_GPU else [])
#: prime (indivisible) extents: 257 shards over no device count, so
#: this prices the padded balanced all-to-all against the 256 divisible
#: step (the CI cpu smoke uses the small prime 31).
SIZES_NH_PRIME = [31] + ([257] if ON_GPU else [])
SIZES_SW = [64] + ([1024, 2048] if ON_GPU else [])

TWO_PI = 2.0 * np.pi
LAT_MAX = 1.4


# ================================================================
#  Timing helpers
# ================================================================
def _first_advance(model) -> float:
    """Run the compile-paying first chunk and return its wall time."""
    jax.block_until_ready(jax.tree_util.tree_leaves(model.state))
    tic = perf_counter()
    model.advance(STEPS)
    jax.block_until_ready(jax.tree_util.tree_leaves(model.state))
    return perf_counter() - tic


def _stepping_case(model, points: float):
    """Package a started model as a benchmark case target."""
    first_advance = _first_advance(model)

    def run():
        model.advance(STEPS)
        return model.state

    return run, (), {"points": points, "steps": float(STEPS),
                     "first_advance_s": first_advance}


# ================================================================
#  3D nonhydrostatic: flat (periodic / walled) and terrain-following
# ================================================================
def _depth(x):
    """Bottom topography of the terrain-following cases."""
    return 1.0 + 0.2 * jnp.sin(x)


def _nh_model(n: int, *, mapped: bool, periodic_x: bool = True,
              periodic_z: bool = False,
              iters: int = 30, advection: bool = False,
              family: str | None = None):
    """Nonhydrostatic f-plane model with a jet-like IC.

    ``advection`` switches the momentum/buoyancy advection on. It also
    switches ``dt`` to a CFL-scaled value: centered advection carries no
    dissipation, so the linear cases' fixed ``dt = 0.02`` goes
    non-finite at 512^3 and the model panics mid-timing.
    """
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI),
                                        periodic=periodic_x, name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI), periodic=True,
                                        name="y")
    mz = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0),
                                        periodic=periodic_z, name="z")
    mapping = None
    if mapped:
        mapping = fr.spatial.CoordinateMapping(
            maps={"zp": lambda z, H: z * H},  # noqa: N803 — H(x), math name
            params={"H": _depth})
    grid = fr.spatial.Grid((mx, my, mz), mapping=mapping)
    # |u| ~ 1 and dx = 2*pi/n, so 0.25 * dx keeps the advective cases
    # finite at every n; the linear cases keep their original dt.
    dt = 0.25 * TWO_PI / n if advection else 0.02
    model = nh.Model(grid=grid, dt=dt, advection=advection,
                     coriolis=nh.FPlaneCoriolis(f0=1.0), dsqr=0.25,
                     pressure_iterations=iters, chunk_size=STEPS,
                     family=family)
    if periodic_x:
        hor = (np.arange(n) + 0.5) * (TWO_PI / n)
        ver = (np.arange(n) + 0.5) / n
        x, y, z = np.meshgrid(hor, hor, ver, indexing="ij")
        model.set_fields(u=np.sin(x) * np.cos(y), v=0.3 * np.cos(x),
                         b=0.01 * np.cos(np.pi * z))
    else:
        # A walled x staggers u onto n-1 x-faces, so the flat meshgrid
        # (n points on every axis) no longer matches u's true shape;
        # evaluate the same ICs on each field's own coordinates instead.
        # init= callables must name every coordinate (x, y, z) even
        # where a component does not depend on all three.
        model.set_fields(
            u=lambda x, y, z: jnp.sin(x) * jnp.cos(y),  # noqa: ARG005
            v=lambda x, y, z: 0.3 * jnp.cos(x),  # noqa: ARG005
            b=lambda x, y, z: 0.01 * jnp.cos(jnp.pi * z))  # noqa: ARG005
    return model


@benchmark_case(params={"n": SIZES_NH_FLAT}, reps=5, warmup=0,
                measure_compile=False)
def nh_flat_periodic(n):
    """Linear step on the triply periodic flat grid (spectral solve)."""
    model = _nh_model(n, mapped=False, periodic_z=True)
    return _stepping_case(model, float(n) ** 3)


@benchmark_case(params={"n": SIZES_NH_PRIME}, reps=5, warmup=0,
                measure_compile=False)
def nh_flat_prime(n):
    """Linear step on a triply periodic PRIME flat grid (257^3).

    The indivisible-extent guard: a prime domain shards over no device
    count, so before the padded balanced all-to-all (Phase 2) the
    distributed solve declined and replicated the spectral cube (~2.8x
    the divisible step on 4 GPUs, 257 vs 256). This case prices the
    padded distributed solve so the prime path cannot regress silently;
    it lands near the 260 divisible step (the residual gap is the cuFFT
    Bluestein cost of a prime length, which the 1-GPU run pays too).
    Baseline recorded post-merge on the A100 nodes.
    """
    model = _nh_model(n, mapped=False, periodic_z=True)
    return _stepping_case(model, float(n) ** 3)


@benchmark_case(params={"n": SIZES_NH_FLAT}, reps=5, warmup=0,
                measure_compile=False)
def nh_flat_advective(n):
    """Advective step on the triply periodic flat grid.

    The production configuration, and the one every other nonhydro case
    here is blind to: they all run ``advection=False``. That gap let a
    4% advective regression pass a green suite (2026-07-14, the
    in-place halo write), because the advective step's cost structure is
    genuinely different -- its stencil consumers can absorb a ghost fill
    into their own fusion, which the spectral solve's FFT cannot. Any
    change that moves a fusion boundary must be priced HERE, not only on
    the linear cases.
    """
    model = _nh_model(n, mapped=False, periodic_z=True, advection=True)
    return _stepping_case(model, float(n) ** 3)


@benchmark_case(params={"n": SIZES_NH_FLAT}, reps=5, warmup=0,
                measure_compile=False)
def nh_flat_periodic_nodal(n):
    """Price the nodal (FD) sibling of ``nh_flat_periodic``.

    Since the F3 default flip the periodic flat cases run the FV
    C-grid; this pins ``family="nodal"`` so FV-vs-FD step parity is a
    standing, mechanical comparison (the two are bitwise-identical
    trajectories, so any timing gap is a compiler/fusion artifact to
    hunt, not physics).
    """
    model = _nh_model(n, mapped=False, periodic_z=True, family="nodal")
    return _stepping_case(model, float(n) ** 3)


@benchmark_case(params={"n": SIZES_NH_FLAT}, reps=5, warmup=0,
                measure_compile=False)
def nh_flat_advective_nodal(n):
    """Price the nodal (FD) sibling of ``nh_flat_advective``.

    The advective step has its own fusion structure (see
    ``nh_flat_advective``), so FV-vs-FD parity must be priced here
    too, not only on the linear case.
    """
    model = _nh_model(n, mapped=False, periodic_z=True, advection=True,
                      family="nodal")
    return _stepping_case(model, float(n) ** 3)


@benchmark_case(params={"n": SIZES_NH_WALLED}, reps=5, warmup=0,
                measure_compile=False)
def nh_flat_walled(n):
    """Linear step with walls in z (trig transform on the column).

    On multiple devices the distributed transform declines the mixed
    (trig) plan and the solve falls back to the replicated composite;
    this case prices that fallback.
    """
    model = _nh_model(n, mapped=False, periodic_z=False)
    return _stepping_case(model, float(n) ** 3)


@benchmark_case(params={"n": SIZES_NH_WALLED}, reps=5, warmup=0,
                measure_compile=False)
def nh_flat_walled_x(n):
    """Linear step with walls in x (trig transform on the sharded axis).

    The sibling of ``nh_flat_walled``, which walls z -- off the default
    sharded axis, so it never exercises a trig transform on the split
    dimension. Here the wall is on x, and after the shard-axis-selection
    merge the decomposition shards y for this case (x's staggering cost
    demotes it): this prices the distributed walled-SHARDED-axis solve,
    the path the indivisible-shard campaign fixed.
    """
    model = _nh_model(n, mapped=False, periodic_x=False, periodic_z=True)
    return _stepping_case(model, float(n) ** 3)


@benchmark_case(params={"n": SIZES_NH_MAPPED, "iters": [30, 12]},
                reps=5, warmup=0, measure_compile=False)
def nh_mapped(n, iters):
    """Linear step on the terrain-following grid (PCG pressure solve).

    ``iters`` pins the PCG budget explicitly so the case tracks the
    solver cost, not the packaging default; differencing the 30- and
    12-iteration instances prices a single CG iteration.
    """
    model = _nh_model(n, mapped=True, iters=iters)
    return _stepping_case(model, float(n) ** 3)


# ================================================================
#  2D shallow water: flat periodic and lat-lon sphere chart
# ================================================================
@benchmark_case(params={"n": SIZES_SW}, reps=5, warmup=0,
                measure_compile=False)
def sw_flat(n):
    """Shallow-water step (Sadourny advection) on the flat grid."""
    mx = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), periodic=True,
                                        name="x")
    my = fr.spatial.meshes.IntervalMesh(n, (0.0, 1.0), periodic=True,
                                        name="y")
    model = sw.Model(grid=fr.spatial.Grid((mx, my)), csqr=0.01,
                     coriolis=sw.modules.FPlaneCoriolis(f0=1.0),
                     chunk_size=STEPS,
                     time_stepper=fr.model.time_steppers.AdamBashforth(
                         2.0 / n, order=3))
    g = (np.arange(n) + 0.5) / n
    x, y = np.meshgrid(g, g, indexing="ij")
    model.set_fields(
        p=1e-3 * np.exp(-((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.01))
    return _stepping_case(model, float(n) ** 2)


@benchmark_case(params={"n": SIZES_SW}, reps=5, warmup=0,
                measure_compile=False)
def sw_sphere(n):
    """Shallow-water step (Sadourny advection) on the sphere chart."""
    mlon = fr.spatial.meshes.IntervalMesh(n, (0.0, TWO_PI), name="lon")
    mlat = fr.spatial.meshes.IntervalMesh(n, (-LAT_MAX, LAT_MAX),
                                          periodic=False, name="lat")
    mapping = fr.spatial.CoordinateMapping(chart={
        "X": lambda lon, lat: (jnp.cos(lat) * jnp.cos(lon),
                               jnp.cos(lat) * jnp.sin(lon),
                               jnp.sin(lat))})
    grid = fr.spatial.Grid((mlon, mlat), mapping=mapping)
    grid.merge_overrides({
        "raise_index": fr.spatial.operators.RaiseIndex(
            ("lon", "lat"), diagonal=True),
        "lower_index": fr.spatial.operators.LowerIndex(
            ("lon", "lat"), diagonal=True)})
    model = sw.Model(grid=grid, coords=("lon", "lat"), csqr=0.01,
                     coriolis=sw.modules.RotationCoriolis(
                         omega=(0.0, 0.0, 1.0), coords=("lon", "lat"),
                         metric_weight="csqr"),
                     chunk_size=STEPS,
                     # dt ~ 1/n keeps the zonal CFL ~0.09 at every n
                     # (the cos(lat_max) squeeze shrinks the effective
                     # spacing 6x; a fixed dt goes unstable at 2048^2
                     # after ~300 steps and trips the finite guard)
                     time_stepper=fr.model.time_steppers.AdamBashforth(
                         1.0 / n, order=3))
    lon = (np.arange(n) + 0.5) * (TWO_PI / n)
    lat = -LAT_MAX + (np.arange(n) + 0.5) * (2 * LAT_MAX / n)
    lo, la = np.meshgrid(lon, lat, indexing="ij")
    model.set_fields(
        p=1e-3 * np.exp(-((lo - np.pi) ** 2 + la ** 2) / 0.5))
    return _stepping_case(model, float(n) ** 2)
