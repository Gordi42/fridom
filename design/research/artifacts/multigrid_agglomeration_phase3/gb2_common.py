"""Shared builders for the agglomeration Phase-3 measurement legs.

Copy of design/research/artifacts/multigrid_gspmd_validation/gb2_common.py
EXTENDED to thread the new ``multigrid_agglomerate`` (tau) knob through
the model / solver builders. Lives in the job dir, never the repo.

Import AFTER setting JAX_PLATFORMS / XLA_FLAGS in the environment
(fridom touches the backend at import).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr  # noqa: F401  (ensures x64 enabled)
import fridom.nonhydro2 as nh
from fridom.model.modules.coriolis import FPlaneCoriolis
from fridom.nonhydro2.modules.mapped_pressure import MappedPressureSolver
from fridom.nonhydro2.modules.immersed_pressure import ImmersedPressureSolver
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.grid import Grid
from fridom.spatial.immersed_domain import ImmersedDomain
from fridom.spatial.meshes.interval import IntervalMesh

TWO_PI = 2.0 * np.pi
DSQR = 0.25


# ================================================================
#  Mapped GB-2 case
# ================================================================
def steep_depth(x):
    """Steep periodic water depth H(x) = 1 + 0.8 sin(x) (ratio 9)."""
    return 1.0 + 0.8 * jnp.sin(x)


def build_mapped_grid(n, *, device_ids=None):
    """The GB-2 terrain-following grid (x,y periodic; z walled)."""
    mx = IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(n, (0.0, 1.0), periodic=False, name="z")
    mapping = CoordinateMapping(
        maps={"zp": lambda z, H: z * H},
        params={"H": steep_depth})
    return Grid((mx, my, mz), mapping=mapping, device_ids=device_ids)


def build_mapped_model(n, *, preconditioner="spectral", method="auto",
                       levels=None, device_ids=None, budget=100,
                       coarsen_vertical=True, agglomerate=None):
    """nh.Model for the GB-2 mapped case (linear, FV auto family)."""
    grid = build_mapped_grid(n, device_ids=device_ids)
    return nh.Model(
        grid=grid, dsqr=DSQR, advection=False,
        coriolis=FPlaneCoriolis(f0=1.0),
        pressure_iterations=budget, pressure_tolerance=1e-8,
        pressure_preconditioner=preconditioner,
        multigrid_levels=levels,
        multigrid_tridiagonal_method=method,
        multigrid_coarsen_vertical=coarsen_vertical,
        multigrid_agglomerate=agglomerate,
        dt=0.02, chunk_size=20)


def set_mapped_ic(model):
    """The record's smooth IC: u=sin(x)cos(y), v=0.3cos(x), b=0.01cos(pi z)."""
    model.set_fields(
        u=lambda x, y, z: jnp.sin(x) * jnp.cos(y),
        v=lambda x, y, z: 0.3 * jnp.cos(x),
        b=lambda x, y, z: 0.01 * jnp.cos(np.pi * z))


def build_mapped_solver(n, *, preconditioner="multigrid", method="auto",
                        levels=None, device_ids=None, budget=100,
                        coarsen_vertical=True, agglomerate=None):
    """Standalone MappedPressureSolver mirroring core.py construction."""
    model = build_mapped_model(
        n, preconditioner=preconditioner, method=method, levels=levels,
        device_ids=device_ids, budget=budget,
        coarsen_vertical=coarsen_vertical, agglomerate=agglomerate)
    grid = model.grid
    p_space = model.state["p"].function_space
    solver = MappedPressureSolver(
        grid, p_space, iterations=budget, tolerance=1e-8,
        weights={"z": 1.0 / DSQR},
        preconditioner=preconditioner, multigrid_levels=levels,
        multigrid_tridiagonal_method=method,
        multigrid_coarsen_vertical=coarsen_vertical,
        multigrid_agglomerate=agglomerate)
    return solver, grid, p_space, model


def mapped_mean_free_rhs(grid, space):
    """A smooth, decomposition-invariant mean-free rhs on the p-space."""
    rhs = grid.create_field(
        space,
        init=lambda x, y, z: jnp.exp(
            -((x - 3.0) ** 2 + (y - 3.0) ** 2 + (z - 0.5) ** 2) * 3.0))
    return rhs - rhs.mean()


# ================================================================
#  Immersed case (kernel study Phase A: slope geometry)
# ================================================================
def slope_fraction(x, y, z):
    """Tilted-slope wet fraction (canonical immersed slope, genuine partials)."""
    return jnp.clip(
        ((0.6 + 0.15 * jnp.sin(y) + 0.1 * z) - x / TWO_PI) * 8.0 + 0.5,
        0.0, 1.0)


def immersed_rhs_from_velocity(solver, seed=0):
    """A wet-supported, compatible rhs = masked divergence of random vel."""
    rng = np.random.default_rng(seed)
    vel = {
        a: solver._alpha[a].with_data(
            jnp.asarray(rng.standard_normal(solver._alpha[a].data.shape)))
        for a in solver.axes}
    return solver.divergence(vel)


def build_immersed_grid(n, *, device_ids=None):
    """Immersed grid with a diagonal slope (x,y periodic; z walled)."""
    mx = IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="x")
    my = IntervalMesh(n, (0.0, TWO_PI), periodic=True, name="y")
    mz = IntervalMesh(n, (0.0, 1.0), periodic=False, name="z")
    return Grid((mx, my, mz),
                immersed=ImmersedDomain(slope_fraction, order=4),
                device_ids=device_ids)


def build_immersed_model(n, *, preconditioner="spectral", method="auto",
                         levels=None, device_ids=None, budget=100,
                         agglomerate=None):
    """nh.Model for the immersed slope case (linear, FV)."""
    grid = build_immersed_grid(n, device_ids=device_ids)
    return nh.Model(
        grid=grid, dsqr=DSQR, advection=False,
        coriolis=FPlaneCoriolis(f0=1.0),
        pressure_iterations=budget, pressure_tolerance=1e-8,
        pressure_preconditioner=preconditioner,
        multigrid_levels=levels,
        multigrid_tridiagonal_method=method,
        multigrid_agglomerate=agglomerate,
        dt=0.02, chunk_size=20)


def build_immersed_solver(n, *, preconditioner="multigrid", method="auto",
                          levels=None, device_ids=None, budget=100,
                          agglomerate=None):
    """Standalone ImmersedPressureSolver mirroring core.py construction."""
    model = build_immersed_model(
        n, preconditioner=preconditioner, method=method, levels=levels,
        device_ids=device_ids, budget=budget, agglomerate=agglomerate)
    grid = model.grid
    p_space = model.state["p"].function_space
    solver = ImmersedPressureSolver(
        grid, p_space, vertical="z", dsqr=DSQR,
        iterations=budget, tolerance=1e-8,
        preconditioner=preconditioner, multigrid_levels=levels,
        multigrid_tridiagonal_method=method,
        multigrid_agglomerate=agglomerate)
    return solver, grid, p_space, model


def set_immersed_ic(model):
    """A reproducible smooth IC for the immersed step timing/parity."""
    model.set_fields(
        u=lambda x, y, z: jnp.sin(x) * jnp.cos(y),
        v=lambda x, y, z: 0.3 * jnp.cos(x),
        b=lambda x, y, z: 0.01 * jnp.cos(np.pi * z))


# ================================================================
#  Timing (GB-2 protocol)
# ================================================================
def sync(model):
    jax.block_until_ready(jax.tree_util.tree_leaves(model._carry))


def time_step_ms(model, *, reps=6, steps=20):
    """Median ms/step over `reps` advances of `steps`; compile excluded."""
    t0 = time.perf_counter()
    model.advance(steps)
    sync(model)
    compile_s = time.perf_counter() - t0
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        model.advance(steps)
        sync(model)
        times.append((time.perf_counter() - t0) / steps * 1e3)
    times = sorted(times)
    med = times[len(times) // 2] if len(times) % 2 else (
        0.5 * (times[len(times) // 2 - 1] + times[len(times) // 2]))
    return {"median": med, "min": times[0], "max": times[-1],
            "compile_s": compile_s, "all": times}


def peak_mem_gib():
    """Per-device peak bytes in GiB (bytes_in_use high-water if available)."""
    out = []
    for i, d in enumerate(jax.local_devices()):
        try:
            st = d.memory_stats()
            peak = st.get("peak_bytes_in_use") or st.get("bytes_in_use")
            out.append((i, None if peak is None else peak / 2**30))
        except Exception as e:  # noqa: BLE001
            out.append((i, f"err:{e}"))
    return out
