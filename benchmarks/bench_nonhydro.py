"""Macro benchmarks for the nonhydrostatic model."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import fridom.nonhydro as nh
from fridom.benchmarking import benchmark_case

SIZES = [32, 64, 128]
# the largest size is too expensive for cpu runs (and the ci smoke job)
if jax.default_backend() == "gpu":
    SIZES += [256]


def _make_model(n: int, advection: bool) -> nh.Model:
    """Set up a started nonhydrostatic model with a jet initial state."""
    f0 = 1e-4
    grid = nh.grid.cartesian.Grid(
        shape=(n, n, n), domain_size=(10_000.0, 10_000.0, 100.0))
    mset = nh.ModelSettings(
        grid, f0=f0, stratification_n2=(50 * f0) ** 2)
    mset.time_stepper.dt = np.timedelta64(20, "s")
    if not advection:
        mset.tendencies.advection.disable()
    mset.setup()

    _, y, z_mesh = grid.x_mesh
    _, ly, lz = grid.domain_size
    z = nh.State(mset)
    z.u.arr = (jnp.exp(-(y - ly / 2) ** 2 / (0.2 * ly) ** 2)
               * jnp.exp(-(z_mesh - lz / 2) ** 2 / (0.2 * lz) ** 2))
    z.sync()

    model = nh.Model(mset)
    model.z = z
    model.start()
    return model


# warmup=4: the Adams-Bashforth stepper ramps its order over the
# first steps, each triggering a fresh jit compilation
@benchmark_case(
    params={"n": SIZES}, reps=20, warmup=4, measure_compile=False)
def bench_nonhydro_step(n):
    """One full time step (advection, pressure solve, diagnostics)."""
    model = _make_model(n, advection=True)

    def run():
        model.step()
        return model.z

    return run, (), {"points": float(n**3)}


@benchmark_case(
    params={"n": SIZES}, reps=20, warmup=4, measure_compile=False)
def bench_nonhydro_linear_step(n):
    """One time step of the linearized model (advection disabled)."""
    model = _make_model(n, advection=False)

    def run():
        model.step()
        return model.z

    return run, (), {"points": float(n**3)}
