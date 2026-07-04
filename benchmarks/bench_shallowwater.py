"""Macro benchmarks for the shallow water model."""
from __future__ import annotations

import fridom.shallowwater as sw
from fridom.benchmarking import benchmark_case

SIZES = [256, 512]


# warmup=4: the Adams-Bashforth stepper ramps its order over the
# first steps, each triggering a fresh jit compilation
@benchmark_case(
    params={"n": SIZES}, reps=5, warmup=4, measure_compile=False)
def bench_shallowwater_step(n):
    """One full time step of the shallow water model."""
    grid = sw.grid.cartesian.Grid(shape=(n, n), domain_size=(1.0, 1.0))
    mset = sw.ModelSettings(grid=grid, f0=1.0, csqr=0.01)
    mset.time_stepper.dt = 2 / n
    mset.setup()

    z = sw.initial_conditions.Jet(
        mset, width=1 / 20, wavenum=2, pos=0.5, waveamp=1e-2)

    model = sw.Model(mset)
    model.z = z
    model.start()

    def run():
        model.step()
        return model.z

    return run, (), {"points": float(n * n)}
