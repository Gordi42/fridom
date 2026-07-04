"""Micro benchmarks for grid operators on scalar fields."""
from __future__ import annotations

import jax.numpy as jnp

import fridom.framework as fr
from fridom.benchmarking import benchmark_case

SIZES = [256, 1024]


def _make_field(n: int) -> fr.ScalarField:
    """Create a 2D scalar field with smooth test data."""
    grid = fr.grid.cartesian.Grid(
        shape=(n, n), domain_size=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid)
    mset.halo = 2
    mset.setup()
    field = fr.ScalarField(mset)
    x, y = grid.x_mesh
    field.arr = jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    return field.sync()


@benchmark_case(params={"n": SIZES})
def bench_diff(n):
    """Partial derivative along the first axis."""
    field = _make_field(n)

    def run(field):
        return field.diff(0)

    return run, (field,), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES})
def bench_laplacian(n):
    """Laplacian of a scalar field."""
    field = _make_field(n)

    def run(field):
        return field.laplacian()

    return run, (field,), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES})
def bench_fft(n):
    """Forward FFT of a scalar field."""
    field = _make_field(n)

    def run(field):
        return field.fft()

    return run, (field,), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES})
def bench_sync(n):
    """Halo synchronization of a scalar field."""
    field = _make_field(n)

    def run(field):
        return field.sync()

    return run, (field,), {"points": float(n * n)}
