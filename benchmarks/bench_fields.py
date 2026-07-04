"""Micro benchmarks for scalar field arithmetic."""
from __future__ import annotations

import jax.numpy as jnp

import fridom.framework as fr
from fridom.benchmarking import benchmark_case

SIZES = [256, 1024]


def _make_fields(n: int) -> tuple[fr.ScalarField, fr.ScalarField]:
    """Create two 2D scalar fields with smooth test data."""
    grid = fr.grid.cartesian.Grid(
        shape=(n, n), domain_size=(1.0, 1.0))
    mset = fr.ModelSettingsBase(grid)
    mset.halo = 2
    mset.setup()
    f = fr.ScalarField(mset)
    g = fr.ScalarField(mset)
    x, y = grid.x_mesh
    f.arr = jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)
    g.arr = jnp.cos(2 * jnp.pi * x) + jnp.sin(2 * jnp.pi * y)
    return f.sync(), g.sync()


@benchmark_case(params={"n": SIZES})
def bench_arithmetic_chain(n):
    """Evaluate a chain of elementwise field operations."""
    f, g = _make_fields(n)

    def run(f, g):
        return 2.0 * f + f * g - g / 2.0 + f * f

    return run, (f, g), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES})
def bench_reduction(n):
    """Integration of a scalar field over the domain."""
    f, _ = _make_fields(n)

    def run(f):
        return f.integrate()

    return run, (f,), {"points": float(n * n)}
