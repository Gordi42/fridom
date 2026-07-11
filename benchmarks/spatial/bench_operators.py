"""Field-level operator benchmarks for fridom.spatial (application path).

End-to-end companions of the pure-kernel cases in
``bench_kernels.py``: every case runs the full operator application
path (dispatch, consumption-side halo sync, kernel, layout
re-attachment) on 2-D fields, at the sizes of the old-stack cases in
``benchmarks/bench_operators.py`` — so old-vs-new comparisons come
from the same infrastructure. Note the semantic difference when
comparing: fridom.spatial applications sync their *operand* at first
consumption (task-1.8 consumption-side contract; results carry
validity claims instead of a post-sync), the old stack syncs
separately (``bench_sync``).
"""
from __future__ import annotations

import jax.numpy as jnp

from fridom.benchmarking import benchmark_case
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.select import Where
from fridom.spatial.operators.weno import WenoReconstruction
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
)

# same sizes as benchmarks/bench_operators.py (old stack)
SIZES = [256, 1024, 4096]

# halo 3 covers the widest kernel in play (order-5 WENO); the old
# stack cases run with halo 2 — the storage-extent difference is
# negligible at these sizes
_HALO = 3


def _grid(n: int) -> tuple[Grid, IntervalMesh, IntervalMesh]:
    """2-D periodic grid with smooth test data helpers."""
    mx = IntervalMesh(n, (0.0, 1.0), name="x")
    my = IntervalMesh(n, (0.0, 1.0), name="y")
    grid = Grid((mx, my))
    grid.negotiate(halo=HaloSpec({"x": _HALO, "y": _HALO}))
    return grid, mx, my


def _wave(x, y):
    return jnp.sin(2 * jnp.pi * x) * jnp.cos(2 * jnp.pi * y)


# reps=50: repetitions of the small cases are tens of us; the higher
# count pulls the medians out of the per-call jitter noise band
@benchmark_case(params={"n": SIZES}, reps=50)
def bench_diff(n):
    """Staggered derivative along x (Center -> Right), synced."""
    grid, _, _ = _grid(n)
    f = grid.create_field(init=_wave)

    def run(f):
        return f.diff("x")

    return run, (f,), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES}, reps=50)
def bench_add(n):
    """Field + field on one space (join, combine, sync)."""
    grid, _, _ = _grid(n)
    f = grid.create_field(init=_wave)
    g = grid.create_field(
        init=lambda x, y: jnp.cos(2 * jnp.pi * x)
        * jnp.sin(2 * jnp.pi * y))

    def run(f, g):
        return f + g

    return run, (f, g), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES}, reps=50)
def bench_mul(n):
    """Dispatched collocation product on one space, synced."""
    grid, _, _ = _grid(n)
    f = grid.create_field(init=_wave)
    g = grid.create_field(
        init=lambda x, y: jnp.cos(2 * jnp.pi * x)
        * jnp.sin(2 * jnp.pi * y))

    def run(f, g):
        return f * g

    return run, (f, g), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES}, reps=50)
def bench_sync(n):
    """Halo fill of a 2-D center field (periodic wrap both axes)."""
    grid, _, _ = _grid(n)
    f = grid.create_field(init=_wave)

    def run(f):
        return grid.sync(f)

    return run, (f,), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES}, reps=50)
def bench_fv_diff(n):
    """FV derivative chain on CellAvg (flux_diff @ reconstruct)."""
    grid, mx, my = _grid(n)
    q = grid.create_field(
        TensorProductSpace.of(mx.cell_avg, my.cell_avg), init=_wave)

    def run(q):
        return q.diff("x")

    return run, (q,), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES}, reps=50)
def bench_tendency(n):
    """Run a representative tendency (interp + product + diff)."""
    grid, mx, my = _grid(n)
    f = grid.create_field(init=_wave)
    u = grid.create_field(
        TensorProductSpace.of(mx.right, my.center),
        init=lambda x, y: 1.0 + 0.1 * jnp.sin(2 * jnp.pi * (x + y)))

    def run(f, u):
        flux_x = u * f.to(u)
        div_x = flux_x.diff("x")
        dfy = f.diff("y")
        return div_x + dfy.diff("y") * 0.1

    return run, (f, u), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES}, reps=50)
def bench_weno_pair_field(n):
    """Field-level order-5 upwind pair + ("select", ...) dispatch."""
    grid, mx, my = _grid(n)
    q = grid.create_field(
        TensorProductSpace.of(mx.cell_avg, my.cell_avg), init=_wave)
    cond = grid.create_field(
        TensorProductSpace.of(mx.right, my.cell_avg),
        init=lambda x, y: jnp.sign(
            1.0 + 0.1 * jnp.sin(2 * jnp.pi * (x + y))))
    left = WenoReconstruction(5, bias="left")["x"]
    right = WenoReconstruction(5, bias="right")["x"]
    where = Where()

    def run(q, cond):
        return where(cond, left(q), right(q))

    return run, (q, cond), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES}, reps=50)
def bench_fft_roundtrip(n):
    """Fourier forward + backward on a 2-D center field."""
    grid, _, _ = _grid(n)
    f = grid.create_field(init=_wave)
    fourier = Fourier(grid)

    def run(f):
        return fourier.backward(fourier.forward(f))

    return run, (f,), {"points": float(n * n)}
