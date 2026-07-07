"""Kernel benchmarks for framework2.grid (hot stencil kernels).

Suite convention (same as the top-level ``benchmarks/`` suite): this
directory is a ``fridom.benchmarking`` suite; cases are declared in
``bench_*.py`` files with the ``@bm.benchmark_case`` decorator and
discovered with

    uv run python -m fridom.benchmarking list benchmarks/framework2
    uv run python -m fridom.benchmarking run benchmarks/framework2

Wave 1 seeds the pure stencil-kernel cases below at the sizes of the
old-stack cases in ``benchmarks/bench_operators.py`` (n x n points),
so old-vs-new comparisons come from the same infrastructure. Wave 4
adds the remaining hot kernels (FV flux differences, WENO,
transforms) and runs the optimization pass. Merge gates for kernel
optimizations (implementation plan): runtime, compile time, and
memory vs the pre-optimization commit and vs the old ``framework``
kernel.
"""
from __future__ import annotations

import jax.numpy as jnp

from fridom.benchmarking import benchmark_case
from fridom.framework2.grid.operators import stencil_kernels as sk
from fridom.framework2.grid.operators.weno import weno_reconstruct

# 4096 provides a compute-bound regime on gpu, where the smaller
# sizes are dominated by per-call and kernel-launch overhead
# (same sizes as benchmarks/bench_operators.py)
SIZES = [256, 1024, 4096]


def _make_input(rows: int, cols: int) -> jnp.ndarray:
    """Smooth 2D test data; extra rows act as the stencil halo."""
    x = jnp.arange(rows) / cols
    y = jnp.arange(cols) / cols
    return (jnp.sin(2 * jnp.pi * x)[:, None]
            * jnp.cos(2 * jnp.pi * y)[None, :])


# reps=50: repetitions are ~100 us each; the higher count pulls the
# medians of these tiny cases out of the per-call jitter noise band
@benchmark_case(params={"n": SIZES, "order": [2, 4, 6]}, reps=50)
def bench_staggered_diff(n, order):
    """Staggered FD derivative along the first axis (n x n output)."""
    arr = _make_input(n + order - 1, n)
    spacing = 1.0 / n

    def run(arr):
        return sk.staggered_diff(arr, axis=0, spacing=spacing,
                                 order=order)

    return run, (arr,), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES}, reps=50)
def bench_staggered_diff_measure(n):
    """Order-2 staggered FD with a per-point measure denominator."""
    arr = _make_input(n + 1, n)
    spacing = jnp.full((n, 1), 1.0 / n)

    def run(arr, spacing):
        return sk.staggered_diff(arr, axis=0, spacing=spacing, order=2)

    return run, (arr, spacing), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES}, reps=50)
def bench_linear_interp(n):
    """Two-point interpolation along the first axis (n x n output)."""
    arr = _make_input(n + 1, n)

    def run(arr):
        return sk.linear_interp(arr, axis=0)

    return run, (arr,), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES, "order": [3, 5]}, reps=50)
def bench_weno_reconstruct(n, order):
    """Biased WENO reconstruction along the first axis (n x n out)."""
    arr = _make_input(n + order - 1, n)

    def run(arr):
        return weno_reconstruct(arr, axis=0, order=order, bias="left")

    return run, (arr,), {"points": float(n * n)}


@benchmark_case(params={"n": SIZES}, reps=50)
def bench_weno_upwind_pair(n):
    """Order-5 upwind path: both biased kernels + the sign select."""
    order = 5
    arr = _make_input(n + order - 1, n)
    sign = jnp.sign(_make_input(n, n))

    def run(arr, sign):
        left = weno_reconstruct(arr, axis=0, order=order,
                                bias="left")
        right = weno_reconstruct(arr, axis=0, order=order,
                                 bias="right")
        return jnp.where(sign > 0, left, right)

    return run, (arr, sign), {"points": float(n * n)}
