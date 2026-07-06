"""Kernel benchmarks for framework2.grid (Wave-4 slot, currently empty).

Suite convention (same as the top-level ``benchmarks/`` suite): this
directory is a ``fridom.benchmarking`` suite; cases are declared in
``bench_*.py`` files with the ``@bm.benchmark_case`` decorator and
discovered with

    uv run python -m fridom.benchmarking list benchmarks/framework2
    uv run python -m fridom.benchmarking run benchmarks/framework2

Wave 4 fills this file with the hot-kernel cases (finite differences,
interpolation, FV flux differences, WENO, transforms) mirroring the
old-stack cases in ``benchmarks/bench_operators.py`` at identical
sizes, so old-vs-new comparisons come from the same infrastructure.
Merge gates for kernel optimizations (implementation plan): runtime,
compile time, and memory vs the pre-optimization commit and vs the
old ``framework`` kernel.
"""
from __future__ import annotations
