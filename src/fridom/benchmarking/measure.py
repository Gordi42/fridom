"""Measure wall time, compile time, and memory of jax functions."""
from __future__ import annotations

from contextlib import suppress
from time import perf_counter
from typing import TYPE_CHECKING, Any

import jax

from fridom.benchmarking.result import BenchmarkResult

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


# ================================================================
#  Helper functions
# ================================================================
def _get_int_attr(obj: Any, attr: str) -> int | None:
    """Return an integer attribute of an object, or None if missing."""
    value = getattr(obj, attr, None)
    return None if value is None else int(value)


def _collect_memory_analysis(result: BenchmarkResult, compiled: Any) -> None:
    """
    Collect the static memory analysis of a compiled executable.

    Description
    -----------
    Fills the memory-related fields of the result from the compiled
    executable's memory analysis. Backends that do not support memory
    analysis leave the fields at None.

    Parameters
    ----------
    result : BenchmarkResult
        The result to fill.
    compiled : Any
        The compiled executable (jax.stages.Compiled).
    """
    mem = None
    # not all backends implement the memory analysis; the raised
    # exception type is backend-defined, so suppress everything
    with suppress(Exception):
        mem = compiled.memory_analysis()
    if mem is None:
        return
    result.temp_bytes = _get_int_attr(mem, "temp_size_in_bytes")
    result.argument_bytes = _get_int_attr(mem, "argument_size_in_bytes")
    result.output_bytes = _get_int_attr(mem, "output_size_in_bytes")
    result.code_bytes = _get_int_attr(mem, "generated_code_size_in_bytes")


def _collect_cost_analysis(result: BenchmarkResult, compiled: Any) -> None:
    """
    Collect the flop estimate of a compiled executable.

    Parameters
    ----------
    result : BenchmarkResult
        The result to fill.
    compiled : Any
        The compiled executable (jax.stages.Compiled).
    """
    cost = None
    # not all backends implement the cost analysis; the raised
    # exception type is backend-defined, so suppress everything
    with suppress(Exception):
        cost = compiled.cost_analysis()
    # older jax versions return a list with a single dictionary
    if isinstance(cost, list):
        cost = cost[0] if cost else None
    if not isinstance(cost, dict):
        return
    flops = cost.get("flops")
    if flops is not None:
        result.flops = float(flops)


def _peak_device_memory() -> int | None:
    """
    Return the peak device memory usage across all local devices.

    Returns
    -------
    int | None
        The maximum "peak_bytes_in_use" over all local devices in
        bytes, or None on backends that do not report memory
        statistics (e.g. cpu).
    """
    peaks = []
    for device in jax.local_devices():
        stats = None
        # memory statistics are not available on all backends
        with suppress(Exception):
            stats = device.memory_stats()
        if stats and "peak_bytes_in_use" in stats:
            peaks.append(int(stats["peak_bytes_in_use"]))
    return max(peaks) if peaks else None


# ================================================================
#  Benchmark
# ================================================================
def benchmark(
    fn: Callable[..., Any],
    *args: Any,
    reps: int = 10,
    warmup: int = 2,
    measure_compile: bool = True,
    name: str | None = None,
) -> BenchmarkResult:
    """
    Benchmark a function on the current jax backend.

    Description
    -----------
    Measures the wall time of `fn(*args)` over a number of timed
    repetitions. Since jax dispatches asynchronously, every timed call
    blocks on its result (`jax.block_until_ready`) before the timer is
    stopped.

    If `measure_compile` is True (the default), the function is
    jit-compiled ahead of time: the trace time, compile time, static
    memory analysis, and flop estimate are recorded, and the timed
    repetitions run the compiled executable. Set `measure_compile` to
    False to benchmark the function as-is (e.g. functions that are not
    jit-compatible or that manage their own jit).

    On backends that report memory statistics (e.g. gpu), the peak
    device memory usage is recorded after the timed repetitions. Note
    that the peak is accumulated per process; run benchmarks in
    separate processes to obtain independent peaks.

    Parameters
    ----------
    fn : Callable
        The function to benchmark.
    *args : Any
        The arguments to call the function with.
    reps : int, optional
        The number of timed repetitions (default: 10).
    warmup : int, optional
        The number of untimed warmup calls before the timed
        repetitions (default: 2).
    measure_compile : bool, optional
        Whether to jit-compile the function ahead of time and measure
        the compile-related metrics (default: True).
    name : str | None, optional
        The name of the benchmark; defaults to the function name
        (default: None).

    Returns
    -------
    BenchmarkResult
        The measurements.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import fridom.framework as fr

        def laplacian(f):
            return sum(jnp.gradient(g, axis=i)
                       for i, g in enumerate(jnp.gradient(f)))

        f = jnp.ones((256, 256))
        result = fr.benchmarking.benchmark(laplacian, f)
        print(result.wall_median, result.compile_time)
    """
    if reps < 1:
        raise ValueError(f"reps must be at least 1, got {reps}")
    if warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {warmup}")

    result = BenchmarkResult(name=name or getattr(fn, "__name__", "<fn>"))

    run = fn
    if measure_compile:
        jitted = jax.jit(fn)
        tic = perf_counter()
        lowered = jitted.lower(*args)
        result.trace_time = perf_counter() - tic
        tic = perf_counter()
        compiled = lowered.compile()
        result.compile_time = perf_counter() - tic
        _collect_memory_analysis(result, compiled)
        _collect_cost_analysis(result, compiled)
        run = compiled

    # make sure the inputs are ready before any timing starts
    jax.block_until_ready(args)

    for _ in range(warmup):
        jax.block_until_ready(run(*args))

    wall_times = []
    for _ in range(reps):
        tic = perf_counter()
        out = run(*args)
        # jax dispatch is asynchronous: block on the result before
        # stopping the timer
        jax.block_until_ready(out)
        wall_times.append(perf_counter() - tic)
    result.wall_times = wall_times

    result.peak_bytes = _peak_device_memory()
    return result
