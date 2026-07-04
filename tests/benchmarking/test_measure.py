"""Tests for the benchmark measurement core."""
import jax
import jax.numpy as jnp
import pytest

from fridom.benchmarking.measure import (
    _collect_cost_analysis,
    _collect_memory_analysis,
    _peak_device_memory,
    benchmark,
)
from fridom.benchmarking.result import BenchmarkResult


def square_sum(x):
    return (x * x).sum()


# ================================================================
#  benchmark
# ================================================================
def test_benchmark_jitted():
    x = jnp.ones((32, 32))
    result = benchmark(square_sum, x, reps=3, warmup=1)

    assert result.name == "square_sum"
    assert result.params == {}
    assert len(result.wall_times) == 3
    assert all(t > 0 for t in result.wall_times)
    assert result.trace_time > 0
    assert result.compile_time > 0
    # the static memory analysis is available on cpu and gpu
    assert result.temp_bytes >= 0
    assert result.argument_bytes > 0
    assert result.output_bytes > 0
    assert result.code_bytes >= 0
    assert result.flops > 0


def test_benchmark_without_compile_measurement():
    side_effects = []

    def impure(x):
        side_effects.append(1)
        return x + 1

    result = benchmark(
        impure, jnp.ones(4), reps=2, warmup=1, measure_compile=False)

    assert len(result.wall_times) == 2
    assert result.trace_time is None
    assert result.compile_time is None
    assert result.temp_bytes is None
    assert result.flops is None
    # warmup + timed repetitions all called the plain function
    assert len(side_effects) == 3


def test_benchmark_no_args_returning_none():
    def do_nothing():
        return None

    result = benchmark(do_nothing, reps=1, warmup=0)
    assert len(result.wall_times) == 1


def test_benchmark_custom_name():
    result = benchmark(square_sum, jnp.ones(4), reps=1, name="custom")
    assert result.name == "custom"


def test_benchmark_default_name_without_dunder():
    class CallableWithoutName:
        def __call__(self):
            return None

    fn = CallableWithoutName()
    result = benchmark(fn, reps=1, measure_compile=False)
    assert result.name == "<fn>"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        pytest.param({"reps": 0}, "reps must be at least 1", id="reps"),
        pytest.param(
            {"warmup": -1}, "warmup must be non-negative", id="warmup"),
    ],
)
def test_benchmark_invalid_arguments(kwargs, match):
    with pytest.raises(ValueError, match=match):
        benchmark(square_sum, jnp.ones(4), **kwargs)


# ================================================================
#  Memory and cost analysis helpers
# ================================================================
class FakeCompiled:
    def __init__(self, mem=None, cost=None, raise_error=False):
        self._mem = mem
        self._cost = cost
        self._raise = raise_error

    def memory_analysis(self):
        if self._raise:
            raise RuntimeError("not supported")
        return self._mem

    def cost_analysis(self):
        if self._raise:
            raise RuntimeError("not supported")
        return self._cost


class FakeMemoryStats:
    temp_size_in_bytes = 128
    argument_size_in_bytes = 256
    # output_size_in_bytes is intentionally missing


def test_collect_memory_analysis():
    result = BenchmarkResult(name="case")
    _collect_memory_analysis(result, FakeCompiled(mem=FakeMemoryStats()))
    assert result.temp_bytes == 128
    assert result.argument_bytes == 256
    assert result.output_bytes is None
    assert result.code_bytes is None


@pytest.mark.parametrize(
    "compiled",
    [
        pytest.param(FakeCompiled(raise_error=True), id="raises"),
        pytest.param(FakeCompiled(mem=None), id="returns-none"),
    ],
)
def test_collect_memory_analysis_unavailable(compiled):
    result = BenchmarkResult(name="case")
    _collect_memory_analysis(result, compiled)
    assert result.temp_bytes is None


@pytest.mark.parametrize(
    ("cost", "flops"),
    [
        pytest.param({"flops": 5.0}, 5.0, id="dict"),
        pytest.param([{"flops": 5.0}], 5.0, id="legacy-list"),
        pytest.param([], None, id="empty-list"),
        pytest.param({"bytes accessed": 1.0}, None, id="no-flops"),
        pytest.param(None, None, id="none"),
    ],
)
def test_collect_cost_analysis(cost, flops):
    result = BenchmarkResult(name="case")
    _collect_cost_analysis(result, FakeCompiled(cost=cost))
    assert result.flops == flops


def test_collect_cost_analysis_raises():
    result = BenchmarkResult(name="case")
    _collect_cost_analysis(result, FakeCompiled(raise_error=True))
    assert result.flops is None


# ================================================================
#  Peak device memory
# ================================================================
class FakeDevice:
    def __init__(self, stats):
        self._stats = stats

    def memory_stats(self):
        if isinstance(self._stats, Exception):
            raise self._stats
        return self._stats


def test_peak_device_memory(monkeypatch):
    devices = [
        FakeDevice({"peak_bytes_in_use": 100}),
        FakeDevice({"peak_bytes_in_use": 300}),
    ]
    monkeypatch.setattr(jax, "local_devices", lambda: devices)
    assert _peak_device_memory() == 300


@pytest.mark.parametrize(
    "stats",
    [
        pytest.param(None, id="returns-none"),
        pytest.param({"bytes_in_use": 1}, id="no-peak-entry"),
        pytest.param(RuntimeError("not supported"), id="raises"),
    ],
)
def test_peak_device_memory_unavailable(monkeypatch, stats):
    monkeypatch.setattr(
        jax, "local_devices", lambda: [FakeDevice(stats)])
    assert _peak_device_memory() is None
