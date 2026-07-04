"""Tests for the benchmark comparison and rendering."""
import pytest

from fridom.benchmarking.compare import (
    CaseComparison,
    MetricDelta,
    _format_bytes,
    _format_rel,
    _format_seconds,
    _memory_bytes,
    compare,
    format_comparison,
    format_suite,
)
from fridom.benchmarking.result import (
    BenchmarkResult,
    RunMetadata,
    SuiteResult,
)


def make_suite(results, commit="a" * 40):
    mdata = RunMetadata(
        commit=commit, branch="main", backend="cpu",
        device_kind="cpu", device_count=1)
    return SuiteResult(metadata=mdata, results=results)


# ================================================================
#  MetricDelta
# ================================================================
@pytest.mark.parametrize(
    ("base", "new", "rel"),
    [
        pytest.param(1.0, 1.5, 0.5, id="increase"),
        pytest.param(2.0, 1.0, -0.5, id="decrease"),
        pytest.param(None, 1.0, None, id="no-base"),
        pytest.param(1.0, None, None, id="no-new"),
        pytest.param(0.0, 1.0, None, id="zero-base"),
    ],
)
def test_metric_delta_rel(base, new, rel):
    assert MetricDelta(base, new).rel == rel


# ================================================================
#  Memory metric
# ================================================================
def test_memory_bytes_prefers_peak():
    result = BenchmarkResult(name="c", peak_bytes=100, temp_bytes=50)
    assert _memory_bytes(result) == 100


def test_memory_bytes_falls_back_to_temp():
    result = BenchmarkResult(name="c", temp_bytes=50)
    assert _memory_bytes(result) == 50


# ================================================================
#  Case status
# ================================================================
def case(base_wall=None, new_wall=None, base_error=None, new_error=None):
    base = None
    if base_wall is not None or base_error is not None:
        base = BenchmarkResult(
            name="c", wall_times=[base_wall] if base_wall else [],
            error=base_error)
    new = None
    if new_wall is not None or new_error is not None:
        new = BenchmarkResult(
            name="c", wall_times=[new_wall] if new_wall else [],
            error=new_error)
    return CaseComparison(full_name="c", base=base, new=new)


@pytest.mark.parametrize(
    ("comparison", "status"),
    [
        pytest.param(case(1.0, 1.01), "ok", id="ok"),
        pytest.param(case(1.0, 2.0), "slower", id="slower"),
        pytest.param(case(2.0, 1.0), "faster", id="faster"),
        pytest.param(case(new_wall=1.0), "added", id="added"),
        pytest.param(case(base_wall=1.0), "removed", id="removed"),
        pytest.param(
            case(1.0, new_error="boom"), "error", id="new-error"),
        pytest.param(
            case(base_error="boom", new_wall=1.0), "error", id="base-error"),
        pytest.param(
            case(base_error="boom"), "error", id="error-beats-added"),
        pytest.param(
            CaseComparison(
                full_name="c",
                base=BenchmarkResult(name="c"),
                new=BenchmarkResult(name="c", wall_times=[1.0]),
            ),
            "ok", id="no-base-walltimes"),
    ],
)
def test_case_status(comparison, status):
    assert comparison.status() == status


def test_case_status_custom_threshold():
    assert case(1.0, 1.2).status(threshold=0.5) == "ok"
    assert case(1.0, 1.2).status(threshold=0.1) == "slower"


# ================================================================
#  Suite comparison
# ================================================================
def test_compare_matches_and_orders_cases():
    base = make_suite([
        BenchmarkResult(name="a", wall_times=[1.0]),
        BenchmarkResult(name="removed", wall_times=[1.0]),
    ])
    new = make_suite([
        BenchmarkResult(name="added", wall_times=[1.0]),
        BenchmarkResult(name="a", wall_times=[2.0]),
    ], commit="b" * 40)

    comparison = compare(base, new)
    assert [c.full_name for c in comparison.cases] == [
        "added", "a", "removed",
    ]
    statuses = {c.full_name: c.status() for c in comparison.cases}
    assert statuses == {"added": "added", "a": "slower",
                        "removed": "removed"}


def test_compare_matches_by_full_name():
    base = make_suite(
        [BenchmarkResult(name="a", params={"n": 2}, wall_times=[1.0])])
    new = make_suite(
        [BenchmarkResult(name="a", params={"n": 4}, wall_times=[1.0])])
    comparison = compare(base, new)
    statuses = {c.full_name: c.status() for c in comparison.cases}
    assert statuses == {"a[n=4]": "added", "a[n=2]": "removed"}


def test_regressions():
    base = make_suite([
        BenchmarkResult(name="a", wall_times=[1.0]),
        BenchmarkResult(name="b", wall_times=[1.0]),
        BenchmarkResult(name="c", wall_times=[1.0]),
    ])
    new = make_suite([
        BenchmarkResult(name="a", wall_times=[2.0]),
        BenchmarkResult(name="b", error="boom"),
        BenchmarkResult(name="c", wall_times=[1.0]),
    ])
    regressions = compare(base, new).regressions()
    assert sorted(c.full_name for c in regressions) == ["a", "b"]


# ================================================================
#  Value formatting
# ================================================================
@pytest.mark.parametrize(
    ("value", "formatted"),
    [
        pytest.param(None, "-", id="none"),
        pytest.param(2.5, "2.50 s", id="seconds"),
        pytest.param(3.2e-3, "3.20 ms", id="milliseconds"),
        pytest.param(4.5e-6, "4.50 us", id="microseconds"),
        pytest.param(6.0e-9, "6.00 ns", id="nanoseconds"),
        pytest.param(0.0, "0.00 ns", id="zero"),
    ],
)
def test_format_seconds(value, formatted):
    assert _format_seconds(value) == formatted


@pytest.mark.parametrize(
    ("value", "formatted"),
    [
        pytest.param(None, "-", id="none"),
        pytest.param(100, "100.0 B", id="bytes"),
        pytest.param(2048, "2.0 KiB", id="kibibytes"),
        pytest.param(3 * 1024**3, "3.0 GiB", id="gibibytes"),
        pytest.param(2 * 1024**4, "2.0 TiB", id="tebibytes"),
    ],
)
def test_format_bytes(value, formatted):
    assert _format_bytes(value) == formatted


@pytest.mark.parametrize(
    ("value", "formatted"),
    [
        pytest.param(None, "-", id="none"),
        pytest.param(0.153, "+15.3%", id="positive"),
        pytest.param(-0.05, "-5.0%", id="negative"),
    ],
)
def test_format_rel(value, formatted):
    assert _format_rel(value) == formatted


# ================================================================
#  Rendering
# ================================================================
def test_format_suite_plain():
    suite = make_suite([
        BenchmarkResult(
            name="a", wall_times=[1.0], compile_time=0.5, temp_bytes=1024),
        BenchmarkResult(name="b", error="boom"),
    ])
    report = format_suite(suite)
    assert "run: aaaaaaaa (main) on cpu [cpu x1]" in report
    assert "1.00 s" in report
    assert "1.0 KiB" in report
    assert "error" in report


def test_format_suite_markdown():
    suite = make_suite([BenchmarkResult(name="a", wall_times=[1.0])])
    report = format_suite(suite, markdown=True)
    assert "| case |" in report


def test_format_suite_empty():
    report = format_suite(make_suite([]))
    assert "case" in report


def test_format_comparison_plain():
    base = make_suite([BenchmarkResult(name="a", wall_times=[1.0])])
    new = make_suite([BenchmarkResult(name="a", wall_times=[2.0])])
    report = format_comparison(compare(base, new))
    assert "base: aaaaaaaa" in report
    assert "+100.0%" in report
    assert "slower" in report
    assert "1 cases: 1 slower" in report


def test_format_comparison_dirty_marker():
    base = make_suite([])
    base.metadata.dirty = True
    report = format_comparison(compare(base, make_suite([])))
    assert "aaaaaaaa+" in report


def test_format_comparison_unknown_metadata():
    comparison = compare(
        SuiteResult(metadata=RunMetadata(), results=[]),
        SuiteResult(metadata=RunMetadata(), results=[]),
    )
    report = format_comparison(comparison)
    assert "unknown on unknown" in report


def test_format_comparison_markdown():
    base = make_suite([BenchmarkResult(name="a", wall_times=[1.0])])
    new = make_suite([BenchmarkResult(name="a", wall_times=[1.0])])
    report = format_comparison(compare(base, new), markdown=True)
    assert "| case |" in report
    assert "| --- |" in report
