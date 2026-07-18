"""Tests for the benchmark comparison and rendering."""
import pytest

from fridom.benchmarking.compare import (
    ABSOLUTE_FLOOR,
    NOISE_TOLERANCE_K,
    CaseComparison,
    EnvMismatch,
    MetricDelta,
    _format_bytes,
    _format_rel,
    _format_seconds,
    _memory_bytes,
    compare,
    env_mismatches,
    format_comparison,
    format_env_mismatches,
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
#  Environment guard
# ================================================================
def env_metadata(**overrides):
    fields = {
        "backend": "gpu",
        "device_count": 4,
        "device_kind": "A100",
        "jax_version": "0.10.2",
    }
    fields.update(overrides)
    return RunMetadata(**fields)


def test_env_mismatches_match():
    assert env_mismatches(env_metadata(), env_metadata()) == []


@pytest.mark.parametrize(
    "field",
    ["backend", "device_count", "device_kind", "jax_version"],
)
def test_env_mismatches_single_field(field):
    base = env_metadata()
    new = env_metadata(**{field: "other" if field != "device_count" else 1})
    mismatches = env_mismatches(base, new)
    assert [m.field for m in mismatches] == [field]
    assert mismatches[0].base == getattr(base, field)
    assert mismatches[0].new == getattr(new, field)


def test_env_mismatches_missing_field_is_mismatch():
    # a field that is None on both sides is still a mismatch: an
    # incompletely recorded run must never silently pass the guard
    mismatches = env_mismatches(
        env_metadata(jax_version=None), env_metadata(jax_version=None))
    assert [m.field for m in mismatches] == ["jax_version"]


def test_env_mismatches_missing_on_one_side():
    mismatches = env_mismatches(
        env_metadata(), env_metadata(backend=None))
    assert [m.field for m in mismatches] == ["backend"]


def test_env_mismatches_multiple_and_order():
    mismatches = env_mismatches(
        env_metadata(),
        env_metadata(backend="cpu", jax_version="0.9.0"))
    # order follows ENV_GUARD_FIELDS
    assert [m.field for m in mismatches] == ["backend", "jax_version"]


def test_format_env_mismatches():
    text = format_env_mismatches([EnvMismatch("backend", "gpu", "cpu")])
    assert "environment mismatch" in text
    assert "backend" in text
    assert "'gpu'" in text
    assert "'cpu'" in text


# ================================================================
#  Min estimator
# ================================================================
def test_wall_delta_uses_min_not_median():
    base = BenchmarkResult(name="c", wall_times=[3.0, 1.0, 2.0])
    new = BenchmarkResult(name="c", wall_times=[6.0, 4.0, 5.0])
    delta = CaseComparison(full_name="c", base=base, new=new).wall
    # min: base 1.0, new 4.0 -> rel 3.0 (median would give 2.0/5.0 -> 1.5)
    assert delta.base == 1.0
    assert delta.new == 4.0
    assert delta.rel == pytest.approx(3.0)


def test_status_uses_min_not_median():
    # base min 1.0 / median 9.0; new min 8.0 / median 9.0. On the min
    # the case is +700% (slower); on the median it is 0% (ok).
    base = BenchmarkResult(name="c", wall_times=[1.0, 9.0, 9.0, 9.0, 9.0])
    new = BenchmarkResult(name="c", wall_times=[8.0, 9.0, 9.0, 9.0, 9.0])
    comparison = CaseComparison(full_name="c", base=base, new=new)
    assert comparison.status() == "slower"


# ================================================================
#  Per-case tolerance
# ================================================================
def test_cov_base_degenerate():
    new = BenchmarkResult(name="c", wall_times=[1.0])
    assert CaseComparison("c", None, new).cov_base == 0.0
    assert CaseComparison(
        "c", BenchmarkResult(name="c", wall_times=[1.0]), new).cov_base == 0.0
    assert CaseComparison(
        "c", BenchmarkResult(name="c", wall_times=[]), new).cov_base == 0.0
    assert CaseComparison(
        "c", BenchmarkResult(name="c", wall_times=[0.0, 0.0]),
        new).cov_base == 0.0


def test_cov_base_value():
    base = BenchmarkResult(name="c", wall_times=[1.0, 2.0, 3.0])
    cov = CaseComparison("c", base, None).cov_base
    assert cov == pytest.approx(base.wall_std / base.wall_mean)


def test_effective_tolerance_global_rule():
    base = BenchmarkResult(name="c", wall_times=[1.0, 1.0, 1.0])
    tol, rule = CaseComparison("c", base, None).effective_tolerance(0.05)
    assert rule == "global"
    assert tol == 0.05


def test_effective_tolerance_noise_rule():
    base = BenchmarkResult(name="c", wall_times=[1.0, 1.0, 1.0, 1.0, 1.10])
    case_cmp = CaseComparison("c", base, None)
    tol, rule = case_cmp.effective_tolerance(0.05)
    assert rule == "noise"
    assert tol == pytest.approx(NOISE_TOLERANCE_K * case_cmp.cov_base)
    assert tol > 0.05


def test_status_noise_band_absorbs_delta():
    # 3*cov ~= 13% here; an 8% delta clears the global 5% threshold but
    # stays inside the noise band -> "ok".
    base = BenchmarkResult(name="c", wall_times=[1.0, 1.0, 1.0, 1.0, 1.10])
    new = BenchmarkResult(name="c", wall_times=[1.08])
    comparison = CaseComparison("c", base, new)
    assert comparison.effective_tolerance(0.05)[0] > 0.08
    assert comparison.status(0.05) == "ok"


def test_status_delta_exceeds_noise_band():
    # same jittery base, but a 20% delta exceeds the ~13% band -> slower.
    base = BenchmarkResult(name="c", wall_times=[1.0, 1.0, 1.0, 1.0, 1.10])
    new = BenchmarkResult(name="c", wall_times=[1.20])
    comparison = CaseComparison("c", base, new)
    assert comparison.effective_tolerance(0.05)[0] < 0.20
    assert comparison.status(0.05) == "slower"


def test_status_low_cov_uses_global_threshold():
    # a stable base has no noise band; an 8% delta is flagged slower.
    base = BenchmarkResult(name="c", wall_times=[1.0, 1.0, 1.0])
    new = BenchmarkResult(name="c", wall_times=[1.08])
    comparison = CaseComparison("c", base, new)
    assert comparison.effective_tolerance(0.05) == (0.05, "global")
    assert comparison.status(0.05) == "slower"


# ================================================================
#  Absolute floor
# ================================================================
def test_status_floor_suppresses_tiny_absolute_delta():
    # a 15 ms case bumped by the ~0.8 ms per-process slow mode: rel is
    # +5.3% (past the global 5% band) but the absolute delta is below
    # the 1.2 ms floor, so it stays "ok".
    comparison = case(15e-3, 15.8e-3)
    assert comparison.wall.rel > comparison.effective_tolerance(0.05)[0]
    assert ABSOLUTE_FLOOR > 15.8e-3 - 15e-3
    assert comparison.status(0.05) == "ok"


def test_status_floor_never_masks_real_tiny_regression():
    # same 15 ms case, but a +2 ms bump clears both the relative band
    # and the absolute floor -> a real regression is still flagged.
    comparison = case(15e-3, 17e-3)
    assert comparison.wall.rel > comparison.effective_tolerance(0.05)[0]
    assert ABSOLUTE_FLOOR < 17e-3 - 15e-3
    assert comparison.status(0.05) == "slower"


def test_status_floor_irrelevant_on_large_case():
    # a 250 ms case at +6%: the absolute delta (15 ms) dwarfs the floor,
    # so the floor plays no role and the regression is flagged.
    comparison = case(250e-3, 265e-3)
    assert comparison.status(0.05) == "slower"


def test_status_floor_suppresses_symmetric_faster():
    # a slow-mode-contaminated baseline can make a fresh run look
    # spuriously faster; a sub-floor absolute delta is equally
    # meaningless, so the suppression is symmetric.
    comparison = case(15e-3, 14.2e-3)
    assert comparison.wall.rel < -comparison.effective_tolerance(0.05)[0]
    assert ABSOLUTE_FLOOR > 15e-3 - 14.2e-3
    assert comparison.status(0.05) == "ok"


def test_format_comparison_floor_annotation():
    # a floor-suppressed case renders "(floor)" in the tol column so the
    # suppression stays visible in the report rather than being silent.
    base = make_suite([BenchmarkResult(name="tiny", wall_times=[15e-3])])
    new = make_suite([BenchmarkResult(name="tiny", wall_times=[15.8e-3])])
    report = format_comparison(compare(base, new))
    assert "(floor)" in report
    for line in report.splitlines():
        if line.startswith("tiny"):
            assert "(floor)" in line
            assert "ok" in line


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


def test_format_comparison_shows_effective_tolerance():
    base = make_suite([
        BenchmarkResult(name="stable", wall_times=[1.0, 1.0, 1.0]),
        BenchmarkResult(
            name="jittery", wall_times=[1.0, 1.0, 1.0, 1.0, 1.10]),
    ])
    new = make_suite([
        BenchmarkResult(name="stable", wall_times=[2.0]),
        BenchmarkResult(name="jittery", wall_times=[1.08]),
    ])
    report = format_comparison(compare(base, new))
    # tol header and both rules appear; the slower (non-ok) stable case
    # carries its global tol, the jittery case its noise band
    assert "tol" in report
    assert "(global)" in report
    assert "(noise)" in report


def test_format_comparison_tolerance_dash_for_non_comparable():
    base = make_suite([BenchmarkResult(name="removed", wall_times=[1.0])])
    new = make_suite([BenchmarkResult(name="added", wall_times=[1.0])])
    report = format_comparison(compare(base, new))
    # added/removed rows have no tolerance band
    for line in report.splitlines():
        if line.startswith(("added", "removed")):
            assert "(global)" not in line
            assert "(noise)" not in line
