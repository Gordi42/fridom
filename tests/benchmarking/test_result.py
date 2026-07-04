"""Tests for the benchmark result containers and run metadata."""
import subprocess

import pytest

from fridom.benchmarking.result import (
    RESULT_FORMAT_VERSION,
    BenchmarkResult,
    RunMetadata,
    SuiteResult,
    _run_git,
    collect_metadata,
)


# ================================================================
#  BenchmarkResult statistics
# ================================================================
def test_wall_time_statistics():
    result = BenchmarkResult(name="case", wall_times=[1.0, 2.0, 6.0])
    assert result.wall_median == 2.0
    assert result.wall_min == 1.0
    assert result.wall_mean == 3.0
    assert result.wall_std == pytest.approx(2.6457513110645907)


def test_wall_time_statistics_empty():
    result = BenchmarkResult(name="case")
    assert result.wall_median is None
    assert result.wall_min is None
    assert result.wall_mean is None
    assert result.wall_std is None


def test_wall_time_std_single_repetition():
    result = BenchmarkResult(name="case", wall_times=[1.0])
    assert result.wall_std == 0.0


# ================================================================
#  BenchmarkResult full name
# ================================================================
def test_full_name_without_params():
    result = BenchmarkResult(name="case")
    assert result.full_name == "case"


def test_full_name_with_params():
    result = BenchmarkResult(name="case", params={"n": 64, "dim": 3})
    assert result.full_name == "case[n=64,dim=3]"


# ================================================================
#  Serialization round trips
# ================================================================
def test_benchmark_result_round_trip():
    result = BenchmarkResult(
        name="case",
        params={"n": 64},
        wall_times=[0.1, 0.2],
        trace_time=0.01,
        compile_time=0.5,
        temp_bytes=1024,
        argument_bytes=2048,
        output_bytes=8,
        code_bytes=4096,
        flops=1.0e6,
        peak_bytes=None,
        extras={"points_per_second": 1.0e9},
    )
    restored = BenchmarkResult.from_dict(result.to_dict())
    assert restored == result


def test_suite_result_round_trip(tmp_path):
    suite = SuiteResult(
        metadata=RunMetadata(commit="abc", device_count=1),
        results=[BenchmarkResult(name="case", wall_times=[0.1])],
    )
    path = tmp_path / "results" / "suite.json"
    suite.save(path)
    assert path.exists()
    restored = SuiteResult.load(path)
    assert restored == suite


def test_suite_result_format_version():
    suite = SuiteResult(metadata=RunMetadata())
    assert suite.to_dict()["format_version"] == RESULT_FORMAT_VERSION


# ================================================================
#  Metadata collection
# ================================================================
def test_collect_metadata():
    mdata = collect_metadata()
    assert isinstance(mdata.commit, str)
    assert len(mdata.commit) == 40
    assert isinstance(mdata.dirty, bool)
    assert isinstance(mdata.branch, str)
    assert mdata.fridom_version is not None
    assert mdata.jax_version is not None
    assert mdata.python_version is not None
    assert mdata.backend is not None
    assert mdata.device_kind is not None
    assert mdata.device_count >= 1
    assert mdata.hostname is not None
    assert "T" in mdata.timestamp


def test_collect_metadata_without_git(monkeypatch):
    monkeypatch.setattr(
        "fridom.benchmarking.result._run_git", lambda *_: None)
    mdata = collect_metadata()
    assert mdata.commit is None
    assert mdata.dirty is None
    assert mdata.branch is None
    # non-git fields are still populated
    assert mdata.jax_version is not None


# ================================================================
#  _run_git
# ================================================================
def test_run_git_success():
    assert _run_git("rev-parse", "HEAD") is not None


def test_run_git_failing_command():
    assert _run_git("not-a-real-subcommand") is None


def test_run_git_missing_executable(monkeypatch):
    def raise_oserror(*_args, **_kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(subprocess, "run", raise_oserror)
    assert _run_git("rev-parse", "HEAD") is None
