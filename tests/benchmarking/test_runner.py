"""Tests for the benchmark suite runner."""
import subprocess
import textwrap

import pytest

from fridom.benchmarking.runner import _run_isolated, run_suite
from fridom.benchmarking.suite import BenchmarkCase, CaseInstance

BENCH_FILE = textwrap.dedent('''
    """A benchmark suite file used in tests."""
    import jax.numpy as jnp

    from fridom.benchmarking.suite import benchmark_case


    @benchmark_case(params={"n": [2, 4]}, reps=2, warmup=0)
    def bench_square(n):
        x = jnp.ones(n)
        return lambda: x * x


    @benchmark_case(reps=1, warmup=0)
    def bench_scalar():
        return lambda: 1.0
''')

FAILING_FILE = textwrap.dedent('''
    """A suite file whose case fails during setup."""
    from fridom.benchmarking.suite import benchmark_case


    @benchmark_case(reps=1, warmup=0)
    def bench_broken():
        raise RuntimeError("setup exploded")
''')


@pytest.fixture
def suite_dir(tmp_path):
    (tmp_path / "bench_test.py").write_text(BENCH_FILE)
    return tmp_path


# ================================================================
#  In-process runs
# ================================================================
def test_run_suite_in_process(suite_dir):
    messages = []
    suite = run_suite(
        suite_dir, isolate=False, progress=messages.append)

    assert [result.full_name for result in suite.results] == [
        "bench_square[n=2]", "bench_square[n=4]", "bench_scalar",
    ]
    assert all(result.error is None for result in suite.results)
    assert suite.metadata.commit is not None
    assert messages == [
        "[1/3] bench_square[n=2]",
        "[2/3] bench_square[n=4]",
        "[3/3] bench_scalar",
    ]


def test_run_suite_pattern_filter(suite_dir):
    suite = run_suite(suite_dir, isolate=False, pattern="n=4")
    assert [result.full_name for result in suite.results] == [
        "bench_square[n=4]",
    ]


def test_run_suite_overrides(suite_dir):
    suite = run_suite(suite_dir, isolate=False, reps=1, warmup=0)
    assert all(len(result.wall_times) == 1 for result in suite.results)


def test_run_suite_failing_case(suite_dir):
    (suite_dir / "bench_fail.py").write_text(FAILING_FILE)
    messages = []
    suite = run_suite(
        suite_dir, isolate=False, progress=messages.append)

    by_name = {result.full_name: result for result in suite.results}
    assert "setup exploded" in by_name["bench_broken"].error
    # the other cases still ran
    assert by_name["bench_scalar"].error is None
    assert "[1/4] bench_broken FAILED" in messages


def test_run_suite_first_only(suite_dir):
    suite = run_suite(suite_dir, isolate=False, first_only=True)
    assert [result.full_name for result in suite.results] == [
        "bench_square[n=2]", "bench_scalar",
    ]


def test_run_suite_without_progress(suite_dir):
    suite = run_suite(suite_dir, isolate=False, pattern="bench_scalar")
    assert len(suite.results) == 1


# ================================================================
#  Subprocess isolation
# ================================================================
def test_run_suite_isolated(suite_dir):
    suite = run_suite(
        suite_dir, pattern="bench_scalar", reps=1, warmup=0)
    assert len(suite.results) == 1
    result = suite.results[0]
    assert result.error is None
    assert len(result.wall_times) == 1


def test_run_isolated_requires_source_file():
    case = BenchmarkCase(setup=lambda: (lambda: None), name="bench_x")
    instance = CaseInstance(case=case, params={})
    with pytest.raises(ValueError, match="requires cases discovered"):
        _run_isolated(instance, reps=None, warmup=None, timeout=None)


def _make_instance(tmp_path):
    case = BenchmarkCase(
        setup=lambda: (lambda: None),
        name="bench_x",
        source_file=tmp_path / "bench_x.py",
    )
    return CaseInstance(case=case, params={})


def test_run_isolated_child_failure(tmp_path, monkeypatch):
    def fake_run(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="boom\n")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = _run_isolated(
        _make_instance(tmp_path), reps=None, warmup=None, timeout=None)
    assert result.error == "boom"
    assert result.name == "bench_x"


def test_run_isolated_timeout(tmp_path, monkeypatch):
    def fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd=[], timeout=5)

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = _run_isolated(
        _make_instance(tmp_path), reps=None, warmup=None, timeout=5)
    assert "timed out after 5 seconds" in result.error
