"""Tests for the benchmarking command line interface."""
import json
import subprocess
import sys
import textwrap

import pytest

from fridom.benchmarking import __main__ as benchmarking_main
from fridom.benchmarking.cli import main
from fridom.benchmarking.result import (
    BenchmarkResult,
    RunMetadata,
    SuiteResult,
)

BENCH_FILE = textwrap.dedent('''
    """A benchmark suite file used in tests."""
    from fridom.benchmarking.suite import benchmark_case


    @benchmark_case(params={"n": [2, 4]}, reps=2, warmup=0)
    def bench_scalar(n):
        return lambda: float(n)
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


def save_suite(path, results):
    mdata = RunMetadata(commit="a" * 40, backend="cpu",
                        device_kind="cpu", device_count=1)
    SuiteResult(metadata=mdata, results=results).save(path)


# ================================================================
#  run
# ================================================================
def test_run(suite_dir, capsys):
    output = suite_dir / "out.json"
    exit_code = main([
        "run", str(suite_dir), "--no-isolate", "--reps", "1",
        "-o", str(output),
    ])
    assert exit_code == 0
    suite = SuiteResult.load(output)
    assert [r.full_name for r in suite.results] == [
        "bench_scalar[n=2]", "bench_scalar[n=4]",
    ]
    out = capsys.readouterr().out
    assert "[1/2] bench_scalar[n=2]" in out
    assert f"results written to {output}" in out


def test_run_default_output(suite_dir, capsys):
    exit_code = main(["run", str(suite_dir), "--no-isolate", "--reps", "1"])
    assert exit_code == 0
    results = list((suite_dir / "results").glob("*.json"))
    assert len(results) == 1
    # default file name contains the short commit hash
    assert "-" in results[0].stem
    capsys.readouterr()


def test_run_with_filter(suite_dir, capsys):
    output = suite_dir / "out.json"
    exit_code = main([
        "run", str(suite_dir), "--no-isolate", "--reps", "1",
        "--filter", "n=4", "-o", str(output),
    ])
    assert exit_code == 0
    suite = SuiteResult.load(output)
    assert [r.full_name for r in suite.results] == ["bench_scalar[n=4]"]
    capsys.readouterr()


def test_run_markdown(suite_dir, capsys):
    exit_code = main([
        "run", str(suite_dir), "--no-isolate", "--reps", "1", "--markdown",
        "-o", str(suite_dir / "out.json"),
    ])
    assert exit_code == 0
    assert "| case |" in capsys.readouterr().out


def test_run_failing_case(suite_dir, capsys):
    (suite_dir / "bench_fail.py").write_text(FAILING_FILE)
    exit_code = main([
        "run", str(suite_dir), "--no-isolate", "--reps", "1",
        "-o", str(suite_dir / "out.json"),
    ])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "case bench_broken failed:" in out
    assert "setup exploded" in out


# ================================================================
#  compare
# ================================================================
def test_compare(tmp_path, capsys):
    save_suite(tmp_path / "base.json",
               [BenchmarkResult(name="a", wall_times=[1.0])])
    save_suite(tmp_path / "new.json",
               [BenchmarkResult(name="a", wall_times=[2.0])])

    exit_code = main([
        "compare", str(tmp_path / "base.json"), str(tmp_path / "new.json"),
    ])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "+100.0%" in out
    assert "slower" in out


def test_compare_fail_on_regression(tmp_path, capsys):
    save_suite(tmp_path / "base.json",
               [BenchmarkResult(name="a", wall_times=[1.0])])
    save_suite(tmp_path / "new.json",
               [BenchmarkResult(name="a", wall_times=[2.0])])

    exit_code = main([
        "compare", str(tmp_path / "base.json"), str(tmp_path / "new.json"),
        "--fail-on-regression",
    ])
    assert exit_code == 1
    capsys.readouterr()


def test_compare_threshold_and_markdown(tmp_path, capsys):
    save_suite(tmp_path / "base.json",
               [BenchmarkResult(name="a", wall_times=[1.0])])
    save_suite(tmp_path / "new.json",
               [BenchmarkResult(name="a", wall_times=[2.0])])

    exit_code = main([
        "compare", str(tmp_path / "base.json"), str(tmp_path / "new.json"),
        "--fail-on-regression", "--threshold", "2.0", "--markdown",
    ])
    assert exit_code == 0
    assert "| case |" in capsys.readouterr().out


# ================================================================
#  list
# ================================================================
def test_list(suite_dir, capsys):
    exit_code = main(["list", str(suite_dir)])
    assert exit_code == 0
    assert capsys.readouterr().out.splitlines() == [
        "bench_scalar[n=2]", "bench_scalar[n=4]",
    ]


# ================================================================
#  parser
# ================================================================
def test_missing_subcommand():
    with pytest.raises(SystemExit):
        main([])


def test_main_module_wires_up_cli():
    assert benchmarking_main.main is main


def test_module_entry_point(suite_dir):
    proc = subprocess.run(  # noqa: S603
        [sys.executable, "-m", "fridom.benchmarking", "list",
         str(suite_dir)],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    assert "bench_scalar[n=2]" in proc.stdout


def test_result_json_is_valid(suite_dir):
    output = suite_dir / "out.json"
    main(["run", str(suite_dir), "--no-isolate", "--reps", "1",
          "-o", str(output)])
    data = json.loads(output.read_text())
    assert data["format_version"] == 1
