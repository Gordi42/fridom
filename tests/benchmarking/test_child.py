"""Tests for the benchmark child-process entry point."""
import json
import textwrap

from fridom.benchmarking._child import main
from fridom.benchmarking.result import BenchmarkResult

BENCH_FILE = textwrap.dedent('''
    """A benchmark suite file used in tests."""
    from fridom.benchmarking.suite import benchmark_case


    @benchmark_case(params={"n": [2]}, reps=3, warmup=0)
    def bench_scalar(n):
        return lambda: float(n)
''')


def test_child_success(tmp_path):
    file = tmp_path / "bench_test.py"
    file.write_text(BENCH_FILE)
    output = tmp_path / "result.json"

    exit_code = main([
        str(file),
        "bench_scalar",
        json.dumps({"n": 2}),
        json.dumps({"reps": 1, "warmup": None}),
        str(output),
    ])

    assert exit_code == 0
    result = BenchmarkResult.from_dict(json.loads(output.read_text()))
    assert result.name == "bench_scalar"
    assert result.params == {"n": 2}
    # the reps override took precedence over the case default
    assert len(result.wall_times) == 1


def test_child_case_not_found(tmp_path, capsys):
    file = tmp_path / "bench_test.py"
    file.write_text(BENCH_FILE)

    exit_code = main([
        str(file),
        "bench_missing",
        json.dumps({}),
        json.dumps({}),
        str(tmp_path / "result.json"),
    ])

    assert exit_code == 1
    assert "not found" in capsys.readouterr().err


def test_child_wrong_argument_count(capsys):
    exit_code = main(["only", "three", "args"])
    assert exit_code == 1
    assert "usage:" in capsys.readouterr().err
