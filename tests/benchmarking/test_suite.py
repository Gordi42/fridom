"""Tests for the benchmark suite definition and discovery."""
import textwrap

import jax.numpy as jnp
import pytest

from fridom.benchmarking.suite import (
    BenchmarkCase,
    benchmark_case,
    discover_cases,
    load_cases,
)


# ================================================================
#  benchmark_case decorator
# ================================================================
def test_decorator_defaults():
    @benchmark_case()
    def bench_something():
        return lambda: None

    assert isinstance(bench_something, BenchmarkCase)
    assert bench_something.name == "bench_something"
    assert bench_something.params == {}
    assert bench_something.reps == 10
    assert bench_something.warmup == 2
    assert bench_something.measure_compile is True
    assert bench_something.source_file is None


def test_decorator_custom_settings():
    @benchmark_case(
        params={"n": [1, 2]}, reps=3, warmup=0,
        measure_compile=False, name="custom")
    def bench_something(n):
        return lambda: n

    assert bench_something.name == "custom"
    assert bench_something.params == {"n": [1, 2]}
    assert bench_something.reps == 3
    assert bench_something.warmup == 0
    assert bench_something.measure_compile is False


# ================================================================
#  Parameter grid expansion
# ================================================================
def test_instances_without_params():
    @benchmark_case()
    def bench_something():
        return lambda: None

    instances = bench_something.instances()
    assert len(instances) == 1
    assert instances[0].params == {}
    assert instances[0].full_name == "bench_something"


def test_instances_cartesian_product():
    @benchmark_case(params={"a": [1, 2], "b": ["x"]})
    def bench_something(a, b):
        return lambda: (a, b)

    instances = bench_something.instances()
    assert [inst.params for inst in instances] == [
        {"a": 1, "b": "x"},
        {"a": 2, "b": "x"},
    ]
    assert instances[0].full_name == "bench_something[a=1,b=x]"


# ================================================================
#  Instance execution
# ================================================================
def test_instance_run():
    calls = []

    @benchmark_case(params={"n": [4]}, reps=2, warmup=0)
    def bench_square(n):
        calls.append(n)
        x = jnp.ones(n)

        def run():
            return x * x

        return run

    instance = bench_square.instances()[0]
    result = instance.run()
    # the setup ran exactly once
    assert calls == [4]
    assert result.name == "bench_square"
    assert result.params == {"n": 4}
    assert len(result.wall_times) == 2
    assert result.compile_time > 0


def test_instance_run_with_overrides():
    @benchmark_case(reps=10, warmup=5)
    def bench_something():
        return lambda: None

    result = bench_something.instances()[0].run(reps=1, warmup=0)
    assert len(result.wall_times) == 1


def test_instance_run_with_args():
    @benchmark_case(reps=1, warmup=0)
    def bench_with_args():
        x = jnp.ones(4)

        def run(x):
            return x * 2

        return run, (x,)

    result = bench_with_args.instances()[0].run()
    assert len(result.wall_times) == 1
    # the argument was traced, not embedded as a constant
    assert result.argument_bytes > 0


def test_instance_run_with_extras():
    @benchmark_case(reps=1, warmup=0)
    def bench_with_extras():
        def run():
            return None

        return run, (), {"points": 42.0}

    result = bench_with_extras.instances()[0].run()
    assert result.extras == {"points": 42.0}


def test_instance_run_invalid_tuple():
    @benchmark_case(reps=1, warmup=0)
    def bench_invalid():
        return (lambda: None), (), {}, "extra"

    with pytest.raises(ValueError, match="tuple of length 4"):
        bench_invalid.instances()[0].run()


# ================================================================
#  Discovery
# ================================================================
BENCH_FILE = textwrap.dedent('''
    """A benchmark suite file used in tests."""
    from fridom.benchmarking.suite import benchmark_case


    @benchmark_case(params={"n": [2, 4]}, reps=1, warmup=0)
    def bench_alpha(n):
        return lambda: n


    @benchmark_case(reps=1, warmup=0)
    def bench_beta():
        return lambda: None
''')

DUPLICATE_FILE = textwrap.dedent('''
    """A suite file with duplicate case names."""
    from fridom.benchmarking.suite import benchmark_case


    @benchmark_case(name="bench_same")
    def bench_a():
        return lambda: None


    @benchmark_case(name="bench_same")
    def bench_b():
        return lambda: None
''')


def test_load_cases(tmp_path):
    file = tmp_path / "bench_test.py"
    file.write_text(BENCH_FILE)
    cases = load_cases(file)
    assert [case.name for case in cases] == ["bench_alpha", "bench_beta"]
    assert all(case.source_file == file for case in cases)


def test_load_cases_duplicate_names(tmp_path):
    file = tmp_path / "bench_dup.py"
    file.write_text(DUPLICATE_FILE)
    with pytest.raises(ValueError, match="duplicate benchmark case names"):
        load_cases(file)


def test_load_cases_empty_file(tmp_path):
    file = tmp_path / "bench_empty.py"
    file.write_text('"""No cases here."""\n')
    assert load_cases(file) == []


def test_discover_cases(tmp_path):
    (tmp_path / "bench_b.py").write_text(BENCH_FILE)
    (tmp_path / "bench_a.py").write_text(
        BENCH_FILE.replace("alpha", "gamma").replace("beta", "delta"))
    # files not matching bench_*.py are ignored
    (tmp_path / "helper.py").write_text("x = 1\n")

    cases = discover_cases(tmp_path)
    # files are visited in sorted order
    assert [case.name for case in cases] == [
        "bench_gamma", "bench_delta", "bench_alpha", "bench_beta",
    ]


def test_discover_cases_empty_directory(tmp_path):
    assert discover_cases(tmp_path) == []
