"""Run benchmark suites, isolating cases in subprocesses."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from fridom.benchmarking.result import (
    BenchmarkResult,
    SuiteResult,
    collect_metadata,
)
from fridom.benchmarking.suite import discover_cases

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.benchmarking.suite import CaseInstance


# ================================================================
#  Helper functions
# ================================================================
def _error_result(instance: CaseInstance, message: str) -> BenchmarkResult:
    """Create a result marking a failed case instance."""
    return BenchmarkResult(
        name=instance.case.name,
        params=dict(instance.params),
        error=message,
    )


def _run_in_process(
    instance: CaseInstance,
    *,
    reps: int | None,
    warmup: int | None,
) -> BenchmarkResult:
    """Run a case instance in the current process."""
    try:
        return instance.run(reps=reps, warmup=warmup)
    # the suite must survive arbitrary case failures
    except Exception:  # noqa: BLE001
        return _error_result(instance, traceback.format_exc())


def _run_isolated(
    instance: CaseInstance,
    *,
    reps: int | None,
    warmup: int | None,
    timeout: float | None,
) -> BenchmarkResult:
    """Run a case instance in a fresh child process."""
    if instance.case.source_file is None:
        raise ValueError(
            "subprocess isolation requires cases discovered from a "
            "suite file")
    overrides = {"reps": reps, "warmup": warmup}
    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp) / "result.json"
        cmd = [
            sys.executable,
            "-m", "fridom.benchmarking._child",
            str(instance.case.source_file),
            instance.case.name,
            json.dumps(instance.params),
            json.dumps(overrides),
            str(output),
        ]
        try:
            # the command is fully determined above; no user input is
            # passed to the shell
            proc = subprocess.run(  # noqa: S603
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return _error_result(
                instance, f"timed out after {timeout} seconds")
        if proc.returncode != 0:
            return _error_result(instance, proc.stderr.strip())
        return BenchmarkResult.from_dict(json.loads(output.read_text()))


# ================================================================
#  Suite runner
# ================================================================
def run_suite(
    directory: Path | str,
    *,
    pattern: str | None = None,
    reps: int | None = None,
    warmup: int | None = None,
    isolate: bool = True,
    timeout: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> SuiteResult:
    """
    Run all benchmark cases of a suite directory.

    Description
    -----------
    Discovers the cases of all `bench_*.py` files in the directory,
    expands their parameter grids, and runs every instance. By default
    each instance runs in a fresh child process so that peak-memory
    counters are independent, jit caches do not interact, and a
    crashing case (e.g. out-of-memory) does not abort the suite.
    Failed cases are recorded on the result via their `error` field.

    Parameters
    ----------
    directory : Path | str
        The suite directory containing the `bench_*.py` files.
    pattern : str | None, optional
        Only run instances whose full name contains this substring
        (default: None).
    reps : int | None, optional
        Override for the number of timed repetitions of all cases;
        None uses the case defaults (default: None).
    warmup : int | None, optional
        Override for the number of warmup calls of all cases; None
        uses the case defaults (default: None).
    isolate : bool, optional
        Whether to run each instance in a fresh child process
        (default: True).
    timeout : float | None, optional
        Timeout per isolated case instance in seconds (default: None).
    progress : Callable[[str], None] | None, optional
        Callback receiving a progress message per instance
        (default: None).

    Returns
    -------
    SuiteResult
        The results of all instances together with the run metadata.
    """
    cases = discover_cases(directory)
    instances = [
        instance for case in cases for instance in case.instances()
    ]
    if pattern is not None:
        instances = [
            instance for instance in instances
            if pattern in instance.full_name
        ]
    report = progress if progress is not None else lambda _msg: None

    results = []
    total = len(instances)
    for i, instance in enumerate(instances):
        report(f"[{i + 1}/{total}] {instance.full_name}")
        if isolate:
            result = _run_isolated(
                instance, reps=reps, warmup=warmup, timeout=timeout)
        else:
            result = _run_in_process(instance, reps=reps, warmup=warmup)
        if result.error is not None:
            report(f"[{i + 1}/{total}] {instance.full_name} FAILED")
        results.append(result)
    return SuiteResult(metadata=collect_metadata(), results=results)
