"""Result containers and run metadata for benchmark measurements."""
from __future__ import annotations

import json
import platform
import statistics
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import jax

RESULT_FORMAT_VERSION = 1


# ================================================================
#  Run metadata
# ================================================================
@dataclass
class RunMetadata:

    """
    Metadata describing the environment of a benchmark run.

    Description
    -----------
    Captures everything needed to interpret and compare benchmark
    results later: the git commit the benchmarks ran on, the software
    versions, and the hardware (jax backend and devices). All fields
    default to None so that partially available information (e.g.
    outside a git repository) is not an error.

    Parameters
    ----------
    commit : str | None, optional
        The git commit hash (default: None).
    dirty : bool | None, optional
        Whether the working tree had uncommitted changes
        (default: None).
    branch : str | None, optional
        The git branch name (default: None).
    fridom_version : str | None, optional
        The installed fridom version (default: None).
    jax_version : str | None, optional
        The installed jax version (default: None).
    python_version : str | None, optional
        The python version (default: None).
    backend : str | None, optional
        The default jax backend, e.g. "cpu" or "gpu" (default: None).
    device_kind : str | None, optional
        The device kind of the first jax device (default: None).
    device_count : int | None, optional
        The number of jax devices (default: None).
    hostname : str | None, optional
        The network name of the machine (default: None).
    timestamp : str | None, optional
        The UTC timestamp of the run in ISO format (default: None).
    """

    commit: str | None = None
    dirty: bool | None = None
    branch: str | None = None
    fridom_version: str | None = None
    jax_version: str | None = None
    python_version: str | None = None
    backend: str | None = None
    device_kind: str | None = None
    device_count: int | None = None
    hostname: str | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the metadata as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunMetadata:
        """Construct a RunMetadata from a plain dictionary."""
        return cls(**data)


def _run_git(*args: str) -> str | None:
    """
    Run a git command and return its stripped stdout.

    Parameters
    ----------
    *args : str
        The git command line arguments (without the leading "git").

    Returns
    -------
    str | None
        The stripped stdout of the command, or None if git is not
        available or the command failed.
    """
    try:
        # the command is fully determined by the caller; no user input
        # is passed to the shell
        proc = subprocess.run(  # noqa: S603
            ["git", *args],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except OSError:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def collect_metadata() -> RunMetadata:
    """
    Collect metadata about the current benchmark environment.

    Description
    -----------
    Queries git for the current commit, branch, and dirty state, and
    jax for the backend and device information. Fields that cannot be
    determined (e.g. the commit outside a git repository) are set to
    None.

    Returns
    -------
    RunMetadata
        The collected metadata.
    """
    try:
        fridom_version = version("fridom")
    except PackageNotFoundError:  # pragma: no cover
        fridom_version = None

    status = _run_git("status", "--porcelain")
    dirty = None if status is None else bool(status)

    devices = jax.devices()
    return RunMetadata(
        commit=_run_git("rev-parse", "HEAD"),
        dirty=dirty,
        branch=_run_git("rev-parse", "--abbrev-ref", "HEAD"),
        fridom_version=fridom_version,
        jax_version=jax.__version__,
        python_version=platform.python_version(),
        backend=jax.default_backend(),
        device_kind=devices[0].device_kind,
        device_count=jax.device_count(),
        hostname=platform.node(),
        timestamp=datetime.now(tz=UTC).isoformat(),
    )


# ================================================================
#  Benchmark result
# ================================================================
@dataclass
class BenchmarkResult:

    """
    The measurements of a single benchmark case.

    Description
    -----------
    Wall times are stored per repetition; summary statistics are
    exposed as properties. Compile-related metrics (trace time,
    compile time, memory analysis, flop estimate) are only available
    when the benchmarked function was jit-compiled; the runtime peak
    device memory is only available on backends that report memory
    statistics (e.g. gpu). Unavailable metrics are None.

    Parameters
    ----------
    name : str
        The name of the benchmark case.
    params : dict[str, Any], optional
        The parameter values of this case (default: {}).
    wall_times : list[float], optional
        The wall time of each timed repetition in seconds
        (default: []).
    trace_time : float | None, optional
        The time spent tracing the function in seconds
        (default: None).
    compile_time : float | None, optional
        The time spent compiling the function in seconds
        (default: None).
    temp_bytes : int | None, optional
        The size of temporary buffers of the compiled executable in
        bytes (default: None).
    argument_bytes : int | None, optional
        The size of the arguments of the compiled executable in bytes
        (default: None).
    output_bytes : int | None, optional
        The size of the outputs of the compiled executable in bytes
        (default: None).
    code_bytes : int | None, optional
        The size of the generated code in bytes (default: None).
    flops : float | None, optional
        The estimated number of floating point operations of one call
        (default: None).
    peak_bytes : int | None, optional
        The peak device memory usage in bytes (default: None).
    extras : dict[str, float], optional
        Additional user-defined metrics, e.g. a throughput
        (default: {}).
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    wall_times: list[float] = field(default_factory=list)
    trace_time: float | None = None
    compile_time: float | None = None
    temp_bytes: int | None = None
    argument_bytes: int | None = None
    output_bytes: int | None = None
    code_bytes: int | None = None
    flops: float | None = None
    peak_bytes: int | None = None
    extras: dict[str, float] = field(default_factory=dict)

    # ----------------------------------------------------------------
    #  Properties
    # ----------------------------------------------------------------
    @property
    def full_name(self) -> str:
        """The case name including the parameter values."""
        if not self.params:
            return self.name
        params = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}[{params}]"

    @property
    def wall_median(self) -> float | None:
        """The median wall time in seconds."""
        if not self.wall_times:
            return None
        return statistics.median(self.wall_times)

    @property
    def wall_min(self) -> float | None:
        """The minimum wall time in seconds."""
        if not self.wall_times:
            return None
        return min(self.wall_times)

    @property
    def wall_mean(self) -> float | None:
        """The mean wall time in seconds."""
        if not self.wall_times:
            return None
        return statistics.fmean(self.wall_times)

    @property
    def wall_std(self) -> float | None:
        """The standard deviation of the wall times in seconds."""
        if not self.wall_times:
            return None
        if len(self.wall_times) < 2:  # noqa: PLR2004
            return 0.0
        return statistics.stdev(self.wall_times)

    # ----------------------------------------------------------------
    #  Serialization
    # ----------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        """Return the result as a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkResult:
        """Construct a BenchmarkResult from a plain dictionary."""
        return cls(**data)


# ================================================================
#  Suite result
# ================================================================
@dataclass
class SuiteResult:

    """
    The results of a benchmark suite run.

    Description
    -----------
    Bundles the results of all benchmark cases of one run together
    with the metadata of the environment they were measured in.
    Serializes to and from JSON files for later comparison.

    Parameters
    ----------
    metadata : RunMetadata
        The metadata of the run.
    results : list[BenchmarkResult], optional
        The results of the individual benchmark cases (default: []).
    """

    metadata: RunMetadata
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return the suite result as a plain dictionary."""
        return {
            "format_version": RESULT_FORMAT_VERSION,
            "metadata": self.metadata.to_dict(),
            "results": [result.to_dict() for result in self.results],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SuiteResult:
        """Construct a SuiteResult from a plain dictionary."""
        return cls(
            metadata=RunMetadata.from_dict(data["metadata"]),
            results=[
                BenchmarkResult.from_dict(result)
                for result in data["results"]
            ],
        )

    def save(self, path: Path | str) -> None:
        """
        Save the suite result to a JSON file.

        Parameters
        ----------
        path : Path | str
            The path of the JSON file. Parent directories are created
            if necessary.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as file:
            json.dump(self.to_dict(), file, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> SuiteResult:
        """
        Load a suite result from a JSON file.

        Parameters
        ----------
        path : Path | str
            The path of the JSON file.

        Returns
        -------
        SuiteResult
            The loaded suite result.
        """
        with Path(path).open() as file:
            return cls.from_dict(json.load(file))
