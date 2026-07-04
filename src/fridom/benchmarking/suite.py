"""Benchmark suite definition and discovery."""
from __future__ import annotations

import importlib.util
import itertools
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fridom.benchmarking.measure import benchmark

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable
    from types import ModuleType

    from fridom.benchmarking.result import BenchmarkResult


# ================================================================
#  Benchmark case
# ================================================================
@dataclass
class BenchmarkCase:

    """
    A single benchmark case with an optional parameter grid.

    Description
    -----------
    A benchmark case wraps a setup function together with measurement
    settings and a parameter grid. The setup function receives one
    value per parameter and returns the callable to be measured (the
    setup itself is not timed). It may alternatively return a tuple
    `(run, args)` or `(run, args, extras)`:

    - `args` is a tuple of arguments the callable is measured with.
      When the case is jit-compiled (`measure_compile=True`), the
      arguments are traced. Pass arrays through `args` instead of
      closing over them, otherwise the compiler embeds them as
      constants and may fold the computation away.
    - `extras` is a dictionary of additional metrics (e.g. the number
      of grid points) that is attached to the result.

    The measured callable should return the arrays (or pytrees of
    arrays, e.g. fields) produced by the computation: jax dispatches
    asynchronously, and the measurement blocks on the return value.

    Benchmark cases are usually created with the
    :func:`benchmark_case` decorator.

    Parameters
    ----------
    setup : Callable
        The setup function; returns the callable to be measured, or a
        tuple `(run, args)` or `(run, args, extras)`.
    name : str
        The name of the case.
    params : dict[str, list[Any]], optional
        The parameter grid; the case is expanded to one instance per
        entry of the cartesian product. Values must be
        JSON-serializable (default: {}).
    reps : int, optional
        The default number of timed repetitions (default: 10).
    warmup : int, optional
        The default number of untimed warmup calls (default: 2).
    measure_compile : bool, optional
        Whether to jit-compile the measured callable ahead of time and
        record compile-related metrics (default: True).
    source_file : Path | None, optional
        The file the case was discovered from; set by
        :func:`load_cases` (default: None).
    """

    setup: Callable[..., Any]
    name: str
    params: dict[str, list[Any]] = field(default_factory=dict)
    reps: int = 10
    warmup: int = 2
    measure_compile: bool = True
    source_file: Path | None = None

    def instances(self) -> list[CaseInstance]:
        """
        Expand the parameter grid into case instances.

        Returns
        -------
        list[CaseInstance]
            One instance per entry of the cartesian product of the
            parameter grid; a single instance if the grid is empty.
        """
        if not self.params:
            return [CaseInstance(case=self, params={})]
        keys = list(self.params)
        return [
            CaseInstance(
                case=self,
                params=dict(zip(keys, values, strict=True)),
            )
            for values in itertools.product(
                *(self.params[key] for key in keys))
        ]


@dataclass
class CaseInstance:

    """
    A benchmark case bound to concrete parameter values.

    Parameters
    ----------
    case : BenchmarkCase
        The benchmark case.
    params : dict[str, Any]
        The parameter values of this instance.
    """

    case: BenchmarkCase
    params: dict[str, Any]

    @property
    def full_name(self) -> str:
        """The case name including the parameter values."""
        if not self.params:
            return self.case.name
        params = ",".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.case.name}[{params}]"

    def run(
        self,
        reps: int | None = None,
        warmup: int | None = None,
    ) -> BenchmarkResult:
        """
        Run the case instance in the current process.

        Parameters
        ----------
        reps : int | None, optional
            Override for the number of timed repetitions; None uses
            the case default (default: None).
        warmup : int | None, optional
            Override for the number of warmup calls; None uses the
            case default (default: None).

        Returns
        -------
        BenchmarkResult
            The measurements, with the parameter values attached.
        """
        target = self.case.setup(**self.params)
        args: tuple[Any, ...] = ()
        extras: dict[str, float] = {}
        if isinstance(target, tuple):
            if len(target) == 2:  # noqa: PLR2004
                target, args = target
            elif len(target) == 3:  # noqa: PLR2004
                target, args, extras = target
            else:
                raise ValueError(
                    "the setup function must return the callable to "
                    "be measured, or a tuple (run, args) or "
                    f"(run, args, extras); got a tuple of length "
                    f"{len(target)}")
        result = benchmark(
            target,
            *args,
            reps=self.case.reps if reps is None else reps,
            warmup=self.case.warmup if warmup is None else warmup,
            measure_compile=self.case.measure_compile,
            name=self.case.name,
        )
        result.params = dict(self.params)
        result.extras.update(extras)
        return result


# ================================================================
#  Case definition decorator
# ================================================================
def benchmark_case(
    params: dict[str, list[Any]] | None = None,
    *,
    reps: int = 10,
    warmup: int = 2,
    measure_compile: bool = True,
    name: str | None = None,
) -> Callable[[Callable[..., Any]], BenchmarkCase]:
    """
    Declare a benchmark case in a suite file.

    Description
    -----------
    Decorator for setup functions in benchmark suite files
    (`bench_*.py`). The decorated function receives one value per
    parameter and returns the callable to be measured (or a tuple
    `(run, args)` or `(run, args, extras)`, see
    :class:`BenchmarkCase`).

    Parameters
    ----------
    params : dict[str, list[Any]] | None, optional
        The parameter grid; one case instance per entry of the
        cartesian product (default: None).
    reps : int, optional
        The default number of timed repetitions (default: 10).
    warmup : int, optional
        The default number of untimed warmup calls (default: 2).
    measure_compile : bool, optional
        Whether to jit-compile the measured callable ahead of time
        (default: True).
    name : str | None, optional
        The case name; defaults to the function name (default: None).

    Returns
    -------
    Callable
        A decorator that turns the setup function into a
        :class:`BenchmarkCase`.

    Examples
    --------
    .. code-block:: python

        import jax.numpy as jnp
        import fridom.benchmarking as bm

        @bm.benchmark_case(params={"n": [64, 128, 256]})
        def bench_square_sum(n):
            x = jnp.ones((n, n))

            def run(x):
                return (x * x).sum()

            return run, (x,), {"points": float(n * n)}
    """
    def decorator(fn: Callable[..., Any]) -> BenchmarkCase:
        return BenchmarkCase(
            setup=fn,
            name=name or fn.__name__,
            params=dict(params or {}),
            reps=reps,
            warmup=warmup,
            measure_compile=measure_compile,
        )
    return decorator


# ================================================================
#  Discovery
# ================================================================
_module_counter = itertools.count()


def _import_file(file: Path) -> ModuleType:
    """Import a python file under a unique module name."""
    module_name = f"fridom_benchmark_suite_{next(_module_counter)}"
    spec = importlib.util.spec_from_file_location(module_name, file)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise ImportError(f"cannot import benchmark file {file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_cases(file: Path | str) -> list[BenchmarkCase]:
    """
    Load all benchmark cases from a suite file.

    Parameters
    ----------
    file : Path | str
        The suite file to import.

    Returns
    -------
    list[BenchmarkCase]
        The cases defined in the file, with `source_file` set.

    Raises
    ------
    ValueError
        If two cases in the file share the same name.
    """
    file = Path(file)
    module = _import_file(file)
    cases = [
        obj for obj in vars(module).values()
        if isinstance(obj, BenchmarkCase)
    ]
    names = [case.name for case in cases]
    duplicates = {name for name in names if names.count(name) > 1}
    if duplicates:
        raise ValueError(
            f"duplicate benchmark case names in {file}: "
            f"{sorted(duplicates)}")
    for case in cases:
        case.source_file = file
    return cases


def discover_cases(directory: Path | str) -> list[BenchmarkCase]:
    """
    Discover all benchmark cases in a suite directory.

    Description
    -----------
    Imports every `bench_*.py` file in the directory (sorted by file
    name) and collects the benchmark cases defined in them.

    Parameters
    ----------
    directory : Path | str
        The suite directory.

    Returns
    -------
    list[BenchmarkCase]
        All discovered cases.
    """
    directory = Path(directory)
    cases = []
    for file in sorted(directory.glob("bench_*.py")):
        cases.extend(load_cases(file))
    return cases
