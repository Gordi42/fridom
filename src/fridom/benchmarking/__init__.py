"""Infrastructure to benchmark jax functions and fridom models."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import cli

    # import all classes and functions
    from .compare import (
        CaseComparison,
        MetricDelta,
        SuiteComparison,
        compare,
        format_comparison,
        format_suite,
    )
    from .measure import benchmark
    from .result import (
        BenchmarkResult,
        RunMetadata,
        SuiteResult,
        collect_metadata,
    )
    from .runner import run_suite
    from .suite import (
        BenchmarkCase,
        CaseInstance,
        benchmark_case,
        discover_cases,
        load_cases,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.benchmarking"

all_modules_by_origin = {
    base: ["cli"],
}

all_imports_by_origin = {
    f"{base}.compare": [
        "CaseComparison",
        "MetricDelta",
        "SuiteComparison",
        "compare",
        "format_comparison",
        "format_suite",
    ],
    f"{base}.measure": ["benchmark"],
    f"{base}.result": [
        "BenchmarkResult",
        "RunMetadata",
        "SuiteResult",
        "collect_metadata",
    ],
    f"{base}.runner": ["run_suite"],
    f"{base}.suite": [
        "BenchmarkCase",
        "CaseInstance",
        "benchmark_case",
        "discover_cases",
        "load_cases",
    ],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
