"""Infrastructure to benchmark jax functions and fridom models."""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all classes and functions
    from .measure import benchmark
    from .result import (
        BenchmarkResult,
        RunMetadata,
        SuiteResult,
        collect_metadata,
    )

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.benchmarking"

all_modules_by_origin = {}

all_imports_by_origin = {
    f"{base}.measure": ["benchmark"],
    f"{base}.result": [
        "BenchmarkResult",
        "RunMetadata",
        "SuiteResult",
        "collect_metadata",
    ],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
