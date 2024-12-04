"""Utility functions and classes for the FRIDOM framework."""
from typing import TYPE_CHECKING
from lazypimp import setup

# for the sphinx autodoc to work, we must import jaxify here in non-lazy mode
# During the doc build process, this module will be imported twice,
# once for the module exploration and once for the actual doc build.
# During module exploration jaxify will fail to import because jax is not available.
# This is not a problem because the exploration import only needs the
# all_modules_by_origin and all_imports_by_origin dictionaries.
# So we can safely ignore the ImportError here.
try:
    from .jax_utils import jaxify
except ImportError:
    # We are in the module exploration phase of doc building
    pass

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:
    # import content
    from .mpi import I_AM_MAIN_RANK, MPI_AVAILABLE, mpi_barrier, get_mpi_size
    from .printing import print_bar, print_job_init_info
    from .formatting import humanize_length, humanize_time, humanize_number
    from .filesystem import chdir_to_submit_dir, stdout_is_file
    from .array_ops import SliceableAttribute, modify_array, random_array
    from .numpy_utils import to_numpy, to_seconds
    from .jax_utils import jaxjit, jaxify, inspect_jitted_function, free_memory
    from .decorators import skip_on_doc_build, cache_figure

# ================================================================
#  Setup lazy loading
# ================================================================

all_modules_by_origin = {}

BASE = "fridom.framework.utils"
all_imports_by_origin = {
    f"{BASE}.mpi": ["I_AM_MAIN_RANK", "MPI_AVAILABLE", "mpi_barrier", "get_mpi_size"],
    f"{BASE}.printing": ["print_bar", "print_job_init_info"],
    f"{BASE}.formatting": ["humanize_length", "humanize_time", "humanize_number"],
    f"{BASE}.filesystem": ["chdir_to_submit_dir", "stdout_is_file"],
    f"{BASE}.array_ops": ["SliceableAttribute", "modify_array", "random_array"],
    f"{BASE}.numpy_utils": ["to_numpy", "to_seconds"],
    f"{BASE}.jax_utils": [
        "jaxjit", "jaxify", "inspect_jitted_function", "free_memory"
        ],
    f"{BASE}.decorators": ["skip_on_doc_build", "cache_figure"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
