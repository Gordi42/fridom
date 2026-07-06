"""
Transitional parallel framework package for the grid redesign.

Description
-----------
Hosts the ``framework2.grid`` package (renamed to
``fridom.framework.grid`` at cutover); see
``notes/framework2/classes/README.md`` for the design contract.
``framework2`` may import ``fridom.framework.utils`` but must not
import the old model/grid stack.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import grid

# ================================================================
#  Setup lazy loading
# ================================================================
all_modules_by_origin = {
    "fridom.framework2": ["grid"],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
