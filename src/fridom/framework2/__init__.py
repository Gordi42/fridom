"""
Transitional parallel framework package for the grid redesign.

Description
-----------
Hosts the ``framework2.grid`` package and the Phase-2 model layer
(``model``, ``transforms``, ``io``, ``ops``), all renamed to
``fridom.framework.*`` at cutover; see
``notes/framework2/classes/README.md`` and
``notes/framework2/model/classes/README.md`` for the design
contracts. ``framework2`` may import ``fridom.framework.utils`` but
must not import the old model/grid stack.

Top-level re-exports (``fr.Model``, ``fr.Ramp``, ``fr.params``,
``fr.StateTransform``, ``fr.every``/``fr.at``, ...) are added wave by
wave with the Phase-2 implementation plan
(``notes/framework2/model/implementation_plan.md``).
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    # import all modules
    from . import grid, io, model, ops, transforms

# ================================================================
#  Setup lazy loading
# ================================================================
all_modules_by_origin = {
    "fridom.framework2": ["grid", "model", "transforms", "io", "ops"],
}

all_imports_by_origin = {}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
