"""
Shallow-water tendency modules (framework2 port).

Description
-----------
The module library for the shallow-water model:

- :class:`DynamicalCore` — declares ``u``/``v``/``p``, owns
  ``csqr`` and the Rossby scaling, contributes the linear physics;
- :class:`~fridom.framework2.modules.FPlaneCoriolis` /
  :class:`~fridom.framework2.modules.BetaPlaneCoriolis` — the shared
  framework Coriolis modules (re-exported from ``fr.modules``), which
  declare ``f_coriolis`` and carry the rotation term;
- :class:`SadournyAdvection` — the energy/enstrophy-conserving
  nonlinear advection.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.modules import (
        BetaPlaneCoriolis,
        FPlaneCoriolis,
    )

    from .core import DynamicalCore
    from .sadourny import SadournyAdvection

# ================================================================
#  Setup lazy loading
# ================================================================
base = "fridom.shallowwater2.modules"

all_modules_by_origin: dict[str, list[str]] = {}

# The Coriolis family is the shared framework module library
# (fr.modules), re-exported here so sw.modules.FPlaneCoriolis keeps
# working.
all_imports_by_origin = {
    "fridom.framework2.modules": [
        "FPlaneCoriolis", "BetaPlaneCoriolis"],
    f"{base}.core": ["DynamicalCore"],
    f"{base}.sadourny": ["SadournyAdvection"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
