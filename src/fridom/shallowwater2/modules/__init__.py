"""
Shallow-water tendency modules (framework2 port).

Description
-----------
The module library for the shallow-water model:

- :class:`ShallowWaterCore` — declares ``u``/``v``/``h``, owns
  ``csqr`` and the Rossby scaling, contributes the linear physics;
- :class:`FPlaneCoriolis` / :class:`BetaPlaneCoriolis` — the Coriolis
  field providers (flagged for consolidation into a shared framework
  module library);
- :class:`SadournyAdvection` — the energy/enstrophy-conserving
  nonlinear advection.
"""
from fridom.shallowwater2.modules.core import ShallowWaterCore
from fridom.shallowwater2.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
)
from fridom.shallowwater2.modules.sadourny import SadournyAdvection

__all__ = [
    "BetaPlaneCoriolis",
    "FPlaneCoriolis",
    "SadournyAdvection",
    "ShallowWaterCore",
]
