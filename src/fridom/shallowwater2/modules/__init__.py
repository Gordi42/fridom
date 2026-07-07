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
from fridom.framework2.modules.coriolis import (
    BetaPlaneCoriolis,
    FPlaneCoriolis,
)
from fridom.shallowwater2.modules.core import DynamicalCore
from fridom.shallowwater2.modules.sadourny import SadournyAdvection

__all__ = [
    "BetaPlaneCoriolis",
    "DynamicalCore",
    "FPlaneCoriolis",
    "SadournyAdvection",
]
