"""
Shallow-water tendency modules.

Description
-----------
The module library for the shallow-water model:

- :class:`DynamicalCore` — declares ``u``/``v``/``p``, owns
  ``csqr`` and the Rossby scaling, contributes the linear physics;
- :class:`~fridom.model.modules.FPlaneCoriolis` /
  :class:`~fridom.model.modules.BetaPlaneCoriolis` /
  :class:`~fridom.model.modules.RotationCoriolis` (chart grids:
  ``f = 2 Omega . n_hat``) — the shared framework Coriolis family
  (re-exported from ``fr.modules``), which declare ``f_coriolis``
  and carry the rotation term. Rotation is opt-in: a model
  assembled without one of these simply does not rotate;
- :class:`SadournyAdvection` — the energy/enstrophy-conserving
  nonlinear advection;
- :class:`CoriolisEnergyCorrection` — the optional term that makes the
  rotation conserve the **thickness-weighted** energy exactly (route
  A: assembled next to a linear Coriolis module; the linear operator
  ``L`` is untouched), and
  :class:`NonlinearFPlaneCoriolis` /
  :class:`NonlinearBetaPlaneCoriolis` /
  :class:`NonlinearRotationCoriolis` — the same conserving rotation
  carried whole, *instead* of a linear Coriolis module (route B:
  simpler and cheaper, but ``L`` then has no rotation at all, so the
  eigenmode / projection / balance machinery refuses the model). See
  ``sw.modules.coriolis``.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

# ================================================================
#  Disable lazy loading for type checking
# ================================================================
if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.modules import (
        BetaPlaneCoriolis,
        FPlaneCoriolis,
        RotationCoriolis,
    )

    from .core import DynamicalCore
    from .coriolis import (
        CoriolisEnergyCorrection,
        NonlinearBetaPlaneCoriolis,
        NonlinearFPlaneCoriolis,
        NonlinearRotationCoriolis,
        carries_linear_rotation,
        conserving_rotation,
    )
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
    "fridom.model.modules": [
        "FPlaneCoriolis", "BetaPlaneCoriolis", "RotationCoriolis"],
    f"{base}.core": ["DynamicalCore"],
    f"{base}.coriolis": [
        "CoriolisEnergyCorrection", "NonlinearFPlaneCoriolis",
        "NonlinearBetaPlaneCoriolis", "NonlinearRotationCoriolis",
        "conserving_rotation", "carries_linear_rotation"],
    f"{base}.sadourny": ["SadournyAdvection"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
