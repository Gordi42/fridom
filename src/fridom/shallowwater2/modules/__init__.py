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
  ``f = 2 Omega . n_hat``) / :class:`~fridom.model.modules.
  SphericalCoriolis` / :class:`~fridom.model.modules.NoCoriolis` —
  the shared framework Coriolis family (re-exported from
  ``fr.modules``), which declare ``f_coriolis`` and carry the
  rotation term;
- :class:`SadournyAdvection` — the energy/enstrophy-conserving
  nonlinear advection.
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
        NoCoriolis,
        RotationCoriolis,
        SphericalCoriolis,
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
    "fridom.model.modules": [
        "FPlaneCoriolis", "BetaPlaneCoriolis", "SphericalCoriolis",
        "RotationCoriolis", "NoCoriolis"],
    f"{base}.core": ["DynamicalCore"],
    f"{base}.sadourny": ["SadournyAdvection"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
