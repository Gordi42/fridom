"""
The framework's shared tendency-module library.

Description
-----------
Reusable, model-agnostic ``fr.Module`` subclasses shared across the
model ports (D2.1 module-library sharing). Wave 6 seeds it with the
Coriolis family (``fr.modules.FPlaneCoriolis`` /
``fr.modules.BetaPlaneCoriolis``); both the nonhydrostatic and
shallow-water packages import from here instead of carrying a copy.
The forcing port adds the generic ``fr.modules.Relaxation``.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from .coriolis import (
        BetaPlaneCoriolis,
        FPlaneCoriolis,
        SphericalCoriolis,
    )
    from .relaxation import Relaxation

base = "fridom.model.modules"

all_modules_by_origin: dict[str, list[str]] = {}

all_imports_by_origin = {
    f"{base}.coriolis": ["FPlaneCoriolis", "BetaPlaneCoriolis",
                          "SphericalCoriolis"],
    f"{base}.relaxation": ["Relaxation"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
