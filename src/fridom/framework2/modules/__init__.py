"""
The framework's shared tendency-module library.

Description
-----------
Reusable, model-agnostic ``fr.Module`` subclasses shared across the
model ports (D2.1 module-library sharing). Wave 6 seeds it with the
Coriolis family (``fr.modules.FPlaneCoriolis`` /
``fr.modules.BetaPlaneCoriolis``); both the nonhydrostatic and
shallow-water packages import from here instead of carrying a copy.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from .coriolis import BetaPlaneCoriolis, FPlaneCoriolis

base = "fridom.framework2.modules"

all_modules_by_origin: dict[str, list[str]] = {}

all_imports_by_origin = {
    f"{base}.coriolis": ["FPlaneCoriolis", "BetaPlaneCoriolis"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
