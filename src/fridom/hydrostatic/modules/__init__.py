"""Hydrostatic modules.

Description
-----------
The concrete ``fr.Module`` subclasses of the hydrostatic model: the
dynamical core (declarations + the two DIAGNOSE stages + the pressure
gradient term), constant stratification, and the explicit free
surface. The Coriolis family is the shared framework module library
(``fr.model.modules``), re-exported here so
``hy.modules.FPlaneCoriolis`` keeps working.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.closures import VerticalMixing
    from fridom.model.modules import (
        BetaPlaneCoriolis,
        FPlaneCoriolis,
        RotationCoriolis,
    )

    from .core import HydrostaticCore
    from .free_surface import ExplicitFreeSurface, ImplicitFreeSurface
    from .stratification import ConstantStratification

base = "fridom.hydrostatic.modules"

all_modules_by_origin: dict[str, list[str]] = {}

# The Coriolis family (fr.model.modules) and the implicit vertical
# mixing closure (fr.model.closures) are shared framework module
# libraries, re-exported here so hy.modules.FPlaneCoriolis /
# hy.modules.VerticalMixing keep working.
all_imports_by_origin = {
    "fridom.model.modules": [
        "FPlaneCoriolis", "BetaPlaneCoriolis", "RotationCoriolis"],
    "fridom.model.closures": ["VerticalMixing"],
    f"{base}.core": ["HydrostaticCore"],
    f"{base}.stratification": ["ConstantStratification"],
    f"{base}.free_surface": [
        "ExplicitFreeSurface", "ImplicitFreeSurface"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
