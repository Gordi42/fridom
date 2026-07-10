"""Nonhydrostatic modules (framework2 port).

Description
-----------
The concrete ``fr.Module`` subclasses of the nonhydrostatic model:
the dynamical core (declarations + pressure projection), the Coriolis
family, constant stratification, and centered advection.
"""
from typing import TYPE_CHECKING

from lazypimp import setup

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.modules import (
        BetaPlaneCoriolis,
        FPlaneCoriolis,
    )

    from .advection import CenteredAdvection
    from .core import DynamicalCore
    from .pressure import SpectralPressureSolver
    from .smagorinsky_lilly import SmagorinskyLilly
    from .stratification import (
        ConstantStratification,
        MeridionalStratification,
    )

base = "fridom.nonhydro2.modules"

all_modules_by_origin: dict[str, list[str]] = {}

# The Coriolis family is the shared framework module library
# (fr.modules), re-exported here so nh.modules.FPlaneCoriolis keeps
# working after the wave-6 consolidation.
all_imports_by_origin = {
    "fridom.framework2.modules": [
        "FPlaneCoriolis", "BetaPlaneCoriolis"],
    f"{base}.core": ["DynamicalCore"],
    f"{base}.stratification": [
        "ConstantStratification", "MeridionalStratification"],
    f"{base}.advection": ["CenteredAdvection"],
    f"{base}.pressure": ["SpectralPressureSolver"],
    f"{base}.smagorinsky_lilly": ["SmagorinskyLilly"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
