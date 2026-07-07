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
    from .advection import CenteredAdvection
    from .core import DynamicalCore
    from .coriolis import BetaPlaneCoriolis, FPlaneCoriolis
    from .pressure import SpectralPressureSolver
    from .stratification import ConstantStratification

base = "fridom.nonhydro2.modules"

all_modules_by_origin: dict[str, list[str]] = {}

all_imports_by_origin = {
    f"{base}.core": ["DynamicalCore"],
    f"{base}.coriolis": ["FPlaneCoriolis", "BetaPlaneCoriolis"],
    f"{base}.stratification": ["ConstantStratification"],
    f"{base}.advection": ["CenteredAdvection"],
    f"{base}.pressure": ["SpectralPressureSolver"],
}

setup(__name__, all_modules_by_origin, all_imports_by_origin)
