"""Canonical parameter names owned by the hydrostatic package.

Description
-----------
Package-specific ``ParamName`` constants (D2.1): the squared
barotropic phase speed ``hydrostatic.csqr`` (:math:`c^2 = g H`) and
the Rossby number ``scaling.rossby`` both live on the hydrostatic
core. Coriolis and stratification names are framework-canonical
(``fr.params.CORIOLIS_F0`` / ``fr.params.STRATIFICATION_N2``) and are
re-exported here for convenience.
"""
from __future__ import annotations

from typing import Final

from fridom.model.params import (
    CORIOLIS_BETA,
    CORIOLIS_F0,
    SCALING_ROSSBY,
    STRATIFICATION_N2,
    ParamName,
)

# ================================================================
#  Package-owned names
# ================================================================
CSQR: Final[ParamName] = ParamName(
    "hydrostatic.csqr", units="m^2/s^2",
    hint="provided by the hydrostatic core, e.g. "
         "hy.HydrostaticCore(csqr=...)")

SHEAR: Final[ParamName] = ParamName(
    "hydrostatic.shear", units="1/s",
    hint="provided by the thermal-wind background, e.g. "
         "hy.ThermalWindBackground(shear=...)")

# ``scaling.rossby`` is framework-canonical but owned (provided) by
# hy.HydrostaticCore; re-exported so recipes can spell hy.params.ROSSBY.
ROSSBY: Final[ParamName] = SCALING_ROSSBY

__all__ = [
    "CORIOLIS_BETA",
    "CORIOLIS_F0",
    "CSQR",
    "ROSSBY",
    "SHEAR",
    "STRATIFICATION_N2",
]
