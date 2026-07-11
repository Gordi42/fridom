"""Canonical parameter names owned by the nonhydro package.

Description
-----------
Package-specific ``ParamName`` constants (D2.1): the nondimensional
squared aspect ratio ``nonhydro.dsqr`` and the Rossby number
``scaling.rossby`` both live on the dynamical-core module. Coriolis
and stratification names are framework-canonical
(``fr.params.CORIOLIS_F0`` / ``fr.params.STRATIFICATION_N2``) and are
re-exported here for convenience.
"""
from __future__ import annotations

from fridom.framework2.model.params import (
    CORIOLIS_BETA,
    CORIOLIS_F0,
    SCALING_ROSSBY,
    STRATIFICATION_N2,
    ParamName,
)

# ================================================================
#  Package-owned names
# ================================================================
DSQR: ParamName = ParamName(
    "nonhydro.dsqr", units="1",
    hint="declared by nh.DynamicalCore(dsqr=...)")

# ``scaling.rossby`` is framework-canonical but owned (provided) by
# nh.DynamicalCore; re-exported so recipes can spell nh.params.ROSSBY.
ROSSBY: ParamName = SCALING_ROSSBY

# ----------------------------------------------------------------
#  Smagorinsky-Lilly closure (nh.SmagorinskyLilly)
# ----------------------------------------------------------------
SMAG_CS: ParamName = ParamName(
    "smagorinsky.cs", units="1",
    hint="provided by nh.SmagorinskyLilly(smagorinsky_constant=...)")

SMAG_PRANDTL: ParamName = ParamName(
    "smagorinsky.prandtl", units="1",
    hint="provided by "
         "nh.SmagorinskyLilly(turbulent_prandtl_number=...)")

SMAG_BACKGROUND_NU: ParamName = ParamName(
    "smagorinsky.background_nu", units="m^2/s",
    hint="provided by nh.SmagorinskyLilly(background_viscosity=...)")

SMAG_BACKGROUND_KAPPA: ParamName = ParamName(
    "smagorinsky.background_kappa", units="m^2/s",
    hint="provided by "
         "nh.SmagorinskyLilly(background_diffusivity=...)")

SMAG_BUOYANCY_MULTIPLIER: ParamName = ParamName(
    "smagorinsky.buoyancy_multiplier", units="1",
    hint="provided by nh.SmagorinskyLilly(buoyancy_multiplier=...)")

__all__ = [
    "CORIOLIS_BETA",
    "CORIOLIS_F0",
    "DSQR",
    "ROSSBY",
    "SMAG_BACKGROUND_KAPPA",
    "SMAG_BACKGROUND_NU",
    "SMAG_BUOYANCY_MULTIPLIER",
    "SMAG_CS",
    "SMAG_PRANDTL",
    "STRATIFICATION_N2",
]
