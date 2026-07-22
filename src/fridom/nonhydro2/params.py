"""Canonical parameter names owned by the nonhydro package.

Description
-----------
Package-specific ``ParamName`` constants (D2.1): the aspect ratio
``nonhydro.aspect_ratio`` lives on the dynamical core (``nh.Core``).
Coriolis and stratification names are framework-canonical
(``fr.params.CORIOLIS_F0`` / ``fr.params.STRATIFICATION_N2`` /
``fr.params.STRATIFICATION_FROUDE``) and are re-exported here for
convenience.

The retired ``nonhydro.dsqr`` name (the squared aspect ratio the old
``DynamicalCore`` provided) raises a taught ``AttributeError``: the
core now provides the aspect ratio delta itself
(``nh.Core(aspect_ratio=...)``) and every consumer squares it at the
use site (``delta**2`` -- the nondimensionalization refactor).
"""
from __future__ import annotations

from fridom.model.params import (
    CORIOLIS_BETA,
    CORIOLIS_F0,
    CORIOLIS_ROSSBY,
    SCALING_NONLINEARITY,
    STRATIFICATION_FROUDE,
    STRATIFICATION_N2,
    ParamName,
)

# ================================================================
#  Package-owned names
# ================================================================
ASPECT_RATIO: ParamName = ParamName(
    "nonhydro.aspect_ratio", units="1",
    hint="provided by the nonhydrostatic core, e.g. "
         "nh.Core(aspect_ratio=...)")

# ``scaling.nonlinearity`` is framework-canonical; re-exported so
# recipes can spell nh.params.ROSSBY (it aliases the epsilon name).
ROSSBY: ParamName = SCALING_NONLINEARITY

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


def __getattr__(name: str) -> ParamName:
    """Teach the retired ``DSQR`` name its replacement."""
    if name == "DSQR":
        raise AttributeError(
            "nh.params.DSQR ('nonhydro.dsqr') is retired: the core "
            "provides the aspect ratio itself — read "
            "nh.params.ASPECT_RATIO ('nonhydro.aspect_ratio', "
            "provided by nh.Core(aspect_ratio=...)) and square it "
            "at the use site (dsqr = delta**2)")
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ASPECT_RATIO",
    "CORIOLIS_BETA",
    "CORIOLIS_F0",
    "CORIOLIS_ROSSBY",
    "ROSSBY",
    "SMAG_BACKGROUND_KAPPA",
    "SMAG_BACKGROUND_NU",
    "SMAG_BUOYANCY_MULTIPLIER",
    "SMAG_CS",
    "SMAG_PRANDTL",
    "STRATIFICATION_FROUDE",
    "STRATIFICATION_N2",
]
