"""Canonical parameter names owned by the hydrostatic package.

Description
-----------
Package-specific ``ParamName`` constants (D2.1): the gravitational
acceleration ``hydrostatic.gravity`` (the physical constant,
centralized on the core — ``hy.Core(gravity=...)``) and the external
(free-surface) Froude number ``hydrostatic.froude`` (provided by a
nondimensional free-surface module). Coriolis and stratification
names are framework-canonical (``fr.params.CORIOLIS_F0`` /
``fr.params.STRATIFICATION_N2`` / ``fr.params.STRATIFICATION_FROUDE``)
and are re-exported here for convenience.

The retired ``hydrostatic.csqr`` name (the squared barotropic phase
speed the old core provided) raises
a taught ``AttributeError``: gravity is the physical constant and
every column depth in the step path is genuine geometry, so there is
no reference-depth parameter left to fold into a ``c^2``.
"""
from __future__ import annotations

from typing import Final

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
GRAVITY: Final[ParamName] = ParamName(
    "hydrostatic.gravity", units="m/s^2",
    hint="provided by the dimensional hydrostatic core, e.g. "
         "hy.Core(gravity=...)")

FROUDE: Final[ParamName] = ParamName(
    "hydrostatic.froude", units="1",
    hint="provided by a nondimensional free-surface module, e.g. "
         "hy.ExplicitFreeSurface(froude_number=...)")

SHEAR: Final[ParamName] = ParamName(
    "hydrostatic.shear", units="1/s",
    hint="provided by the thermal-wind background, e.g. "
         "hy.ThermalWindBackground(shear=...)")

# ``scaling.nonlinearity`` is framework-canonical; re-exported so
# recipes can spell hy.params.ROSSBY (it aliases the epsilon name).
ROSSBY: Final[ParamName] = SCALING_NONLINEARITY


def __getattr__(name: str) -> ParamName:
    """Teach the retired ``CSQR`` name its replacement."""
    if name == "CSQR":
        raise AttributeError(
            "hy.params.CSQR ('hydrostatic.csqr') is retired: the "
            "physical constant is gravity — read hy.params.GRAVITY "
            "('hydrostatic.gravity', provided by "
            "hy.Core(gravity=...)); where an analytic path genuinely "
            "needs a barotropic c^2, form g * H with the physical "
            "vertical extent (the nondimensional free surface "
            "provides hy.params.FROUDE instead)")
    raise AttributeError(
        f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CORIOLIS_BETA",
    "CORIOLIS_F0",
    "CORIOLIS_ROSSBY",
    "FROUDE",
    "GRAVITY",
    "ROSSBY",
    "SHEAR",
    "STRATIFICATION_FROUDE",
    "STRATIFICATION_N2",
]
