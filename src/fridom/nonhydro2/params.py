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

__all__ = [
    "CORIOLIS_BETA",
    "CORIOLIS_F0",
    "DSQR",
    "ROSSBY",
    "STRATIFICATION_N2",
]
