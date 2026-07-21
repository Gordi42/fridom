"""
Package-local canonical parameter names for shallowwater2.

Description
-----------
The shallow-water core owns the package-specific dotted parameter
names — ``shallowwater.gravity`` / ``shallowwater.depth``
(dimensional) and ``shallowwater.froude`` (nondimensional) — built
with the framework's :class:`ParamName` class so they interoperate
with ``fr.params.*`` in exactly the same way (the string spelling and
the constant hit one mapping key).
The framework-owned names (``coriolis.f0`` / ``coriolis.beta`` /
``scaling.nonlinearity`` / ``stepper.dt``) live in ``fr.params`` and are
re-used verbatim (module-library sharing, D2.1).
"""
from __future__ import annotations

from typing import Final

from fridom.model.params import (
    CORIOLIS_F0,
    SCALING_NONLINEARITY,
    ParamName,
)

# ================================================================
#  Package-owned names
# ================================================================
# The retired ``shallowwater.csqr`` provide is replaced by the
# variant primitives below (nondimensionalization plan): consumers
# assemble the effective c^2 = g*D (dimensional) or
# (epsilon/Fr)^2 * depth-ratio (nondimensional) themselves.

# The dimensional core primitives (provides-implies-constancy: a
# Ramped or profile-valued leaf is still a provide; a variable-depth
# NONDIM D-tilde profile provides no constant DEPTH).
GRAVITY: Final[ParamName] = ParamName(
    "shallowwater.gravity",
    units="m/s^2",
    hint="provided by the dimensional shallow-water core, e.g. "
         "sw.Core(gravity=..., depth=...)")

DEPTH: Final[ParamName] = ParamName(
    "shallowwater.depth",
    units="m",
    hint="provided by the dimensional shallow-water core, e.g. "
         "sw.Core(gravity=..., depth=...)")

FROUDE: Final[ParamName] = ParamName(
    "shallowwater.froude",
    units="n/a",
    hint="provided by the nondimensional shallow-water core, e.g. "
         "sw.Core(froude_number=...)")

# ``scaling.nonlinearity`` is framework-canonical but owned (provided) by
# sw.DynamicalCore; re-exported so recipes can spell sw.params.ROSSBY.
ROSSBY: Final[ParamName] = SCALING_NONLINEARITY

__all__ = [
    "CORIOLIS_F0",
    "DEPTH",
    "FROUDE",
    "GRAVITY",
    "ROSSBY",
]
