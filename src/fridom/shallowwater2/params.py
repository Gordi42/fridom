"""
Package-local canonical parameter names for shallowwater2.

Description
-----------
The shallow-water core owns one package-specific dotted parameter
name — ``shallowwater.csqr`` (the squared gravity-wave phase speed
:math:`c^2`) — built with the framework's :class:`ParamName` class so
it interoperates with ``fr.params.*`` in exactly the same way
(``params["shallowwater.csqr"]`` and
``update_parameters({sw.params.CSQR: ...})`` hit one mapping key).
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
# c^2 is the whole truth only when the depth is constant; a
# variable-depth core simply does not provide it (provides-implies-
# constancy, 02_rules), so the eigenmode/host read surface can rely
# on its presence as a constancy check.
CSQR: Final[ParamName] = ParamName(
    "shallowwater.csqr",
    units="m^2/s^2",
    hint="provided by the shallow-water core, e.g. "
         "sw.modules.DynamicalCore(csqr=...)")

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
    "CSQR",
    "DEPTH",
    "FROUDE",
    "GRAVITY",
    "ROSSBY",
]
