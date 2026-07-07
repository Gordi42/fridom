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
``scaling.rossby`` / ``stepper.dt``) live in ``fr.params`` and are
re-used verbatim (module-library sharing, D2.1).
"""
from __future__ import annotations

from typing import Final

from fridom.framework2.model.params import ParamName

# c^2 is the whole truth only when the depth is constant; a
# variable-depth core simply does not provide it (provides-implies-
# constancy, 02_rules), so the eigenmode/host read surface can rely
# on its presence as a constancy check.
CSQR: Final[ParamName] = ParamName(
    "shallowwater.csqr",
    units="m^2/s^2",
    hint="provided by the shallow-water core, e.g. "
         "sw.modules.DynamicalCore(csqr=...)")
