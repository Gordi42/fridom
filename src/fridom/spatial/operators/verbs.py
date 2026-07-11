"""
The seeded operator verbs: ``diff``, ``interpolate``, ``integrate``.

Description
-----------
Owning decision: ``design/specs/grid/classes/operator_algebra_merge.md``
(D3b). The standard single-kind verbs are module-level ``Dispatched``
singletons on ``fr.operators`` — the discoverable surface the field
forwarders route through (``f.diff("x")`` is
``fr.operators.diff["x"](f)``). The ``Dispatched(kind)`` constructor
stays public as the extension escape-hatch for custom kinds; an
unknown kind is a clean ``DispatchError`` at resolution.
"""
# Wave 2: seeded verbs (D3b)
from __future__ import annotations

from fridom.spatial.operators.base import Dispatched

#: default derivative verb: resolves ``("diff", factor)``
diff = Dispatched("diff")

#: staggering-interpolation verb: resolves ``("interpolate", factor)``
interpolate = Dispatched("interpolate")

#: weighted-integral verb: resolves ``("integrate", factor)``
#: (the default ``Integral`` rows land in Wave 3)
integrate = Dispatched("integrate")
