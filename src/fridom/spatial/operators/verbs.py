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
# Wave: boundary embed/as_profile verbs + n-ary scatter functions (§3)
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.operators.base import Dispatched

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.operators.base import FieldLike

#: default derivative verb: resolves ``("diff", factor)``
diff = Dispatched("diff")

#: staggering-interpolation verb: resolves ``("interpolate", factor)``
interpolate = Dispatched("interpolate")

#: weighted-integral verb: resolves ``("integrate", factor)``
#: (the default ``Integral`` rows land in Wave 3)
integrate = Dispatched("integrate")

#: running-integral verb: resolves ``("cumint", factor)`` — the
#: staggered partial integral along a bounded axis (stage H1). The
#: seeded ``CumulativeIntegral`` rows land the bottom-up face form;
#: the top-down / co-located variants are constructed explicitly.
cumint = Dispatched("cumint")

#: constant-physical-coordinate derivative verb on mapped grids:
#: resolves the ``"physical_diff"`` builder the grid seeds when a
#: ``CoordinateMapping`` couples coordinates (stage C1)
physical_diff = Dispatched("physical_diff")

#: boundary-embed verb: resolves ``("embed", trace_factor)`` — the
#: sparse-3D materialization of a 2D trace back into its parent row
#: (the side/depth ride the operand's ``TraceSpace``)
embed = Dispatched("embed")

#: trace-to-profile conversion verb: resolves ``("as_profile", trace)``
#: — the opt-in bridge into the Constant-z machinery
as_profile = Dispatched("as_profile")


def scatter_add(
    target: FieldLike, trace: FieldLike,
) -> FieldLike:
    """
    Row-scatter-ADD a 2D trace into ``target``'s boundary row.

    Description
    -----------
    An n-ary boundary verb (no single-field method form): builds and
    applies :class:`~fridom.spatial.operators.boundary.BoundaryScatterAdd`.
    The traced axis, side, and parent node set are read off ``trace``'s
    ``TraceSpace``; the operands share a grid and horizontal layout
    (the binary base enforces it).

    Parameters
    ----------
    target : FieldLike
        The full field the trace is added into (its space is kept).
    trace : FieldLike
        The 2D boundary-trace field.

    Returns
    -------
    FieldLike
        ``target`` with the trace added into its boundary row.
    """
    from fridom.spatial.operators.boundary import (  # noqa: PLC0415 — keep boundary off the verbs import path
        BoundaryScatterAdd,
    )
    return BoundaryScatterAdd()(target, trace)


def scatter_set(
    target: FieldLike, trace: FieldLike,
) -> FieldLike:
    """
    Row-scatter-SET (overwrite) a 2D trace into ``target``'s row.

    Description
    -----------
    The overwrite twin of :func:`scatter_add`, applying
    :class:`~fridom.spatial.operators.boundary.BoundaryScatterSet` — the
    boundary row is replaced by the trace value (the sanctioned
    replacement for raw ``.data`` boundary surgery, plan §6).

    Parameters
    ----------
    target : FieldLike
        The full field whose boundary row is overwritten.
    trace : FieldLike
        The 2D boundary-trace field.

    Returns
    -------
    FieldLike
        ``target`` with its boundary row overwritten by the trace.
    """
    from fridom.spatial.operators.boundary import (  # noqa: PLC0415 — keep boundary off the verbs import path
        BoundaryScatterSet,
    )
    return BoundaryScatterSet()(target, trace)
