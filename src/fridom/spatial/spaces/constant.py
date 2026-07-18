"""
``ConstantSpace``: the one-DOF broadcast factor.

Description
-----------
Owning class doc: ``design/specs/grid/classes/spaces.md``
(``ConstantSpace``). Replaces ``topo=False`` axes: broadcasting a
``ConstantSpace`` factor against a full factor is exact and
unambiguous — one of the two sanctioned exceptions to strict
algebra. Obtained as ``mesh.constant`` on any mesh type.
"""
# Wave 1: ConstantSpace
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.spaces.function_space import (
    FunctionSpace,
    space_key,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.decomposition.layout import Layout
    from fridom.spatial.scalars import Scalars, Variance


class ConstantSpace(FunctionSpace):

    """
    Constant along this mesh factor: a single broadcast DOF.

    Description
    -----------
    The codomain of ``integrate`` along a factor and the broadcast
    vehicle for per-factor symbols and coordinate fields across a
    product. Distinct from ``PointValues`` on a boundary mesh: a
    ``ConstantSpace`` is a geometry-less bulk reduction that *does*
    broadcast into the interior; a trace space must not. Interned
    one per (mesh, scalars); ``bc`` is the free structure.
    """

    _repr_label = "Constant"

    @property
    def is_constant(self) -> bool:
        """Always ``True``: this is the constant/broadcast factor."""
        return True

    @property
    def collapses_axis(self) -> bool:
        """Always ``True``: a constant collapses its axis to size 1."""
        return True

    @property
    def shape(self) -> tuple[int, ...]:
        """Always (1,): a single broadcast DOF."""
        return (1,)

    def _variant_key(self, scalars: Scalars,
                     layout: Layout | None,
                     variance: Variance | None) -> tuple:
        """Return the (scalars, layout, variance) variant's key.

        One entry per (mesh, scalars), plus the layout and variance
        when set; bc is always the free structure.
        """
        return space_key(type(self), scalars, layout=layout,
                         variance=variance)
