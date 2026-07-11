"""
Nodal spaces: the ``NodeSet`` marker and the nodal space family.

Description
-----------
Owning class doc: ``design/specs/grid/classes/spaces.md`` (``NodalSpace``
and friends); the ``NodeSet`` enum is specified with the static
markers in ``design/specs/grid/classes/meshes.md``. Node-set names are
topological; the mesh fixes physical placement (equispaced on an
``IntervalMesh``, Gauss-Lobatto on a ``ChebyshevMesh``).
"""
# Wave 0: NodeSet
# Wave 1: NodalSpace (ABC), Center, Left, Right, Outer, Inner, PointValues
from __future__ import annotations

from enum import Enum, auto
from typing import ClassVar

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.spaces.function_space import FunctionSpace


class NodeSet(Enum):

    """
    Topological node-set vocabulary of the nodal spaces.

    Description
    -----------
    The interning-key component of every nodal space factory (there
    is deliberately no string-keyed variant). ``CENTER`` / ``LEFT`` /
    ``RIGHT`` carry n DOFs, ``OUTER`` n + 1 (bounded meshes only),
    ``INNER`` n - 1 (bounded meshes only), ``POINTS`` the point
    values of a ``PointMesh``.
    """

    CENTER = auto()
    LEFT = auto()
    RIGHT = auto()
    OUTER = auto()
    INNER = auto()
    POINTS = auto()


# base DOF count per node set, as an offset from the cell count n
_COUNT_OFFSET: dict[NodeSet, int] = {
    NodeSet.CENTER: 0,
    NodeSet.LEFT: 0,
    NodeSet.RIGHT: 0,
    NodeSet.OUTER: 1,
    NodeSet.INNER: -1,
}

# whether the (left, right) boundary DOF is a member of the node set
# on a bounded mesh; only a Dirichlet constraint drops a DOF, and
# only when the constrained boundary DOF is in the set (spaces.md
# shape note, owner decision 2026-07-07)
_BOUNDARY_MEMBERSHIP: dict[NodeSet, tuple[bool, bool]] = {
    NodeSet.CENTER: (False, False),
    NodeSet.LEFT: (True, False),
    NodeSet.RIGHT: (False, True),
    NodeSet.OUTER: (True, True),
    NodeSet.INNER: (False, False),
}


class NodalSpace(FunctionSpace):

    """
    Point values at a node set of the mesh.

    Description
    -----------
    The finite-difference representation (xgcm vocabulary). The
    concrete class *is* the node-set tag for dispatch, with
    ``node_set`` as the enum mirror used in interning keys and
    generic code.
    """

    # set by each concrete subclass; the enum mirror of the class
    _node_set: ClassVar[NodeSet]

    @property
    def node_set(self) -> NodeSet:
        """The topological node-set tag."""
        return self._node_set

    @property
    def shape(self) -> tuple[int, ...]:
        """DOF count of the node set.

        The node-set base count, reduced by one per
        Dirichlet-constrained boundary component whose DOF is in the
        set: a Dirichlet condition eliminates a boundary *value*
        DOF. Neumann structure never reduces the shape — it
        constrains a derivative combination, not a nodal DOF (owner
        decision 2026-07-07; Neumann ``Outer`` keeps all n + 1
        nodes, the shape-honest DCT-I origin).
        """
        node_set = self._node_set
        base = self._mesh.n_cells + _COUNT_OFFSET[node_set]
        membership = _BOUNDARY_MEMBERSHIP[node_set]
        drop = sum(
            1 for kind, present in
            zip(self._bc.components, membership, strict=False)
            if present and kind is BC.DIRICHLET)
        return (base - drop,)


class Center(NodalSpace):

    """Nodal values at the n cell centers."""

    _node_set = NodeSet.CENTER


class Left(NodalSpace):

    """Nodal values at the n left cell edges."""

    _node_set = NodeSet.LEFT


class Right(NodalSpace):

    """Nodal values at the n right cell edges."""

    _node_set = NodeSet.RIGHT


class Outer(NodalSpace):

    """Nodal values at all n + 1 faces (bounded meshes)."""

    _node_set = NodeSet.OUTER


class Inner(NodalSpace):

    """Nodal values at the n - 1 interior faces (bounded meshes)."""

    _node_set = NodeSet.INNER


class PointValues(NodalSpace):

    """
    One nodal DOF per point of a ``PointMesh``.

    Description
    -----------
    The factor of trace product spaces, obtained as
    ``mesh.boundary.points``. It is a *located* boundary space
    (coordinates, generally > 1 DOF, possibly 0 DOFs on periodic
    factors) restricted to the boundary manifold; unlike a
    ``ConstantSpace`` it must not broadcast into the interior.
    """

    _node_set = NodeSet.POINTS

    @property
    def shape(self) -> tuple[int, ...]:
        """One DOF per point: ``(mesh.n_points,)``."""
        return (self._mesh.n_points,)


# concrete nodal class per (structured-1D) node set, used by the
# mesh factories; POINTS is minted by PointMesh.points instead
NODAL_CLASS: dict[NodeSet, type[NodalSpace]] = {
    NodeSet.CENTER: Center,
    NodeSet.LEFT: Left,
    NodeSet.RIGHT: Right,
    NodeSet.OUTER: Outer,
    NodeSet.INNER: Inner,
    NodeSet.POINTS: PointValues,
}
