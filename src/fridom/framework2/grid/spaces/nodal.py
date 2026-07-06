"""
Nodal spaces: the ``NodeSet`` marker and the nodal space family.

Description
-----------
Owning class doc: ``notes/framework2/classes/spaces.md`` (``NodalSpace``
and friends); the ``NodeSet`` enum is specified with the static
markers in ``notes/framework2/classes/meshes.md``. Node-set names are
topological; the mesh fixes physical placement.
"""
# Wave 0: NodeSet
# Wave 1: NodalSpace (ABC), Center, Left, Right, Outer, Inner, PointValues
from __future__ import annotations

from enum import Enum, auto


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
