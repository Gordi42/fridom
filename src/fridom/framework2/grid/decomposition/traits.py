"""
Per-mesh decomposition traits.

Description
-----------
Owning class doc: ``notes/framework2/classes/decomposition.md``
(``HaloStrategy`` / ``MeshDecompositionTraits``). This module is the
single definition of both types; the mesh cluster imports them and
declares traits through the per-space seam
``Mesh.decomposition_traits(space)``.
"""
# Wave 1: HaloStrategy, MeshDecompositionTraits
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


class HaloStrategy(Enum):

    """
    How a factor space can be distributed across devices.

    Description
    -----------
    ``GHOST``: shard with ghost-cell halo exchange. ``TRANSPOSE``:
    shard via transpose-based transforms (a contiguous dimension is
    reached at operator time, not by keeping the factor unsharded).
    ``LOCAL``: keep this factor on-device. ``GRAPH``: graph
    partition (designed-for, unstructured meshes).
    """

    GHOST = auto()
    TRANSPOSE = auto()
    LOCAL = auto()
    GRAPH = auto()


@dataclass(frozen=True)
class MeshDecompositionTraits:

    """
    Shardability declaration for one factor space of a mesh.

    Description
    -----------
    Returned by ``Mesh.decomposition_traits(space)``, per space (the
    nodal and coefficient spaces of one mesh differ). ``strategies``
    is ordered by preference; shardability is derivable as
    ``strategies != (HaloStrategy.LOCAL,)`` — there is no separate
    flag. ``min_local_size`` guards against shards smaller than a
    halo, as a negotiation constraint instead of a runtime error.

    Parameters
    ----------
    strategies : tuple[HaloStrategy, ...]
        Admissible distribution strategies, ordered by preference.
    min_local_size : int, optional
        Smallest admissible local shard extent (default: 1).
    """

    strategies: tuple[HaloStrategy, ...]
    min_local_size: int = 1
