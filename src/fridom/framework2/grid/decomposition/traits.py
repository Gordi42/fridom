"""
Per-mesh decomposition traits.

Description
-----------
Owning class doc: ``design/specs/grid/classes/decomposition.md``. This
module owns the single definition of ``HaloStrategy`` and
``MeshDecompositionTraits``; the meshes cluster imports from here for
the per-space declaring seam ``Mesh.decomposition_traits(space)``.
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
    Declared per factor space (not per mesh: nodal and coefficient
    spaces of one mesh differ). ``GHOST``/``TRANSPOSE``/``LOCAL`` are
    iteration 1; ``GRAPH`` is the designed-for unstructured tag.
    """

    GHOST = auto()      # shard with ghost-cell halo exchange
    TRANSPOSE = auto()  # shard via transpose-based transforms
    LOCAL = auto()      # keep this factor on-device
    GRAPH = auto()      # graph partition (designed-for)


@dataclass(frozen=True)
class MeshDecompositionTraits:

    """
    Shardability declaration for one factor space of a mesh.

    Description
    -----------
    A static, hashable descriptor collected by negotiation together
    with the operator demands. ``strategies`` is ordered by
    preference; shardability is derivable as
    ``traits.strategies != (HaloStrategy.LOCAL,)`` — there is no
    separate shardable flag. ``min_local_size`` guards against shards
    smaller than a halo, as a negotiation constraint instead of a
    runtime error.

    Parameters
    ----------
    strategies : tuple[HaloStrategy, ...]
        The supported distribution strategies, ordered by preference;
        must be non-empty (any sequence is normalized to a tuple).
    min_local_size : int, optional
        The smallest admissible per-device extent along this factor
        (default: 1).
    """

    strategies: tuple[HaloStrategy, ...]
    min_local_size: int = 1

    def __post_init__(self) -> None:
        """Normalize `strategies` to a tuple and validate."""
        object.__setattr__(self, "strategies", tuple(self.strategies))
        if not self.strategies:
            raise ValueError(
                "strategies must name at least one HaloStrategy")
        if self.min_local_size < 1:
            raise ValueError(
                f"min_local_size must be >= 1, "
                f"got {self.min_local_size}")
