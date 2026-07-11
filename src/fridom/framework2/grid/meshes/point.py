"""
``PointMesh``: a 0D mesh of located points.

Description
-----------
Owning class doc: ``notes/framework2/classes/meshes.md``. The
boundary of an interval: two located endpoints, or the empty
``PointMesh`` for periodic meshes — one type covers both, and
"empty boundary" needs no special case. Point positions are a
static tuple-of-tuples of floats: hashable by value, tiny, and
host-side.
"""
# Wave 1: PointMesh
from __future__ import annotations

from functools import cached_property

from fridom.framework2.grid.decomposition.traits import (
    HaloStrategy,
    MeshDecompositionTraits,
)
from fridom.framework2.grid.meshes.mesh import Mesh
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.function_space import (
    _FACTORY_TOKEN,
    FunctionSpace,
    space_key,
)
from fridom.framework2.grid.spaces.nodal import PointValues


class PointMesh(Mesh):

    """
    0D mesh: a finite tuple of located points.

    Description
    -----------
    Iteration-1 minimal: exists so ``mesh.boundary`` is total. A
    boundary ``PointMesh`` inherits its name from the parent factor,
    so trace spaces print as ``boundary(x)``.

    Parameters
    ----------
    positions : tuple[tuple[float, ...], ...]
        The points, given by their ambient coordinates; may be
        empty (the boundary of a periodic mesh).
    name : str
        The mandatory coordinate name.
    """

    def __init__(self, positions: tuple[tuple[float, ...], ...],
                 *, name: str) -> None:
        """Points given by their ambient coordinates; may be empty."""
        super().__init__((name,))
        normalized = tuple(
            tuple(float(coordinate) for coordinate in position)
            for position in positions)
        for position in normalized:
            if len(position) != len(normalized[0]):
                raise ValueError(
                    "all points need the same number of ambient "
                    f"coordinates, got {normalized}")
        self._positions: tuple[tuple[float, ...], ...] = normalized

    # ================================================================
    #  Geometry / topology
    # ================================================================
    @property
    def n_points(self) -> int:
        """Number of points (0 on a periodic mesh's boundary)."""
        return len(self._positions)

    @property
    def positions(self) -> tuple[tuple[float, ...], ...]:
        """Ambient coordinates of the points (static descriptor)."""
        return self._positions

    @property
    def dim(self) -> int:
        """Always 0."""
        return 0

    @cached_property
    def boundary(self) -> PointMesh:
        """The empty ``PointMesh`` (a 0D mesh has no boundary)."""
        name = self._names[0]
        mesh = PointMesh((), name=name)
        mesh._label = f"boundary({self._label})"
        return mesh

    @property
    def _n_boundary_components(self) -> int:
        """Always 0: a 0D mesh has no boundary."""
        return 0

    def decomposition_traits(
        self, space: FunctionSpace,
    ) -> MeshDecompositionTraits:
        """
        Return ``(LOCAL,)`` always: point meshes are replicated.

        Parameters
        ----------
        space : FunctionSpace
            A space interned on this mesh.

        Returns
        -------
        MeshDecompositionTraits
            The local-only traits record.
        """
        self._check_owned(space)
        return MeshDecompositionTraits((HaloStrategy.LOCAL,))

    # ================================================================
    #  Space factory
    # ================================================================
    @cached_property
    def points(self) -> PointValues:
        """One nodal DOF per point (the trace space factor).

        Named ``points``, not ``nodal``: the structured meshes'
        ``nodal`` is a method, and one spelling must not have two
        arities across the mesh family.
        """
        key = space_key(PointValues, self._free_bc, Scalars.REAL)
        return self._intern(
            key,
            lambda: PointValues(self, Scalars.REAL, self._free_bc,
                                _token=_FACTORY_TOKEN))

    def __repr__(self) -> str:
        """E.g. ``PointMesh(boundary(x): 2 points)``."""
        return f"PointMesh({self._label}: {self.n_points} points)"
