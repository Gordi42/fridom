"""
The ``StructuredMesh1D`` ABC: shared 1D cell structure.

Description
-----------
Owning class doc: ``design/specs/grid/classes/meshes.md``. Owns the
full space-factory surface of interval-like meshes: node-set,
average, and coefficient space factories. Node-set names are
topological; the concrete mesh fixes physical placement.
"""
# Wave 1: StructuredMesh1D (ABC)
from __future__ import annotations

from abc import abstractmethod
from fractions import Fraction
from functools import cached_property
from typing import TYPE_CHECKING, Self

from fridom.spatial.bc import BC, BCStructure
from fridom.spatial.decomposition.traits import (
    HaloStrategy,
    MeshDecompositionTraits,
)
from fridom.spatial.meshes.mesh import Mesh
from fridom.spatial.meshes.point import PointMesh
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import (
    AverageSpace,
    CellAvg,
    FaceAvg,
)
from fridom.spatial.spaces.coefficient import (
    CoefficientSpace,
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.function_space import (
    _FACTORY_TOKEN,
    FunctionSpace,
    space_key,
)
from fridom.spatial.spaces.nodal import (
    NODAL_CLASS,
    Center,
    Inner,
    Left,
    NodalSpace,
    NodeSet,
    Outer,
    Right,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.spaces.galerkin import GalerkinSpace


class StructuredMesh1D(Mesh):

    """
    1D mesh with n cells and the full space-factory surface.

    Description
    -----------
    The constructor argument is named ``shape`` for continuity with
    the sketches, but it is the **cell count**, exposed as
    ``n_cells``: the mesh has no DOF shape (shapes live on spaces).
    Factory validity is checked at call time: ``outer``/``inner``
    raise on periodic meshes, ``fourier`` on bounded ones,
    ``sine``/``cosine`` on periodic ones.

    Parameters
    ----------
    shape : int
        The number of primal cells n.
    extent : tuple[float, float]
        The physical interval (x_min, x_max).
    periodic : bool
        Whether the interval is periodic (mesh topology, not a BC).
    name : str
        The mandatory coordinate name.
    """

    def __init__(self, shape: int, extent: tuple[float, float],
                 periodic: bool, *, name: str) -> None:
        """Store cell count, extent, periodicity, and name."""
        super().__init__((name,))
        if isinstance(shape, bool) or not isinstance(shape, int):
            raise TypeError(
                f"shape must be an integer cell count, got {shape!r}")
        if shape < 1:
            raise ValueError(
                f"shape must be a positive cell count, got {shape}")
        try:
            x_min, x_max = extent
        except ValueError:
            raise ValueError(
                "extent must be an increasing (x_min, x_max) pair, "
                f"got {tuple(extent)}") from None
        if not x_max > x_min:
            raise ValueError(
                "extent must be an increasing (x_min, x_max) pair, "
                f"got {(x_min, x_max)}")
        self._n_cells: int = shape
        self._extent: tuple[float, float] = (x_min, x_max)
        self._periodic: bool = bool(periodic)
        self._refined_from: Self | None = None
        self._refined_cache: dict[Fraction, Self] = {}

    # ================================================================
    #  Geometry descriptors
    # ================================================================
    @property
    def n_cells(self) -> int:
        """Number of primal cells n.

        The only count the mesh knows; DOF counts live on spaces.
        """
        return self._n_cells

    @property
    def extent(self) -> tuple[float, float]:
        """Physical interval (x_min, x_max)."""
        return self._extent

    @property
    def periodic(self) -> bool:
        """Whether the interval is periodic (topology, not a BC)."""
        return self._periodic

    # ================================================================
    #  Mesh interface
    # ================================================================
    @property
    def dim(self) -> int:
        """Always 1."""
        return 1

    @cached_property
    def boundary(self) -> PointMesh:
        """Two located endpoints (empty ``PointMesh`` if periodic).

        Stable: repeated access returns the identical object.
        """
        name = self._names[0]
        positions: tuple[tuple[float, ...], ...]
        if self._periodic:
            positions = ()
        else:
            positions = ((self._extent[0],), (self._extent[1],))
        mesh = PointMesh(positions, name=name)
        mesh._label = f"boundary({name})"  # noqa: SLF001 — repr seam
        return mesh

    @property
    def _n_boundary_components(self) -> int:
        """0 when periodic, 2 (left, right) when bounded."""
        return 0 if self._periodic else 2

    def decomposition_traits(
        self, space: FunctionSpace,
    ) -> MeshDecompositionTraits:
        """
        Return the sharding/halo traits of one of this mesh's spaces.

        Description
        -----------
        GHOST-first for nodal/average spaces; (LOCAL, TRANSPOSE)
        for coefficient spaces (local preferred); (LOCAL,) for
        ``ConstantSpace``. ``ChebyshevMesh`` overrides
        (transpose-first).

        Parameters
        ----------
        space : FunctionSpace
            A space interned on this mesh.

        Returns
        -------
        MeshDecompositionTraits
            The preference-ordered strategies record.
        """
        self._check_owned(space)
        if isinstance(space, ConstantSpace):
            return MeshDecompositionTraits((HaloStrategy.LOCAL,))
        if isinstance(space, CoefficientSpace):
            return MeshDecompositionTraits(
                (HaloStrategy.LOCAL, HaloStrategy.TRANSPOSE))
        return MeshDecompositionTraits(
            (HaloStrategy.GHOST, HaloStrategy.TRANSPOSE))

    # ================================================================
    #  Refinement (padded-transform target, section 3.12)
    # ================================================================
    def refined(self, factor: Fraction | int) -> Self:
        """
        Return a same-type mesh with the cell count scaled.

        Description
        -----------
        The result is a new, distinct, first-class mesh.
        The finer-mesh factory behind padded transforms. Results are
        memoized per (mesh, factor), so repeated requests return the
        identical finer mesh and its spaces stay
        identity-comparable. Iteration 1 on ``IntervalMesh``;
        designed-for on the other 1D meshes.

        Parameters
        ----------
        factor : Fraction | int
            The exact cell-count ratio (e.g. ``Fraction(3, 2)``);
            ``n_cells * factor`` must be integral.

        Returns
        -------
        Self
            The memoized finer (or coarser) mesh, with
            ``refined_from`` set to this mesh.
        """
        if isinstance(factor, bool) or not isinstance(
                factor, int | Fraction):
            raise TypeError(
                "refinement factors are exact Fractions (or ints), "
                f"got {factor!r}")
        factor = Fraction(factor)
        if factor <= 0:
            raise ValueError(
                f"refinement factors must be positive, got {factor}")
        cells = self._n_cells * factor
        if cells.denominator != 1:
            raise ValueError(
                f"n_cells * factor must be integral, got "
                f"{self._n_cells} * {factor} = {cells}")
        mesh = self._refined_cache.get(factor)
        if mesh is None:
            mesh = self._make_refined(int(cells))
            mesh._refined_from = self  # noqa: SLF001 — same family
            self._refined_cache[factor] = mesh
        return mesh

    @property
    def refined_from(self) -> Self | None:
        """The refinement parent (None on unrefined meshes)."""
        return self._refined_from

    @abstractmethod
    def _make_refined(self, n_cells: int) -> Self:
        """Construct (not memoize) the scaled same-type mesh.

        Raises ``NotImplementedError`` where refinement is
        designed-for only.
        """
        ...

    # ================================================================
    #  Nodal factories (xgcm vocabulary)
    # ================================================================
    @cached_property
    def center(self) -> Center:
        """Nodal cell centers (n DOFs); BC-free, real."""
        return self.nodal(NodeSet.CENTER)

    @cached_property
    def left(self) -> Left:
        """Nodal left cell edges (n DOFs)."""
        return self.nodal(NodeSet.LEFT)

    @cached_property
    def right(self) -> Right:
        """Nodal right cell edges (n DOFs)."""
        return self.nodal(NodeSet.RIGHT)

    @cached_property
    def outer(self) -> Outer:
        """All faces (n + 1 DOFs); bounded meshes only."""
        return self.nodal(NodeSet.OUTER)

    @cached_property
    def inner(self) -> Inner:
        """Interior faces (n - 1 DOFs); bounded meshes only."""
        return self.nodal(NodeSet.INNER)

    def nodal(self, node_set: NodeSet, *,
              bc: BC | BCStructure | tuple[BC, ...] = BC.NONE,
              ) -> NodalSpace:
        """
        General nodal factory with BC structure.

        Description
        -----------
        The zero-argument properties above are sugar for
        ``bc=BC.NONE``. BC-structured variants are the origins of
        the iteration-1 Sine/Cosine spaces (Dirichlet ``Center`` →
        DST-II, Dirichlet ``Inner`` → DST-I, Neumann ``Center`` →
        DCT-II, Neumann ``Outer`` → DCT-I).

        Parameters
        ----------
        node_set : NodeSet
            The topological node set (no string-keyed variant).
        bc : BC | BCStructure | tuple[BC, ...], optional
            The homogeneous BC structure; a single kind applies to
            every boundary component (default: BC.NONE).

        Returns
        -------
        NodalSpace
            The interned nodal space.
        """
        if not isinstance(node_set, NodeSet):
            raise TypeError(
                f"node_set must be a NodeSet member, got {node_set!r}")
        if node_set is NodeSet.POINTS:
            raise ValueError(
                "POINTS is the PointMesh node set; structured 1D "
                "meshes have no point-values space")
        if self._periodic and node_set in (
                NodeSet.OUTER, NodeSet.INNER):
            raise ValueError(
                f"{node_set.name} spaces need a bounded mesh: a "
                "periodic mesh has no boundary faces")
        self._validate_node_set(node_set)
        structure = self._normalize_bc(bc)
        cls = NODAL_CLASS[node_set]
        key = space_key(cls, structure, Scalars.REAL)
        return self._intern(
            key,
            lambda: cls(self, Scalars.REAL, structure,
                        _token=_FACTORY_TOKEN))

    def _validate_node_set(self, node_set: NodeSet) -> None:
        """Reject node sets a restricted mesh does not carry.

        No-op here; the ``ChebyshevMesh`` override raises.
        """

    def _normalize_bc(
        self, bc: BC | BCStructure | tuple[BC, ...],
    ) -> BCStructure:
        """Normalize a BC spec against this mesh's boundary."""
        if self._periodic and isinstance(bc, BC) and bc is not BC.NONE:
            raise ValueError(
                "periodic meshes have no boundary to constrain; "
                f"got bc={bc}")
        return BCStructure.normalize(bc, self._n_boundary_components)

    # ================================================================
    #  Average factories (section 3.9)
    # ================================================================
    @cached_property
    def cell_avg(self) -> CellAvg:
        """Primal-cell averages (n DOFs)."""
        return self._average(CellAvg)

    @cached_property
    def face_avg(self) -> FaceAvg:
        """Dual-cell averages (n periodic / n - 1 bounded DOFs)."""
        return self._average(FaceAvg)

    def _average(self, cls: type[AverageSpace]) -> AverageSpace:
        """Intern the BC-free average space of the given class."""
        self._validate_average()
        structure = self._free_bc
        key = space_key(cls, structure, Scalars.REAL)
        return self._intern(
            key,
            lambda: cls(self, Scalars.REAL, structure,
                        _token=_FACTORY_TOKEN))

    def _validate_average(self) -> None:
        """Reject averages on meshes without a cell family.

        No-op here; the ``ChebyshevMesh`` override raises.
        """

    # ================================================================
    #  Coefficient factories (section 3.2)
    # ================================================================
    def fourier(self, origin: FunctionSpace) -> FourierSpace:
        """
        Fourier coefficient space of the given origin.

        Description
        -----------
        Periodic meshes only. The origin is explicit — no default —
        and constitutive: Fourier spaces of distinct origins are
        distinct interned spaces.

        Parameters
        ----------
        origin : FunctionSpace
            The nodal or average origin space (on this mesh).

        Returns
        -------
        FourierSpace
            The interned coefficient space (scalars follow the
            origin: real origin → Hermitian half spectrum).
        """
        if not self._periodic:
            raise ValueError(
                "fourier spaces need a periodic mesh; use sine/"
                "cosine (or chebyshev) on bounded ones")
        self._validate_origin(origin)
        return self._coefficient(FourierSpace, origin)

    def sine(self, origin: FunctionSpace) -> SineSpace:
        """
        DST coefficient space of a Dirichlet-structured origin.

        Parameters
        ----------
        origin : FunctionSpace
            A nodal/average origin on this (bounded) mesh whose BC
            structure is Dirichlet at every boundary component.

        Returns
        -------
        SineSpace
            The interned coefficient space.
        """
        if self._periodic:
            raise ValueError(
                "sine spaces need a bounded mesh; use fourier on "
                "periodic ones")
        self._validate_origin(origin)
        components = origin.bc.components
        if any(kind is not BC.DIRICHLET for kind in components):
            raise ValueError(
                "sine origins must be Dirichlet-structured at every "
                f"boundary component, got {origin.bc!r}")
        return self._coefficient(SineSpace, origin)

    def cosine(self, origin: FunctionSpace) -> CosineSpace:
        """
        DCT coefficient space of a Neumann-structured origin.

        Parameters
        ----------
        origin : FunctionSpace
            A nodal/average origin on this (bounded) mesh whose BC
            structure is Neumann at every boundary component.

        Returns
        -------
        CosineSpace
            The interned coefficient space.
        """
        if self._periodic:
            raise ValueError(
                "cosine spaces need a bounded mesh; use fourier on "
                "periodic ones")
        self._validate_origin(origin)
        components = origin.bc.components
        if any(kind is not BC.NEUMANN for kind in components):
            raise ValueError(
                "cosine origins must be Neumann-structured at every "
                f"boundary component, got {origin.bc!r}")
        return self._coefficient(CosineSpace, origin)

    def galerkin(self, *, bc: BC | BCStructure,
                 extended: bool = False) -> GalerkinSpace:
        """
        Modal Galerkin space with baked-in BCs (designed-for).

        Parameters
        ----------
        bc : BC | BCStructure
            The BC structure baked into the basis.
        extended : bool, optional
            Whether to include boundary modes (default: False).

        Returns
        -------
        GalerkinSpace
            Never returns in iteration 1.
        """
        raise NotImplementedError(
            "Galerkin/Shen spaces are designed-for, not iteration 1")

    def _validate_origin(self, origin: FunctionSpace) -> None:
        """Reject origins that cannot key a coefficient space."""
        if not isinstance(origin, NodalSpace | AverageSpace):
            raise TypeError(
                "coefficient origins are nodal or average spaces, "
                f"got {origin!r}")
        self._check_owned(origin)
        if origin.layout is not None:
            raise ValueError(
                "coefficient origins are bare spaces; got a laid-out "
                f"origin {origin!r}")

    def _coefficient(self, cls: type[CoefficientSpace],
                     origin: FunctionSpace) -> CoefficientSpace:
        """Intern the coefficient space of one class/origin pair.

        The origin's identity is the key; scalars follow it.
        """
        key = space_key(cls, origin)
        return self._intern(
            key,
            lambda: cls(self, origin, _token=_FACTORY_TOKEN))

    def __repr__(self) -> str:
        """E.g. ``IntervalMesh(x: n=256, extent=(0, 1), periodic)``."""
        topology = "periodic" if self._periodic else "bounded"
        return (f"{type(self).__name__}({self._label}: "
                f"n={self._n_cells}, extent={self._extent}, "
                f"{topology})")
