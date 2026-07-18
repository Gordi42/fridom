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
from fridom.spatial.spaces.trace import Side, TraceSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

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
        self._coarsened_cache: dict[int, Self] = {}

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

    @property
    def coordinate_map(
        self,
    ) -> Callable[[jax.Array], jax.Array] | None:
        r"""
        Optional map from computational to physical coordinates.

        Description
        -----------
        The mesh-side geometry seam of the grid's node/measure
        materializers (concepts section 2.7): a pure, jnp-traceable,
        strictly increasing callable from the computational
        coordinate :math:`s \in [0, 1]` to the physical coordinate.
        ``None`` — this default — declares the uniform affine
        placement read off ``extent`` and the mesh's scalar ``dx``
        descriptor (the constant special case XLA folds). Non-affine
        meshes (``MappedIntervalMesh``, ``ChebyshevMesh``) return
        the mapping callable; the mesh stores the *function*, never
        materialized node arrays — the grid composes and shards on
        demand.
        """
        return None

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
        for coefficient spaces (local preferred); (LOCAL,) for the
        collapsed factors (``ConstantSpace`` / ``TraceSpace``).
        ``ChebyshevMesh`` overrides (transpose-first).

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
        if space.collapses_axis:
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

    def coarsened(self, factor: int) -> Self:
        """
        Return an independent same-family mesh with n_cells // factor.

        Description
        -----------
        The coarse sibling for grid hierarchies (multigrid, regridding,
        MG-D3). Unlike :meth:`refined`, the result carries **no**
        ``refined_from`` link: it is not adopted by any grid owning this
        mesh, so cross-resolution use is always explicit through
        ``GridTransfer``. Memoized per factor in its own cache, so
        repeated requests return the identical coarser mesh and its
        spaces stay identity-comparable.

        ``factor == 1`` is the semicoarsening pass-through (MG-D4): it
        returns ``self`` unchanged, so a ``ChebyshevMesh`` kept at full
        resolution along the vertical is never asked to coarsen. A
        ``factor >= 2`` on a mesh whose ``_make_refined`` is designed-for
        (``ChebyshevMesh``) raises there, exactly as ``refined`` does.

        Parameters
        ----------
        factor : int
            The integer cell-count divisor (>= 1); ``n_cells`` must be
            divisible by it.

        Returns
        -------
        Self
            The pass-through ``self`` (factor 1) or the memoized
            coarser mesh (``refined_from`` unset).
        """
        if isinstance(factor, bool) or not isinstance(factor, int):
            raise TypeError(
                "coarsening factors are positive integers, got "
                f"{factor!r}")
        if factor < 1:
            raise ValueError(
                f"coarsening factors must be >= 1, got {factor}")
        if factor == 1:
            return self  # semicoarsening pass-through (MG-D4)
        if self._n_cells % factor != 0:
            raise ValueError(
                f"n_cells must be divisible by the coarsening factor, "
                f"got {self._n_cells} % {factor} = "
                f"{self._n_cells % factor}")
        mesh = self._coarsened_cache.get(factor)
        if mesh is None:
            mesh = self._make_refined(self._n_cells // factor)
            # no ``refined_from`` link (MG-D3): an independent mesh, so
            # the fine grid never silently adopts these coarse spaces
            self._coarsened_cache[factor] = mesh
        return mesh

    @property
    def coarsenable(self) -> bool:
        """Whether this mesh family can build a coarser sibling.

        ``True`` on the structured interval meshes (their
        ``_make_refined`` scales the cell count); the ``ChebyshevMesh``
        override returns ``False``. Consulted by the multigrid hierarchy
        builder to keep a non-coarsenable axis at full resolution rather
        than fault (GM-D9 graceful degradation).
        """
        return True

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

    def trace(self, node_set: NodeSet, side: Side,
              depth: int = 0) -> TraceSpace:
        """
        Boundary-trace factory: the values on one boundary row.

        Description
        -----------
        Mints the interned ``TraceSpace`` of the given parent node
        set at the given wall — a size-1, non-broadcasting factor on
        *this* mesh (``boundary_trace_plan.md`` §2). A trace needs a
        boundary, so periodic meshes and the ``POINTS`` node set are
        rejected; ``depth != 0`` (interior slices) raises at
        construction (only the boundary row is implemented).

        Parameters
        ----------
        node_set : NodeSet
            The topological node set the trace is taken from (any
            structured 1D set; ``POINTS`` is rejected).
        side : Side
            The boundary side (``Side.LOW`` / ``Side.HIGH``).
        depth : int, optional
            The signed true-node index from ``side``; only ``0`` (the
            boundary row) is implemented (default: 0).

        Returns
        -------
        TraceSpace
            The interned trace space.
        """
        if not isinstance(node_set, NodeSet):
            raise TypeError(
                f"node_set must be a NodeSet member, got {node_set!r}")
        if node_set is NodeSet.POINTS:
            raise ValueError(
                "POINTS is the PointMesh node set; structured 1D "
                "meshes have no point-values space to trace")
        if not isinstance(side, Side):
            raise TypeError(
                f"side must be a Side member (Side.LOW / Side.HIGH), "
                f"got {side!r}")
        if self._periodic:
            raise ValueError(
                "trace spaces need a bounded mesh: a periodic mesh "
                "has no boundary to trace")
        self._validate_node_set(node_set)
        key = space_key(TraceSpace, node_set, side, depth,
                        Scalars.REAL)
        return self._intern(
            key,
            lambda: TraceSpace(
                self, Scalars.REAL, self._free_bc,
                parent_node_set=node_set, side=side, depth=depth,
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
        """Primal-cell averages (n DOFs); BC-free sugar for ``average``."""
        return self.average(CellAvg)

    @cached_property
    def face_avg(self) -> FaceAvg:
        """Dual-cell averages (n periodic / n - 1 bounded DOFs)."""
        return self.average(FaceAvg)

    def average(self, kind: type[AverageSpace], *,
                bc: BC | BCStructure | tuple[BC, ...] = BC.NONE,
                ) -> AverageSpace:
        """
        General average factory with BC structure.

        Description
        -----------
        The zero-argument ``cell_avg`` / ``face_avg`` properties are
        sugar for ``bc=BC.NONE``. The average class is the topological
        discriminator — there is no ``AverageSet`` enum, the two
        average classes *are* the vocabulary — mirroring
        :meth:`nodal`'s ``node_set`` argument. A BC tag on an average
        origin is the same wall-value claim about the represented
        function as on a nodal one and changes no DOF count
        (``CellAvg`` keeps shape ``(n,)`` under every tag: averages
        have no boundary DOF in the set), so a tagged ``CellAvg``
        origin carries the walled DCT/DST transforms (Neumann
        ``CellAvg`` → DCT-II, Dirichlet ``CellAvg`` → DST-II).
        ``FaceAvg`` stays untaggable (FV-D2: the dual-cell average
        family is a dead-end with no diff / flux_diff rows), so a
        non-NONE bc on it is a taught error.

        Parameters
        ----------
        kind : type[AverageSpace]
            The average space class (``CellAvg`` or ``FaceAvg``).
        bc : BC | BCStructure | tuple[BC, ...], optional
            The homogeneous BC structure; a single kind applies to
            every boundary component (default: BC.NONE).

        Returns
        -------
        AverageSpace
            The interned average space.
        """
        if not (isinstance(kind, type)
                and issubclass(kind, AverageSpace)):
            raise TypeError(
                "kind must be an AverageSpace subclass (CellAvg or "
                f"FaceAvg), got {kind!r}")
        self._validate_average()
        structure = self._normalize_bc(bc)
        if issubclass(kind, FaceAvg) and not structure.is_free:
            raise ValueError(
                "FaceAvg is untaggable (FV-D2): the dual-cell average "
                "family is a dead-end (no diff / flux_diff rows), so a "
                f"non-NONE bc on it has no consumer, got bc={bc}; tag "
                "the primal CellAvg instead")
        key = space_key(kind, structure, Scalars.REAL)
        return self._intern(
            key,
            lambda: kind(self, Scalars.REAL, structure,
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
