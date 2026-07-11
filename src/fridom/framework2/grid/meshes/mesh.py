"""
The ``Mesh`` ABC: atomic factor of the domain geometry.

Description
-----------
Owning class doc: ``design/specs/grid/classes/meshes.md``. A mesh owns
geometry and topology only — no discretization, no arrays. It is the
factory and interning registry of its function spaces: value-equal
factory requests return the identical space object, so the strict
algebra's equality check is ``a is b``. Meshes themselves are *not*
interned by value: each construction is a new, distinct domain
factor.
"""
# Wave 1: Mesh (ABC)
from __future__ import annotations

from abc import ABC, abstractmethod
from functools import cached_property
from typing import TYPE_CHECKING

from fridom.framework2.grid.bc import BC, BCStructure
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.function_space import (
    _FACTORY_TOKEN,
    FunctionSpace,
    space_key,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Hashable

    from fridom.framework2.grid.decomposition.traits import (
        MeshDecompositionTraits,
    )


class Mesh(ABC):

    """
    Atomic factor of the domain geometry (geometry + topology).

    Description
    -----------
    Static, identity-hashed, and never interned by value (two
    value-equal meshes are distinct domain factors). Coordinate
    names are mandatory at construction. The base class declares no
    nodal/average/coefficient factories — those exist only where
    the node-set vocabulary is meaningful; only ``constant`` is
    universal.

    Parameters
    ----------
    names : tuple[str, ...]
        The coordinate names this factor contributes (1 for 1D
        factors, 2 for a sphere); unique, non-empty strings.
    """

    def __init__(self, names: tuple[str, ...]) -> None:
        """Create the factor; coordinate names are mandatory."""
        names = tuple(names)
        if not names:
            raise ValueError("a mesh needs at least one coordinate "
                             "name")
        for name in names:
            if not isinstance(name, str) or not name:
                raise TypeError(
                    "coordinate names must be non-empty strings, "
                    f"got {name!r}")
        if len(set(names)) != len(names):
            raise ValueError(
                f"coordinate names must be unique, got {names}")
        self._names: tuple[str, ...] = names
        # label used by space reprs; boundary meshes override it
        # (e.g. 'boundary(x)') so trace spaces print accordingly
        self._label: str = ", ".join(names)
        # the mesh-owned space registry: interning happens per mesh,
        # so spaces of two distinct-but-equal meshes never conflate;
        # a plain dict — interned spaces live as long as the mesh
        self._registry: dict[Hashable, FunctionSpace] = {}

    # ================================================================
    #  Identity (explicit, see cluster rules)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Identity: return ``self is other``.

        Explicit so fridom's structural-equality walk falls through
        to plain ``==``.
        """
        return self is other

    def __hash__(self) -> int:
        """Identity hash, matching ``__eq__``."""
        return id(self)

    # ================================================================
    #  Geometry / topology
    # ================================================================
    @property
    @abstractmethod
    def dim(self) -> int:
        """Intrinsic dimension of the factor (>= 0)."""
        ...

    @property
    @abstractmethod
    def boundary(self) -> Mesh:
        """The conforming boundary as a mesh of dimension dim - 1.

        Stable: repeated access returns the identical object.
        """
        ...

    # ================================================================
    #  Coordinate names
    # ================================================================
    @property
    def names(self) -> tuple[str, ...]:
        """Coordinate names, fixed at construction."""
        return self._names

    # ================================================================
    #  Decomposition traits (seam: types owned by doc 04)
    # ================================================================
    @abstractmethod
    def decomposition_traits(
        self, space: FunctionSpace,
    ) -> MeshDecompositionTraits:
        """
        Sharding/halo traits for one of this mesh's spaces.

        Parameters
        ----------
        space : FunctionSpace
            A space interned on this mesh (per-space query: nodal
            and coefficient spaces of one mesh differ).

        Returns
        -------
        MeshDecompositionTraits
            The frozen traits record with preference-ordered
            strategies.
        """
        ...

    def _check_owned(self, space: FunctionSpace) -> None:
        """Raise unless ``space`` is interned on this mesh."""
        if space.mesh is not self:
            raise ValueError(
                f"{space!r} lives on {space.mesh!r}, not on {self!r}")

    # ================================================================
    #  Space factory / interning registry
    # ================================================================
    @cached_property
    def constant(self) -> ConstantSpace:
        """The one-DOF broadcast space on this mesh.

        Universal: 'constant along this factor' makes sense for
        every mesh.
        """
        key = space_key(ConstantSpace, Scalars.REAL)
        return self._intern(
            key,
            lambda: ConstantSpace(self, Scalars.REAL, self._free_bc,
                                  _token=_FACTORY_TOKEN))

    @cached_property
    def _free_bc(self) -> BCStructure:
        """The all-free BC structure of this mesh.

        One ``BC.NONE`` per boundary component.
        """
        return BCStructure(
            (BC.NONE,) * self._n_boundary_components)

    @property
    @abstractmethod
    def _n_boundary_components(self) -> int:
        """Number of boundary components.

        2 for a bounded interval, 0 for periodic and point meshes.
        """
        ...

    def _intern(self, key: Hashable,
                factory: Callable[[], FunctionSpace]) -> FunctionSpace:
        """
        Return the registry entry for ``key``.

        Description
        -----------
        Builds the entry on first request: the interning mechanism
        behind every space factory.

        Parameters
        ----------
        key : Hashable
            The value-hashable interning key (``space_key(...)``).
        factory : Callable[[], FunctionSpace]
            Zero-argument builder invoked only on the first request.

        Returns
        -------
        FunctionSpace
            The interned space: value-equal requests return the
            identical object.
        """
        space = self._registry.get(key)
        if space is None:
            space = factory()
            self._registry[key] = space
        return space

    def __repr__(self) -> str:
        """Render a generic mesh repr; concrete meshes override."""
        return f"{type(self).__name__}({self._label})"
