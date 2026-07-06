"""
Homogeneous boundary-condition structure markers.

Description
-----------
Owning class doc: ``notes/framework2/classes/meshes.md`` (static
markers). ``BC`` is the per-boundary-component condition kind
(re-exported at top level as ``fr.BC``); ``BCStructure`` is the
normalized per-component tuple used inside the mesh interning keys.
Periodicity is deliberately *not* a ``BC`` member: it is mesh
topology (``mesh.periodic``), fixed at mesh construction.
"""
# Wave 0: BC, BCStructure
from __future__ import annotations

from enum import Enum, auto


class BC(Enum):

    """Homogeneous boundary-condition kind at one boundary component."""

    NONE = auto()  # BC-free: boundary DOFs stay in the space
    DIRICHLET = auto()
    NEUMANN = auto()


class BCStructure:

    """
    Normalized per-boundary-component tuple of BC kinds.

    Description
    -----------
    A small *value* type used inside space interning keys; unlike
    spaces (identity-hashed) it is deliberately hashed and compared
    by value. Factories accept the sugar ``bc=fr.BC.DIRICHLET``
    (meaning: at every boundary component) and normalize through
    :meth:`normalize`.

    Parameters
    ----------
    components : tuple[BC, ...]
        The BC kind per boundary component ((left, right) for 1D
        meshes).
    """

    def __init__(self, components: tuple[BC, ...]) -> None:
        """Store the (left, right) tuple for 1D meshes."""
        components = tuple(components)
        for component in components:
            if not isinstance(component, BC):
                raise TypeError(
                    "BCStructure components must be BC members, got "
                    f"{component!r}")
        self._components: tuple[BC, ...] = components

    @classmethod
    def normalize(cls, spec: BC | BCStructure | tuple[BC, ...],
                  n_components: int) -> BCStructure:
        """
        Expand a single BC to all components; validate length.

        Parameters
        ----------
        spec : BC | BCStructure | tuple[BC, ...]
            A single kind (applied to every boundary component), an
            explicit per-component tuple, or an existing structure.
        n_components : int
            The number of boundary components of the mesh.

        Returns
        -------
        BCStructure
            The normalized structure with `n_components` components.
        """
        if isinstance(spec, BC):
            structure = cls((spec,) * n_components)
        elif isinstance(spec, BCStructure):
            structure = spec
        else:
            structure = cls(spec)
        if len(structure.components) != n_components:
            raise ValueError(
                f"expected {n_components} boundary components, got "
                f"{len(structure.components)}: {structure!r}")
        return structure

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def components(self) -> tuple[BC, ...]:
        """BC kind per boundary component (left, right for 1D)."""
        return self._components

    @property
    def n_constraints(self) -> int:
        """Number of constrained DOFs (non-NONE components)."""
        return sum(
            1 for component in self._components if component is not BC.NONE)

    @property
    def is_free(self) -> bool:
        """True if every component is BC.NONE."""
        return all(
            component is BC.NONE for component in self._components)

    # ================================================================
    #  Value equality (unlike spaces, which use identity)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Value equality (unlike spaces, which use identity)."""
        if not isinstance(other, BCStructure):
            return NotImplemented
        return self._components == other._components

    def __hash__(self) -> int:
        """Value hash; used inside the mesh interning keys."""
        return hash((type(self), self._components))

    def __repr__(self) -> str:
        """E.g. 'BCStructure(DIRICHLET, NONE)'."""
        names = ", ".join(
            component.name for component in self._components)
        return f"BCStructure({names})"
