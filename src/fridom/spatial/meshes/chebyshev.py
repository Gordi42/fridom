"""
``ChebyshevMesh``: bounded 1D mesh on Gauss-Lobatto geometry.

Description
-----------
Owning class doc: ``design/specs/grid/classes/meshes.md``. Host of
the Chebyshev space family. The space family is restricted (owner
decision): ``outer``/``lobatto``, the ``chebyshev``/``galerkin``
coefficient spaces, and ``constant`` — no cell family. The nodal
``center``/``left``/``right``/``inner`` and average factories raise
until an FV-on-Chebyshev consumer exists.
"""
# Wave 1: ChebyshevMesh (meshes.md tags it iteration 1; the Wave-0
# stub marker said Wave 3, which was wrong)
from __future__ import annotations

from typing import TYPE_CHECKING, Self

import jax.numpy as jnp

from fridom.spatial.decomposition.traits import (
    HaloStrategy,
    MeshDecompositionTraits,
)
from fridom.spatial.meshes.structured_1d import StructuredMesh1D
from fridom.spatial.spaces.coefficient import ChebyshevSpace
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import (
    NodalSpace,
    NodeSet,
    Outer,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )


class ChebyshevMesh(StructuredMesh1D):

    """
    1D interval with Gauss-Lobatto node geometry.

    Description
    -----------
    Bounded by construction. The xgcm node-set vocabulary is
    topological; this mesh places the n + 1 faces at Gauss-Lobatto
    locations. ``fourier`` raises (bounded); ``sine``/``cosine`` are
    admissible but the natural bases here are Chebyshev/Shen.

    Parameters
    ----------
    shape : int
        The cell count n (n + 1 Lobatto points).
    extent : tuple[float, float]
        The physical interval (x_min, x_max).
    name : str
        The mandatory coordinate name.
    """

    def __init__(self, shape: int, extent: tuple[float, float],
                 *, name: str) -> None:
        """Bounded by construction (periodic is always False)."""
        super().__init__(shape, extent, periodic=False, name=name)

    # ================================================================
    #  Restricted space family (owner decision)
    # ================================================================
    def _validate_node_set(self, node_set: NodeSet) -> None:
        """Only the Lobatto/outer family exists on this mesh."""
        if node_set is not NodeSet.OUTER:
            raise ValueError(
                f"ChebyshevMesh has no {node_set.name} spaces: the "
                "family is restricted to outer/lobatto, chebyshev "
                "coefficients, and constant")

    def _validate_average(self) -> None:
        """No cell family until an FV-on-Chebyshev consumer exists."""
        raise ValueError(
            "ChebyshevMesh has no average spaces: the family is "
            "restricted to outer/lobatto, chebyshev coefficients, "
            "and constant")

    @property
    def coordinate_map(self) -> Callable[[jax.Array], jax.Array]:
        r"""
        The Gauss-Lobatto placement as a coordinate map.

        Description
        -----------
        The geometry seam (concepts section 2.7) of this mesh:
        :math:`x(s) = x_{mid} - \tfrac{L}{2} \cos(\pi s)`, mapping
        the uniform computational faces ``s = j / n`` to the
        ascending Gauss-Lobatto points (storage order). Strictly
        increasing on [0, 1] with ``x(0) = x_min``, ``x(1) = x_max``.
        """
        x_min, x_max = self.extent
        mid = 0.5 * (x_min + x_max)
        half = 0.5 * (x_max - x_min)

        def lobatto_map(s: jax.Array) -> jax.Array:
            return mid - half * jnp.cos(jnp.pi * s)

        return lobatto_map

    @property
    def lobatto(self) -> Outer:
        """Alias for ``self.outer`` (the identical interned object).

        The Gauss-Lobatto collocation space: n + 1 points including
        the endpoints. Spectral users think 'Lobatto points', not
        'outer faces'.
        """
        return self.outer

    def chebyshev(self, origin: FunctionSpace) -> ChebyshevSpace:
        """
        Chebyshev coefficient space of the given origin.

        Parameters
        ----------
        origin : FunctionSpace
            The Gauss-Lobatto (outer-family) origin space on this
            mesh; explicit, no default.

        Returns
        -------
        ChebyshevSpace
            The interned coefficient space.
        """
        self._validate_origin(origin)
        # defensive: the restricted family means every nodal/average
        # space of this mesh is outer-family already; this guard
        # only fires if the family ever grows
        if not (isinstance(origin, NodalSpace)
                and origin.node_set is NodeSet.OUTER):  # pragma: no cover
            raise ValueError(
                "chebyshev origins are Gauss-Lobatto (outer-family) "
                f"spaces, got {origin!r}")
        return self._coefficient(ChebyshevSpace, origin)

    def _make_refined(self, n_cells: int) -> Self:
        """Refinement is designed-for on this mesh."""
        raise NotImplementedError(
            "refined() is designed-for on ChebyshevMesh (iteration 1 "
            "only on IntervalMesh)")

    # ================================================================
    #  Decomposition traits (owner directive: TRANSPOSE-capable)
    # ================================================================
    def decomposition_traits(
        self, space: FunctionSpace,
    ) -> MeshDecompositionTraits:
        """
        Return the transpose-first traits of this mesh's spaces.

        Description
        -----------
        The mesh is shardable: contiguous-axis needs (transform,
        banded solves) are met via transpose layouts.

        Parameters
        ----------
        space : FunctionSpace
            A space interned on this mesh.

        Returns
        -------
        MeshDecompositionTraits
            ``(TRANSPOSE, LOCAL)``; ``(LOCAL,)`` for
            ``ConstantSpace``.
        """
        self._check_owned(space)
        if isinstance(space, ConstantSpace):
            return MeshDecompositionTraits((HaloStrategy.LOCAL,))
        return MeshDecompositionTraits(
            (HaloStrategy.TRANSPOSE, HaloStrategy.LOCAL))
