"""``MappedIntervalMesh``: stretched 1D interval via a coordinate map."""
# Coordinate-systems plan, stage C0: MappedIntervalMesh
from __future__ import annotations

from typing import TYPE_CHECKING, Self

import numpy as np

from fridom.spatial.meshes.structured_1d import StructuredMesh1D

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.spatial.spaces.coefficient import (
        CosineSpace,
        FourierSpace,
        SineSpace,
    )
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )

#: relative tolerance of the construction-time endpoint check
_ENDPOINT_RTOL = 1e-6


class MappedIntervalMesh(StructuredMesh1D):

    r"""
    1D interval with a monotone self-coordinate mapping.

    Description
    -----------
    Owning class doc: ``design/specs/grid/classes/meshes.md``. The
    stretched 1D interval (vertical coordinates): a pure,
    jnp-traceable, strictly increasing ``mapping`` from the
    computational coordinate :math:`s \in [0, 1]` to the physical
    coordinate. The mesh stores the *function*, never materialized
    node arrays — the grid composes the mapping with the uniform
    computational nodes on demand (concepts section 2.7), and the
    two staggered ``dx`` measures become genuinely different metric
    fields on their spaces. Deliberately **no scalar ``dx``**
    property exists here: asking for one is the bug the metric-field
    design prevents. The mapping may depend only on this mesh's own
    coordinate; cross-factor mappings (terrain-following) are
    grid-level metric data (rules section 3.8), not mesh structure.

    The spectral coefficient factories (``fourier``/``sine``/
    ``cosine``) raise: those bases diagonalize derivatives of the
    *computational* coordinate, so their seeded derivative rows
    would silently not be the physical derivative (deferred to the
    coordinate-systems plan, stage C2).

    Parameters
    ----------
    shape : int
        The number of primal cells n.
    extent : tuple[float, float]
        The physical interval (x_min, x_max); the mapping endpoints
        must match it.
    mapping : Callable
        Pure, jnp-traceable, strictly increasing map from
        computational ``s`` in [0, 1] to the physical coordinate
        (a static descriptor, hashed by identity).
    periodic : bool, optional
        Whether the interval is periodic (default: False).
    name : str
        The mandatory coordinate name.
    """

    def __init__(self, shape: int, extent: tuple[float, float],
                 mapping: Callable[[jax.Array], jax.Array],
                 periodic: bool = False, *, name: str) -> None:
        """Store the mapped interval; validate the mapping once."""
        super().__init__(shape, extent, periodic, name=name)
        if not callable(mapping):
            raise TypeError(
                "mapping must be a callable s in [0, 1] -> physical "
                f"coordinate, got {mapping!r}")
        self._mapping: Callable[[jax.Array], jax.Array] = mapping
        self._validate_mapping()

    # ================================================================
    #  Geometry descriptors
    # ================================================================
    @property
    def mapping(self) -> Callable[[jax.Array], jax.Array]:
        """The coordinate mapping (static descriptor)."""
        return self._mapping

    @property
    def coordinate_map(self) -> Callable[[jax.Array], jax.Array]:
        """The mapping, exposed through the geometry seam."""
        return self._mapping

    # ================================================================
    #  Refinement (same mapping, scaled cell count)
    # ================================================================
    def _make_refined(self, n_cells: int) -> Self:
        """Construct the scaled mesh (same extent, mapping, name)."""
        return type(self)(n_cells, self._extent, self._mapping,
                          self._periodic, name=self._names[0])

    # ================================================================
    #  Coefficient factories (deferred: computational-space bases)
    # ================================================================
    def fourier(self, origin: FunctionSpace) -> FourierSpace:
        """Raise: spectral bases on mapped meshes are deferred."""
        raise NotImplementedError(self._spectral_hint("fourier"))

    def sine(self, origin: FunctionSpace) -> SineSpace:
        """Raise: spectral bases on mapped meshes are deferred."""
        raise NotImplementedError(self._spectral_hint("sine"))

    def cosine(self, origin: FunctionSpace) -> CosineSpace:
        """Raise: spectral bases on mapped meshes are deferred."""
        raise NotImplementedError(self._spectral_hint("cosine"))

    def _spectral_hint(self, family: str) -> str:
        """Compose the deferral message of a coefficient factory."""
        return (
            f"{family} spaces on a MappedIntervalMesh are deferred: "
            "the basis diagonalizes computational-coordinate "
            "derivatives, not the physical ones (coordinate-systems "
            "plan, stage C2)")

    # ================================================================
    #  Construction-time mapping validation (host-side, once)
    # ================================================================
    def _validate_mapping(self) -> None:
        """Check monotonicity and the extent endpoints once."""
        n = self._n_cells
        s = np.linspace(0.0, 1.0, 2 * n + 1)
        x = np.asarray(self._mapping(s))
        if x.shape != s.shape:
            raise ValueError(
                "mapping must map coordinates elementwise: shape "
                f"{s.shape} in, {x.shape} out")
        if not np.all(np.isfinite(x)):
            raise ValueError(
                "mapping produced non-finite physical coordinates")
        if not np.all(np.diff(x) > 0):
            raise ValueError(
                "mapping must be strictly increasing on [0, 1] "
                "(sampled at the cell faces and centers)")
        x_min, x_max = self._extent
        tol = _ENDPOINT_RTOL * (x_max - x_min)
        if (abs(float(x[0]) - x_min) > tol
                or abs(float(x[-1]) - x_max) > tol):
            raise ValueError(
                f"mapping endpoints ({float(x[0])}, {float(x[-1])}) "
                f"must match the extent {self._extent}")
