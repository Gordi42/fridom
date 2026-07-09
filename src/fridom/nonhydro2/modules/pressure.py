r"""The spectral pressure Poisson solve.

Description
-----------
The nonhydrostatic incompressibility constraint is enforced by the
fractional-step projection ``div(grad p) = div(u*)`` then
``u = u* - grad p`` (03 §5.6, project-the-state). The elliptic operator
is the grid's own ``dsqr``-weighted discrete Laplacian
``Laplacian(metric={z: 1/dsqr})`` — the block matmul
``Div @ Diag(1, 1, 1/dsqr) @ Grad`` whose ``Divergence`` / ``Gradient``
factors expand into whatever staggered difference scheme the grid
dispatches (the C-grid forward / backward differences). The vertical
``1/dsqr`` weight rides ``ctx.params`` as a traced-but-constant 0-d
leaf, carried on the metric's vertical entry.

The solve is delegated to :class:`SpectralSolve`, which resolves the
transform, queries the operator's ``eigenvalues`` on the coefficient
space, pseudo-inverts (the ``k = 0`` mean-pressure gauge), and applies
the realized-map composition ``backward @ inverse @ forward`` (S2). The
``bwd @ fwd`` round-trip recovers the honest ``-k̂²`` on *every* mode
(Nyquist included), so inverting drives the *discrete* divergence to
machine zero.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework2.grid.operators.composed import Laplacian
from fridom.framework2.grid.operators.spectral_solve import SpectralSolve

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


class SpectralPressureSolver:

    """Grid-bound spectral solve of ``lap(p) = div`` on a Fourier grid.

    Description
    -----------
    Constructed at trace time inside the projection stage from the
    operand's grid; carries no mutable state and is not a pytree leaf.
    Per solve it expands the ``dsqr``-weighted Laplacian
    ``Div @ Diag(1, .., 1/dsqr) @ Grad`` (the live ``1/dsqr`` riding the
    vertical metric entry) and inverts it through :class:`SpectralSolve`.

    Parameters
    ----------
    grid : object
        The grid carrying the transform / dispatch registry.
    space : SpaceLike
        The (cell-centered) function space of the divergence operand.
    vertical : str
        The vertical coordinate name (the ``1/dsqr``-weighted axis).
    """

    def __init__(
        self, grid: object, space: SpaceLike, *, vertical: str,
    ) -> None:
        """Store the grid, operand space, and vertical axis name."""
        self._grid: object = grid
        self._space: SpaceLike = space
        self._vertical: str = vertical

    def solve(
        self, div: ScalarField, *, dsqr: jax.Array,
    ) -> ScalarField:
        """Return ``p`` with ``div(grad p) = div`` (discrete, exact).

        Parameters
        ----------
        div : ScalarField
            The (cell-centered, real) divergence of the provisional
            velocity.
        dsqr : jax.Array
            The live squared-aspect-ratio leaf.

        Returns
        -------
        ScalarField
            The pressure on the same (cell-centered) space as ``div``.
        """
        laplacian = Laplacian(
            metric={self._vertical: 1.0 / dsqr},
        ).expand(self._space.bare, self._grid).scalar()
        return SpectralSolve(laplacian, self._grid, self._space).solve(div)
