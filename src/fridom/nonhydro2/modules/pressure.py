"""The spectral pressure Poisson solve.

Description
-----------
The nonhydrostatic incompressibility constraint is enforced by the
fractional-step projection ``div(grad p) = div(u*)`` then
``u = u* - grad p`` (03 §5.6, project-the-state). The elliptic operator
is the grid's own discrete Laplacian ``∇² = Div @ Diag(1, 1, 1/dsqr) @
Grad``: the ``Divergence``/``Gradient`` builders expand into whatever
staggered difference scheme the grid dispatches (the C-grid forward /
backward differences), and their ``eigenvalues`` supply the per-mode
diagonal ``Symbol`` on the transformed coefficient space — no
hand-written ``k̂²`` (``discrete_laplace_symbol`` retired,
symbol_stack_design.md decision 1).

Because ``eigenvalues`` is layout-faithful (decision 3), the composed
``bwd @ fwd`` symbol is assembled directly on the ``rfftn`` coefficient
layout ``(half, full, full)`` — the wall the old hand-rolled solve hit
is dissolved. The ``bwd @ fwd`` round-trip recovers the honest
``-k̂² = -2(1 - cos k dx)/dx²`` on *every* mode (Nyquist included),
matching the ``staggered_diff`` kernel exactly, so inverting drives the
*discrete* divergence to machine zero.

The vertical ``1/dsqr`` weight lives in the operator: the pressure
Laplacian is the literal block matmul ``Div @ Diag(1, .., 1/dsqr) @
Grad``, a diagonal metric between ``Grad`` and ``Div`` that scales the
vertical gradient component. ``dsqr`` rides ``ctx.params`` as a
traced-but-constant leaf, carried on that metric entry as a 0-d
``ScaledOperator`` coefficient; a *scalar* coefficient is translation-
invariant, so ``ScaledOperator.eigenvalues`` folds it into the diagonal
(only a *field* coefficient would break translation invariance and
raise). The weighted diagonal is inverted by the
realized-map composition ``backward @ weighted_lap.inverse() @
forward`` (S2 — ``SpectralSolve`` is a composition, not a bespoke
orchestrator), the ``k = 0`` nullspace regularized by the inverse's
exact structural-zero test (the mean-pressure gauge).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework2.grid.operators.composed import (
    Diag,
    Divergence,
    Gradient,
)
from fridom.framework2.grid.operators.realized import BoundTransform
from fridom.framework2.grid.spaces.constant import ConstantSpace

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.operators.base import Operator
    from fridom.framework2.grid.operators.symbol import Symbol
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


class SpectralPressureSolver:

    """Grid-bound spectral solve of ``lap(p) = div`` on a Fourier grid.

    Description
    -----------
    Constructed at trace time inside the projection stage from the
    operand's grid; carries no mutable state and is not a pytree leaf.
    At construction it resolves the transform, binds its forward /
    backward directions as :class:`BoundTransform` realized maps, and
    expands the grid's ``Divergence`` / ``Gradient`` blocks (capturing
    the component axis order); per solve it assembles the weighted
    Laplacian ``Div @ Diag(1, .., 1/dsqr) @ Grad`` as one block matmul,
    queries its symbol (live ``1/dsqr`` riding the vertical metric
    entry), and inverts it through the realized-map composition
    ``backward @ weighted_lap.inverse() @ forward``.

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
        """Bind the transform directions and the Div / Grad blocks."""
        bare = space.bare
        self._grid: object = grid
        self._vertical: str = vertical
        self._transform = grid.dispatch.resolve("transform", bare)
        self._coeff: SpaceLike = self._transform.codomain(bare)
        # the forward / backward transform directions as realized maps
        # (the fixed-tag halves of the spectral solve composition)
        self._forward = BoundTransform(self._transform, bare)
        self._backward = BoundTransform(
            self._transform, self._coeff, backward=True)
        # the component axis family (the diagonal metric's layout) and
        # the expanded Divergence / Gradient blocks, captured once; the
        # ``dsqr``-weighted metric is assembled per solve
        self._axes: tuple[str, ...] = tuple(
            name
            for factor in bare.factors
            if not isinstance(factor, ConstantSpace)
            for name in factor.names)
        self._grad: Operator = Gradient().expand(bare, grid.dispatch)
        mid = self._grad.codomain(bare)
        self._mid: tuple[SpaceLike, ...] = (
            mid if isinstance(mid, tuple) else (mid,))
        self._div: Operator = Divergence().expand(
            self._mid, grid.dispatch)

    def _laplacian_symbol(self, dsqr: jax.Array) -> Symbol:
        """Assemble ``-(k̂_x² + k̂_y² + k̂_z²/dsqr)`` on the coeff space."""
        # Div @ Diag(1, .., 1/dsqr) @ Grad: the live 1/dsqr rides the
        # vertical metric entry as a 0-d leaf (ScaledOperator carries
        # its symbol); the block matmul collapses to a 1x1 block whose
        # entry is the weighted discrete Laplacian
        diag = Diag({self._vertical: 1.0 / dsqr}, axes=self._axes)
        lap = (self._div @ diag @ self._grad).rows[0][0]
        return lap.eigenvalues(self._grid, self._coeff)

    def solve(
        self, div: object, *, dsqr: jax.Array,
    ) -> object:
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
        laplace = self._laplacian_symbol(dsqr)
        # the spectral solve *is* backward @ inverse @ forward (S2); the
        # inverse's exact structural-zero test regularizes the k = 0
        # nullspace (the mean-pressure gauge)
        solve = self._backward @ laplace.inverse() @ self._forward
        return solve(div)
