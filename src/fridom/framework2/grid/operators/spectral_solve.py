r"""
``SpectralSolve``: the diagonal spectral elliptic solver.

Description
-----------
Owning design: ``notes/framework2/operator_symbols_plan.md`` section 5
(the diagonal -> banded boundary) and
``notes/framework2/projection_eigenmode_roadmap.md`` Phase D. The typed
port of the inline ``test_spectral_poisson`` pattern and the
hand-rolled v1 spectral pressure solver: a grid-bound operator that
inverts an elliptic operator whose every factor diagonalizes in the
transform (Fourier / sine / cosine) basis.

At construction it materializes the inverse symbol once —
``elliptic.eigenvalues(grid, coeff_space).inverse(where_zero)`` — so the
per-call work is only the transform pair and a Hadamard multiply:

.. code-block:: python

    p = SpectralSolve(laplacian, grid, div.function_space)(div)

Iteration-1 scope is the **pure-diagonal** partition (all Fourier /
sine / cosine factors); the mixed ``Fourier(x, y) x Chebyshev(z)``
block-diagonal case (a per-mode banded z-solve on the lifted
``grid.operators.banded`` kernel) is designed-for and deferred.
"""
# Wave 9B: SpectralSolve (pure-diagonal; banded partition designed-for)
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.operators.base import FieldLike, Operator
    from fridom.framework2.grid.operators.symbol import Symbol
    from fridom.framework2.grid.operators.transform import Transform
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


class SpectralSolve:

    r"""
    Grid-bound diagonal solve of an elliptic operator ``elliptic``.

    Description
    -----------
    Constructed once (a setup / trace-time object, carrying no mutable
    state and not a pytree leaf): it resolves the transform for the
    operand space, queries ``elliptic``'s symbol on the coefficient
    space, and stores its pseudo-inverse. Applying it forward-transforms
    the right-hand side, Hadamard-multiplies by the inverse symbol, and
    backward-transforms — the fractional-step Poisson / Helmholtz solve.

    The nullspace (the Poisson ``k = 0`` mode, the real-FFT
    Nyquist-zeroed modes) is regularized to ``where_zero`` by the
    exact structural-zero test of ``Symbol.inverse`` (no caller-side
    masking); Helmholtz ``(nabla^2 - lambda)`` with ``lambda != 0`` has
    no nullspace and inverts everywhere.

    Parameters
    ----------
    elliptic : Operator
        The elliptic operator to invert; every factor must diagonalize
        in the transform basis (it carries an ``eigenvalues`` symbol on
        the coefficient space).
    grid : Grid
        The grid mediating the transform, wavenumbers, and measures.
    space : SpaceLike
        The (nodal) function space of the operand the solve consumes.
    where_zero : complex, optional
        The inverse value at structural zeros of the symbol — the
        nullspace gauge (default: 0.0, the mean-free Poisson gauge).
    """

    def __init__(
        self,
        elliptic: Operator,
        grid: object,
        space: SpaceLike,
        *,
        where_zero: complex = 0.0,
    ) -> None:
        """Resolve the transform and materialize the inverse symbol."""
        bare = space.bare
        self._transform: Transform = grid.dispatch.resolve(
            "transform", bare)
        coeff = self._transform.codomain(bare)
        self._inverse: Symbol = elliptic.eigenvalues(
            grid, coeff).inverse(where_zero)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def transform(self) -> Transform:
        """The bound nodal <-> coefficient transform."""
        return self._transform

    @property
    def inverse_symbol(self) -> Symbol:
        """The materialized inverse diagonal (the per-mode ``1/lambda``)."""
        return self._inverse

    # ================================================================
    #  Application
    # ================================================================
    def __call__(self, rhs: FieldLike) -> FieldLike:
        """
        Solve ``elliptic(x) = rhs`` for ``x`` (diagonal, exact).

        Parameters
        ----------
        rhs : FieldLike
            The right-hand-side field on the solve's operand space.

        Returns
        -------
        FieldLike
            The solution on the same space (real for a real transform).
        """
        rhs_hat = self._transform.forward(rhs)
        return self._transform.backward(self._inverse(rhs_hat))

    def solve(self, rhs: FieldLike) -> FieldLike:
        """
        Alias of ``__call__`` — solve ``elliptic(x) = rhs``.

        Parameters
        ----------
        rhs : FieldLike
            The right-hand-side field on the solve's operand space.

        Returns
        -------
        FieldLike
            The solution on the same space.
        """
        return self(rhs)
