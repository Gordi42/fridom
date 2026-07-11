r"""
``SpectralSolve``: the diagonal spectral elliptic solve, as a composition.

Description
-----------
Owning design: ``notes/framework2/symbol_stack_design.md`` (the
realized-map algebra — ``SpectralSolve`` "is not a class, it is a
composition ``backward @ symbol.inverse() @ forward``") and
``notes/framework2/composition_refactor_plan.md`` (stage S2). The typed
port of the inline ``test_spectral_poisson`` pattern and the
hand-rolled v1 spectral pressure solver: a grid-bound solve that
inverts an elliptic operator whose every factor diagonalizes in the
transform (Fourier / sine / cosine) basis.

At construction it materializes the inverse symbol once —
``elliptic.eigenvalues(grid, coeff_space).inverse(where_zero)`` — and
composes the realized-map chain ``backward @ inverse @ forward`` into a
:class:`RealizedComposite`, so the per-call work is only the transform
pair and a Hadamard multiply:

.. code-block:: python

    p = SpectralSolve(laplacian, grid, div.function_space)(div)

The object is now a thin constructor over the composition (S2): it
holds the composite and delegates application to it. Iteration-1 scope
is the **pure-diagonal** partition (all Fourier / sine / cosine
factors); the mixed ``Fourier(x, y) x Chebyshev(z)`` block-diagonal
case (a per-mode banded z-solve on the lifted ``grid.operators.banded``
kernel) is designed-for and deferred.
"""
# S2: SpectralSolve reframed as backward @ inverse @ forward composition
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework2.grid.operators.mixed import resolve_transform
from fridom.framework2.grid.operators.realized import BoundTransform
from fridom.framework2.grid.operators.symbol import Symbol

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.operators.base import FieldLike, Operator
    from fridom.framework2.grid.operators.mixed import ComposedTransform
    from fridom.framework2.grid.operators.realized import RealizedMap
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
    space, pseudo-inverts it, and composes the realized-map chain
    ``BoundTransform(backward) @ inverse @ BoundTransform(forward)`` into
    a :class:`RealizedComposite`. Applying it threads the right-hand side
    through that composite (forward-transform, Hadamard by the inverse
    symbol, backward-transform) — the fractional-step Poisson / Helmholtz
    solve, now the realized-map composition rather than a hand-rolled
    orchestration.

    The nullspace (the Poisson ``k = 0`` mode, the real-FFT
    Nyquist-zeroed modes) is regularized to ``where_zero`` by the
    exact structural-zero test of ``Symbol.inverse`` (no caller-side
    masking); Helmholtz ``(nabla^2 - lambda)`` with ``lambda != 0`` has
    no nullspace and inverts everywhere.

    Parameters
    ----------
    elliptic : Operator | Symbol
        The elliptic operator to invert (every factor must diagonalize
        in the transform basis, carrying an ``eigenvalues`` symbol on
        the coefficient space), or a pre-assembled coefficient-space
        ``Symbol``. The symbol form is the seam for a metric that
        cannot fold into a static operator — the nonhydro pressure
        Laplacian, whose ``1/dsqr`` vertical weight is a *traced*
        (``ctx.params``) leaf and so is scaled in at the symbol level
        (``Symbol x field``), not via ``ScaledOperator``
        (symbol_stack_design.md decision 2).
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
        elliptic: Operator | Symbol,
        grid: object,
        space: SpaceLike,
        *,
        where_zero: complex = 0.0,
    ) -> None:
        """Materialize the inverse symbol and compose the solve chain."""
        bare = space.bare
        self._transform: Transform | ComposedTransform = (
            resolve_transform(grid, bare))
        coeff = self._transform.codomain(bare)
        symbol = (elliptic if isinstance(elliptic, Symbol)
                  else elliptic.eigenvalues(grid, coeff))
        self._inverse: Symbol = symbol.inverse(where_zero)
        forward = BoundTransform(self._transform, bare)
        backward = BoundTransform(self._transform, coeff, backward=True)
        # SpectralSolve *is* this composition (symbol_stack_design.md):
        # backward @ inverse @ forward, a lazy RealizedComposite
        self._composite: RealizedMap = backward @ self._inverse @ forward

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def transform(self) -> Transform | ComposedTransform:
        """The bound nodal <-> coefficient transform."""
        return self._transform

    @property
    def inverse_symbol(self) -> Symbol:
        """The materialized inverse diagonal (the per-mode ``1/lambda``)."""
        return self._inverse

    @property
    def composite(self) -> RealizedMap:
        """The realized-map chain ``backward @ inverse @ forward``."""
        return self._composite

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
        return self._composite(rhs)

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
