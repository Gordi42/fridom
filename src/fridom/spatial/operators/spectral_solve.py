r"""
``SpectralSolve``: the diagonal spectral elliptic solve, as a composition.

Description
-----------
Owning design: ``design/decisions/symbol_stack_design.md`` (the
realized-map algebra — ``SpectralSolve`` "is not a class, it is a
composition ``backward @ symbol.inverse() @ forward``") and
``design/plans/active/composition_refactor_plan.md`` (stage S2). The typed
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

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.base import EigenbasisError
from fridom.spatial.operators.mixed import resolve_transform
from fridom.spatial.operators.realized import BoundTransform
from fridom.spatial.operators.slab_fft import (
    SlabSolve,
    resolve_slab_plan,
    symbol_fits,
)
from fridom.spatial.operators.symbol import Symbol

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.operators.base import FieldLike, Operator
    from fridom.spatial.operators.mixed import ComposedTransform
    from fridom.spatial.operators.realized import RealizedMap
    from fridom.spatial.operators.transform import Transform
    from fridom.spatial.spaces.tensor_product import SpaceLike


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

    On a multi-device grid an eligible solve (pure unpadded Fourier
    transform, divisible extents — see ``operators/slab_fft.py``)
    additionally resolves a distributed :class:`SlabSolve`: the
    whole ``backward @ inverse @ forward`` runs slab-decomposed
    inside one ``jax.shard_map`` region, with the eigenvalue
    diagonal materialized on the plan's internal coefficient space
    and sliced per shard. Ineligible solves — and mismatched-layout
    operands — keep the replicated composite; on one device the
    program is bitwise unchanged.

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
        self._grid: object = grid
        self._elliptic: Operator | Symbol = elliptic
        self._where_zero: complex = where_zero
        self._domain: SpaceLike = bare
        self._transform: Transform | ComposedTransform = (
            resolve_transform(grid, bare))
        self._coeff: SpaceLike = self._transform.codomain(bare)
        self._inverse: Symbol | None = None
        self._composite: RealizedMap | None = None
        # the distributed slab pipeline (multi-device only; None on
        # one device, keeping the single-device program bitwise
        # unchanged) — see operators/slab_fft.py
        self._slab: SlabSolve | None = self._resolve_slab()
        if self._slab is None:
            self._materialize()

    def _resolve_slab(self) -> SlabSolve | None:
        """
        Resolve the distributed slab solve, or None (fallback).

        Description
        -----------
        The distributed path needs an :class:`Operator` recipe (a
        pre-assembled ``Symbol`` is bound to the replicated codomain
        layout), a resolvable :class:`SlabPlan`, and eigenvalues
        that materialize on the plan's internal coefficient space as
        an endomorphic broadcast-shaped diagonal; anything else
        falls back to the replicated composite.
        """
        if isinstance(self._elliptic, Symbol):
            return None
        plan = resolve_slab_plan(self._grid, self._domain)
        if plan is None:
            return None
        try:
            symbol = self._elliptic.eigenvalues(self._grid,
                                                plan.coeff)
        except (EigenbasisError, SpaceMismatchError):
            return None
        if not symbol_fits(plan, symbol):
            return None
        return SlabSolve(plan, symbol.inverse(self._where_zero))

    def _materialize(self) -> None:
        """Build the replicated ``backward @ inverse @ forward``."""
        symbol = (self._elliptic
                  if isinstance(self._elliptic, Symbol)
                  else self._elliptic.eigenvalues(self._grid,
                                                  self._coeff))
        self._inverse = symbol.inverse(self._where_zero)
        forward = BoundTransform(self._transform, self._domain)
        backward = BoundTransform(self._transform, self._coeff,
                                  backward=True)
        # SpectralSolve *is* this composition (symbol_stack_design.md):
        # backward @ inverse @ forward, a lazy RealizedComposite
        self._composite = backward @ self._inverse @ forward

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
        if self._inverse is None:
            self._materialize()
        return self._inverse

    @property
    def composite(self) -> RealizedMap:
        """
        The realized-map chain ``backward @ inverse @ forward``.

        Description
        -----------
        Built lazily when a distributed slab solve is active (the
        replicated chain then only serves mismatched-layout
        operands); on a single device it is built eagerly at
        construction, exactly as before.
        """
        if self._composite is None:
            self._materialize()
        return self._composite

    @property
    def slab(self) -> SlabSolve | None:
        """The distributed slab solve, or None (replicated path)."""
        return self._slab

    # ================================================================
    #  Application
    # ================================================================
    def __call__(self, rhs: FieldLike) -> FieldLike:
        """
        Solve ``elliptic(x) = rhs`` for ``x`` (diagonal, exact).

        Description
        -----------
        With an active distributed slab solve and a matching operand
        (same bare space and layout) the whole pipeline runs inside
        one ``jax.shard_map`` region — no device ever gathers the
        spectral cube; otherwise the replicated composite applies.

        Parameters
        ----------
        rhs : FieldLike
            The right-hand-side field on the solve's operand space.

        Returns
        -------
        FieldLike
            The solution on the same space (real for a real transform).
        """
        if self._slab is not None and self._slab.applies(rhs):
            return self._slab(rhs)
        return self.composite(rhs)

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
