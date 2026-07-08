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

The vertical ``1/dsqr`` weight lives in the operator (``Diag``); but
``dsqr`` rides ``ctx.params`` as a traced-but-constant leaf, so the
scaling happens at the **symbol** level — the assembled vertical term
is multiplied by a degenerate all-``Constant`` coefficient field
(``Symbol x field``, the ``dsqr`` case), never via
``ScaledOperator.eigenvalues`` (which rightly refuses a non-constant
operator coefficient). The weighted diagonal is handed to a grid-bound
``SpectralSolve`` (forward transform → ``Symbol.inverse(where_zero=0)``
→ backward), the ``k = 0`` nullspace regularized by its exact
structural-zero test (the mean-pressure gauge).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.framework2.grid.fields.metadata import FieldMetadata
from fridom.framework2.grid.fields.scalar_field import ScalarField
from fridom.framework2.grid.fields.storage import store
from fridom.framework2.grid.operators.base import OperatorSum
from fridom.framework2.grid.operators.composed import Laplacian
from fridom.framework2.grid.operators.spectral_solve import SpectralSolve
from fridom.framework2.grid.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.operators.base import Operator
    from fridom.framework2.grid.operators.symbol import Symbol
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


def _constant_field(
    grid: object, coeff: SpaceLike, value: jax.Array,
) -> ScalarField:
    """
    Wrap a (possibly traced) scalar as an all-``Constant`` field.

    Description
    -----------
    The degenerate ``Symbol x field`` coefficient (symbol_stack_design.md
    §"``Symbol x field``"): a field constant on every axis, so it is
    trivially ``Constant`` on the symbol's transformed factors and
    scales the diagonal by ``value`` in the mixed representation. The
    array is a dynamic leaf, so a traced ``value`` (``1/dsqr``) flows
    through untouched.

    Parameters
    ----------
    grid : object
        The grid mediating the decomposition.
    coeff : SpaceLike
        The coefficient space whose factor family the field mirrors.
    value : jax.Array
        The (possibly traced) scalar coefficient.

    Returns
    -------
    ScalarField
        The all-``Constant`` coefficient field carrying ``value``.
    """
    bare = TensorProductSpace.of(
        *(factor.mesh.constant for factor in coeff.factors))
    data = jnp.broadcast_to(
        jnp.asarray(value, dtype=dtype_real()), (1,) * len(coeff.factors))
    stored = store(grid.decomposition, bare, data)
    return ScalarField(grid, bare, stored,
                       FieldMetadata.create(name="dsqr_metric"))


class SpectralPressureSolver:

    """Grid-bound spectral solve of ``lap(p) = div`` on a Fourier grid.

    Description
    -----------
    Constructed at trace time inside the projection stage from the
    operand's grid; carries no mutable state and is not a pytree leaf.
    At construction it resolves the transform and expands the grid's
    ``Divergence @ Gradient`` into its per-axis discrete
    second-difference terms; per solve it assembles the ``dsqr``-weighted
    Laplacian ``Symbol`` (live ``1/dsqr`` on the vertical term) and
    inverts it through a ``SpectralSolve``.

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
        """Bind the transform and the per-axis Laplacian terms."""
        bare = space.bare
        self._grid: object = grid
        self._vertical: str = vertical
        self._transform = grid.dispatch.resolve("transform", bare)
        self._coeff: SpaceLike = self._transform.codomain(bare)
        # div @ grad collapses to a 1x1 block whose entry is the sum of
        # per-axis ``bwd @ fwd`` chains (the discrete Laplacian)
        entry: Operator = Laplacian().expand(
            bare, grid.dispatch).rows[0][0]
        self._terms: tuple[Operator, ...] = (
            entry.terms if isinstance(entry, OperatorSum) else (entry,))

    def _laplacian_symbol(self, dsqr: jax.Array) -> Symbol:
        """Assemble ``-(k̂_x² + k̂_y² + k̂_z²/dsqr)`` on the coeff space."""
        inv_dsqr = _constant_field(self._grid, self._coeff, 1.0 / dsqr)
        total: Symbol | None = None
        for term in self._terms:
            sym = term.eigenvalues(self._grid, self._coeff)
            if term.bound_axis == self._vertical:
                # the traced 1/dsqr scaled in at the symbol level
                sym = sym * inv_dsqr
            total = sym if total is None else total + sym
        return total

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
        # ``SpectralSolve`` wraps forward -> Symbol.inverse(0) -> backward;
        # the exact structural-zero test regularizes the k = 0 nullspace
        # (the mean-pressure gauge)
        return SpectralSolve(laplace, self._grid, div.function_space)(div)
