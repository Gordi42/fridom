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

import jax.numpy as jnp

import fridom.framework as fr
from fridom.spatial.fields.storage import storage_dtype
from fridom.spatial.operators.mixed import resolve_transform
from fridom.spatial.operators.realized import (
    BoundTransform,
    realized_matmul,
    realized_rmatmul,
    realized_sum,
)
from fridom.spatial.operators.symbol import Symbol

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.operators.base import FieldLike, Operator
    from fridom.spatial.operators.mixed import ComposedTransform
    from fridom.spatial.operators.realized import RealizedMap
    from fridom.spatial.operators.transform import Transform
    from fridom.spatial.spaces.tensor_product import SpaceLike


def _reduced_dtype(space: SpaceLike) -> jnp.dtype:
    """
    Single-precision twin of a space's derived storage dtype.

    Description
    -----------
    ``complex64`` for a complex/Fourier space, ``float32`` for a real
    one — the reduced-precision solve's per-space cast target.

    Parameters
    ----------
    space : SpaceLike
        The (bare) function space whose storage dtype to reduce.

    Returns
    -------
    jnp.dtype
        ``complex64`` or ``float32``.
    """
    full = storage_dtype(space)
    if jnp.issubdtype(full, jnp.complexfloating):
        return jnp.dtype(jnp.complex64)
    return jnp.dtype(jnp.float32)


@fr.utils.jaxify
class _CastMap:

    r"""
    Endo realized map casting a field's data to a fixed dtype.

    Description
    -----------
    The single-precision-solve seam. The transform's ``_deliver``
    re-promotes every stage's output to the *space-derived* storage
    dtype (``complex128`` on a Fourier coefficient space), so a bare
    ``rfftn`` in ``float32`` is silently widened back to
    ``complex128`` before the spectral divide — erasing the win.
    Inserting a ``_CastMap`` right after the forward transform
    downcasts the half-spectrum to ``complex64`` so the divide and
    the backward ``irfftn`` run in single precision, and a second one
    at the innermost position downcasts the real operand to
    ``float32`` so the forward ``rfftn`` itself runs single. Domain
    and codomain are the same coefficient tag (a cast changes only
    the array width, not the space), so it composes transparently in
    the ``backward @ inverse @ cast @ forward @ cast`` chain. Carries
    no dynamic leaves; the target dtype is static treedef aux.

    Parameters
    ----------
    space : SpaceLike
        The (bare) space the cast acts on (its fixed domain and
        codomain tag).
    dtype : object
        The target array dtype (e.g. ``jnp.complex64``).
    """

    def __init__(self, space: SpaceLike, dtype: object) -> None:
        """Bind the endo space tag and the target dtype."""
        self._space: SpaceLike = space.bare
        self._dtype: jnp.dtype = jnp.dtype(dtype)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def domain(self) -> SpaceLike:
        """The fixed domain tag (equals the codomain)."""
        return self._space

    @property
    def codomain(self) -> SpaceLike:
        """The fixed codomain tag (equals the domain)."""
        return self._space

    @property
    def dtype(self) -> jnp.dtype:
        """The target cast dtype."""
        return self._dtype

    # ================================================================
    #  Application
    # ================================================================
    def __call__(self, f: FieldLike) -> FieldLike:
        """Return ``f`` with its data cast to the target dtype."""
        return f.with_data(f.data.astype(self._dtype))

    # ================================================================
    #  Algebra
    # ================================================================
    def __matmul__(self, other: object) -> RealizedMap:
        """Compose ``self @ other`` (flatten, fuse, typecheck)."""
        return realized_matmul(self, other)

    def __rmatmul__(self, other: object) -> RealizedMap:
        """Reflected ``other @ self`` (materialization guard)."""
        return realized_rmatmul(self, other)

    def __add__(self, other: object) -> RealizedMap:
        """Sum ``self + other`` (common-signature)."""
        return realized_sum(self, other)

    def __radd__(self, other: object) -> RealizedMap:
        """Reflected sum ``other + self``."""
        return realized_sum(other, self)

    def inverse(self, where_zero: complex = 0.0) -> RealizedMap:
        """
        Raise: a width cast has no meaningful realized-map inverse.

        Parameters
        ----------
        where_zero : complex, optional
            Unused; present for the uniform ``inverse`` signature
            (default: 0.0).

        Returns
        -------
        RealizedMap
            Never returns.
        """
        raise NotImplementedError(
            "a _CastMap is not inverted (the reduced-precision solve "
            "composite is applied forward only)")

    def conj(self) -> _CastMap:
        """Return ``self`` (a real-linear cast is its own conjugate)."""
        return self


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
    single_precision : bool, optional
        Run the transform pair and the spectral divide in single
        precision (``float32`` / ``complex64``) while the operand and
        the returned solution stay ``dtype_real()`` (``float64``) — a
        performance option for the bandwidth-/FFT-bound solve. The
        operand is cast to ``float32`` before the forward ``rfftn``,
        the half-spectrum is downcast to ``complex64`` for the divide
        and the backward ``irfftn``, and the inverse eigenvalue
        diagonal is materialized once in ``complex64``; the backward
        transform lands back on ``dtype_real()``. On a mixed (walled)
        transform only the Fourier stages and the spectrum-level
        divide are single precision — the trig stages re-widen at
        their storage boundary — so the win is largest on a fully
        periodic (all-Fourier) solve. Off by default (bitwise
        identical to the full-precision solve); on, the solution
        carries the reduced round-off, an opt-in accuracy trade
        (default: False).
    """

    def __init__(
        self,
        elliptic: Operator | Symbol,
        grid: object,
        space: SpaceLike,
        *,
        where_zero: complex = 0.0,
        single_precision: bool = False,
    ) -> None:
        """Materialize the inverse symbol and compose the solve chain."""
        bare = space.bare
        self._single_precision: bool = bool(single_precision)
        self._transform: Transform | ComposedTransform = (
            resolve_transform(grid, bare))
        coeff = self._transform.codomain(bare)
        symbol = (elliptic if isinstance(elliptic, Symbol)
                  else elliptic.eigenvalues(grid, coeff))
        self._inverse: Symbol = symbol.inverse(where_zero)
        forward = BoundTransform(self._transform, bare)
        backward = BoundTransform(self._transform, coeff, backward=True)
        if self._single_precision:
            # SpectralSolve in single precision: cast the operand to
            # float32 (so rfftn runs single), downcast the c128
            # half-spectrum to complex64 (the transform's _deliver
            # re-widens it otherwise), and apply a complex64 inverse
            # diagonal — the backward irfftn then lands on float64.
            inv = self._inverse
            inverse: Symbol = Symbol(
                inv.space, inv.data.astype(_reduced_dtype(inv.space)),
                codomain=inv.codomain)
            cast_operand = _CastMap(bare, _reduced_dtype(bare))
            cast_spectrum = _CastMap(coeff, _reduced_dtype(coeff))
            self._composite: RealizedMap = (
                backward @ inverse @ cast_spectrum
                @ forward @ cast_operand)
        else:
            # SpectralSolve *is* this composition (symbol_stack_design):
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
        return self._inverse

    @property
    def composite(self) -> RealizedMap:
        """The realized-map chain ``backward @ inverse @ forward``."""
        return self._composite

    @property
    def single_precision(self) -> bool:
        """Whether the transform pair and divide run in float32/c64."""
        return self._single_precision

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
