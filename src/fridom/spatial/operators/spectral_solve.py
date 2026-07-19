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
from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.fields.storage import storage_dtype
from fridom.spatial.operators.distributed_solve import (
    resolve_distributed_solve,
)
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
    from fridom.spatial.operators.distributed_solve import SlabSolve
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

    On a multi-device grid an eligible solve (unpadded
    Fourier/Sine/Cosine transforms — including the mixed walled
    product — and divisible extents) additionally resolves a
    distributed fused solve **through the ordinary transform's layout
    plan** (see ``operators/distributed_solve.py``): the whole
    ``backward @ inverse @ forward`` runs slab-decomposed inside one
    ``jax.shard_map`` region, with the eigenvalue diagonal materialized
    on the plan's internal coefficient space and sliced per shard.
    Ineligible solves — and mismatched-layout operands — keep the
    replicated composite; on one device the program is bitwise
    unchanged.

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
        periodic (all-Fourier) solve. Applies to the replicated
        composite only: on a multi-device grid the distributed slab
        solve (full precision) takes precedence — since the mixed
        (walled) transform distributes too, the flag is a no-op on
        any eligible multi-device solve (a single-precision
        distributed solve is future work). Off by default (bitwise
        identical to the full-precision solve); on, the solution
        carries the reduced
        round-off, an opt-in accuracy trade (default: False).
    allow_replicated : bool, optional
        Escape hatch for the **replicated composite** on a
        multi-device grid: when no distributed slab solve resolves
        (a block-diagonal Chebyshev-vertical solve, a family outside
        Fourier/Sine/Cosine, or a symbol that refuses the distributed
        spectral frame) the composite's naive ``forward`` would trip
        the GSPMD transform illegality guard
        (``Transform._reject_sharded_transform`` /
        ``_reject_replicating_transform``) — a silent all-gather is
        illegal by design
        (``design/research/gspmd_naive_transform_illegality.md``).
        With this flag the solve performs an **explicit, honest
        replicate-then-compute**: it gathers the operand to the
        replicated layout (``Layout({})``), applies the composite
        there (every axis local, so the guard is satisfied), and
        reshards the solution back to the operand's layout. The gather
        materializes the whole cube on every device (numerically exact
        but unscalable), so this is a deliberate opt-in for the
        irreducible cases, not the scalable path — an eligible solve
        still takes the distributed slab. On one device (or a
        replicated operand) it is a no-op, bitwise identical to the
        plain composite (default: False).
    """

    def __init__(
        self,
        elliptic: Operator | Symbol,
        grid: object,
        space: SpaceLike,
        *,
        where_zero: complex = 0.0,
        single_precision: bool = False,
        allow_replicated: bool = False,
    ) -> None:
        """Materialize the inverse symbol and compose the solve chain."""
        bare = space.bare
        self._single_precision: bool = bool(single_precision)
        self._allow_replicated: bool = bool(allow_replicated)
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
        # unchanged) — see operators/distributed_solve.py. The distributed
        # solve runs full precision; ``single_precision`` applies to
        # the replicated composite path only.
        self._slab: SlabSolve | None = self._resolve_slab()
        if self._slab is None:
            self._materialize()

    def _resolve_slab(self) -> SlabSolve | None:
        """
        Resolve the distributed solve, or None (replicated fallback).

        Description
        -----------
        Distribution now flows through the ordinary transform: the fused
        pipeline's geometry is resolved from the transform's
        ``distributed_forward_plan``
        (``operators/distributed_solve.py``) rather than a solve-scoped
        ``resolve_slab_plan``. A pre-assembled ``Symbol`` (bound to the
        replicated codomain) and any operand the transform plan declines
        keep the replicated composite.
        """
        if isinstance(self._elliptic, Symbol):
            return None
        return resolve_distributed_solve(
            self._elliptic, self._transform, self._grid,
            self._domain, self._where_zero)

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
        if self._single_precision:
            # single precision: cast the operand to float32 (so rfftn
            # runs single), downcast the c128 half-spectrum to
            # complex64 (the transform's _deliver re-widens it
            # otherwise), and apply a complex64 inverse diagonal — the
            # backward irfftn then lands back on float64.
            inv = self._inverse
            inverse = Symbol(
                inv.space, inv.data.astype(_reduced_dtype(inv.space)),
                codomain=inv.codomain)
            cast_operand = _CastMap(
                self._domain, _reduced_dtype(self._domain))
            cast_spectrum = _CastMap(
                self._coeff, _reduced_dtype(self._coeff))
            self._composite = (
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
    def single_precision(self) -> bool:
        """Whether the transform pair and divide run in float32/c64."""
        return self._single_precision

    @property
    def allow_replicated(self) -> bool:
        """Whether the composite may gather to a replicated layout."""
        return self._allow_replicated

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
        if self._allow_replicated:
            return self._replicated_composite(rhs)
        return self.composite(rhs)

    def _replicated_composite(self, rhs: FieldLike) -> FieldLike:
        """
        Apply the composite via an explicit replicate-then-compute.

        Description
        -----------
        The ``allow_replicated`` escape: gather the operand to the
        replicated layout (``Layout({})``), run the composite there
        (every axis device-local, so the naive transform's guard is
        satisfied), and reshard the solution back to the operand's own
        layout. A single-device / already-replicated / layout-free
        operand needs no gather and applies the composite directly —
        bitwise the plain path.

        Parameters
        ----------
        rhs : FieldLike
            The right-hand-side field.

        Returns
        -------
        FieldLike
            The solution on the operand's own space and layout.
        """
        layout = rhs.function_space.layout
        replicated = Layout({})
        if (layout is None or layout == replicated
                or getattr(self._grid.decomposition,
                           "device_count", 1) <= 1):
            return self.composite(rhs)
        solution = self.composite(rhs.reshard(replicated))
        return solution.reshard(layout)

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
