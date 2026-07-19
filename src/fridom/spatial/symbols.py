r"""
``GridSymbols``: the per-component operator-symbol kit of a grid.

Description
-----------
A thin, physics-free convenience layer over the eigenvalue queries:
a model names its staggered components once (``u``, ``v``, ``w``,
``p``, ...) and the kit threads every symbol / transform query
through the right per-component coefficient space — the
``SpectralSolve`` resolution pattern (``grid.dispatch.resolve`` on
the operand factor, ``eigenvalues`` on the transform codomain),
packaged so eigenmode and projection assemblies stop repeating it.

``rayleigh_dual`` builds the Rayleigh dual family of a diagonal
column ``q`` (the biorthonormal row ``p`` with
``sum_c conj(p_c) q_c == 1`` off the structural nullspace) in the
symbol algebra alone.

``ModeChart`` aligns per-component coefficient data across the
bounded trig families' different slot lattices: it embeds each
component's ``(mode_offset, shape)`` layout onto the shared
``0..n`` union mode lattice (and restricts back), identity on
periodic / Fourier axes.
"""
from __future__ import annotations

from functools import reduce
from operator import matmul
from typing import TYPE_CHECKING

from fridom.spatial.operators.base import Identity
from fridom.spatial.operators.mixed import resolve_transform
from fridom.spatial.operators.realized import BoundTransform
from fridom.spatial.operators.symbol import _reindex
from fridom.spatial.spaces.average import AverageSpace
from fridom.spatial.spaces.coefficient import (
    CosineSpace,
    SineSpace,
)
from fridom.spatial.spaces.nodal import NodalSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.spatial.grid import Grid
    from fridom.spatial.operators.mixed import (
        ComposedTransform,
    )
    from fridom.spatial.operators.symbol import Symbol
    from fridom.spatial.operators.transform import Transform
    from fridom.spatial.spaces.tensor_product import SpaceLike


class GridSymbols:

    r"""
    Operator symbols of a grid, threaded per component.

    Description
    -----------
    Binds a grid and a named family of (staggered) operand spaces —
    e.g. ``{"u": right * center, "p": center * center}`` — and
    answers every symbol query on the matching **coefficient** space:
    the transform is resolved once per component at construction
    (``resolve_transform(grid, space.bare)``, which composes the
    per-family transforms of a mixed walled product), and
    ``diff`` / ``interp`` / ``move`` thread
    ``transform.codomain(space.bare)`` through the operators'
    ``eigenvalues``, exactly like ``SpectralSolve``. No physics lives
    here: the kit only names the plumbing.

    Parameters
    ----------
    grid : Grid
        The grid mediating dispatch, transforms, and wavenumbers.
    spaces : Mapping[str, SpaceLike]
        The named component operand spaces (stored bare).
    coeff_spaces : Mapping[str, SpaceLike] | None, optional
        A per-component **coefficient-frame override**: when given,
        :meth:`coeff` (and hence every ``diff`` / ``interp`` symbol,
        which threads its eigenvalue query on that frame) reads the
        supplied space instead of the transform's own
        ``codomain(space)``. This is the frame hook the distributed
        eigenmode route uses to rebuild the operator symbols on the
        transpose engine's internal coefficient frame (the half axis
        re-designated). ``forward`` / ``backward`` keep the real
        transform and are undefined against an override frame — the
        distributed route never calls them, it fuses the transform
        inside a ``jax.shard_map`` region instead (default: None).
    """

    def __init__(
        self, grid: Grid, spaces: Mapping[str, SpaceLike],
        coeff_spaces: Mapping[str, SpaceLike] | None = None,
    ) -> None:
        """Intern the bare spaces and resolve their transforms."""
        self._grid: Grid = grid
        self._spaces: dict[str, SpaceLike] = {
            name: space.bare for name, space in spaces.items()}
        self._transforms: dict[
            str, Transform | ComposedTransform] = {
            name: resolve_transform(grid, bare)
            for name, bare in self._spaces.items()}
        self._coeff_override: dict[str, SpaceLike] | None = (
            None if coeff_spaces is None
            else {name: space.bare
                  for name, space in coeff_spaces.items()})

    # ================================================================
    #  Spaces and transforms
    # ================================================================
    def coeff(self, name: str) -> SpaceLike:
        """
        Coefficient space of a component (the transform codomain).

        Description
        -----------
        The override frame supplied at construction (the distributed
        eigenmode route's internal coefficient frame), else the
        transform's own ``codomain(space.bare)``.

        Parameters
        ----------
        name : str
            The component name.

        Returns
        -------
        SpaceLike
            The component's coefficient space (override or derived).
        """
        known = self._known(name)
        if self._coeff_override is not None:
            return self._coeff_override[known]
        return self._transforms[known].codomain(self._spaces[known])

    def forward(self, name: str) -> BoundTransform:
        """
        Bound forward transform of a component (nodal -> coefficient).

        Parameters
        ----------
        name : str
            The component name.

        Returns
        -------
        BoundTransform
            The forward direction on the component's bare space.
        """
        return BoundTransform(self._transforms[self._known(name)],
                              self._spaces[name])

    def backward(self, name: str) -> BoundTransform:
        """
        Bound backward transform of a component (coefficient -> nodal).

        Parameters
        ----------
        name : str
            The component name.

        Returns
        -------
        BoundTransform
            The backward direction on the component's coefficient
            space.
        """
        return BoundTransform(self._transforms[self._known(name)],
                              self.coeff(name), backward=True)

    # ================================================================
    #  Per-axis symbols
    # ================================================================
    def diff(self, axis: str, *, on: str) -> Symbol:
        """
        First-derivative symbol along ``axis``, threaded on ``on``.

        Parameters
        ----------
        axis : str
            The coordinate the derivative acts along.
        on : str
            The component whose coefficient space the symbol reads.

        Returns
        -------
        Symbol
            The (retagging) derivative diagonal.
        """
        return self._axis_symbol("diff", axis, on)

    def interp(self, axis: str, *, on: str) -> Symbol:
        """
        Interpolation symbol along ``axis``, threaded on ``on``.

        Parameters
        ----------
        axis : str
            The coordinate the interpolation acts along.
        on : str
            The component whose coefficient space the symbol reads.

        Returns
        -------
        Symbol
            The (retagging) averaging diagonal.
        """
        return self._axis_symbol("interpolate", axis, on)

    def move(self, frm: str, to: str) -> Symbol:
        """
        Staggering symbol moving component ``frm`` onto ``to``.

        Description
        -----------
        Composes (``@``) the per-axis interpolation symbols over
        every axis where the two components' bare factors differ —
        the tensor-product corner move. When no factor differs the
        result is the all-ones identity diagonal (the neutral of
        ``@``), so ``move`` is uniformly composable.

        Parameters
        ----------
        frm : str
            The source component name.
        to : str
            The target component name.

        Returns
        -------
        Symbol
            The composed inter-component staggering diagonal.
        """
        src = self._spaces[self._known(frm)]
        dst = self._spaces[self._known(to)]
        axes = [axis for axis in src.names
                if src.factor(axis) is not dst.factor(axis)]
        if not axes:
            return Identity().eigenvalues(self._grid, self.coeff(frm))
        return reduce(matmul, (self.interp(axis, on=frm)
                               for axis in axes))

    # ================================================================
    #  Internal helpers
    # ================================================================
    def _known(self, name: str) -> str:
        """Validate a component name (helpful ``KeyError``)."""
        if name not in self._spaces:
            known = ", ".join(sorted(self._spaces))
            raise KeyError(
                f"unknown component {name!r}: this kit threads "
                f"components {known}")
        return name

    def _axis_symbol(self, kind: str, axis: str, on: str) -> Symbol:
        """Resolve ``kind`` on the factor, query on the coeff space.

        Description
        -----------
        FV C-grid staggering (FV-D3, stage F3): an ``"interpolate"``
        along a **nodal face** factor of an **average-family** field
        (a C-grid velocity, ``CellAvg`` transverse ⊗ ``Right`` normal)
        must land on the cell average (``Right -> CellAvg``), matching
        the ``FluxDifference`` diff leg — so the symbol composes with
        the average-origin cell symbols. That is the ``"average"``
        reconstruct kind, not the nodal ``("interpolate", Right) ->
        Center`` row (which the global registry keeps, since a nodal
        scalar's ``.to`` on the same grid still needs it). The
        redirect is inferred per field, so no grid override is
        required and the mixed corner is untouched. Nodal fields (no
        average factor) and average-origin factors (the ``G4``
        ``("interpolate", CellAvg) -> Right`` row) are unaffected.
        """
        grid = self._grid
        space = self._spaces[self._known(on)]
        factor = space.factor(axis)
        if (kind == "interpolate"
                and isinstance(factor, NodalSpace)
                and any(isinstance(fac, AverageSpace)
                        for fac in space.factors)):
            kind = "average"
        op = grid.dispatch.resolve(kind, factor)
        return op[axis].eigenvalues(grid, self.coeff(on))


class ModeChart:

    r"""
    Union mode lattice of the bounded trig families, per axis.

    Description
    -----------
    On a walled axis the per-component coefficient factors live on
    *different* slot lattices — DST-I holds modes ``1..n-1``, DST-II
    modes ``1..n``, DCT-II modes ``0..n-1``, DCT-I modes ``0..n`` —
    while cross-component accumulations (eigen-projector amplitudes,
    biorthonormality sums) are indexed by the *physical* mode. The
    chart is the shared ``0..n`` union lattice per bounded axis:
    ``embed`` moves component-layout data onto the union lattice
    (modes the component lacks are exact zero-fills), ``restrict``
    moves union data back (surplus modes drop). Both are static
    pad/slice moves derived from ``(mode_offset, shape)`` — the same
    index maps as the derived-shift ``Symbol`` alignment. Fourier,
    nodal and constant factors are identity, so on a fully periodic
    grid both methods return their input unchanged (bitwise).

    Parameters
    ----------
    grid : Grid
        The grid whose bounded axes the chart spans.
    """

    def __init__(self, grid: Grid) -> None:
        """Bind the grid (the lattices derive from each space)."""
        self._grid: Grid = grid

    def embed(self, data: jax.Array, space: SpaceLike) -> jax.Array:
        """
        Embed component-layout ``data`` onto the union lattice.

        Parameters
        ----------
        data : jax.Array
            An array in ``space``'s slot layout (size-1 broadcast
            axes pass through untouched).
        space : SpaceLike
            The component's coefficient space.

        Returns
        -------
        jax.Array
            The data on the ``0..n`` union lattice per trig axis;
            absent modes are exact zeros.
        """
        return _reindex(data, self._shifts(space, to_union=True))

    def restrict(self, data: jax.Array, space: SpaceLike) -> jax.Array:
        """
        Restrict union-lattice ``data`` to the component layout.

        Parameters
        ----------
        data : jax.Array
            An array on the union lattice (size-1 broadcast axes
            pass through untouched).
        space : SpaceLike
            The component's coefficient space.

        Returns
        -------
        jax.Array
            The data in ``space``'s slot layout; modes the component
            lacks are dropped.
        """
        return _reindex(data, self._shifts(space, to_union=False))

    def _shifts(
        self, space: SpaceLike, *, to_union: bool,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """Per-axis slot moves between ``space`` and union layouts."""
        factors = space.bare.factors
        rank = len(factors)
        shifts = []
        for i, factor in enumerate(factors):
            if not isinstance(factor, SineSpace | CosineSpace):
                continue
            union = factor.mesh.n_cells + 1
            offset = factor.mode_offset
            slots = factor.shape[0]
            if (offset, slots) == (0, union):
                continue
            if to_union:
                shifts.append((i - rank, offset, slots, union))
            else:
                shifts.append((i - rank, -offset, union, slots))
        return tuple(shifts)


def rayleigh_dual(
    q: Mapping[str, Symbol], weights: Mapping[str, float],
) -> dict[str, Symbol]:
    r"""
    Rayleigh dual (biorthonormal row) of a diagonal column ``q``.

    Description
    -----------
    The weighted pseudo-inverse row of the per-component diagonals:

    .. math::

        p_c = w_c q_c \, \Big(\sum_c w_c |q_c|^2\Big)^{-1}

    i.e. ``p_c = (w_c * q_c) @ norm.inverse()`` with the norm
    ``sum_c w_c * q_c.magnitude ** 2``. Off the norm's structural
    nullspace ``sum_c conj(p_c) q_c == 1``; on it (where every
    ``q_c`` is a structural zero) the dual is exactly zero — the
    ``Symbol.inverse`` regularization, no caller-side masking.

    Parameters
    ----------
    q : Mapping[str, Symbol]
        The per-component diagonal column.
    weights : Mapping[str, float]
        The per-component quadratic weights (the energy metric).

    Returns
    -------
    dict[str, Symbol]
        The dual diagonals, keyed like ``q``.
    """
    norm = sum(weights[c] * q[c].magnitude ** 2 for c in q)
    inverse = norm.inverse()
    return {c: (weights[c] * q[c]) @ inverse for c in q}
