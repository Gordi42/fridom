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
"""
from __future__ import annotations

from functools import reduce
from operator import matmul
from typing import TYPE_CHECKING

from fridom.framework2.grid.operators.base import Identity
from fridom.framework2.grid.operators.mixed import resolve_transform
from fridom.framework2.grid.operators.realized import BoundTransform

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.operators.mixed import (
        ComposedTransform,
    )
    from fridom.framework2.grid.operators.symbol import Symbol
    from fridom.framework2.grid.operators.transform import Transform
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


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
    """

    def __init__(
        self, grid: Grid, spaces: Mapping[str, SpaceLike],
    ) -> None:
        """Intern the bare spaces and resolve their transforms."""
        self._grid: Grid = grid
        self._spaces: dict[str, SpaceLike] = {
            name: space.bare for name, space in spaces.items()}
        self._transforms: dict[
            str, Transform | ComposedTransform] = {
            name: resolve_transform(grid, bare)
            for name, bare in self._spaces.items()}

    # ================================================================
    #  Spaces and transforms
    # ================================================================
    def coeff(self, name: str) -> SpaceLike:
        """
        Coefficient space of a component (the transform codomain).

        Parameters
        ----------
        name : str
            The component name.

        Returns
        -------
        SpaceLike
            ``transform.codomain(space.bare)`` of the component.
        """
        return self._transforms[self._known(name)].codomain(
            self._spaces[name])

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
        """Resolve ``kind`` on the factor, query on the coeff space."""
        grid = self._grid
        factor = self._spaces[self._known(on)].factor(axis)
        op = grid.dispatch.resolve(kind, factor)
        return op[axis].eigenvalues(grid, self.coeff(on))


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
