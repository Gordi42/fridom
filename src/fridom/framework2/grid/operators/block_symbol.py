r"""
``BlockSymbol``: a per-mode ``m x m`` matrix of diagonal operators.

Description
-----------
Owning class docs: ``notes/framework2/blocksymbol_l_assembly.md`` (the
``BlockSymbol`` type, item 3 of section 2) and
``notes/framework2/operator_symbols_plan.md`` (section 3, the scalar
``Symbol`` contract this generalizes). The block generalization of the
scalar :class:`~fridom.framework2.grid.operators.symbol.Symbol`: where a
``Symbol`` is one per-mode scalar diagonal from ``space`` to
``codomain``, a ``BlockSymbol`` is a per-mode ``m_out x m_in`` matrix of
such diagonals, carrying the *tuples* of column (``in_spaces``) and row
(``out_spaces``) coefficient spaces as static pytree aux and the batched
matrix ``_data`` of shape ``(*n_modes, m_out, m_in)`` as the single
dynamic leaf.

A ``BlockSymbol`` is what a linear *block* operator's per-mode system
matrix ``A(k)`` looks like: ``@`` is the per-mode matrix product,
``+``/``-`` the matrix sum, ``*`` a scalar scale, ``conj`` the Hermitian
adjoint. It is a **separate** ``@final`` type from the scalar ``Symbol``
(which stays ``@final`` and unchanged): the scalar chain / sum / scale
algebra runs *inside* each entry (reused unchanged), and the block layer
only assembles those entry symbols into a matrix and does the matrix
calculus on top. Its ``matrix`` accessor exposes ``_data`` batched over
the leading mode axes for :func:`jax.numpy.linalg.eigh`.
"""
# Wave 9B: BlockSymbol
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, final

import jax.numpy as jnp

import fridom.framework as fr
from fridom.framework2.grid.errors import SpaceMismatchError

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.operators.symbol import Symbol
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


@final
@partial(fr.utils.jaxify, dynamic=("_data",))
class BlockSymbol:

    r"""
    Per-mode ``m_out x m_in`` matrix of diagonal operators.

    Description
    -----------
    The per-mode system matrix ``A(k)`` of a linear block operator.
    Entry ``(i, j)`` is the scalar diagonal mapping the ``j``-th column
    coefficient space ``in_spaces[j]`` to the ``i``-th row coefficient
    space ``out_spaces[i]`` (the block generalization of the scalar
    symbol's ``space`` / ``codomain`` tags). The space tuples are the
    static pytree aux; ``data`` (shape ``(*n_modes, m_out, m_in)``) is
    the dynamic leaf.

    Parameters
    ----------
    in_spaces : tuple[SpaceLike, ...]
        The column (domain) coefficient spaces (stored bare).
    out_spaces : tuple[SpaceLike, ...]
        The row (codomain) coefficient spaces (stored bare).
    data : jax.Array
        The batched per-mode matrix, shape ``(*n_modes, m_out, m_in)``.
    """

    def __init__(
        self,
        in_spaces: tuple[SpaceLike, ...],
        out_spaces: tuple[SpaceLike, ...],
        data: jax.Array,
    ) -> None:
        """Wrap the per-mode matrix ``data`` on the space tuples."""
        self._in_spaces: tuple[SpaceLike, ...] = tuple(
            space.bare for space in in_spaces)
        self._out_spaces: tuple[SpaceLike, ...] = tuple(
            space.bare for space in out_spaces)
        self._data: jax.Array = jnp.asarray(data)

    # ================================================================
    #  Assembly from scalar symbols
    # ================================================================
    @classmethod
    def from_blocks(
        cls,
        blocks: tuple[tuple[Symbol | None, ...], ...],
        in_spaces: tuple[SpaceLike, ...],
        out_spaces: tuple[SpaceLike, ...],
    ) -> BlockSymbol:
        r"""
        Assemble a block symbol from an ``m x n`` grid of scalars.

        Description
        -----------
        Each non-``None`` entry is a scalar :class:`Symbol` whose
        diagonal is broadcast to the shared mode grid (the broadcast of
        the entry diagonal shapes — the spectral grid, which the
        real-FFT half-spectrum makes distinct from the physical
        resolution) and scattered into ``data[..., i, j]``; a ``None``
        entry is a zero block. The scalar entries carry the
        per-component staggering *phases* in their complex values (and
        their lifted ``(space, codomain)`` tags), so the assembler only
        has to broadcast and stack — no phase bookkeeping.

        Parameters
        ----------
        blocks : tuple[tuple[Symbol | None, ...], ...]
            The ``m_out x m_in`` grid of scalar symbols; ``None`` is a
            structural zero block.
        in_spaces : tuple[SpaceLike, ...]
            The ``m_in`` column coefficient spaces.
        out_spaces : tuple[SpaceLike, ...]
            The ``m_out`` row coefficient spaces.

        Returns
        -------
        BlockSymbol
            The assembled per-mode matrix.
        """
        shapes = [entry.data.shape for row in blocks
                  for entry in row if entry is not None]
        if not shapes:
            raise ValueError(
                "a block symbol needs at least one non-zero entry")
        n_modes = jnp.broadcast_shapes(*shapes)
        comp = fr.utils.dtype_comp()
        rows_data = []
        for row in blocks:
            cols = []
            for entry in row:
                if entry is None:
                    leaf = jnp.zeros(n_modes, dtype=comp)
                else:
                    leaf = jnp.broadcast_to(
                        entry.data.astype(comp), n_modes)
                cols.append(leaf)
            rows_data.append(jnp.stack(cols, axis=-1))
        data = jnp.stack(rows_data, axis=-2)
        return cls(in_spaces, out_spaces, data)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def in_spaces(self) -> tuple[SpaceLike, ...]:
        """The column (domain) coefficient spaces."""
        return self._in_spaces

    @property
    def out_spaces(self) -> tuple[SpaceLike, ...]:
        """The row (codomain) coefficient spaces."""
        return self._out_spaces

    @property
    def data(self) -> jax.Array:
        """The batched per-mode matrix (dynamic leaf)."""
        return self._data

    @property
    def matrix(self) -> jax.Array:
        r"""
        The per-mode matrix batched for :func:`jax.numpy.linalg.eigh`.

        Description
        -----------
        The thin eigendecomposition accessor (H1 consumer hook):
        returns ``data`` of shape ``(*n_modes, m, m)``, ready to be
        passed to a batched ``eigh`` over the leading mode axes. The
        metric / eigendecomposition itself lives one layer up.

        Returns
        -------
        jax.Array
            The batched matrix ``(*n_modes, m_out, m_in)``.
        """
        return self._data

    # ================================================================
    #  Matrix algebra (never the physical product)
    # ================================================================
    def __matmul__(self, other: BlockSymbol) -> BlockSymbol:
        r"""
        Per-mode matrix product ``(A @ B)(f) == A(B(f))``.

        Description
        -----------
        Requires ``B.out_spaces == A.in_spaces`` (the shared mid
        coefficient tuple), the block generalization of the scalar
        ``B.codomain == A.space`` — and per mode each scalar entry
        product ``A_ij @ B_jk`` then obeys that same tag rule. Contracts
        the mid index (``(A @ B)_ik = sum_j A_ij B_jk``) and threads the
        tuples ``in_spaces = B.in_spaces``, ``out_spaces = A.out_spaces``
        — the block matmul from which the Laplacian's scalar symbol
        falls out of the ``1 x 1`` ``div @ grad`` block.

        Parameters
        ----------
        other : BlockSymbol
            The inner block (applied first).

        Returns
        -------
        BlockSymbol
            The composed per-mode matrix.
        """
        if not isinstance(other, BlockSymbol):
            return NotImplemented
        if not _spaces_match(other._out_spaces, self._in_spaces):
            raise SpaceMismatchError(
                "cannot compose block symbols: inner out_spaces "
                f"{other._out_spaces!r} != outer in_spaces "
                f"{self._in_spaces!r}",
                left=other._out_spaces, right=self._in_spaces,
                operation="BlockSymbol.__matmul__")
        data = jnp.einsum("...ij,...jk->...ik", self._data, other._data)
        return BlockSymbol(other._in_spaces, self._out_spaces, data)

    def __add__(self, other: BlockSymbol) -> BlockSymbol:
        """Matrix sum on matched space tuples."""
        return self._matrix_add(other, jnp.add, "+")

    def __sub__(self, other: BlockSymbol) -> BlockSymbol:
        """Matrix difference on matched space tuples."""
        return self._matrix_add(other, jnp.subtract, "-")

    def __mul__(self, other: complex) -> BlockSymbol:
        """Scalar scale of the per-mode matrix."""
        if not isinstance(other, int | float | complex):
            return NotImplemented
        return BlockSymbol(self._in_spaces, self._out_spaces,
                           self._data * other)

    def __rmul__(self, other: complex) -> BlockSymbol:
        """Scalar scale (commutative in the coefficient)."""
        return self.__mul__(other)

    def __neg__(self) -> BlockSymbol:
        """Negation of the per-mode matrix."""
        return BlockSymbol(self._in_spaces, self._out_spaces,
                           -self._data)

    def conj(self) -> BlockSymbol:
        r"""
        Hermitian adjoint of the per-mode matrix.

        Description
        -----------
        Conjugate-transposes each per-mode matrix
        (``(A*)_ij = conj(A_ji)``) and swaps the domain / codomain
        tuples — the block generalization of the scalar adjoint (a
        ``1 x 1`` block reduces to the scalar ``conj``).

        Returns
        -------
        BlockSymbol
            The adjoint block (``in_spaces`` and ``out_spaces``
            swapped).
        """
        data = jnp.conj(jnp.swapaxes(self._data, -1, -2))
        return BlockSymbol(self._out_spaces, self._in_spaces, data)

    # ================================================================
    #  Internal helpers
    # ================================================================
    def _matrix_add(
        self, other: BlockSymbol, op: object, name: str,
    ) -> BlockSymbol:
        """Matrix ``+`` / ``-`` on strictly matched space tuples."""
        if not isinstance(other, BlockSymbol):
            return NotImplemented
        if not (_spaces_match(self._in_spaces, other._in_spaces)
                and _spaces_match(self._out_spaces, other._out_spaces)):
            raise SpaceMismatchError(
                "cannot combine block symbols on mismatched spaces: "
                f"{(self._out_spaces, self._in_spaces)!r} vs "
                f"{(other._out_spaces, other._in_spaces)!r}",
                left=self._in_spaces, right=other._in_spaces,
                operation=f"BlockSymbol.{name}")
        return BlockSymbol(self._in_spaces, self._out_spaces,
                           op(self._data, other._data))


def _spaces_match(
    a: tuple[SpaceLike, ...], b: tuple[SpaceLike, ...],
) -> bool:
    """Element-wise interned-identity equality of two space tuples."""
    return len(a) == len(b) and all(
        sa is sb for sa, sb in zip(a, b, strict=True))
