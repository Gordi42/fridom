"""
Exception types of the grid cluster.

Description
-----------
Owning class doc: ``notes/framework2/classes/product_spaces.md``
(``SpaceMismatchError``, ``GridMismatchError``). Both subclass
``TypeError``: the operand *combination* is unsupported, the moral
analogue of ``unsupported operand type(s)``.
"""
# Wave 0: SpaceMismatchError, GridMismatchError
from __future__ import annotations


class SpaceMismatchError(TypeError):

    """
    Raised when fields on incompatible function spaces are combined.

    Description
    -----------
    The strict-algebra exception: raised by all binary arithmetic
    when the operands' spaces cannot be joined by the sanctioned
    lifts, by ``.to`` when the registered conversion's codomain does
    not equal the requested target factor, and by operator
    application to a field outside the operator's domain. It is *not*
    raised for a missing dispatch entry (that is a
    registry-resolution error).

    Parameters
    ----------
    msg : str
        The error message (intended format: a per-factor diff over
        the mismatched names plus a conversion hint).
    left : object | None, optional
        The space of the left operand (default: None).
    right : object | None, optional
        The space of the right operand; None if unary
        (default: None).
    operation : str | None, optional
        The offending operation, e.g. "+", "*", "to", "forward"
        (default: None).
    mismatched_names : tuple[str, ...], optional
        The factor names whose factors differ beyond the sanctioned
        lifts (default: ()).
    """

    def __init__(
        self,
        msg: str,
        *,
        left: object | None = None,
        right: object | None = None,
        operation: str | None = None,
        mismatched_names: tuple[str, ...] = (),
    ) -> None:
        """Store the offending spaces for programmatic inspection."""
        super().__init__(msg)
        self.left: object | None = left
        self.right: object | None = right
        self.operation: str | None = operation
        self.mismatched_names: tuple[str, ...] = tuple(mismatched_names)


class GridMismatchError(TypeError):

    """
    Raised when fields on different grids are combined.

    Description
    -----------
    Checked *before* the space join: meshes — and therefore interned
    spaces — may legally be shared across grids, so the space-join
    check alone cannot detect operands living under different
    decompositions. The grid is static aux data with identity
    hashing, so the ``f.grid is g.grid`` check is exact inside and
    outside jit.

    Parameters
    ----------
    msg : str
        The error message.
    left : object | None, optional
        The grid of the left operand (default: None).
    right : object | None, optional
        The grid of the right operand (default: None).
    operation : str | None, optional
        The offending operation, e.g. "+", "*", "to"
        (default: None).
    """

    def __init__(
        self,
        msg: str,
        *,
        left: object | None = None,
        right: object | None = None,
        operation: str | None = None,
    ) -> None:
        """Store the offending grids for programmatic inspection."""
        super().__init__(msg)
        self.left: object | None = left
        self.right: object | None = right
        self.operation: str | None = operation
