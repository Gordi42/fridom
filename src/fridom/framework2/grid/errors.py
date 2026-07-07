"""
Exception types of the grid cluster.

Description
-----------
Owning class docs: ``notes/framework2/classes/product_spaces.md``
(``SpaceMismatchError``, ``GridMismatchError``) and
``notes/framework2/classes/grid.md`` ("Merge call site" resolution,
``GridFrozenError``). The mismatch errors subclass ``TypeError``: the
operand *combination* is unsupported, the moral analogue of
``unsupported operand type(s)``. ``GridFrozenError`` subclasses
``RuntimeError``: the operation is fine, the grid's *lifecycle
phase* is not.
"""
# Wave 0: SpaceMismatchError, GridMismatchError -- Phase 2:
#    GridFrozenError
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


class GridFrozenError(RuntimeError):

    """
    Raised when a frozen grid receives an unsatisfiable demand.

    Description
    -----------
    The frozen-grid lifecycle exception (grid.md "Merge call site"
    resolution, model D4/D5): ``freeze()`` ends the assembly phase
    and records the negotiation fingerprint. Post-freeze *mutators*
    (``merge_overrides``, ``with_immersed``) raise unconditionally;
    a post-freeze ``negotiate`` instead *verifies* its demands
    against the record — demand satisfaction (subset /
    less-or-equal, never equality), with ConstantSpace-broadcast
    state spaces adopted — and raises only on genuinely larger
    demands ("assemble the most demanding model first"). It
    subclasses ``RuntimeError``: the operation itself is fine, the
    grid's lifecycle phase is not.

    Parameters
    ----------
    msg : str
        The error message (verify failures carry a diff-style
        listing of the demands exceeding the frozen record).
    """
