"""
``FiniteDifference``: the staggered finite-difference derivative.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_stencils.md``.
A ``SeparableOperator`` wrapping the pure ``staggered_diff`` kernel;
the default ``("diff", ...)`` entry on nodal spaces. Per-factor
signatures: periodic ``Center -> Right``, ``Right -> Center``;
bounded ``Center -> Inner``, ``Outer/Inner -> Center``. Nodal only —
the FV derivative on average spaces is ``FVDerivative`` (Wave 3).
"""
# Wave 2: FiniteDifference
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.operators.base import (
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
)
from fridom.framework2.grid.operators.interned import interned
from fridom.framework2.grid.operators.staggering import (
    apply_staggered,
    uniform_spacing,
)
from fridom.framework2.grid.operators.stencil_kernels import (
    staggered_diff,
    staggered_diff_weights,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )


@final
@interned
class FiniteDifference(SeparableOperator):

    """
    Staggered finite-difference derivative (order 2, 4, ...).

    Description
    -----------
    The stencil pattern (from ``order``) is static identity; the
    spacing denominator is the mesh's uniform cell width read at
    trace time (iteration-1 stand-in for the ``grid.measure`` dual
    measure field). ``eigenvalues`` (the ``i k_hat`` retagging
    symbol) is designed-for and inherits the raising base until the
    ``Symbol`` cluster lands (Wave 3).

    Parameters
    ----------
    order : int, optional
        The even order of accuracy = stencil size (default: 2).
    """

    dispatch_kind: ClassVar[str | None] = "diff"

    def __init__(self, order: int = 2) -> None:
        """Create an FD kernel of the given even order."""
        staggered_diff_weights(order)  # validates even, >= 2
        self._order: int = order

    def _intern_key(self) -> tuple:
        """Structural key: the stencil order (D6)."""
        return (self._order,)

    @property
    def order(self) -> int:
        """Order of accuracy of the stencil."""
        return self._order

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        diff: Center -> Right | Inner; Right/Outer/Inner -> Center.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D nodal factor space.

        Returns
        -------
        FunctionSpace
            The staggered codomain factor (scalars preserved).
        """
        if not isinstance(domain, NodalSpace):
            raise SpaceMismatchError(
                f"FiniteDifference is nodal-only, got {domain!r}; "
                "the FV derivative on average spaces is FVDerivative",
                left=domain, operation="diff")
        if not domain.bc.is_free:
            raise SpaceMismatchError(
                "FiniteDifference covers BC-free nodal spaces in "
                f"iteration 1, got {domain!r}",
                left=domain, operation="diff")
        mesh = domain.mesh
        node_set = domain.node_set
        if mesh.periodic:
            result = {NodeSet.CENTER: "right",
                      NodeSet.RIGHT: "center"}.get(node_set)
        else:
            result = {NodeSet.CENTER: "inner",
                      NodeSet.OUTER: "center",
                      NodeSet.INNER: "center"}.get(node_set)
        if result is None:
            raise SpaceMismatchError(
                f"no diff signature on {domain!r}: Center -> Right "
                "(periodic) / Inner (bounded); Right/Outer/Inner -> "
                "Center", left=domain, operation="diff")
        try:
            codomain: FunctionSpace = getattr(mesh, result)
        except ValueError as exc:  # mesh lacks the codomain family
            raise SpaceMismatchError(
                f"no diff signature on {domain!r}: {mesh!r} has no "
                f"{result} space", left=domain,
                operation="diff") from exc
        if domain.scalars is Scalars.COMPLEX:
            codomain = codomain.as_complex()
        return codomain

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — order-dependent only
    ) -> OperatorRequirements:
        """
        Declare halo = order // 2, layout "any".

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=self._order // 2)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Differentiate along ``axis`` (window-aligned kernel).

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The derivative field (default metadata: new quantity).
        """
        spacing = uniform_spacing(f.function_space.bare.factor(axis))
        order = self._order

        def kernel(arr: Array, axis_index: int) -> Array:
            return staggered_diff(arr, axis_index, spacing=spacing,
                                  order=order)

        return apply_staggered(self, f, axis, order, kernel,
                               metadata=None)
