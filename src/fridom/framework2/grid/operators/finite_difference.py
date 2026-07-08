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
    EigenbasisError,
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
    _resolve_axis,
)
from fridom.framework2.grid.operators.interned import interned
from fridom.framework2.grid.operators.spectral import (
    finite_difference_symbol,
    fourier_partner,
)
from fridom.framework2.grid.operators.staggering import (
    apply_staggered,
    uniform_spacing,
)
from fridom.framework2.grid.operators.stencil_kernels import (
    staggered_diff,
    staggered_diff_weights,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.coefficient import FourierSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from fridom.framework2.grid.operators.symbol import Symbol
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike

#: the only staggered-FD order carrying a closed-form Fourier symbol
#: in iteration 1
_SYMBOL_ORDER = 2


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
    measure field). ``eigenvalues`` is the ``i k_hat`` retagging
    symbol on a periodic mesh (order 2, Wave 9A); bounded meshes and
    higher orders raise ``EigenbasisError``.

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
        if isinstance(domain, FourierSpace):
            # layout-faithful eigenvalue threading: retag the Fourier
            # factor through the staggered origin (decision 3)
            return domain.mesh.fourier(origin=self.codomain(domain.origin))
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

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — the factor carries the mesh
        space: SpaceLike,
    ) -> Symbol:
        r"""
        Return the retagging ``i k_hat`` diagonal (periodic mesh).

        Description
        -----------
        The order-2 staggered-difference Fourier symbol
        ``2i sin(k dx/2)/dx`` composed with the half-cell inter-origin
        phase, retagging ``Fourier(Center) -> Fourier(Right)`` (and
        back). Bounded meshes diagonalize in the sine/cosine basis and
        higher orders are not grounded in iteration 1, so both raise
        ``EigenbasisError``.

        Parameters
        ----------
        grid : object
            The grid (unused: the nodal factor carries the mesh).
        space : SpaceLike
            The nodal coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The ``i k_hat`` diagonal on the Fourier factor.
        """
        if self._order != _SYMBOL_ORDER:
            raise EigenbasisError(
                "the iteration-1 FiniteDifference symbol is order 2 "
                f"only, got order {self._order}")
        bare = space.bare
        axis = _resolve_axis(self, bare)
        factor = bare.factor(axis)
        # resolve the source Fourier factor from the threaded layout
        # (nodal operand, or a transformed rfftn coefficient factor)
        src, nodal_origin = fourier_partner(factor, "FiniteDifference")
        codomain_nodal = self.codomain(nodal_origin)
        return finite_difference_symbol(
            bare, axis, src, nodal_origin, codomain_nodal)

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
