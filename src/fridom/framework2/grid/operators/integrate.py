"""
``Integral``: the quadrature-weighted reduction to ``ConstantSpace``.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_products.md``
("Reductions"). The default ``("integrate", ...)`` entry on nodal and
average factors: contracts the field against the space's
quadrature-weight measure field (``grid.measure(space, name=...)``,
read at trace time), landing in the factor's ``ConstantSpace`` so
the result broadcasts back under the strict algebra (rules section
3.3). Coefficient factors deliberately have no default row —
transform back first. There is no unweighted ``sum`` operator
(``f.data.sum()`` is the escape hatch). ``CumulativeIntegral``
(``"cumint"``) is deferred within Wave 3 (see the wave report).
"""
# Wave 3: Integral -- deferred: CumulativeIntegral
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.fields.storage import store
from fridom.framework2.grid.operators.base import (
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
    resolve_codomain,
)
from fridom.framework2.grid.operators.interned import interned
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.average import AverageSpace
from fridom.framework2.grid.spaces.coefficient import CoefficientSpace
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )


@final
@interned
class Integral(SeparableOperator):

    """
    Weighted integral of a factor, landing in ConstantSpace.

    Description
    -----------
    Exact on average spaces (sum of average times cell measure), the
    node-set quadrature rule on nodal spaces (the measure field of
    the space, halved dual cells at boundary-member nodes). The
    cross-shard sum is declared through ``collective=True``
    (informational, no layout constraint). Application along a
    ``ConstantSpace`` factor is the identity (rules section 3.3,
    realized by the separable base).
    """

    dispatch_kind: ClassVar[str | None] = "integrate"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        integrate: S(m) -> ConstantSpace(m); Constant -> Constant.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D nodal or average factor space.

        Returns
        -------
        FunctionSpace
            The factor's ``ConstantSpace`` (scalars preserved).
        """
        if isinstance(domain, ConstantSpace):
            return domain
        if isinstance(domain, CoefficientSpace):
            raise SpaceMismatchError(
                "the honest coefficient-space integral is the "
                "zero-mode extraction (designed-for); transform "
                f"back first, got {domain!r}",
                left=domain, operation="integrate")
        if not isinstance(domain, NodalSpace | AverageSpace):
            raise SpaceMismatchError(
                f"no integrate signature on {domain!r}: nodal and "
                "average factors only in iteration 1",
                left=domain, operation="integrate")
        constant: FunctionSpace = domain.mesh.constant
        if domain.scalars is Scalars.COMPLEX:
            constant = constant.as_complex()
        return constant

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — pointwise contraction
    ) -> OperatorRequirements:
        """
        Declare halo = 0, layout "any", collective = True.

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=0, collective=True)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Contract the field against the axis' measure field.

        Description
        -----------
        Reads ``grid.measure(space, name=axis)`` at trace time (a
        uniform mesh constant-folds it) and sums the weighted true
        DOFs along the axis, keeping the singleton ``ConstantSpace``
        dimension.

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The integral field on the reduced space (default
            metadata: new quantity).
        """
        space = f.function_space
        bare = space.bare
        weight = f.grid.measure(bare, name=axis)
        axis_index = bare.names.index(axis)
        data = (f.data * weight.data).sum(axis=axis_index,
                                          keepdims=True)
        codomain = resolve_codomain(self, space)
        stored = store(f.grid.decomposition, codomain, data)
        return type(f)(f.grid, codomain, stored, None)
