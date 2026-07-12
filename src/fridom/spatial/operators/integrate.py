r"""
``Integral``: the quadrature-weighted reduction to ``ConstantSpace``.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_products.md``
("Reductions"). The default ``("integrate", ...)`` entry on nodal and
average factors: contracts the field against the space's
quadrature-weight measure field (``grid.measure(space, name=...)``,
read at trace time), landing in the factor's ``ConstantSpace`` so
the result broadcasts back under the strict algebra (rules section
3.3). Coefficient factors deliberately have no default row —
transform back first. There is no unweighted ``sum`` operator
(``f.data.sum()`` is the escape hatch). ``CumulativeIntegral``
(``"cumint"``) is deferred within Wave 3 (see the wave report).

Chart grids (coordinate-systems plan, stage C2): quadrature weights
reuse the metric measures (rules 3.13), so on a grid whose
``CoordinateMapping`` carries an embedding chart the seeded rows hold
``Integral(jacobian=<chart coords>)`` — the computational measure
times the ``sqrt_g`` Jacobian on the querying space,
:math:`\int f\,\sqrt{g}\,du\,dv`. The Jacobian enters exactly once
per area integral: on the reduction of a chart coordinate while the
operand space still resolves *every* chart coordinate (the first
chart reduction of a sequential ``f.integrate()``); once a chart
factor is constant the remaining reductions contract against the
plain computational measure. A field *born* constant along a chart
coordinate therefore integrates against the computational measure
only — consistent with the flat-grid convention that constant
factors carry no geometry.
"""
# Wave 3: Integral -- Stage C2: the sqrt_g Jacobian weight
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.storage import store
from fridom.spatial.operators.base import (
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
    resolve_codomain,
)
from fridom.spatial.operators.interned import interned
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import AverageSpace
from fridom.spatial.spaces.coefficient import CoefficientSpace
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.spaces.function_space import (
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
    realized by the separable base). With ``jacobian=`` set the
    reduction of a chart coordinate additionally contracts against
    the ``sqrt_g`` metric derived on the operand space (module
    docstring; the weight enters once per area integral).

    Parameters
    ----------
    jacobian : tuple[str, ...] | None, optional
        The chart-coupled coordinate names whose reduction picks up
        the ``sqrt_g`` Jacobian weight; None keeps the plain
        computational measure (default: None).
    """

    dispatch_kind: ClassVar[str | None] = "integrate"

    def __init__(
        self, jacobian: tuple[str, ...] | None = None,
    ) -> None:
        """Store the (optional) chart coordinate family."""
        if jacobian is not None:
            jacobian = tuple(jacobian)
            if not jacobian or not all(
                    isinstance(name, str) for name in jacobian):
                raise TypeError(
                    "jacobian names chart coordinates: a non-empty "
                    f"tuple of strings, got {jacobian!r}")
        self._jacobian: tuple[str, ...] | None = jacobian

    @property
    def jacobian(self) -> tuple[str, ...] | None:
        """Chart coordinates carrying the sqrt_g weight, or None."""
        return self._jacobian

    def _intern_key(self) -> tuple:
        """Structural key: the Jacobian coordinate family (D6)."""
        return (self._jacobian,)

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
        dimension. On a Jacobian row (``jacobian=`` set) the
        reduction of a chart coordinate additionally weighs by the
        ``sqrt_g`` metric derived on the operand space — exactly
        when every chart coordinate is still resolved by the space,
        so sequential reductions apply the area element once
        (module docstring).

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
        data = f.data * weight.data
        if (self._jacobian is not None
                and axis in self._jacobian
                and _resolves(bare, self._jacobian)):
            data = data * f.grid.metric(bare, "sqrt_g").data
        axis_index = bare.names.index(axis)
        data = data.sum(axis=axis_index, keepdims=True)
        codomain = resolve_codomain(self, space)
        stored = store(f.grid.decomposition, codomain, data)
        return type(f)(f.grid, codomain, stored, None)


def _resolves(space: FunctionSpace | object,
              names: tuple[str, ...]) -> bool:
    """
    Whether ``space`` resolves every name through a live factor.

    Parameters
    ----------
    space : SpaceLike
        The (bare) operand space.
    names : tuple[str, ...]
        The chart coordinate names.

    Returns
    -------
    bool
        True iff every name is contributed by a non-constant,
        non-coefficient factor of ``space``.
    """
    for name in names:
        try:
            factor = space.factor(name)
        except KeyError:
            return False
        if isinstance(factor, ConstantSpace | CoefficientSpace):
            return False
    return True
