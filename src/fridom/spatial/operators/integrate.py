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

Mapped grids (coordinate-systems plan, stage C2): quadrature weights
reuse the metric measures (rules 3.13), so on **any** grid whose
``CoordinateMapping`` derives a volume element the seeded rows hold
``Integral(jacobian=<family>)`` — the computational measure times the
metric Jacobian on the querying space (``grid._reduction_jacobian``).
Both mapping forms reduce physically alike: an embedding ``chart=``
grid carries the ``sqrt_g`` area element
:math:`\int f\,\sqrt{g}\,du\,dv` on the chart's base coordinates, and
an analytic ``maps=`` (terrain-following) grid the column Jacobian
``d<mapped>_d<base>`` on the mapped physical coordinates
:math:`\int f\,\mathrm{d}z_p`. For an embedding chart the Jacobian
enters exactly once per area integral: on the reduction of a chart
coordinate while the operand space still resolves *every* chart
coordinate (the first chart reduction of a sequential
``f.integrate()``); once a chart factor is constant the remaining
reductions contract against the plain computational measure. A field
*born* constant along a chart coordinate therefore integrates against
the computational measure only — consistent with the flat-grid
convention that constant factors carry no geometry. (The ``maps=``
column Jacobian instead varies over the map's *parameter* axes, so the
seeded ``f.integrate()`` verb reduces the single base axis first —
``scalar_field._bases_first`` — and a chartless multi-base analytic
map, deriving no unambiguous column, keeps the computational measure.)

The ``jacobian=`` names *chart coordinates* (``jacobian_weight``);
besides an embedding chart's base coordinates above, a name may be an
analytic-``maps=`` grid's mapped physical coordinate ``p`` (e.g.
``"zp"`` for ``zp = sigma * H``), whose reduction over its single
base axis ``b`` carries the column Jacobian ``d<p>_d<b>`` — the
terrain-following column integral :math:`\int f\,dz_p`, equal to the
single-column ``sqrt_g`` restriction. A ``jacobian=`` name that names
neither an embedding chart coordinate nor a mapped physical
coordinate raises a taught error instead of silently no-opping.
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
from fridom.spatial.operators.jacobian_weight import jacobian_factor
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
    the metric Jacobian derived on the operand space — the ``sqrt_g``
    area element (embedding chart) or a terrain column's
    ``d<p>_d<b>`` (analytic ``maps=``) — module docstring.

    Parameters
    ----------
    jacobian : tuple[str, ...] | None, optional
        Chart coordinate names whose reduction picks up the metric
        Jacobian weight — an embedding chart's base coordinates
        (``sqrt_g``) or an analytic-``maps=`` physical coordinate
        (its column Jacobian). None keeps the plain computational
        measure; a name off every chart raises (default: None).
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
        metric Jacobian derived on the operand space
        (``jacobian_weight``): an embedding chart's ``sqrt_g`` once,
        while every chart coordinate is still resolved by the space,
        or a terrain column's ``d<p>_d<b>`` on its base axis
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
        factor = jacobian_factor(f, axis, self._jacobian)
        if factor is not None:
            data = data * factor
        axis_index = bare.names.index(axis)
        data = data.sum(axis=axis_index, keepdims=True)
        codomain = resolve_codomain(self, space)
        stored = store(f.grid.decomposition, codomain, data)
        return type(f)(f.grid, codomain, stored, None)
