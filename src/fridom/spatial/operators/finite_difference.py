"""
``FiniteDifference``: the staggered finite-difference derivative.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_stencils.md``.
A ``SeparableOperator`` wrapping the pure ``staggered_diff`` kernel;
the default ``("diff", ...)`` entry on nodal spaces. Per-factor
signatures: periodic ``Center -> Right``, ``Right -> Center``;
bounded ``Center -> Inner``, ``Outer/Inner -> Center``. BC-tagged
bounded domains are accepted when the tag drops no DOFs (the tag
governs only the ghost fill); the codomain is always the BC-free
sibling of the table. Exterior-needing bounded signatures follow
the R1 legality rule (boundary_plan.md): every needy side must
carry BC structure — BC-free ``Inner -> Center`` (and any wider
boundary window) raises with the ``boundary="one_sided"`` hint.
Nodal only — the FV derivative on average spaces is
``FVDerivative`` (Wave 3).
"""
# Wave 2: FiniteDifference
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.base import (
    EigenbasisError,
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
    _resolve_axis,
)
from fridom.spatial.operators.interned import interned
from fridom.spatial.operators.spectral import (
    finite_difference_symbol,
    fourier_partner,
    in_trig_family,
    trig_diff_codomain,
    trig_finite_difference_symbol,
    trig_partner,
)
from fridom.spatial.operators.staggering import (
    apply_staggered,
    divide_by_codomain_measure,
    mapped_factor,
    patch_one_sided_edges,
    reach_or,
    require_dof_preserving_bc,
    require_grounded_bounded_sides,
    require_local_axis,
    uniform_spacing,
)
from fridom.spatial.operators.stencil_kernels import (
    staggered_diff,
    staggered_diff_weights,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.coefficient import (
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from fridom.spatial.operators.symbol import Symbol
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike

#: the only staggered-FD order carrying a closed-form Fourier symbol
#: in iteration 1
_SYMBOL_ORDER = 2

#: the only order whose spacing route is grounded on mapped meshes:
#: the staggered measure field *is* the two-point difference of the
#: node positions (rules section 3.9)
_MEASURE_ORDER = 2


@final
@interned
class FiniteDifference(SeparableOperator):

    """
    Staggered finite-difference derivative (order 2, 4, ...).

    Description
    -----------
    The stencil pattern (from ``order``) is static identity; the
    spacing denominator is the codomain's ``grid.measure`` metric
    field, derived at trace time (concepts section 2.7): on a
    uniform mesh the constant ``dx`` scalar fast path (folded by
    XLA), on a mapped mesh the staggered measure field dividing the
    unit-spacing stencil (order 2 only — higher orders need the
    computational-space chain rule and are deferred).
    ``eigenvalues`` is the ``i k_hat`` retagging
    symbol on a periodic mesh (order 2, Wave 9A) and the real
    ``±k_hat`` sine/cosine diagonal on the walled trig families
    (C4); BC-free bounded factors and higher orders raise
    ``EigenbasisError``.

    Parameters
    ----------
    order : int, optional
        The even order of accuracy = stencil size (default: 2).
    boundary : str, optional
        ``"closed"`` (default): bounded signatures follow the R1
        legality rule (exterior-needing windows exist where every
        needy side carries BC structure). ``"one_sided"``: the
        explicit opt-in closure of boundary_plan.md 2d — BC-free
        bounded exterior-needing signatures become legal, with the
        boundary outputs patched from ``order + 1`` one-sided
        true-DOF stencils; demands the applied axis undistributed
        (``layout="local"``).
    """

    dispatch_kind: ClassVar[str | None] = "diff"

    def __init__(self, order: int = 2,
                 boundary: str = "closed") -> None:
        """Create an FD kernel of the given even order."""
        staggered_diff_weights(order)  # validates even, >= 2
        if boundary not in ("closed", "one_sided"):
            raise ValueError(
                f"boundary must be 'closed' or 'one_sided', got "
                f"{boundary!r}")
        self._order: int = order
        self._boundary: str = boundary

    def _intern_key(self) -> tuple:
        """Structural key: the order and boundary mode (D6)."""
        return (self._order, self._boundary)

    @property
    def order(self) -> int:
        """Order of accuracy of the stencil."""
        return self._order

    @property
    def boundary(self) -> str:
        """The bounded-boundary closure mode."""
        return self._boundary

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        diff: Center -> Right | Inner; Right/Outer/Inner -> Center.

        Description
        -----------
        BC-tagged bounded domains resolve through the same table;
        the codomain is the **BC-free** sibling (nodal outputs are
        BC-free — the input's tag governs only the ghost fill).
        Tags that drop a member boundary DOF (Dirichlet on
        Left/Right/Outer) are rejected loudly.

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
        if isinstance(domain, SineSpace | CosineSpace):
            # coefficient-side pairing (constitutive tags): the
            # family and BC kind flip, the node set staggers — unlike
            # the BC-free nodal codomains below (two-representations
            # rule, see the tables in ``operators.spectral``)
            return trig_diff_codomain(domain)
        if not isinstance(domain, NodalSpace):
            raise SpaceMismatchError(
                f"FiniteDifference is nodal-only, got {domain!r}; "
                "the FV derivative on average spaces is FVDerivative",
                left=domain, operation="diff")
        require_dof_preserving_bc(domain, "diff")
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
        if not mesh.periodic:
            # R1 legality (boundary_plan.md 2c): exterior-needing
            # signatures exist only where every needy side carries
            # BC structure — the row un-seeds itself otherwise; the
            # one-sided variant (R2) reopens the BC-free rows
            require_grounded_bounded_sides(
                domain, codomain, self._order, "diff",
                "FiniteDifference(boundary='one_sided')",
                one_sided=self._boundary == "one_sided")
        if domain.scalars is Scalars.COMPLEX:
            codomain = codomain.as_complex()
        return codomain

    def requirements(
        self,
        domain: FunctionSpace,
    ) -> OperatorRequirements:
        """
        Declare reach ``(below, above)``, halo = order // 2.

        Description
        -----------
        The staggered difference is asymmetric: a ``Center -> Right``
        derivative reaches one cell up, a ``Right -> Center`` one cell
        down (order 2), so the two-sided reach keeps the chain from
        over-provisioning. ``halo`` (the per-side maximum) is unchanged
        at ``order // 2``.

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        half = self._order // 2
        if self._boundary == "one_sided":
            # the boundary patches write static physical-edge indices
            # from wider one-sided true-DOF stencils, so the interior
            # midpoint reach does not describe them: keep the symmetric
            # declaration, and demand the axis undistributed
            return OperatorRequirements(halo=half, layout="local")
        reach = reach_or(self, domain, self._order, half)
        return OperatorRequirements(reach=reach)

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — the factor carries the mesh
        space: SpaceLike,
    ) -> Symbol:
        r"""
        Return the retagging ``i k_hat`` / ``±k_hat`` diagonal.

        Description
        -----------
        On a periodic mesh: the order-2 staggered-difference Fourier
        symbol ``2i sin(k dx/2)/dx`` composed with the half-cell
        inter-origin phase, retagging ``Fourier(Center) ->
        Fourier(Right)`` (and back). On a walled mesh (sine/cosine
        factors, or their BC-tagged nodal origins): the real
        ``±2 sin(k dz/2)/dz`` derived-shift diagonal on the paired
        trig family (``+`` on sine -> cosine, ``-`` on cosine ->
        sine). BC-free bounded factors and higher orders are not
        grounded in iteration 1 and raise ``EigenbasisError``.

        Parameters
        ----------
        grid : object
            The grid (unused: the nodal factor carries the mesh).
        space : SpaceLike
            The nodal or coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The staggering diagonal on the coefficient factor.
        """
        if self._order != _SYMBOL_ORDER:
            raise EigenbasisError(
                "the iteration-1 FiniteDifference symbol is order 2 "
                f"only, got order {self._order}")
        bare = space.bare
        axis = _resolve_axis(self, bare)
        factor = bare.factor(axis)
        if in_trig_family(factor):
            # bounded staggering diagonalizes in the sine/cosine
            # basis (C4); the codomain carries the constitutive tag
            coeff, _origin = trig_partner(factor, "FiniteDifference")
            return trig_finite_difference_symbol(
                bare, axis, coeff, self.codomain(coeff))
        # resolve the source Fourier factor from the threaded layout
        # (nodal operand, or a transformed rfftn coefficient factor)
        src, nodal_origin = fourier_partner(factor, "FiniteDifference")
        codomain_nodal = self.codomain(nodal_origin)
        return finite_difference_symbol(
            bare, axis, src, nodal_origin, codomain_nodal)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Differentiate along ``axis`` (window-aligned kernel).

        Description
        -----------
        Uniform meshes fold the scalar cell width into the static
        weights (the constant special case XLA folds); mapped
        meshes run the unit-spacing kernel and divide by the
        codomain's measure field (rules sections 2.7, 3.9) — order
        2 only, where the staggered measure *is* the two-point
        difference of the node positions.

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
        factor = f.function_space.bare.factor(axis)
        order = self._order
        mapped = mapped_factor(factor)
        if mapped and order != _MEASURE_ORDER:
            raise NotImplementedError(
                "FiniteDifference on a mapped mesh is grounded at "
                "order 2 (the staggered measure is the two-point "
                "difference of the node positions); higher orders "
                "need the computational-space chain rule "
                "(coordinate-systems plan, stage C1+)")
        if mapped and self._boundary == "one_sided":
            raise NotImplementedError(
                "the one-sided boundary closure solves its patch "
                "weights on uniform node offsets; not grounded on "
                "mapped meshes")
        spacing = 1.0 if mapped else uniform_spacing(factor)

        def kernel(arr: Array, axis_index: int) -> Array:
            return staggered_diff(arr, axis_index, spacing=spacing,
                                  order=order)

        result = apply_staggered(self, f, axis, order, kernel,
                                 metadata=None)
        if mapped:
            return divide_by_codomain_measure(result, f, axis)
        if (self._boundary == "one_sided" and factor.bc.is_free
                and not factor.mesh.periodic):
            require_local_axis(f, axis)
            result = patch_one_sided_edges(
                f, result, axis, size=order, points=order + 1,
                derivative=1, spacing=spacing)
        return result
