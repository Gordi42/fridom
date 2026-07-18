"""
``LinearInterp``: two-point staggering interpolation.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_stencils.md``.
A ``SeparableOperator`` wrapping the pure ``linear_interp`` kernel;
the default ``("interpolate", ...)`` entry on nodal spaces.
Per-factor defaults: periodic ``Center <-> Right``; bounded
``Center -> Inner``, ``Outer/Inner -> Center`` — BC-tagged bounded
domains resolve through the same table to the **BC-free** sibling
(the tag governs only the ghost fill; DOF-dropping tags raise).
Exterior-needing bounded signatures follow the R1 legality rule
(boundary_plan.md): every needy side must carry BC structure. The
``target=NodeSet.OUTER`` variant (bounded ``Center -> Outer``, wall
faces through the BC-structured ghost fill) is per-instance and
never a default row.
"""
# Wave 2: LinearInterp
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
    fourier_partner,
    in_trig_family,
    linear_interp_symbol,
    trig_interp_codomain,
    trig_linear_interp_symbol,
    trig_partner,
)
from fridom.spatial.operators.staggering import (
    apply_staggered,
    patch_one_sided_edges,
    reach_or,
    require_dof_preserving_bc,
    require_grounded_bounded_sides,
    require_local_axis,
)
from fridom.spatial.operators.stencil_kernels import (
    linear_interp,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.coefficient import (
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.operators.symbol import Symbol
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike

_INTERP_SIZE = 2


@final
@interned
class LinearInterp(SeparableOperator):

    """
    Second-order two-point interpolation between nodal node sets.

    Description
    -----------
    Fixed codomain per rules section 3.4: the registered operator
    fixes its own codomain, alternative codomains are per-instance
    via the ``target=`` constructor knob. ``eigenvalues`` is the
    ``one_hat`` averaging symbol on a periodic mesh (Wave 9A) and
    the real ``cos(k dz/2)`` sine/cosine diagonal on the walled
    trig families (C4, DCT-II excluded); BC-free bounded factors
    and the ``target=`` variant raise ``EigenbasisError``.

    Parameters
    ----------
    target : NodeSet | None, optional
        Explicit target node set overriding the default table;
        iteration 1 grounds ``NodeSet.OUTER`` only (default: None).
    boundary : str, optional
        ``"closed"`` (default): bounded signatures follow the R1
        legality rule. ``"one_sided"``: the explicit opt-in closure
        (boundary_plan.md 2d) — BC-free bounded ``Inner -> Center``
        and the ``target=OUTER`` wall faces become legal, patched by
        two-point one-sided value stencils (linear extrapolation,
        matching the interior order); demands the applied axis
        undistributed (``layout="local"``).
    """

    dispatch_kind: ClassVar[str | None] = "interpolate"

    def __init__(self, target: NodeSet | None = None,
                 boundary: str = "closed") -> None:
        """Create the kernel; ``target`` overrides the codomain."""
        if target is not None and not isinstance(target, NodeSet):
            raise TypeError(
                f"target must be a NodeSet member or None, got "
                f"{target!r}")
        if boundary not in ("closed", "one_sided"):
            raise ValueError(
                f"boundary must be 'closed' or 'one_sided', got "
                f"{boundary!r}")
        self._target: NodeSet | None = target
        self._boundary: str = boundary

    def _intern_key(self) -> tuple:
        """Structural key: the target and boundary mode (D6)."""
        return (self._target, self._boundary)

    @property
    def target(self) -> NodeSet | None:
        """Explicit target node set, or None for the default table."""
        return self._target

    @property
    def boundary(self) -> str:
        """The bounded-boundary closure mode."""
        return self._boundary

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve the staggering-interpolation codomain.

        Description
        -----------
        interpolate: Center <-> Right (periodic); Center -> Inner,
        Outer/Inner -> Center (bounded); target= selects Outer.
        BC-tagged bounded domains resolve to the **BC-free** sibling
        (the tag governs only the ghost fill); tags that drop a
        member boundary DOF (Dirichlet on Left/Right/Outer) are
        rejected loudly.

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
            return self._trig_codomain(domain)
        if not isinstance(domain, NodalSpace):
            raise SpaceMismatchError(
                "LinearInterp covers nodal spaces, got "
                f"{domain!r} (average conversions are the "
                "'reconstruct' kind)", left=domain,
                operation="interpolate")
        require_dof_preserving_bc(domain, "interpolate")
        mesh = domain.mesh
        node_set = domain.node_set
        if self._target is not None:
            result = self._target_result(domain)
        elif mesh.periodic:
            result = {NodeSet.CENTER: "right",
                      NodeSet.RIGHT: "center"}.get(node_set)
        else:
            result = {NodeSet.CENTER: "inner",
                      NodeSet.OUTER: "center",
                      NodeSet.INNER: "center"}.get(node_set)
        if result is None:
            raise SpaceMismatchError(
                f"no interpolate signature on {domain!r}: "
                "Center <-> Right (periodic); Center -> Inner, "
                "Outer/Inner -> Center (bounded)",
                left=domain, operation="interpolate")
        try:
            codomain: FunctionSpace = getattr(mesh, result)
        except ValueError as exc:  # mesh lacks the codomain family
            raise SpaceMismatchError(
                f"no interpolate signature on {domain!r}: {mesh!r} "
                f"has no {result} space", left=domain,
                operation="interpolate") from exc
        if not mesh.periodic:
            # R1 legality (boundary_plan.md 2c): exterior-needing
            # signatures exist only where every needy side carries
            # BC structure — the row un-seeds itself otherwise; the
            # one-sided variant (R2) reopens the BC-free rows
            require_grounded_bounded_sides(
                domain, codomain, _INTERP_SIZE, "interpolate",
                "LinearInterp(boundary='one_sided')",
                one_sided=self._boundary == "one_sided")
        if domain.scalars is Scalars.COMPLEX:
            codomain = codomain.as_complex()
        return codomain

    def _target_result(self, domain: FunctionSpace) -> str:
        """
        Resolve the ``target=`` variant's codomain attribute name.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D nodal factor space.

        Returns
        -------
        str
            The mesh factory attribute of the codomain.
        """
        if (self._target is NodeSet.OUTER
                and not domain.mesh.periodic
                and domain.node_set is NodeSet.CENTER):
            return "outer"
        raise SpaceMismatchError(
            "the target= variant grounds bounded "
            "Center -> Outer only in iteration 1; got "
            f"target={self._target} on {domain!r}",
            left=domain, operation="interpolate")

    def _trig_codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Coefficient-side pairing of a sine/cosine factor.

        Description
        -----------
        Constitutive tags (two-representations rule, see the tables
        in ``operators.spectral``): family and BC kind are kept, the
        node set staggers — unlike the BC-free nodal codomains of
        the default table. The DCT-II domain raises
        ``EigenbasisError`` (the eigen-layer skip signal), and the
        ``target=`` variant has no trig pairing.

        Parameters
        ----------
        domain : FunctionSpace
            The bare sine/cosine coefficient factor.

        Returns
        -------
        FunctionSpace
            The same-family, BC-tagged codomain factor.
        """
        if self._target is not None:
            raise SpaceMismatchError(
                "the target= variant grounds bounded "
                "Center -> Outer only in iteration 1; got "
                f"target={self._target} on {domain!r}",
                left=domain, operation="interpolate")
        return trig_interp_codomain(domain)

    def requirements(
        self,
        domain: FunctionSpace,
    ) -> OperatorRequirements:
        """
        Declare reach ``(below, above)``, halo = 1.

        Description
        -----------
        Two-point interpolation is one-sided per direction: a
        ``Center -> Right`` average reaches one cell up, a
        ``Right -> Center`` one cell down. The two-sided reach keeps a
        composed chain (interp then difference) from over-provisioning;
        the symmetric ``halo`` stays 1.

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        reach = reach_or(self, domain, _INTERP_SIZE, 1)
        if self._boundary == "one_sided":
            # the boundary patches write static physical-edge
            # indices: negotiation must keep the axis undistributed
            return OperatorRequirements(reach=reach, layout="local")
        return OperatorRequirements(reach=reach)

    def eigenvalues(
        self,
        grid: object,  # noqa: ARG002 — the factor carries the mesh
        space: SpaceLike,
    ) -> Symbol:
        r"""
        Return the retagging ``one_hat`` diagonal.

        Description
        -----------
        On a periodic mesh: the two-point averaging Fourier symbol
        ``cos(k dx/2)`` composed with the half-cell inter-origin
        phase, retagging ``Fourier(Center) -> Fourier(Right)`` (and
        back). On a walled mesh (sine/cosine factors, or their
        BC-tagged nodal origins): the real ``cos(k dz/2)``
        derived-shift diagonal on the same trig family (the DCT-II
        domain has no grounded codomain family and raises
        ``EigenbasisError`` — the eigen layer skips it). The
        ``target=`` variant and BC-free bounded factors are not
        grounded as diagonalizing symbols in iteration 1, so they
        raise ``EigenbasisError`` too.

        Parameters
        ----------
        grid : object
            The grid (unused: the nodal factor carries the mesh).
        space : SpaceLike
            The nodal or coefficient factor (or product) space.

        Returns
        -------
        Symbol
            The ``one_hat`` diagonal on the coefficient factor.
        """
        if self._target is not None:
            raise EigenbasisError(
                "the target= LinearInterp variant has no diagonalizing "
                "symbol in iteration 1")
        bare = space.bare
        axis = _resolve_axis(self, bare)
        factor = bare.factor(axis)
        if in_trig_family(factor):
            # bounded staggering diagonalizes in the sine/cosine
            # basis (C4); the codomain carries the constitutive tag
            coeff, _origin = trig_partner(factor, "LinearInterp")
            return trig_linear_interp_symbol(
                bare, axis, coeff, self.codomain(coeff))
        # resolve the source Fourier factor from the threaded layout
        # (nodal operand, or a transformed rfftn coefficient factor)
        src, nodal_origin = fourier_partner(factor, "LinearInterp")
        codomain_nodal = self.codomain(nodal_origin)
        return linear_interp_symbol(
            bare, axis, src, nodal_origin, codomain_nodal)

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Interpolate along ``axis`` (window-aligned two-point mean).

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The interpolated field (metadata kept: same quantity).
        """
        result = apply_staggered(self, f, axis, _INTERP_SIZE,
                                 linear_interp, metadata=f.metadata)
        factor = f.function_space.bare.factor(axis)
        if (self._boundary == "one_sided" and factor.bc.is_free
                and not factor.mesh.periodic):
            require_local_axis(f, axis)
            result = patch_one_sided_edges(
                f, result, axis, size=_INTERP_SIZE,
                points=_INTERP_SIZE, derivative=0, spacing=1.0)
        return result
