"""
``LinearInterp``: two-point staggering interpolation.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_stencils.md``.
A ``SeparableOperator`` wrapping the pure ``linear_interp`` kernel;
the default ``("interpolate", ...)`` entry on nodal spaces.
Per-factor defaults: periodic ``Center <-> Right``; bounded
``Center -> Inner``, ``Outer/Inner -> Center`` — BC-tagged bounded
domains resolve through the same table to the **BC-free** sibling
(the tag governs only the ghost fill; DOF-dropping tags raise). The
``target=NodeSet.OUTER`` variant (bounded ``Center -> Outer``,
boundary faces by one-sided extrapolation through the BC-free ghost
fill) is per-instance and never a default row.
"""
# Wave 2: LinearInterp
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
    fourier_partner,
    in_trig_family,
    linear_interp_symbol,
    trig_interp_codomain,
    trig_linear_interp_symbol,
    trig_partner,
)
from fridom.framework2.grid.operators.staggering import (
    apply_staggered,
    require_dof_preserving_bc,
)
from fridom.framework2.grid.operators.stencil_kernels import (
    linear_interp,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.coefficient import (
    CosineSpace,
    FourierSpace,
    SineSpace,
)
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.operators.symbol import Symbol
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike

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
    """

    dispatch_kind: ClassVar[str | None] = "interpolate"

    def __init__(self, target: NodeSet | None = None) -> None:
        """Create the kernel; ``target`` overrides the codomain."""
        if target is not None and not isinstance(target, NodeSet):
            raise TypeError(
                f"target must be a NodeSet member or None, got "
                f"{target!r}")
        self._target: NodeSet | None = target

    def _intern_key(self) -> tuple:
        """Structural key: the explicit target node set (D6)."""
        return (self._target,)

    @property
    def target(self) -> NodeSet | None:
        """Explicit target node set, or None for the default table."""
        return self._target

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
            if (self._target is NodeSet.OUTER and not mesh.periodic
                    and node_set is NodeSet.CENTER):
                result = "outer"
            else:
                raise SpaceMismatchError(
                    "the target= variant grounds bounded "
                    "Center -> Outer only in iteration 1; got "
                    f"target={self._target} on {domain!r}",
                    left=domain, operation="interpolate")
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
        if domain.scalars is Scalars.COMPLEX:
            codomain = codomain.as_complex()
        return codomain

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
        domain: FunctionSpace,  # noqa: ARG002 — fixed two-point halo
    ) -> OperatorRequirements:
        """
        Declare halo = 1, layout "any".

        Parameters
        ----------
        domain : FunctionSpace
            The factor space the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record.
        """
        return OperatorRequirements(halo=1)

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
        return apply_staggered(self, f, axis, _INTERP_SIZE,
                               linear_interp, metadata=f.metadata)
