"""
``LinearInterp``: two-point staggering interpolation.

Description
-----------
Owning class doc: ``notes/framework2/classes/operators_stencils.md``.
A ``SeparableOperator`` wrapping the pure ``linear_interp`` kernel;
the default ``("interpolate", ...)`` entry on nodal spaces.
Per-factor defaults: periodic ``Center <-> Right``; bounded
``Center -> Inner``, ``Outer/Inner -> Center``. The
``target=NodeSet.OUTER`` variant (bounded ``Center -> Outer``,
boundary faces by one-sided extrapolation through the BC-free ghost
fill) is per-instance and never a default row.
"""
# Wave 2: LinearInterp
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.operators.base import (
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
)
from fridom.framework2.grid.operators.interned import interned
from fridom.framework2.grid.operators.staggering import apply_staggered
from fridom.framework2.grid.operators.stencil_kernels import (
    linear_interp,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )

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
    via the ``target=`` constructor knob. ``eigenvalues`` (the
    ``one_hat`` averaging symbol) is designed-for and inherits the
    raising base until the ``Symbol`` cluster lands (Wave 3).

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

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D nodal factor space.

        Returns
        -------
        FunctionSpace
            The staggered codomain factor (scalars preserved).
        """
        if not (isinstance(domain, NodalSpace) and domain.bc.is_free):
            raise SpaceMismatchError(
                "LinearInterp covers BC-free nodal spaces, got "
                f"{domain!r} (average conversions are the "
                "'reconstruct' kind)", left=domain,
                operation="interpolate")
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
