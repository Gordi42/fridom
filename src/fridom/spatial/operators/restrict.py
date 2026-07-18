r"""
``Restriction``: the exact ``Outer -> Inner`` face restriction.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_stencils.md``.
A ``SeparableOperator`` that drops the two boundary faces of the
both-boundary vertical face set: on a bounded mesh ``Outer`` (the
n + 1 faces) contains ``Inner`` (the n - 1 interior faces) as its
shared interior nodes, so the map ``Outer -> Inner`` is the **exact**
selection of those shared values — no interpolation and no metric.
The selection ``Inner[m] == Outer[m + 1]`` reads one face **above**
each output, so on a device-distributed axis it carries the
asymmetric per-shard footprint ``(0, 1)`` (declared in
``requirements``): the last interior face of every shard is the
neighbour shard's boundary face, reached across the seam through the
one ghost slot the exchange fills. It is the ``("restrict", ...)``
default row, the kind
``ScalarField.to`` resolves for the ``Outer -> Inner`` direction (the
``interpolate`` kind is already committed to ``Outer -> Center``, the
half-cell average, and a registry key resolves one codomain).

The single consumer today is the shared flux-form advection family on
the hydrostatic model: the diagnosed vertical velocity ``w`` lives on
``Outer`` (its surface face is a free-surface DOF, ``hy.HydrostaticCore``)
and the vertical flux leg needs it on the interior flux faces
``Inner``. Dropping the two boundary faces is exactly the correct
closure — **zero advective flux through the top and bottom boundary
faces**, the standard fixed-domain treatment under a linear free
surface: the domain does not move, and the volume flux through
``z = 0`` is carried by the surface-pressure equation, not by
advection. The surface velocity ``w(0)`` therefore never enters an
advective flux, which is what makes the transported tracer's mass
conserved to roundoff (the boundary flux is a structural zero, not a
truncation-level one).

Nodal ``Outer -> Inner`` only; every other domain (any other node set,
average spaces, periodic meshes — which carry no ``Outer`` factor)
raises. Metric-free: the restriction is a pure index selection, exact
on uniform and stretched meshes alike, so it carries no mapped-mesh
refusal.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.base import (
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
)
from fridom.spatial.operators.interned import interned
from fridom.spatial.operators.staggering import (
    apply_staggered,
    reach_or,
    require_dof_preserving_bc,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from jax import Array

    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )

#: the restriction is a one-point selection (identity window)
_RESTRICT_SIZE = 1


@final
@interned
class Restriction(SeparableOperator):

    """
    Exact ``Outer -> Inner`` restriction (drops the boundary faces).

    Description
    -----------
    A ``size = 1`` staggered kernel (the identity window): the
    first-node offsets of ``Outer`` (0.0) and ``Inner`` (1.0) place
    output face ``m`` over input face ``m + 1`` in the storage frame,
    so ``Inner[m] == Outer[m + 1]`` for ``m = 0 .. n - 2`` — the
    n - 1 interior faces of ``Outer``, dropping ``Outer[0]`` (the
    bottom boundary face) and ``Outer[n]`` (the top). Exact and
    metric-free. It reads one slot **above** each output
    (``Inner[m]`` takes ``Outer[m + 1]``), so its per-shard footprint
    is the asymmetric reach ``(0, 1)`` — the ghost slot a distributed
    operand must carry so the last interior face of a shard reads the
    neighbour shard's boundary face across the seam (``requirements``
    publishes it, so ``_ensure_valid`` and the negotiation size and
    sync the operand; declaring the naive ``halo 0`` silently elides
    that sync and reads a stale ghost at the shard seam). The output
    is the BC-free ``Inner`` sibling (nodal operator outputs carry no
    BC tag); a complex ``Outer`` restricts to the complex ``Inner``.
    """

    dispatch_kind: ClassVar[str | None] = "restrict"

    def _intern_key(self) -> tuple:
        """Structural key: the restriction carries no parameters (D6)."""
        return ()

    def requirements(
        self,
        domain: FunctionSpace,
    ) -> OperatorRequirements:
        """
        Declare the asymmetric one-slot-above footprint ``(0, 1)``.

        Description
        -----------
        The identity-window selection ``Inner[m] == Outer[m + 1]``
        reads one slot **above** each output, so its per-shard
        halo-exchange footprint is the two-sided reach ``(0, 1)`` (the
        midpoint :func:`~fridom.spatial.operators.staggering.footprint_reach`
        at the ``Outer -> Inner`` alignment). This is the demand
        ``_ensure_valid`` and the negotiation tracer must see: without
        it a device-distributed ``Outer`` operand is never synced along
        the applied axis, and the last interior face of every shard
        reads the unfilled (zero) ghost slot at the shard seam —
        silently wrong on a sharded vertical (the same failure the
        ``staggering.footprint_reach`` docstring warns of for a
        bounded staggered stencil whose exterior reach cancels to
        zero). The symmetric ``halo`` collapses to 1.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D ``Outer`` factor the operator is applied on.

        Returns
        -------
        OperatorRequirements
            The per-factor requirements record (reach ``(0, 1)``).
        """
        return OperatorRequirements(
            reach=reach_or(self, domain, _RESTRICT_SIZE, 1))

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve ``Outer -> Inner`` (bounded nodal only).

        Description
        -----------
        The only signature: a bounded ``Outer`` factor restricts to
        the BC-free ``Inner`` sibling (scalars preserved). A Dirichlet
        tag on an ``Outer`` boundary member drops that value DOF and
        is rejected loudly (``require_dof_preserving_bc``); every other
        domain (other node sets, average factors, periodic meshes with
        no ``Outer``) raises.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D factor space.

        Returns
        -------
        FunctionSpace
            The ``Inner`` codomain factor (BC-free, scalars kept).
        """
        if (not isinstance(domain, NodalSpace)
                or domain.node_set is not NodeSet.OUTER):
            raise SpaceMismatchError(
                "Restriction maps the both-boundary face set Outer "
                f"onto the interior faces Inner only, got {domain!r}",
                left=domain, operation="restrict")
        require_dof_preserving_bc(domain, "restrict")
        try:
            codomain: FunctionSpace = domain.mesh.inner
        except ValueError as exc:  # mesh carries no Inner family
            raise SpaceMismatchError(
                f"no restrict signature on {domain!r}: {domain.mesh!r} "
                "has no Inner space", left=domain,
                operation="restrict") from exc
        if domain.scalars is Scalars.COMPLEX:
            codomain = codomain.as_complex()
        return codomain

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Select the interior faces along ``axis`` (identity window).

        Parameters
        ----------
        f : FieldLike
            The operand field (storage-shaped ``_data``).
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The restricted field on the bare ``Inner`` codomain
            (metadata kept: the same quantity, fewer nodes).
        """
        def kernel(arr: Array, axis_index: int) -> Array:  # noqa: ARG001
            return arr

        return apply_staggered(self, f, axis, _RESTRICT_SIZE, kernel,
                               metadata=f.metadata)
