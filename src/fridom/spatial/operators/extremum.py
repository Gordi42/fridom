"""
``Maximum`` / ``Minimum``: the extremum reductions to ``ConstantSpace``.

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_products.md``
("Reductions"). The default ``("amax", ...)`` / ``("amin", ...)``
entries on nodal, average and BC-tagged factors: reduce a factor to
its extremal true-DOF value, landing in the factor's
``ConstantSpace`` so the result broadcasts back under the strict
algebra (rules section 3.3) — ``f / f.max()`` stays in the algebra.

Unlike ``Integral`` the reduction is un-weighted: an extremum is an
order statistic of the nodal values, so quadrature measures (and the
metric Jacobian of mapped grids) play no role. The measure field is
used only as a **validity mask**: it is zero exactly on the non-DOF
rows (halo layers, decomposition padding, BC-claimed wall rows), so
masking those entries to the opposite-infinity sentinel before the
reduction keeps them from ever winning — and, because halos hold
synced *copies* of true values, keeps a tie from splitting the VJP
cotangent between a DOF and its copy. The masked ``jnp.where`` is a
select (no product with the branch value), and a sentinel entry never
equals the finite extremum, so the subgradient lands on valid DOFs
only — no double-``where`` guard is needed (differentiability
policy).

Complex factors deliberately have no rows: complex values carry no
order — reduce ``abs(f)`` first. Coefficient factors have no rows
either — transform back first.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

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
    import jax

    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )


def _extremum_codomain(
    domain: FunctionSpace, kind: str,
) -> FunctionSpace:
    """
    Shared codomain rule: S(m) -> ConstantSpace(m), real only.

    Parameters
    ----------
    domain : FunctionSpace
        The bare 1D factor space.
    kind : str
        The dispatch kind (for the error text).

    Returns
    -------
    FunctionSpace
        The factor's real ``ConstantSpace``.
    """
    if isinstance(domain, ConstantSpace):
        return domain
    if isinstance(domain, CoefficientSpace):
        raise SpaceMismatchError(
            "an extremum is an order statistic of nodal values; "
            f"transform back first, got {domain!r}",
            left=domain, operation=kind)
    if domain.scalars is Scalars.COMPLEX:
        raise SpaceMismatchError(
            "complex values carry no order; reduce the modulus "
            f"instead (abs(f)), got {domain!r}",
            left=domain, operation=kind)
    if not isinstance(domain, NodalSpace | AverageSpace):
        raise SpaceMismatchError(
            f"no {kind} signature on {domain!r}: nodal and "
            "average factors only",
            left=domain, operation=kind)
    return domain.mesh.constant


def _reduce_factor(
    op: SeparableOperator,
    f: FieldLike,
    axis: str,
    sentinel: float,
    reducer: str,
) -> FieldLike:
    """
    Shared kernel: mask non-DOF rows, reduce, land constant.

    Parameters
    ----------
    op : SeparableOperator
        The applying operator (codomain resolution).
    f : FieldLike
        The operand field.
    axis : str
        The resolved coordinate axis.
    sentinel : float
        The mask fill value (``-inf`` for max, ``+inf`` for min).
    reducer : str
        The array reduction method name (``"max"`` / ``"min"``).

    Returns
    -------
    FieldLike
        The extremum field on the reduced space (default metadata:
        new quantity).
    """
    space = f.function_space
    bare = space.bare
    # the measure is zero exactly on halo / padding / claimed rows —
    # reused here as the validity mask, never as a weight
    weight = f.grid.measure(bare, name=axis)
    data: jax.Array = jnp.where(weight.data > 0, f.data, sentinel)
    axis_index = bare.names.index(axis)
    data = getattr(data, reducer)(axis=axis_index, keepdims=True)
    codomain = resolve_codomain(op, space)
    stored = store(f.grid.decomposition, codomain, data)
    return type(f)(f.grid, codomain, stored, None)


@final
@interned
class Maximum(SeparableOperator):

    """
    Maximum over a factor's true DOFs, landing in ConstantSpace.

    Description
    -----------
    The un-weighted order-statistic reduction (module docstring):
    non-DOF rows are masked to ``-inf`` through the measure's
    positivity pattern, then the factor reduces with ``max``. The
    cross-shard reduction is declared through ``collective=True``
    (informational, no layout constraint). Application along a
    ``ConstantSpace`` factor is the identity (rules section 3.3,
    realized by the separable base).
    """

    dispatch_kind: ClassVar[str | None] = "amax"

    def _intern_key(self) -> tuple:
        """Structural key: parameter-free (D6)."""
        return ()

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        amax: S(m) -> ConstantSpace(m); Constant -> Constant.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D real nodal or average factor space.

        Returns
        -------
        FunctionSpace
            The factor's ``ConstantSpace``.
        """
        return _extremum_codomain(domain, "amax")

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — pointwise reduction
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
        Reduce the axis to its masked maximum.

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The maximum field on the reduced space.
        """
        return _reduce_factor(self, f, axis, -jnp.inf, "max")


@final
@interned
class Minimum(SeparableOperator):

    """
    Minimum over a factor's true DOFs, landing in ConstantSpace.

    Description
    -----------
    The mirror of :class:`Maximum`: non-DOF rows are masked to
    ``+inf`` through the measure's positivity pattern, then the
    factor reduces with ``min`` (module docstring).
    """

    dispatch_kind: ClassVar[str | None] = "amin"

    def _intern_key(self) -> tuple:
        """Structural key: parameter-free (D6)."""
        return ()

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        amin: S(m) -> ConstantSpace(m); Constant -> Constant.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D real nodal or average factor space.

        Returns
        -------
        FunctionSpace
            The factor's ``ConstantSpace``.
        """
        return _extremum_codomain(domain, "amin")

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — pointwise reduction
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
        Reduce the axis to its masked minimum.

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The minimum field on the reduced space.
        """
        return _reduce_factor(self, f, axis, jnp.inf, "min")
