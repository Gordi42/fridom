"""
Shared field helper for the shallow-water modules.

Description
-----------
One helper, :func:`scale`, that multiplies a field by a possibly
*traced* scalar (a ``ctx.params`` value such as the Rossby number) —
the residual traced-scalar-times-field escape the field-algebra
dunders cannot express (they accept only Python scalars). The
Coriolis / ``csqr`` fields are declared on the one-DOF
``fr.Profile()`` and broadcast to the nodal join with the plain
``.to(space)`` operator (the GAP-A ConstantSpace/Profile lift makes
that trace cleanly), so they need no such escape.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import jax

    from fridom.framework2.grid.fields.scalar_field import ScalarField


def scale(field: ScalarField, value: jax.Array) -> ScalarField:
    """
    Scale a field by a possibly-traced scalar (raw-data path).

    Description
    -----------
    ``ScalarField``'s ``*`` dunder accepts only Python scalars, so a
    traced ``ctx.params`` value (a 0-d ``jax.Array``, e.g. the Rossby
    number) must scale the field through its true-shape data — the
    idiom the framework's own tendency terms use for traced scalars.
    The result reclaims its halos at the next ghost-consuming
    operator.

    Parameters
    ----------
    field : ScalarField
        The field to scale.
    value : jax.Array
        The (0-d) scalar multiplier.

    Returns
    -------
    ScalarField
        ``value * field`` on the same space.
    """
    return field.with_data(field.data * value)
