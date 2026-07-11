"""
``Where``: the elementwise ternary select (the ``("select", ...)`` kind).

Description
-----------
Owning class doc: ``design/specs/grid/classes/operators_products.md``
(Where section; the class doc places it in the ``products`` module —
it lives in this sibling module because the Wave-2 ``products.py`` is
read-only for the Wave-4 cluster).

An elementwise ternary select, so that upwind flux-sign selection
stays **inside the operator layer** where the halo-accounting trace
can see it — a raw ``jnp.where`` on ``.data`` would be invisible to
the tracer. Flux-splitting advection combines it with the biased
``WenoReconstruction`` pair (``operators.weno``): sign selection
routed through ``("select", space)`` is what makes the whole upwind
path visible. halo 0, layout "any".
"""
# Wave 4: Where (upwind flux-sign selection)
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.operators.base import (
    BinaryOperator,
    FieldLike,
)
from fridom.framework2.grid.operators.interned import interned

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike


@final
@interned
class Where(BinaryOperator):

    """
    Elementwise ternary select on same-space operands.

    Description
    -----------
    ``Where()(cond, a, b)`` is ``where(cond, a, b)`` on the aligned
    local data: entries where ``cond`` is nonzero (true) read ``a``,
    the rest read ``b``. Ternary via the binary template's ``*more``
    slot; ``codomain`` widens to three domains (recorded, like the
    composed-operator widening). Iteration 1 demands one shared bare
    space across all three operands (the doc-02 ``where`` sugar owns
    the sanctioned lifts and does not exist yet). The result keeps
    the first branch's metadata: a select does not transform the
    branch quantity, it picks.
    """

    dispatch_kind: ClassVar[str | None] = "select"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(
        self,
        domain_cond: SpaceLike,
        domain_a: SpaceLike,
        domain_b: SpaceLike,
    ) -> SpaceLike:
        """
        Return the common space of the three (pre-lifted) operands.

        Parameters
        ----------
        domain_cond : SpaceLike
            The condition operand's bare space.
        domain_a : SpaceLike
            The true-branch operand's bare space.
        domain_b : SpaceLike
            The false-branch operand's bare space.

        Returns
        -------
        SpaceLike
            The shared bare space (the branches' space).
        """
        if domain_a is not domain_b:
            raise SpaceMismatchError(
                "the branches of 'select' must share one space "
                f"after the sanctioned lifts, got {domain_a!r} vs "
                f"{domain_b!r}", left=domain_a, right=domain_b,
                operation="select")
        if domain_cond is not domain_a:
            raise SpaceMismatchError(
                "the condition of 'select' must live on the "
                f"branches' space, got {domain_cond!r} vs "
                f"{domain_a!r}", left=domain_cond, right=domain_a,
                operation="select")
        return domain_a

    def _apply(
        self, cond: FieldLike, a: FieldLike, b: FieldLike,
    ) -> FieldLike:
        """
        Elementwise select on the aligned storage arrays.

        Parameters
        ----------
        cond : FieldLike
            The condition field (nonzero selects the true branch).
        a : FieldLike
            The true-branch field.
        b : FieldLike
            The false-branch field.

        Returns
        -------
        FieldLike
            The selected field (first branch's metadata kept).
        """
        data = jnp.where(
            cond._data,  # noqa: SLF001 — storage seam
            a._data,  # noqa: SLF001 — storage seam
            b._data)  # noqa: SLF001 — storage seam
        # pointwise on aligned storage frames (task 1.8, stage B):
        # the result claims what every operand's ghosts had
        valid = cond.halo_valid.merge_min(
            a.halo_valid).merge_min(b.halo_valid)
        return type(a)(a.grid, a.function_space.bare, data,
                       a.metadata, halo_valid=valid)
