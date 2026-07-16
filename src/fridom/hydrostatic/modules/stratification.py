r"""Stratification module: the buoyancy tracer and its restoring.

Description
-----------
``ConstantStratification`` registers the buoyancy tracer ``b`` and
contributes the **single** linear restoring term ``db/dt = -N^2 w``.
Unlike the nonhydrostatic twin it carries **no** ``buoyancy_force``
term (``+b/dsqr`` in a ``dw/dt`` equation): the hydrostatic model has
no vertical momentum equation, and hydrostatic balance
(``d_z p_hyd = b``, the ``HydrostaticCore`` DIAGNOSE) replaces it. The
energy exchange KE <-> PE flows through the ``p_hyd`` gradient on the
momentum and this restoring on the buoyancy, and is exactly
skew-adjoint under the ``diag(1, 1, 1/N^2, 1/c^2)`` energy metric
because the diagnosed ``w`` lives on the both-boundary face set
(the surface DOF ``w(0)``) and ``p_hyd`` is the half-cell center form
(``hy.energy``).

The restoring interpolates the diagnosed ``w`` (on the vertical
``Outer`` faces) onto the ``b`` cell centres via ``w.to(b)`` — the
registered ``Outer -> Center`` interpolation, the adjoint of the
half-cell hydrostatic-pressure pairing.
"""
from __future__ import annotations

from functools import partial

import fridom as fr
from fridom.framework.utils import jaxify


@partial(jaxify, dynamic=("n2",))
class ConstantStratification(fr.model.Module):

    r"""Registers ``b``; contributes the linear restoring term.

    Parameters
    ----------
    n2 : float | fr.model.Ramp, optional
        The constant squared buoyancy frequency ``N^2`` (default: 1.0);
        may be an ``fr.model.Ramp`` for a spun-up stratification.
    """

    def __init__(
        self,
        n2: float | fr.model.Ramp = 1.0,
    ) -> None:
        """Store the stratification leaf."""
        self.n2 = fr.model.leaf(n2)

    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """The buoyancy tracer ``b`` (collocated, TRACER + ADVECTED)."""
        return (
            fr.model.FieldDeclaration.tracer(
                "b", space=fr.spatial.Collocated(),
                long_name="Buoyancy", units="m/s^2"),
        )

    field_references = (
        fr.model.FieldReference(
            "w", hint="buoyancy couples to the diagnosed vertical "
                      "velocity, declared by a hydrostatic core "
                      "(hy.HydrostaticCore)"),
    )
    parameter_declarations = (
        fr.model.ParameterDeclaration(
            fr.model.params.STRATIFICATION_N2, attr="n2",
            units="1/s^2",
            doc="squared buoyancy frequency N^2"),
    )

    @fr.model.term(advances=("b",), linear=True)
    def restoring(self, state, ctx) -> dict:  # noqa: ANN001
        """``db/dt += -N^2 w`` (w interpolated onto the b cell)."""
        n2 = ctx.params[fr.model.params.STRATIFICATION_N2]
        return {"b": -(n2 * state["w"].to(state["b"]))}
