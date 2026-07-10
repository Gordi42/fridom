"""Constant stratification: buoyancy and the linear coupling.

Description
-----------
``ConstantStratification`` registers the buoyancy tracer ``b`` and
contributes **both** linear coupling terms (D1's driving example):
``+b/dsqr`` in the w-equation (buoyancy force) and ``-N^2 w`` in the
b-equation (restoring). It owns the constant ``n2`` leaf and provides
``stratification.n2``; ``dsqr`` is read from ``ctx.params``. The
terms are pure field arithmetic (``.to`` interpolation across the
staggered w-b face), so their halo stencils are traced normally and
the module declares no ``extra_halo``.

``b`` is declared BC-free on every grid (topology-driven walls, C8):
walls enter through grid periodicity alone, the buoyancy's trig
parity on a walled grid is derived by the physics layers
(eigenmodes/transforms), never by a declaration knob.
"""
from __future__ import annotations

from functools import partial

import fridom.framework2 as fr
from fridom.framework.utils import jaxify
from fridom.nonhydro2.params import DSQR


@partial(jaxify, dynamic=("n2",))
class ConstantStratification(fr.Module):

    """Registers ``b``; contributes both linear coupling terms.

    Parameters
    ----------
    n2 : float | fr.Ramp, optional
        The constant squared buoyancy frequency ``N^2`` (default: 1.0);
        may be an ``fr.Ramp`` for a spun-up stratification.
    """

    def __init__(self, n2: float | fr.Ramp = 1.0) -> None:
        """Store the stratification leaf."""
        self.n2 = fr.leaf(n2)

    field_declarations = (
        fr.FieldDeclaration.tracer(
            "b", space=fr.Collocated(),
            long_name="Buoyancy", units="m/s^2"),
    )
    field_references = (
        fr.FieldReference(
            "w", hint="buoyancy couples to vertical velocity, "
                      "declared by a dynamical core (nh.DynamicalCore)"),
    )
    parameter_declarations = (
        fr.ParameterDeclaration(fr.params.STRATIFICATION_N2, attr="n2",
                                units="1/s^2",
                                doc="squared buoyancy frequency N^2"),
    )
    parameter_references = (
        fr.ParameterReference(DSQR, hint="declared by nh.DynamicalCore"),
    )

    @fr.term(advances=("w",), linear=True)
    def buoyancy_force(self, state, ctx) -> dict:  # noqa: ANN001
        """``dw/dt += b / dsqr`` (buoyancy interpolated onto the w face)."""
        dsqr = ctx.params[DSQR]
        return {"w": state["b"].to(state["w"]) / dsqr}

    @fr.term(advances=("b",), linear=True)
    def restoring(self, state, ctx) -> dict:  # noqa: ANN001
        """``db/dt += -N^2 w`` (w interpolated onto the b cell)."""
        n2 = ctx.params[fr.params.STRATIFICATION_N2]
        return {"b": -(n2 * state["w"].to(state["b"]))}
