r"""
The shallow-water dynamical core module.

Description
-----------
``DynamicalCore`` declares the state vocabulary (``u``, ``v``,
``p``), owns the squared phase speed :math:`c^2` (the AUXILIARY
``csqr`` field and the ``shallowwater.csqr`` scalar) and the Rossby
scaling (``scaling.rossby``), and contributes the single **linear**
pressure-gradient / geopotential-divergence term:

.. math::
    \partial_t \boldsymbol{u} = - \nabla p , \qquad
    \partial_t p = -\nabla\cdot\left(c^2 \boldsymbol{u}\right)

The rotation :math:`f\,\underset{\neg}{\boldsymbol{u}}` is **not** a
core term: it is carried by the shared Coriolis module
(``fr.modules.FPlaneCoriolis`` / ``BetaPlaneCoriolis``), which declares
the ``f_coriolis`` field and the ``+f v`` / ``-f u`` coupling. The
gravity term here is unscaled (the Rossby number multiplies only the
advection, D2.2). The nonlinear Sadourny advection is a separate
module.
"""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp

import fridom.framework2 as fr
from fridom.framework.utils import jaxify
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.diagnostics import DIAGNOSTICS
from fridom.shallowwater2.state import State


@partial(jaxify, dynamic=("csqr", "rossby_number"))
class DynamicalCore(fr.Module):

    r"""
    Shallow-water core: declares ``u``, ``v``, ``p``; linear physics.

    Parameters
    ----------
    csqr : float, optional
        The squared gravity-wave phase speed :math:`c^2` (constant
        depth); published as ``shallowwater.csqr`` and materialized
        into the one-DOF ``csqr`` field (default: 1.0).
    rossby_number : float | fr.Ramp, optional
        The Rossby number scaling the (separate) advection term;
        published as ``scaling.rossby`` (default: 1.0); may be a
        ``fr.Ramp`` for a spun-up nonlinearity.
    """

    #: The vocabulary class this core supplies (D1.3 commitment 4).
    state_type = State

    #: Bound parameterful diagnostics (the D1.3 commitment-4 channel).
    diagnostics = DIAGNOSTICS

    def __init__(
        self, csqr: float = 1.0, rossby_number: float | fr.Ramp = 1.0,
    ) -> None:
        """Store ``csqr`` and the Rossby number as dynamic leaves."""
        self.csqr = fr.leaf(csqr)
        self.rossby_number = fr.leaf(rossby_number)

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_declarations(self) -> tuple[fr.FieldDeclaration, ...]:
        """U (east face), v (north face), p (centre), csqr (AUX)."""
        return (
            fr.FieldDeclaration.velocity(
                "u", "x", space=fr.Staggered("x"),
                long_name="Velocity (x)", units="m/s"),
            fr.FieldDeclaration.velocity(
                "v", "y", space=fr.Staggered("y"),
                long_name="Velocity (y)", units="m/s"),
            fr.FieldDeclaration(
                "p", space=fr.Collocated(),
                long_name="Pressure (g*eta)", units="m^2/s^2"),
            fr.FieldDeclaration(
                "csqr", space=fr.Profile(),
                lifecycle=fr.Lifecycle.AUXILIARY,
                default=self._csqr_default,
                long_name="Squared phase speed", units="m^2/s^2"),
        )

    parameter_declarations = (
        fr.ParameterDeclaration(
            fr.params.SCALING_ROSSBY, attr="rossby_number"),
        fr.ParameterDeclaration(
            sw_params.CSQR, attr="csqr", units="m^2/s^2"))

    def _csqr_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.grid.ScalarField:
        """Owner-method default: fill the one-DOF profile with ``csqr``.

        The field is declared on ``fr.Profile()`` (constant depth is a
        single degree of freedom); the GAP-A ConstantSpace/Profile
        broadcast lifts it to the nodal join wherever a term multiplies
        it (``c.to(u) * u``). No ``grid.sync``
        pre-syncing: the GAP-B fix keeps carry-resident AUXILIARY
        fields scan-treedef-stable without pre-flooding their halos.
        """
        return grid.create_field(
            space, data=jnp.full(space.shape, self.csqr),
            name="csqr")

    # ================================================================
    #  Tendency terms (linear)
    # ================================================================
    @fr.term(advances=("u", "v", "p"), linear=True)
    def gravity(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""Pressure gradient and geopotential divergence.

        .. math::
            \partial_t \boldsymbol{u} = - \nabla p , \qquad
            \partial_t p = -\nabla\cdot\left(c^2 \boldsymbol{u}\right)

        Pure field arithmetic: ``c^2`` sits INSIDE the divergence
        (``(c.to(u) * u).diff("x")``, the flux form) so the discrete
        stencil matches ``diff(c^2 u)``. The ``csqr`` field lifts from
        its one-DOF ``fr.Profile()`` onto each velocity face via the
        ConstantSpace broadcast in ``.to``.
        """
        u, v, p = state["u"], state["v"], state["p"]
        csqr = state["csqr"]
        return {
            "u": -p.diff("x"),
            "v": -p.diff("y"),
            "p": -(csqr.to(u) * u).diff("x") - (csqr.to(v) * v).diff("y"),
        }
