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
from fridom.framework.utils import dtype_real, jaxify
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
    rossby_number : float, optional
        The Rossby number scaling the (separate) advection term;
        published as ``scaling.rossby`` (default: 1.0).
    """

    #: The vocabulary class this core supplies (D1.3 commitment 4).
    state_type = State

    #: Bound parameterful diagnostics (the D1.3 commitment-4 channel).
    diagnostics = DIAGNOSTICS

    def __init__(
        self, csqr: float = 1.0, rossby_number: float = 1.0,
    ) -> None:
        """Store ``csqr`` and the Rossby number as dynamic leaves."""
        self.csqr = jnp.asarray(csqr, dtype=dtype_real())
        self.rossby_number = jnp.asarray(
            rossby_number, dtype=dtype_real())

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_declarations(self) -> tuple[fr.FieldDeclaration, ...]:
        """U (east face), v (north face), p (centre), csqr (AUX)."""
        return (
            fr.FieldDeclaration(
                "u", space=fr.Staggered("x"),
                roles=(fr.roles.Velocity("x"),),
                long_name="Velocity (x)", units="m/s"),
            fr.FieldDeclaration(
                "v", space=fr.Staggered("y"),
                roles=(fr.roles.Velocity("y"),),
                long_name="Velocity (y)", units="m/s"),
            fr.FieldDeclaration(
                "p", space=fr.Collocated(),
                long_name="Pressure (g*eta)", units="m^2/s^2"),
            fr.FieldDeclaration(
                "csqr", space=fr.Profile(),
                lifecycle=fr.Lifecycle.AUXILIARY,
                default=DynamicalCore._csqr_default,
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
        it (``c.to(u.function_space) * u``). No ``grid.sync``
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
        r"""Pressure gradient and geopotential divergence."""
        u, v, p = state["u"], state["v"], state["p"]
        c = state["csqr"]
        du = -p.diff("x")
        dv = -p.diff("y")
        flux_u = c.to(u.function_space) * u
        flux_v = c.to(v.function_space) * v
        dp = -(flux_u.diff("x") + flux_v.diff("y"))
        return {"u": du, "v": dv, "p": dp}
