r"""
The shallow-water dynamical core module.

Description
-----------
``ShallowWaterCore`` declares the state vocabulary (``u``, ``v``,
``h``), owns the squared phase speed :math:`c^2` (the AUXILIARY
``csqr`` field and the ``shallowwater.csqr`` scalar) and the Rossby
scaling (``scaling.rossby``), and contributes the two **linear**
tendency terms:

.. math::
    \partial_t \boldsymbol{u} = f\,\underset{\neg}{\boldsymbol{u}}
                                - \nabla h , \qquad
    \partial_t h = -\nabla\cdot\left(c^2 \boldsymbol{u}\right)

The rotation ``coriolis`` term reads the ``f_coriolis`` field
declared by a Coriolis module (a ``FieldReference``, hinted if
absent); the ``gravity`` term is the pressure gradient + geopotential
divergence. Both are unscaled (the Rossby number multiplies only the
advection, D2.2). The nonlinear Sadourny advection is a separate
module.
"""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp

import fridom.framework2 as fr
from fridom.framework.utils import dtype_real, jaxify
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.state import State


@partial(jaxify, dynamic=("csqr", "rossby_number"))
class ShallowWaterCore(fr.Module):

    r"""
    Shallow-water core: declares ``u``, ``v``, ``h``; linear physics.

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
        """U (east face), v (north face), h (centre), csqr (AUX)."""
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
                "h", space=fr.Collocated(),
                long_name="Geopotential (g*eta)", units="m^2/s^2"),
            fr.FieldDeclaration(
                "csqr", space=fr.Collocated(),
                lifecycle=fr.Lifecycle.AUXILIARY,
                default=ShallowWaterCore._csqr_default,
                long_name="Squared phase speed", units="m^2/s^2"),
        )

    field_references = (
        fr.FieldReference(
            "f_coriolis",
            hint="declared by a Coriolis module, e.g. "
                 "sw.modules.FPlaneCoriolis(f0=...)"),)

    parameter_declarations = (
        fr.ParameterDeclaration(
            fr.params.SCALING_ROSSBY, attr="rossby_number"),
        fr.ParameterDeclaration(
            sw_params.CSQR, attr="csqr", units="m^2/s^2"))

    def _csqr_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.grid.ScalarField:
        """Owner-method default: fill the field with ``csqr``.

        The field is halo-synced so its ghost validity already meets
        every term's stencil requirement: a carry-resident AUXILIARY
        field read (hence sync-mutated) in-step would otherwise leak
        an altered ``halo_valid`` into the scan carry and break its
        treedef stability (a framework2 interaction — see the port
        report).
        """
        field = grid.create_field(
            space, data=jnp.full(space.shape, self.csqr),
            name="csqr")
        return grid.sync(field)

    # ================================================================
    #  Tendency terms (linear)
    # ================================================================
    @fr.term(advances=("u", "v"), linear=True)
    def coriolis(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""Rotation force :math:`f\,\underset{\neg}{\boldsymbol{u}}`."""
        u, v = state["u"], state["v"]
        f = state["f_coriolis"]
        du = f.to(u.function_space) * v.to(u.function_space)
        dv = -(f.to(v.function_space) * u.to(v.function_space))
        return {"u": du, "v": dv}

    @fr.term(advances=("u", "v", "h"), linear=True)
    def gravity(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""Pressure gradient and geopotential divergence."""
        u, v, h = state["u"], state["v"], state["h"]
        c = state["csqr"]
        du = -h.diff("x")
        dv = -h.diff("y")
        flux_u = c.to(u.function_space) * u
        flux_v = c.to(v.function_space) * v
        dh = -(flux_u.diff("x") + flux_v.diff("y"))
        return {"u": du, "v": dv, "h": dh}
