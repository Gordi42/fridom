r"""
The Sadourny [1975] energy/enstrophy-conserving advection scheme.

Description
-----------
The nonlinear shallow-water advection, written in the vector-invariant
form that conserves the discrete total energy (and, in the enstrophy
variant, potential enstrophy):

.. math::
    \partial_t \boldsymbol{u} =
        - \underset{\neg}{\boldsymbol{f_u}}\,q
        - \nabla\, E_\mathrm{kin} , \qquad
    \partial_t h = - \nabla\cdot\left(\boldsymbol{u}\,h\right)

with the mass flux :math:`\boldsymbol{f_u} = h_\mathrm{full}
\boldsymbol{u}`, the potential vorticity
:math:`q = \zeta / h_\mathrm{full}`, and the full geopotential
thickness :math:`h_\mathrm{full} = c^2 + \mathrm{Ro}\,h`. Every term
is scaled by the Rossby number ``scaling.rossby`` (read from
``ctx.params``); the module owns no numeric leaves.

**Signed delta vs the old model (§8.8):** the old scheme read the
*scalar* ``csqr`` in ``h_full`` (an outright bug on variable depth).
This port reads the ``csqr`` **field** — ``state["csqr"]`` — so
``h_full`` is spatially correct. Bitwise-identical to the old scheme
only for constant depth.
"""
from __future__ import annotations

import fridom.framework2 as fr
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.shallowwater2._util import scale


class SadournyAdvection(fr.Module):

    r"""
    Energy/enstrophy-conserving advection for ``u``, ``v``, ``h``.

    Description
    -----------
    Reads ``u``, ``v``, ``h`` and the ``csqr`` field (all owned by
    the core, declared here as references) and the ``scaling.rossby``
    parameter. Contributes one nonlinear term transporting all three
    prognostic components by name (the D1.4 name-coupling; no role
    selection). Uses only field-algebra operators, so its wider
    stencil is followed by the halo trace (no ``extra_halo``).
    """

    # The Rossby scaling multiplies a traced ``ctx.params`` scalar
    # into the tendency (a raw-data op the halo tracer cannot follow),
    # so the module declares its stencil width and is halo-trace
    # exempt (V-N2): the vector-invariant scheme reaches two cells in
    # each direction (nested interpolations to the vorticity corner).
    extra_halo = HaloSpec({"x": 2, "y": 2})

    field_references = (
        fr.FieldReference("u", hint="a shallow-water core"),
        fr.FieldReference("v", hint="a shallow-water core"),
        fr.FieldReference("h", hint="a shallow-water core"),
        fr.FieldReference("csqr", hint="a shallow-water core"))

    parameter_references = (
        fr.Param(fr.params.SCALING_ROSSBY, default=1.0),)

    @fr.term(advances=("u", "v", "h"),
             transports=("u", "v", "h"), linear=False)
    def advect(self, state, ctx) -> dict:  # noqa: ANN001
        """Return the Sadourny vector-invariant tendency (Ro-scaled)."""
        rossby = ctx.params[fr.params.SCALING_ROSSBY]
        u, v, h = state["u"], state["v"], state["h"]
        c = state["csqr"]

        # full geopotential thickness (centre) — the csqr-FIELD fix
        # (c is the centre csqr field, never the scalar; old bug)
        h_full = c.to(h.function_space) + scale(h, rossby)

        u_sp, v_sp, h_sp = (u.function_space, v.function_space,
                            h.function_space)

        # --- thickness tendency  dh = -Ro div(u h_e, v h_n) --------
        flux_u = u * h.to(u_sp)                     # u face (east)
        flux_v = v * h.to(v_sp)                     # v face (north)
        dh = scale(-(flux_u.diff("x") + flux_v.diff("y")), rossby)

        # --- momentum: vorticity flux + kinetic-energy gradient ----
        zeta = v.diff("x") - u.diff("y")           # NE corner
        ne = zeta.function_space
        q = zeta / h_full.to(ne)                   # potential vort.
        fu = (u * h_full.to(u_sp)).to(ne)          # mass flux, NE
        fv = (v * h_full.to(v_sp)).to(ne)
        ekin = 0.5 * ((u * u).to(h_sp) + (v * v).to(h_sp))  # centre
        du = scale((fv * q).to(u_sp) - ekin.diff("x"), rossby)
        dv = scale(-(fu * q).to(v_sp) - ekin.diff("y"), rossby)
        return {"u": du, "v": dv, "h": dh}
