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
    \partial_t p = - \nabla\cdot\left(\boldsymbol{u}\,p\right)

with the mass flux :math:`\boldsymbol{f_u} = p_\mathrm{full}
\boldsymbol{u}`, the potential vorticity
:math:`q = \zeta / p_\mathrm{full}`, and the full geopotential
thickness :math:`p_\mathrm{full} = c^2 + \mathrm{Ro}\,p`. Every term
is scaled by the Rossby number ``scaling.rossby`` (read from
``ctx.params``); the module owns no numeric leaves.

**Signed delta vs the old model (§8.8):** the old scheme read the
*scalar* ``csqr`` in ``p_full`` (an outright bug on variable depth).
This port reads the ``csqr`` **field** — ``state["csqr"]`` — so
``p_full`` is spatially correct. Bitwise-identical to the old scheme
only for constant depth.

**Walled grids are future work**: the vector-invariant flux stencils
near rigid walls (bounded, non-periodic mesh factors) are not covered
yet — genuinely odd*odd products (e.g. the vorticity flux) need a
parity-aware product codomain — so ``bind`` rejects walled grids with
a taught error; build a linear model (``advection=False`` in
``sw.Model``) instead.
"""
from __future__ import annotations

import fridom.framework2 as fr
from fridom.framework2.grid.decomposition.halo import HaloSpec


class SadournyAdvection(fr.Module):

    r"""
    Energy/enstrophy-conserving advection for ``u``, ``v``, ``p``.

    Description
    -----------
    Reads ``u``, ``v``, ``p`` and the ``csqr`` field (all owned by
    the core, declared here as references) and the ``scaling.rossby``
    parameter. Contributes one nonlinear term transporting all three
    prognostic components by name (the D1.4 name-coupling; no role
    selection).
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
        fr.FieldReference("p", hint="a shallow-water core"),
        fr.FieldReference("csqr", hint="a shallow-water core"))

    parameter_references = (
        fr.Param(fr.params.SCALING_ROSSBY, default=1.0),)

    def bind(self, table: object) -> None:
        """Reject walled grids with a taught error (future work).

        Raises
        ------
        NotImplementedError
            On a walled grid (any bounded mesh factor): the
            vector-invariant flux stencils near rigid walls are
            future work, and the natural downstream failure (an
            operator dispatch mismatch deep in the flux chain)
            would be cryptic.
        """
        factors = getattr(table.grid, "factors", ())
        walled = tuple(
            name for mesh in factors for name in mesh.names
            if not getattr(mesh, "periodic", True))
        if walled:
            raise NotImplementedError(
                f"SadournyAdvection does not support walled grids "
                f"yet (bounded coordinates: {walled}); the "
                "vector-invariant flux stencils near rigid walls "
                "are future work. Build a linear model instead "
                "(advection=False in sw.Model) or drop the "
                "advection module")

    @fr.term(advances=("u", "v", "p"),
             transports=("u", "v", "p"), linear=False)
    def advect(self, state, ctx) -> dict:  # noqa: ANN001
        """Return the Sadourny vector-invariant tendency (Ro-scaled)."""
        rossby = ctx.params[fr.params.SCALING_ROSSBY]
        u, v, p = state["u"], state["v"], state["p"]
        c = state["csqr"]

        # full geopotential thickness (centre) — the csqr-FIELD fix
        # (c is the centre csqr field, never the scalar; old bug)
        p_full = c.to(p) + rossby * p

        # --- thickness tendency  dp = -Ro div(u p_e, v p_n) --------
        flux_u = u * p.to(u)                       # u face (east)
        flux_v = v * p.to(v)                       # v face (north)
        dp = rossby * -(flux_u.diff("x") + flux_v.diff("y"))

        # --- momentum: vorticity flux + kinetic-energy gradient ----
        zeta = v.diff("x") - u.diff("y")           # NE corner
        q = zeta / p_full.to(zeta)                 # potential vort.
        fu = (u * p_full.to(u)).to(zeta)           # mass flux, NE
        fv = (v * p_full.to(v)).to(zeta)
        ekin = 0.5 * ((u * u).to(p) + (v * v).to(p))  # centre
        du = rossby * ((fv * q).to(u) - ekin.diff("x"))
        dv = rossby * (-(fu * q).to(v) - ekin.diff("y"))
        return {"u": du, "v": dv, "p": dp}
