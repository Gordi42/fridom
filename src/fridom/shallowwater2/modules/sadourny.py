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

Walled grids (free-slip)
------------------------
On bounded mesh factors (topology-driven walls) the scheme realizes
**impermeable free-slip** walls; the treatment is fully explicit —
no boundary values are patched numerically (the old model's
``nan_to_num(q)`` corresponds to the free-slip corner tag below).

- *Impermeability* is structural: the wall-normal velocity lives on
  interior faces only (``Inner``, Dirichlet), the wall face is a
  boundary condition, not a DOF. Thickness and momentum fluxes
  through the wall vanish through the Dirichlet ghost fill of the
  flux divergences, so **mass is conserved exactly**.
- *Free slip* enters in exactly one place: the vorticity
  :math:`\zeta = \partial_x v - \partial_y u` is assembled from
  BC-free stencil outputs and **retagged onto the Dirichlet corner
  space** (wall-normal factor of each velocity). Declaring
  :math:`\partial_y u` odd across a ``y`` wall asserts zero wall
  shear, i.e. :math:`\zeta = 0` and hence :math:`q = 0` **at** the
  wall — the free-slip condition. Wall corners carry no DOFs; the
  choice acts only through the one-layer ghost fills below.
- The corner chain (:math:`q`, the corner mass fluxes) inherits the
  Dirichlet tag. A tag is a *wall-value claim* consumed by the
  staggered fills, and every wall value the scheme actually reads is
  an exact zero: :math:`(f_v q)`, :math:`(f_u q)` and :math:`v^2` /
  :math:`u^2` all contain a factor that vanishes on the wall (the
  wall-normal mass flux by impermeability, :math:`q` by free slip).
  The corner-interpolated thickness is also Dirichlet-tagged as a
  formal device; its (false) fill is never consumed.

**Discrete budgets on the walled grid** (verified to machine
precision, channel and doubly-walled):

- The vorticity-flux exchange — the ``u`` equation's
  :math:`+\overline{f_v q}` against the ``v`` equation's
  :math:`-\overline{f_u q}` — is an antisymmetric coupling under the
  thickness-weighted energy sum for **any** finite corner ``q``:
  transposing the wall-zero average lands exactly on the interior
  corners, so energy conservation does *not* depend on the
  :math:`q = 0` choice, only on the zero *wall value of the flux*
  (exact, since the wall-normal mass flux vanishes).
- The total energy
  :math:`E = \sum \tfrac12 \bar{h}^x u^2 + \tfrac12 \bar{h}^y v^2
  + \tfrac12 p^2` (with :math:`h = p_\mathrm{full}`) is conserved
  **exactly** (semi-discretely) by this term *together with* the
  core's linear gravity term — the mass flux :math:`u h` is split
  across the two terms, so the invariant belongs to the pair. The
  kinetic-energy gradient / mass-flux pair telescopes because the
  scheme's :math:`E_\mathrm{kin}` uses the same wall-zero average of
  :math:`v^2` (exact: :math:`v = 0` on the wall).
- :math:`q = 0` at the wall governs the **circulation/enstrophy**
  budget instead: the advective vorticity flux through the wall
  corners is identically zero (no boundary vorticity/enstrophy
  source), and the total circulation changes only through the
  along-wall tangential acceleration.
- The split Coriolis module (``f`` outside ``q``) conserves the
  quadratic M-energy exactly but commits an :math:`O(\mathrm{Ro})`
  commutator error in the thickness-weighted energy — as in v1.
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
    selection). Works on periodic, channel, and doubly-walled grids;
    on bounded axes the scheme realizes impermeable free-slip walls
    (module docstring).
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

    @fr.term(advances=("u", "v", "p"),
             transports=("u", "v", "p"), linear=False)
    def advect(self, state, ctx) -> dict:  # noqa: ANN001
        """Return the Sadourny vector-invariant tendency (Ro-scaled).

        Description
        -----------
        On walled grids the only wall-specific step is the free-slip
        retag of the vorticity onto the Dirichlet corner space (and
        the matching retags of the kinetic-energy gradients onto the
        velocities, the nonhydro-projection precedent); every retag
        is the identity on periodic axes, so the periodic scheme is
        reproduced bit for bit.
        """
        rossby = ctx.params[fr.params.SCALING_ROSSBY]
        u, v, p = state["u"], state["v"], state["p"]
        c = state["csqr"]

        # full geopotential thickness (centre) — the csqr-FIELD fix
        # (c is the centre csqr field, never the scalar; old bug)
        p_full = c.to(p) + rossby * p

        # --- thickness tendency  dp = -Ro div(u p_e, v p_n) --------
        # (wall-normal flux lives on interior faces; the Dirichlet
        # fill closes the divergence with a zero wall flux)
        flux_u = u * p.to(u)                       # u face (east)
        flux_v = v * p.to(v)                       # v face (north)
        dp = rossby * -(flux_u.diff("x") + flux_v.diff("y"))

        # --- momentum: vorticity flux + kinetic-energy gradient ----
        # The NE-corner space adopts each velocity's wall tag on the
        # OTHER velocity's axis (Dirichlet on bounded factors);
        # retagging the BC-free stencil outputs onto it is the
        # explicit free-slip claim zeta = 0 at the wall (docstring).
        corner = u.function_space.bare.replace(
            y=v.function_space.bare.factor("y"))
        zeta = (v.diff("x").retag(corner)
                - u.diff("y").retag(corner))
        q = zeta / p_full.to(zeta)                 # potential vort.
        fu = (u * p_full.to(u)).to(zeta)           # mass flux, NE
        fv = (v * p_full.to(v)).to(zeta)
        ekin = 0.5 * ((u * u).to(p) + (v * v).to(p))  # centre
        du = rossby * ((fv * q).to(u)
                       - ekin.diff("x").retag(u))
        dv = rossby * (-(fu * q).to(v)
                       - ekin.diff("y").retag(v))
        return {"u": du, "v": dv, "p": dp}
