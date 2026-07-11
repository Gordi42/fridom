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

Prescribed background flow (``background=``)
--------------------------------------------
With a prescribed, stationary, divergence-free background
:math:`\boldsymbol{u}_b` the scheme contributes a **second, linear**
term (V-S3: a separate ``linear=True`` term, so ``fr.model.linearize``
keeps it while dropping the nonlinear self-advection):

.. math::
    \partial_t \boldsymbol{u} \mathrel{-}=
        \nabla\cdot\left(\boldsymbol{u}_b \otimes \boldsymbol{u}
        \right), \qquad
    \partial_t p \mathrel{-}=
        \nabla\cdot\left(\boldsymbol{u}_b\, p\right)

in the old scheme's flux form: the momentum fluxes are the centred
products ``mean(u_b) * mean(u)`` at the cell centres /
:math:`\zeta`-corners, the pressure flux is ``u_b * mean(p)`` on the
velocity faces. Because the pressure flux is bilinear, the
background term plus the nonlinear term reproduce the single
full-velocity flux :math:`-\nabla\cdot((\boldsymbol{u}' +
\boldsymbol{u}_b)\,p)` exactly (to rounding); the vorticity-flux /
kinetic-energy machinery of the nonlinear term stays pure
perturbation — exactly the old module's term split.

**Scaling (signed delta vs the old model):** the old stack set
``tendencies.advection.scaling = rossby_number`` and that factor
multiplied the background transport too, so old users passed
*pre-scaled* backgrounds (the effective advecting velocity was
``Ro * u_b``). Here the background term carries **no** Rossby
factor: ``u_b`` is an O(1) velocity, like every other linear
module's coefficient. Migrating an old setup, pass the physical
(unscaled) background.

The background samples at each velocity component's own staggered
nodes (AUXILIARY fields ``u_background`` / ``v_background``,
materialized at assembly — the profile-sampling precedent) and is
validated at bind: the **discrete** divergence of the sampled
background must vanish (the term split and the exact pressure/tracer
identities assume :math:`\nabla\cdot\boldsymbol{u}_b = 0`
*discretely* — derive the background from a streamfunction), and on
bounded axes the wall-normal component must vanish at the wall
(impermeability; structural for the sampled field, checked on the
user's input). The wall flux stencils reuse the free-slip retag /
Dirichlet-fill machinery of the nonlinear term: every wall value
the background term reads contains the wall-normal background
factor, an exact zero.

**Energy:** the background term alone conserves the plain quadratic
energy :math:`\sum \tfrac12(u^2 + v^2 + p^2)` exactly
(semi-discretely, periodic and walled): flux-form transport by a
discretely divergence-free field telescopes. The thickness-weighted
invariant of the gravity + self-advection pair is *not* conserved
in the presence of a background (the background exchanges energy
with the perturbation through the :math:`h`-weighting) — expect an
:math:`O(\mathrm{Ro})` drift of that functional, controlled by the
time-integration error.
"""
from __future__ import annotations

import inspect
import numbers
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.spaces.constant import ConstantSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid

#: wall-normal background components must vanish at the wall to this
#: relative tolerance (impermeability)
_WALL_TOL = 1e-12

#: the sampled background's discrete divergence must cancel to this
#: fraction of its individual flux-difference terms
_DIV_TOL = 1e-11

_BG_COMPONENTS = ("u", "v")
_BG_AXES = {"u": "x", "v": "y"}


class SadournyAdvection(fr.model.Module):

    r"""
    Energy/enstrophy-conserving advection for ``u``, ``v``, ``p``.

    Description
    -----------
    Reads ``u``, ``v``, ``p`` and the ``csqr`` field (all owned by
    the core, declared here as references) and the ``scaling.rossby``
    parameter. Contributes one nonlinear term transporting all three
    prognostic components by name (the D1.4 name-coupling; no role
    selection) and, when a background flow is prescribed, one
    additional **linear** background-advection term (module
    docstring). Works on periodic, channel, and doubly-walled grids;
    on bounded axes the scheme realizes impermeable free-slip walls
    (module docstring).

    Parameters
    ----------
    background : Mapping[str, float | Callable] | None, optional
        Prescribed stationary background flow: maps the velocity
        component names ``"u"`` / ``"v"`` to a constant or to a
        coordinate callable (e.g. ``{"u": lambda y: -psi_y(y)}``),
        sampled at each component's own staggered nodes; a missing
        component is zero. Must be discretely divergence-free and
        impermeable on walled axes (validated at bind, taught
        errors). Carries **no** Rossby factor — pass the physical,
        unscaled velocity (the old stack scaled the background by
        ``Ro``; see the module docstring). ``None`` disables the
        background term entirely (default: None).
    """

    # The Rossby scaling multiplies a traced ``ctx.params`` scalar
    # into the tendency (a raw-data op the halo tracer cannot follow),
    # so the module declares its stencil width and is halo-trace
    # exempt (V-N2): the vector-invariant scheme reaches two cells in
    # each direction (nested interpolations to the vorticity corner);
    # the background fluxes stay within the same reach.
    extra_halo = HaloSpec({"x": 2, "y": 2})

    field_references = (
        fr.model.FieldReference("u", hint="a shallow-water core"),
        fr.model.FieldReference("v", hint="a shallow-water core"),
        fr.model.FieldReference("p", hint="a shallow-water core"),
        fr.model.FieldReference("csqr", hint="a shallow-water core"))

    parameter_references = (
        fr.model.Param(fr.model.params.SCALING_ROSSBY, default=1.0),)

    def __init__(
        self,
        background: Mapping[str, float | Callable] | None = None,
    ) -> None:
        """Normalize and locally validate the background slot."""
        if background is None:
            self._background: dict[str, float | Callable] | None = (
                None)
            return
        items = dict(background)
        unknown = sorted(set(items) - set(_BG_COMPONENTS))
        if unknown:
            raise TypeError(
                f"background= keys are the velocity component names "
                f"{_BG_COMPONENTS}, got the unknown {unknown}")
        if not items:
            raise ValueError(
                "background= needs at least one of the components "
                f"{_BG_COMPONENTS}; None disables the background "
                "term")
        for name, value in items.items():
            if not (isinstance(value, numbers.Number)
                    or callable(value)):
                raise TypeError(
                    f"background[{name!r}] must be a constant or a "
                    f"coordinate callable, got {value!r}")
        self._background = {
            name: items.get(name, 0.0) for name in _BG_COMPONENTS}

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def background(self) -> dict[str, float | Callable] | None:
        """The prescribed background flow (or None)."""
        return (None if self._background is None
                else dict(self._background))

    # ================================================================
    #  Background declarations (AUXILIARY, materialized at assembly)
    # ================================================================
    @property
    def field_declarations(self) -> tuple[fr.model.FieldDeclaration, ...]:
        """The sampled background velocities (when prescribed)."""
        if self._background is None:
            return ()
        return (
            fr.model.FieldDeclaration(
                "u_background",
                space=fr.spatial.Staggered("x", wall_bc={"x": BC.DIRICHLET}),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._u_background_default,
                long_name="Background velocity (x)", units="m/s"),
            fr.model.FieldDeclaration(
                "v_background",
                space=fr.spatial.Staggered("y", wall_bc={"y": BC.DIRICHLET}),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._v_background_default,
                long_name="Background velocity (y)", units="m/s"),
        )

    def _u_background_default(
        self, grid, space,  # noqa: ANN001
    ) -> ScalarField:
        """Owner-method default: sample ``u_b`` at the u nodes."""
        return self._sample(grid, space, "u", "u_background")

    def _v_background_default(
        self, grid, space,  # noqa: ANN001
    ) -> ScalarField:
        """Owner-method default: sample ``v_b`` at the v nodes."""
        return self._sample(grid, space, "v", "v_background")

    def _sample(
        self, grid: Grid, space, component: str,  # noqa: ANN001
        name: str,
    ) -> ScalarField:
        """Materialize one background component on its own nodes.

        A constant fills the space; a callable is sampled at the
        component's staggered nodes, with the callable free to name
        any subset of the coordinates (the wrapper stamps the full
        signature ``create_field`` expects — the profile-sampling
        precedent).
        """
        value = self._background[component]
        if not callable(value):
            data = jnp.full(space.shape, value)
            return grid.create_field(space, data=data, name=name)
        fn = value
        wanted = tuple(inspect.signature(fn).parameters)
        names = tuple(
            factor_name
            for factor in space.factors
            if not isinstance(factor, ConstantSpace)
            for factor_name in factor.names)

        def init(**coords: object) -> object:
            return fn(**{key: coords[key] for key in wanted})

        init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(
                factor_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD)
             for factor_name in names])
        return grid.create_field(space, init=init, name=name)

    # ================================================================
    #  Bind-time validation of the background (taught errors)
    # ================================================================
    def bind(self, table) -> None:  # noqa: ANN001
        """Validate the prescribed background against the grid.

        Description
        -----------
        Sampling and both checks are host-side and transient (the
        real AUXILIARY fields materialize at assembly step 8 through
        the declared defaults, from the same initializers).

        Raises
        ------
        ValueError
            If a background callable names a coordinate the grid
            does not have, if a wall-normal background component
            does not vanish on its wall, or if the sampled
            background is not discretely divergence-free.
        """
        if self._background is None:
            return
        grid = table.grid
        self._check_coordinate_names(grid)
        fields = {
            component: self._sample(
                grid, table[f"{component}_background"].space,
                component, f"{component}_background")
            for component in _BG_COMPONENTS}
        self._check_wall_normal(grid, fields)
        self._check_divergence_free(fields)

    def _check_coordinate_names(self, grid: Grid) -> None:
        """Raise if a background callable names a bad coordinate."""
        for component in _BG_COMPONENTS:
            value = self._background[component]
            if not callable(value):
                continue
            wanted = tuple(inspect.signature(value).parameters)
            unknown = sorted(set(wanted) - set(grid.names))
            if unknown:
                raise ValueError(
                    f"the background[{component!r}] callable names "
                    f"the coordinate(s) {unknown}, which the grid "
                    f"does not have (coordinates: {grid.names}); "
                    "background callables vary along the "
                    "coordinates their parameters name")

    def _check_wall_normal(
        self, grid: Grid, fields: dict[str, ScalarField],
    ) -> None:
        """Impermeability: wall-normal background zero at the wall.

        The sampled field is structurally impermeable (the wall face
        is not a DOF), so the check runs on the *user's input*: a
        constant must be zero, a callable must evaluate to zero at
        the wall positions (over the tangential nodes it names).
        """
        for component in _BG_COMPONENTS:
            axis = _BG_AXES[component]
            mesh = next(
                (factor for factor in grid.factors
                 if axis in factor.names), None)
            if mesh is None or getattr(mesh, "periodic", True):
                continue
            value = self._background[component]
            scale = max(
                1.0,
                float(np.max(np.abs(np.asarray(
                    fields[component].data)))))
            if not callable(value):
                worst = abs(float(value))
            else:
                worst = self._wall_values(
                    grid, fields[component], value, axis,
                    mesh.extent)
            if worst > _WALL_TOL * scale:
                raise ValueError(
                    f"background[{component!r}] does not vanish at "
                    f"the {axis!r} wall (max wall value {worst:.3e})"
                    ": the walls are impermeable, so the wall-normal"
                    " background component must be zero on the wall "
                    "— derive the background from a streamfunction "
                    "that is constant along each wall")

    @staticmethod
    def _wall_values(
        grid: Grid, field: ScalarField, fn: Callable, axis: str,
        extent: tuple[float, float],
    ) -> float:
        """Return the largest |fn| over both walls of ``axis``."""
        wanted = tuple(inspect.signature(fn).parameters)
        worst = 0.0
        for wall in extent:
            coords: dict[str, object] = {}
            for name in wanted:
                if name == axis:
                    coords[name] = wall
                else:
                    coords[name] = np.asarray(grid.evaluation_nodes(
                        field.function_space, name).data).ravel()
            worst = max(worst, float(np.max(np.abs(
                np.asarray(fn(**coords))))))
        return worst

    def _check_divergence_free(
        self, fields: dict[str, ScalarField],
    ) -> None:
        """Require a discretely solenoidal sampled background.

        The scheme's term split (and the exact pressure/tracer flux
        identities) assume the *discrete* divergence of the sampled
        background vanishes; the tolerance is relative to the
        individual flux-difference terms (pure cancellation error
        passes, an analytically-but-not-discretely solenoidal field
        does not).
        """
        parts = (fields["u"].diff("x"), fields["v"].diff("y"))
        div = parts[0] + parts[1]
        max_div = float(np.max(np.abs(np.asarray(div.data))))
        scale = max(
            float(np.max(np.abs(np.asarray(part.data))))
            for part in parts)
        if max_div > _DIV_TOL * scale:
            raise ValueError(
                "the sampled background flow is not discretely "
                f"divergence-free (max |div u_b| = {max_div:.3e}, "
                f"flux-difference scale {scale:.3e}): the Sadourny "
                "background term assumes div(u_b) = 0 on the grid — "
                "a non-solenoidal background silently breaks the "
                "pressure/tracer transport identities. Provide a "
                "divergence-free background, e.g. u_b = -d(psi)/dy, "
                "v_b = +d(psi)/dx from a streamfunction psi sampled "
                "at the vorticity corners")

    @fr.model.term(advances=("u", "v", "p"),
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
        rossby = ctx.params[fr.model.params.SCALING_ROSSBY]
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

    @fr.model.term(name="background_advection",
             advances=("u", "v", "p"), linear=True)
    def background_advection(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""Flux-form transport by the prescribed background flow.

        Description
        -----------
        ``-div(u_b (x) u)`` on the velocities (centre / NE-corner
        flux products, the old module's stencils) and
        ``-div(u_b p)`` on the pressure — together with the
        nonlinear term's ``-div(u' p)`` this is exactly the
        bilinear full-velocity pressure flux. **Unscaled**: no
        Rossby factor (module docstring). The wall handling reuses
        the nonlinear term's machinery: the corner space adopts the
        Dirichlet wall tags, so every wall value consumed by the
        flux divergences carries the wall-normal background factor
        — an exact zero (impermeability); the retags onto the
        velocities are the identity on periodic axes.
        """
        u, v, p = state["u"], state["v"], state["p"]
        ub = state["u_background"]
        vb = state["v_background"]
        corner = u.function_space.bare.replace(
            y=v.function_space.bare.factor("y"))

        # --- thickness: dp = -div(u_b p) ---------------------------
        dp = -((ub * p.to(ub)).diff("x")
               + (vb * p.to(vb)).diff("y"))

        # --- momentum: -div(u_b (x) u) -----------------------------
        # (fluxes at the centre along the own axis, at the NE corner
        # along the other axis — the old module's positions)
        fx_u = ub.to(p) * u.to(p)                  # centre
        fy_u = vb.to(corner) * u.to(corner)        # NE corner
        du = -(fx_u.diff("x").retag(u)
               + fy_u.diff("y").retag(u))
        fx_v = ub.to(corner) * v.to(corner)        # NE corner
        fy_v = vb.to(p) * v.to(p)                  # centre
        dv = -(fx_v.diff("x").retag(v)
               + fy_v.diff("y").retag(v))
        return {"u": du, "v": dv, "p": dp}

    def tendency_terms(self) -> tuple[fr.model.TendencyTerm, ...]:
        """Collect the terms; drop the unused background term."""
        terms = super().tendency_terms()
        if self._background is None:
            terms = tuple(term for term in terms
                          if term.name != "background_advection")
        return terms
