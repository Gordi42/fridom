r"""
The shallow-water dynamical core module.

Description
-----------
``DynamicalCore`` declares the state vocabulary (``u``, ``v``,
``p``), owns the squared phase speed :math:`c^2` (the AUXILIARY
``csqr`` field and, when constant, the ``shallowwater.csqr`` scalar)
and the Rossby scaling (``scaling.rossby``), and contributes the
single **linear** pressure-gradient / geopotential-divergence term:

.. math::
    \partial_t \boldsymbol{u} = - \nabla p , \qquad
    \partial_t p = -\nabla\cdot\left(c^2 \boldsymbol{u}\right)

**Variable depth**: ``csqr`` accepts a callable :math:`c^2(y)` (the
coriolis two-type precedent, folded into one core because the core
also owns the whole state vocabulary): the ``csqr`` field is then
declared on a meridional ``fr.spatial.Profile("y")`` and the constant
``shallowwater.csqr`` scalar is **not** provided
(provides-implies-constancy, 02_rules) — analytic consumers keyed on
the provide reject the model, the dense-column channel engine serves
it. The tendency terms are untouched either way: they read the
``csqr`` *field* (:math:`c^2` sits inside the divergence — the flux
form), which is exactly the sampling the variable-depth energy
metric ``diag(c^2, c^2, 1)`` pairs with. Pair a varying ``csqr``
with a Coriolis module carrying ``metric_weight="csqr"`` (the
thickness-weighted rotation) so the rotation stays energy-conserving
under that metric; the ``sw.Model`` preset wires this automatically.

The rotation :math:`f\,\underset{\neg}{\boldsymbol{u}}` is **not** a
core term: it is carried by the shared Coriolis module
(``fr.model.modules.FPlaneCoriolis`` / ``BetaPlaneCoriolis``, or
``RotationCoriolis`` on a chart grid), which declares the
``f_coriolis`` field and the ``+f v`` / ``-f u`` coupling — and is
opt-in: a model assembled without one simply does not rotate. The
gravity term here is unscaled (the Rossby number multiplies only the
advection, D2.2). The nonlinear Sadourny advection is a separate
module.

Chart grids (coordinate-systems plan, stage C2)
-----------------------------------------------
The same module assembles on a chart-coupled grid (a
``CoordinateMapping`` embedding chart, e.g. the lat-lon sphere):
declare the coordinate names via ``coords=`` and the terms select
the metric-aware path (a static grid property, never a traced
value), resolving the seeded ``"grad"`` / ``"div"`` /
``"raise_index"`` kinds through the grid dispatch — the module never
hand-builds metric compositions.

**Velocity convention (recorded per the C2 task):** on chart grids
the prognostic ``u`` / ``v`` are the **contravariant** components
:math:`u^\lambda = \dot\lambda`, :math:`u^\varphi = \dot\varphi`
(units 1/s on the sphere), stored untagged in the state; terms tag
them ``CONTRAVARIANT`` at the seams. The conversion points to
physical (m/s) components are ``State.u_physical`` /
``State.v_physical`` and the metric-aware ``ekin`` diagnostic —
:math:`u_{\rm east} = \sqrt{g_{\lambda\lambda}}\,u^\lambda`,
:math:`v_{\rm north} = \sqrt{g_{\varphi\varphi}}\,u^\varphi`,
derived per call via ``grid.metric``. On flat grids the convention
degenerates to the usual physical velocities and the flat code path
is taken verbatim (bitwise; the hard results-neutrality gate).

The chart gravity term is

.. math::
    \partial_t u^i = -\,g^{ij}\,\partial_j p , \qquad
    \partial_t p = -\frac{1}{\sqrt{g}}\,
        \partial_i\left(\sqrt{g}\, c^2 u^i\right)

via ``grad`` -> ``raise_index`` on the pressure and the flux-form
metric ``div`` on the tagged geopotential flux. A spherical model is
assembled through the same preset (see ``sw.Model``'s ``coords=``
docs for the grid recipe).
"""
from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.model.halo_demand import derive_extra_halo
from fridom.model.scheduled_field import ProfileFunction, profile_coords
from fridom.model.stages import Stage, StageKind
from fridom.model.time_dependent import TimeDependent
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.diagnostics import DIAGNOSTICS
from fridom.shallowwater2.modules.immersed_weighting import (
    mask_field,
    scale_divergence,
    weight_flux,
)
from fridom.shallowwater2.state import State
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.scalars import Variance

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


@partial(jaxify, dynamic=("csqr", "rossby_number", "_csqr_law"))
class DynamicalCore(fr.model.Module):

    r"""
    Shallow-water core: declares ``u``, ``v``, ``p``; linear physics.

    Parameters
    ----------
    csqr : float | Callable | fr.model.ProfileFunction, optional
        The squared gravity-wave phase speed :math:`c^2`. A float is
        the constant depth: published as ``shallowwater.csqr`` and
        materialized into the one-DOF ``csqr`` field. A callable
        ``csqr(y)`` (evaluated on the meridional coordinate) is the
        static variable depth: materialized into a ``csqr`` field on
        ``fr.spatial.Profile("y")``, with **no** ``shallowwater.csqr``
        provide (provides-implies-constancy). A ``ProfileFunction``
        ``c^2(y,t)`` is the time-dependent variable depth (TDF-D7): the
        ``csqr`` field is marked ``time_dependent`` and rewritten each
        substage by a SELF_UPDATE stage (also no scalar provide)
        (default: 1.0).
    rossby_number : float | fr.model.Ramp, optional
        The Rossby number scaling the (separate) advection term;
        published as ``scaling.rossby`` (default: 1.0); may be a
        ``fr.model.Ramp`` for a spun-up nonlinearity.
    coords : tuple[str, str], optional
        The (zonal, meridional) coordinate names, in the grid's
        factor order — ``("lon", "lat")`` on the standard sphere
        chart (default: ``("x", "y")``).
    meridional : str | None, optional
        The meridional coordinate name a callable ``csqr`` varies
        along; None uses ``coords[1]`` (default: None).
    """

    #: The vocabulary class this core supplies (D1.3 commitment 4).
    state_type = State

    #: Bound parameterful diagnostics (the D1.3 commitment-4 channel).
    diagnostics = DIAGNOSTICS

    # The chart-path gravity term resolves the metric-aware kinds,
    # whose multi-row block application the halo tracer cannot
    # follow (it collects traced operands into VectorFields), so the
    # module declares its stencil width and is halo-trace exempt
    # (V-N2, the Sadourny precedent). On an immersed grid the flux-form
    # continuity multiplies the concrete open-area / plan-area fraction
    # fields (IP-D4) — a raw-data op the halo tracer cannot follow (the
    # fraction field is materialized, not traced), exactly the advection
    # precedent — so the immersed path is halo-trace exempt and declares
    # its (order-2 staggered) FD-stencil halo here too. The width is
    # DERIVED at bind from the ``diff`` rows the term applies (width 1 —
    # the cross-interp hop telescopes, see ``_derive_extra_halo``), not
    # a literal. None on a plain flat grid: the unimmersed path stays
    # fully halo-traced (the parity guard).
    @property
    def extra_halo(self) -> HaloSpec | None:
        """Derived staggered ghost width on chart / immersed grids."""
        return self._extra_halo

    def __init__(
        self,
        csqr: float | Callable = 1.0,
        rossby_number: float | fr.model.Ramp = 1.0,
        *,
        coords: tuple[str, str] = ("x", "y"),
        meridional: str | None = None,
    ) -> None:
        """Store the leaves; a callable ``csqr`` stays static.

        Raises
        ------
        TypeError
            On invalid ``coords``, or a *scalar* time-dependent ``csqr``
            (an ``fr.Ramp``) — the phase speed is the AUXILIARY ``csqr``
            FIELD read by several terms, so a spatially varying
            time-dependent ``c^2`` is a ``c^2(y,t)`` law
            (``fr.model.ProfileFunction``, TDF-D7), not a scalar ramp.
        """
        coords = tuple(coords)
        if (len(coords) != 2  # noqa: PLR2004 — zonal + meridional
                or not all(isinstance(c, str) for c in coords)
                or coords[0] == coords[1]):
            raise TypeError(
                "coords names the (zonal, meridional) coordinates: "
                f"two distinct strings, got {coords!r}")
        # a ProfileFunction c^2(y,t) is the general time-dependent path
        # (TDF-D7): the csqr field is marked time_dependent and rewritten
        # each substage by a SELF_UPDATE stage. It is neither callable nor
        # a scalar TimeDependent, so it must be routed before both.
        if isinstance(csqr, ProfileFunction):
            self._csqr_law: ProfileFunction | None = csqr
            self._csqr_fn = None
            self.csqr = None
        elif isinstance(csqr, TimeDependent):
            # a scalar Ramp is callable, so this MUST precede the
            # callable(csqr) profile branch or it would be read as c^2(y)
            raise TypeError(
                f"csqr={csqr!r} is a scalar time-dependent value, but "
                "c^2 is materialized as the AUXILIARY csqr FIELD and "
                "read as a field by several terms (the gravity flux "
                "divergence with c^2 inside the divergence, the Sadourny "
                "advection, and the thickness-weighted rotation), so a "
                "spatially varying time-dependent c^2 is a c^2(y,t) law: "
                "pass an fr.model.ProfileFunction (TDF-D7). A constant "
                "c^2 is a float; a static profile is a callable c^2(y); "
                "ramp scaling.rossby or coriolis.f0 for a scalar "
                "time-dependent run")
        else:
            self._csqr_law = None
            self._csqr_fn = csqr if callable(csqr) else None
            self.csqr = None if callable(csqr) else fr.model.leaf(csqr)
        self.rossby_number = fr.model.leaf(rossby_number)
        self._coords: tuple[str, str] = coords
        self._meridional = (coords[1] if meridional is None
                            else meridional)
        # whether the bound grid is chart-coupled; set by bind()
        # (assembly step 4, before the extra_halo merge of step 7)
        self._charted: bool = False
        # whether the bound grid carries an immersed domain (masked
        # continuity + pressure gradient); set by bind()
        self._immersed: bool = False
        # the chart / immersed gravity term's derived halo substitute
        # (V-N2), computed at bind from the staggered rows it applies;
        # None on a plain flat grid (the traceable path).
        self._extra_halo: HaloSpec | None = None

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def coords(self) -> tuple[str, str]:
        """The (zonal, meridional) coordinate names."""
        return self._coords

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_declarations(self) -> tuple[fr.model.FieldDeclaration, ...]:
        """U (east face), v (north face), p (centre), csqr (AUX)."""
        if self._csqr_law is not None:
            csqr_decl = fr.model.FieldDeclaration(
                "csqr", space=fr.spatial.Profile(self._meridional),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._csqr_law_default,
                long_name="Squared phase speed", units="m^2/s^2",
                time_dependent=True)
        elif self._csqr_fn is None:
            csqr_decl = fr.model.FieldDeclaration(
                "csqr", space=fr.spatial.Profile(),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._csqr_default,
                long_name="Squared phase speed", units="m^2/s^2")
        else:
            csqr_decl = fr.model.FieldDeclaration(
                "csqr", space=fr.spatial.Profile(self._meridional),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._csqr_profile_default,
                long_name="Squared phase speed", units="m^2/s^2")
        zonal, meridional = self._coords
        return (
            fr.model.FieldDeclaration.velocity(
                "u", zonal, space=fr.spatial.Staggered(zonal),
                long_name=f"Velocity ({zonal})", units="m/s"),
            fr.model.FieldDeclaration.velocity(
                "v", meridional,
                space=fr.spatial.Staggered(meridional),
                long_name=f"Velocity ({meridional})", units="m/s"),
            fr.model.FieldDeclaration(
                "p", space=fr.spatial.Collocated(),
                long_name="Pressure (g*eta)", units="m^2/s^2"),
            csqr_decl,
        )

    @property
    def parameter_declarations(
        self,
    ) -> tuple[fr.model.ParameterDeclaration, ...]:
        """Rossby always; ``shallowwater.csqr`` only when constant."""
        decls = (
            fr.model.ParameterDeclaration(
                fr.model.params.SCALING_ROSSBY, attr="rossby_number"),
        )
        # provides-implies-constancy: only the constant depth publishes
        # shallowwater.csqr (a static profile or a c^2(y,t) law does not)
        if self._csqr_fn is None and self._csqr_law is None:
            decls += (
                fr.model.ParameterDeclaration(
                    sw_params.CSQR, attr="csqr", units="m^2/s^2"),
            )
        return decls

    def _csqr_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: fill the one-DOF profile with ``csqr``.

        The field is declared on ``fr.spatial.Profile()`` (constant depth is a
        single degree of freedom); the GAP-A ConstantSpace/Profile
        broadcast lifts it to the nodal join wherever a term multiplies
        it (``c.to(u) * u``). No ``grid.sync``
        pre-syncing: the GAP-B fix keeps carry-resident AUXILIARY
        fields scan-treedef-stable without pre-flooding their halos.
        """
        return grid.create_field(
            space, data=jnp.full(space.shape, self.csqr),
            name="csqr")

    def _csqr_profile_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: materialize the ``csqr(y)`` profile.

        The meridional profile carries a single non-constant
        coordinate, so ``init`` names exactly that coordinate; the
        signature is stamped dynamically to match ``self._meridional``
        (the ``BetaPlaneCoriolis._f_default`` precedent). No
        pre-syncing (GAP-B).
        """
        fn, mer = self._csqr_fn, self._meridional

        def init(**coords: object) -> object:
            return fn(coords[mer])

        init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(
                mer, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        return grid.create_field(space, init=init, name="csqr")

    def _csqr_law_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: sample the ``c^2(y,t)`` law at ``t = 0``.

        The AUXILIARY field is materialized as the ``t = 0`` snapshot so
        it keeps a valid static treedef; the SELF_UPDATE stage rewrites it
        with the stage-time value each substage (the frozen snapshot is
        never read at run time). No pre-syncing (GAP-B).
        """
        coords = profile_coords(grid, space, (self._meridional,))
        data = self._csqr_law.sample(coords, 0.0, space.shape)
        return grid.create_field(space, data=data, name="csqr")

    # ================================================================
    #  The SELF_UPDATE stage (law path only, S1 per substage)
    # ================================================================
    @property
    def stages(self) -> tuple[Stage, ...]:
        """The per-substage ``c^2(y,t)`` recompute (law path only)."""
        if self._csqr_law is None:
            return ()
        return (Stage(
            kind=StageKind.SELF_UPDATE, fn="_update_csqr", name="csqr",
            reads=("csqr",), writes=("csqr",)),)

    def _update_csqr(self, state, ctx) -> dict:  # noqa: ANN001
        """Re-evaluate the ``c^2(y,t)`` law at the substage clock (TDF-D7).

        SELF_UPDATE runs first in every substage (S1), so every ``csqr``
        consumer — the gravity divergence, Sadourny advection, the
        thickness-weighted rotation, the energy metric, diagnostics —
        reads the stage-time field, consistent with ``eval_params``.
        """
        time = getattr(ctx.clock, "time", ctx.clock)
        field = state["csqr"]
        space = field.function_space
        coords = profile_coords(field.grid, space, (self._meridional,))
        value = self._csqr_law.sample(coords, time, space.shape)
        return {"csqr": field.with_data(value)}

    # ================================================================
    #  Bind-time validation (taught errors)
    # ================================================================
    def bind(self, table) -> None:  # noqa: ANN001
        """On chart grids, require ``coords`` to match the chart.

        Raises
        ------
        ValueError
            If the grid carries an embedding chart whose coordinate
            family does not match ``coords`` in the grid's factor
            order (the metric-aware kinds match vector components
            to axes positionally, so the order is load-bearing).
        """
        grid = table.grid
        self._immersed = getattr(grid, "immersed", None) is not None
        chart = grid.chart_coords
        self._charted = chart is not None
        if chart is not None:
            expected = tuple(
                name for name in grid.names if name in set(chart))
            if self._coords != expected:
                raise ValueError(
                    f"DynamicalCore coords={self._coords!r} do not "
                    f"match the grid's chart coordinates {expected!r} "
                    "(in factor order); pass coords=(zonal, meridional) "
                    "matching the grid, e.g. coords=('lon', 'lat') on "
                    "the standard sphere chart")
        self._extra_halo = self._derive_extra_halo(table)

    def _derive_extra_halo(self, table) -> HaloSpec | None:  # noqa: ANN001
        r"""Derive the chart / immersed gravity term's ghost width (V-N2).

        Description
        -----------
        On a plain flat grid the gravity term is a traceable staggered
        difference, so no substitute is declared (``None``) — **unless**
        a ``c^2(y,t)`` law drives a SELF_UPDATE rewrite of ``csqr`` from
        raw data (halo-trace exempt, V-N2), in which case the module
        declares the term's reach itself (one ghost per axis, merged with
        any chart/immersed derivation). On a
        **chart** or **immersed** grid it resolves metric-aware /
        fraction-weighted rows the halo trace cannot follow, so the
        module declares its own width — derived from the order-2
        staggered ``diff`` rows the term applies, not a literal. The
        pressure gradient differences the cell pressure onto the
        velocity faces (reach 1 per coordinate) and the geopotential /
        continuity flux differences the face flux back onto the cell
        (reach 1); the two are parallel outputs, so the demand is their
        per-side max, 1.

        On a **non-orthogonal** chart the contravariant tendency adds a
        ``raise_index`` cross-interpolation hop, but its window is
        opposite-biased to the gradient difference it re-aligns and
        telescopes two-sidedly back to reach 1
        (``[0, +1] ⊕ [-1, 0] = [-1, +1]``, ``storage_halo_width.md``
        §1) — so the chart gravity is 1 whether or not the chart is
        orthogonal (empirically bitwise-verified,
        ``pressure_solver_halo.md``). A registry override of the
        differences moves the value.
        """
        grid = table.grid
        derived: HaloSpec | None = None
        if self._charted or self._immersed:
            registry = grid.dispatch
            p = table["p"].space
            vel = (table["u"].space, table["v"].space)
            grad_leg: dict[str, list[tuple[str, object]]] = {}
            div_leg: dict[str, list[tuple[str, object]]] = {}
            for axis in self._coords:
                centre = p.factor(axis)
                grad_leg[axis] = [("diff", centre)]
                face = next(vs.factor(axis) for vs in vel
                            if vs.factor(axis) is not centre)
                div_leg[axis] = [("diff", face)]
            derived = derive_extra_halo(
                registry, self._coords, [div_leg, grad_leg])
        # a c^2(y,t) law SELF_UPDATE rewrites csqr from raw sampled data
        # (halo-trace exempt, V-N2), so the module declares the gravity
        # term's reach itself: one ghost per axis covers the staggered
        # diff / interp hops (reach 1), merged with any chart/immersed
        # derivation above.
        if self._csqr_law is not None:
            profile = HaloSpec(dict.fromkeys(grid.names, 1))
            derived = (profile if derived is None
                       else derived.merge_max(profile))
        return derived

    # ================================================================
    #  Tendency terms (linear)
    # ================================================================
    @fr.model.term(advances=("u", "v", "p"), linear=True,
                   linear_fields=("csqr",))
    def gravity(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""Pressure gradient and geopotential divergence.

        .. math::
            \partial_t \boldsymbol{u} = - \nabla p , \qquad
            \partial_t p = -\nabla\cdot\left(c^2 \boldsymbol{u}\right)

        Pure field arithmetic: ``c^2`` sits INSIDE the divergence
        (``(c.to(u) * u).diff("x")``, the flux form) so the discrete
        stencil matches ``diff(c^2 u)``. The ``csqr`` field lifts from
        its one-DOF ``fr.spatial.Profile()`` onto each velocity face via the
        ConstantSpace broadcast in ``.to``.

        The pressure-gradient entries retag onto their velocities:
        nodal stencil outputs are BC-free, but on a walled grid each
        wall-normal velocity carries the derived Dirichlet wall tag
        on its own axis, so the entry adopts it (the nonhydro
        projection precedent) — identity on periodic grids. The flux
        entries need no retag: ``csqr.to(u)`` adopts the velocity's
        tag (BC-sibling adoption) and the divergence lands BC-free,
        which is ``p``'s space.

        On a chart grid (module docstring) the same physics resolves
        the seeded metric-aware kinds: ``grad`` -> ``raise_index``
        turns the covariant pressure gradient into the contravariant
        tendency (``-g^{ij} d_j p``), and the flux-form ``div``
        carries the ``sqrt_g``-weighted geopotential flux; the final
        retags strip the variance claim and restore the velocities'
        wall tags. On the identity chart every metric factor is an
        exact 1.0, reproducing the flat path **to rounding** — not
        bitwise: ``extra_halo`` is chart-conditional (2 cells per
        axis on a chart, none on a flat grid), so the two pad their
        storage differently and XLA fuses the stencil differently
        (FMA contraction in one path, multiply-then-add in the
        other). The results-neutrality gate that *is* bitwise is the
        one that matters and is unaffected: a **flat** grid takes the
        flat branch verbatim (module docstring).
        """
        u, v, p = state["u"], state["v"], state["p"]
        csqr = state["csqr"]
        zonal, meridional = self._coords
        if u.grid.chart_coords is None:
            if getattr(u.grid, "immersed", None) is not None:
                return self._gravity_immersed(u, v, p, csqr)
            return {
                "u": (-p.diff(zonal)).retag(u),
                "v": (-p.diff(meridional)).retag(v),
                "p": (-(csqr.to(u) * u).diff(zonal)
                      - (csqr.to(v) * v).diff(meridional)),
            }
        dispatch = u.grid.dispatch
        con = Variance.CONTRAVARIANT
        grad = dispatch.resolve("grad", p.function_space.bare)
        gp = grad(p)
        raise_index = dispatch.resolve(
            "raise_index", gp[zonal].function_space.bare)
        raised = raise_index(gp)
        flux = VectorField({
            zonal: (csqr.to(u) * u).with_variance(con),
            meridional: (csqr.to(v) * v).with_variance(con)})
        div = dispatch.resolve(
            "div", flux[zonal].function_space.bare)
        return {
            "u": (-raised[zonal]).retag(u),
            "v": (-raised[meridional]).retag(v),
            "p": -div(flux),
        }

    def _gravity_immersed(self, u, v, p, csqr) -> dict:  # noqa: ANN001
        r"""Masked pressure gradient and fraction-weighted continuity.

        Description
        -----------
        The cut-cell weighting of the linear physics (IP-D10):

        .. math::
            \partial_t \boldsymbol{u} = -\,m \odot \nabla p , \qquad
            \partial_t p = -\frac{1}{\theta}\,
                \nabla\cdot\left(\alpha \odot c^2 \boldsymbol{u}\right)

        The geopotential flux :math:`c^2 u` is weighted by the open-area
        face fraction :math:`\alpha_f` before the divergence, whose sum
        is then divided by the wet plan-area fraction :math:`\theta_c`
        (guarded: a dry cell stays exactly ``0``) — the ``theta V``-
        weighted mass ``sum_c theta_c V_c p_c`` is conserved to machine
        zero (the open-area flux differences telescope, an
        :math:`\alpha = 0` face carrying none). The pressure gradient is
        the plain two-point difference masked by the boolean per-space
        face mask :math:`m` (``theta > 0``), so no tendency drives a
        closed face. This path runs **only** on an immersed grid; the
        term is halo-trace exempt here (``extra_halo``) because the
        fraction multiplies drop to concrete fields.
        """
        immersed = u.grid.immersed
        zonal, meridional = self._coords
        flux_u = weight_flux(immersed, csqr.to(u) * u)
        flux_v = weight_flux(immersed, csqr.to(v) * v)
        div = -(flux_u.diff(zonal) + flux_v.diff(meridional))
        return {
            "u": mask_field(immersed, (-p.diff(zonal)).retag(u)),
            "v": mask_field(immersed, (-p.diff(meridional)).retag(v)),
            "p": scale_divergence(immersed, div),
        }
