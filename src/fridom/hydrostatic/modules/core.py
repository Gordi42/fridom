r"""The hydrostatic dynamical core module.

Description
-----------
``HydrostaticCore`` is the package's dynamical-core module (D1.3): it
declares the horizontal velocities ``u, v`` (Velocity + ADVECTED, on
the C-grid faces), the **diagnosed** vertical velocity ``w`` and the
**diagnosed** hydrostatic pressure ``p_hyd``; it owns the core
parameters ``hydrostatic.csqr`` and ``scaling.rossby``; it owns the
two pre-tendency **DIAGNOSE** stages (S1'); and it contributes the
single linear pressure-gradient term. It supplies the ``hy.State``
vocabulary class through ``state_type``.

The two DIAGNOSE stages (recomputed from the current state every
substage, so a restart / ``set_state`` sees them fresh before any
term reads — the S1' placement of ``03_time_stepping.md`` §5.2):

.. math::

    w(z)      = -\int_{-H}^{z} (\partial_x u + \partial_y v)\, dz', \\
    p_{hyd}(z) = -\int_{z}^{0} b \, dz'.

``w`` is built with the **face** form of ``CumulativeIntegral``
(``direction="up"``, seeded ``w = 0`` at the flat bottom), landing on
the both-boundary vertical face set ``Outer`` — the
fundamental-theorem-exact form: ``d_z w == -(d_x u + d_y v)`` to
machine precision. Because the barotropic column divergence is
non-zero under a free surface, the surface face ``w(0)`` is a genuine
DOF (``Outer``), not a rigid-lid zero, and keeping it is what makes
the buoyancy conversion exactly energy-conserving (the half-cell
``p_hyd`` at the top cancels the surface boundary term — see
``hy.energy``). ``p_hyd`` uses the **center** form
(``direction="down"``, seeded ``p_hyd = 0`` at the surface), the pyOM
half-cell hydrostatic pressure co-located with ``b``, so its
horizontal gradient reaches the ``u``/``v`` faces through the ordinary
``diff``.

The linear pressure-gradient term reads the **baroclinic** pressure
``p_hyd`` only:

.. math::

    \partial_t u = -\partial_x p_{hyd}, \qquad
    \partial_t v = -\partial_y p_{hyd}.

The barotropic ``-\nabla_h p_s`` momentum coupling is **owned by the
free-surface variant** (H3): each variant owns both sides of its
coupling — ``ExplicitFreeSurface`` carries ``-\nabla_h p_s`` as a
linear term (the adjoint of its ``-c^2\nabla_h\cdot\bar u`` gravity
term), while ``ImplicitFreeSurface`` applies the force inside its
CONSTRAINT-stage 2D projection. The core never reads ``ps``, so an
implicit variant's barotropic force cannot double-count with the
constraint's own velocity correction.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.hydrostatic.diagnostics import DIAGNOSTICS
from fridom.hydrostatic.modules.terrain import (
    discover_column,
    jacobian_name,
)
from fridom.hydrostatic.params import CSQR, ROSSBY
from fridom.hydrostatic.state import State
from fridom.model.halo_demand import derive_extra_halo
from fridom.model.roles import Velocity
from fridom.spatial.fields.scalar_field import _bc_siblings
from fridom.spatial.operators.cumulative import CumulativeIntegral
from fridom.spatial.operators.verbs import scatter_set
from fridom.spatial.spaces.average import AverageSpace
from fridom.spatial.spaces.nodal import NodeSet
from fridom.spatial.spaces.tensor_product import TensorProductSpace
from fridom.spatial.spaces.trace import Side

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.model.context import StepContext
    from fridom.spatial.decomposition.halo import HaloSpec
    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.function_space import FunctionSpace


def _make_w_space_rule(
    vertical: str,
) -> Callable[[Grid], FunctionSpace]:
    """Build the ``w`` space rule: collocated horizontal, Outer vertical.

    Description
    -----------
    The diagnosed vertical velocity lives at the C-grid w-points: the
    cell centres in the horizontal (co-located with ``p_hyd`` / ``b``)
    and the **both-boundary** face set ``Outer`` along ``vertical``
    (n + 1 faces, including the flat bottom ``w = 0`` and the free
    surface). No ``Dof`` tag resolves to ``Outer`` (``Staggered`` is
    the interior ``Inner`` on a bounded axis), so ``w`` is declared
    through this ``SpaceRule`` escape hatch — the space the face-form
    ``CumulativeIntegral`` lands on, so the DIAGNOSE write is exact.

    Parameters
    ----------
    vertical : str
        The vertical coordinate name.

    Returns
    -------
    Callable[[Grid], FunctionSpace]
        The pure per-grid space rule.
    """
    def rule(grid: Grid) -> FunctionSpace:
        factors = []
        for mesh in grid.factors:
            if vertical in mesh.names:
                factors.append(mesh.outer)
            else:
                factors.append(mesh.center)
        return TensorProductSpace.of(*factors)

    return rule


@partial(jaxify, dynamic=("csqr", "rossby"))
class HydrostaticCore(fr.model.Module):

    r"""Declares u, v, w, p_hyd; owns csqr/rossby and the DIAGNOSE.

    Parameters
    ----------
    csqr : float | fr.model.Ramp, optional
        The squared barotropic phase speed :math:`c^2 = g H`, the
        single barotropic parameter, published as
        ``hydrostatic.csqr`` and read by the free-surface term
        (default: 1.0); may be an ``fr.model.Ramp``.
    rossby_number : float | fr.model.Ramp, optional
        The Rossby number scaling the (separate) advection term,
        published as ``scaling.rossby`` (default: 1.0); may be an
        ``fr.model.Ramp`` for a spun-up nonlinearity.
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
    horizontal : tuple[str, str], optional
        The (zonal, meridional) coordinate names of the two velocity
        components (default: ``("x", "y")``).
    """

    state_type = State
    diagnostics = DIAGNOSTICS

    def __init__(
        self,
        csqr: float | fr.model.Ramp = 1.0,
        *,
        rossby_number: float | fr.model.Ramp = 1.0,
        vertical: str = "z",
        horizontal: tuple[str, str] = ("x", "y"),
    ) -> None:
        """Store the core parameter leaves and the geometry names."""
        horizontal = tuple(horizontal)
        if (len(horizontal) != 2  # noqa: PLR2004 — zonal + meridional
                or not all(isinstance(name, str) for name in horizontal)
                or horizontal[0] == horizontal[1]):
            raise TypeError(
                "horizontal names the (zonal, meridional) coordinates: "
                f"two distinct strings, got {horizontal!r}")
        self.csqr = fr.model.leaf(csqr)
        self.rossby = fr.model.leaf(rossby_number)
        self._vertical = vertical
        self._horizontal = horizontal
        # captured at bind: the immersed descriptor (None off a cut-cell
        # grid) and the grid coordinate names. On an immersed grid the
        # DIAGNOSE stages weight the horizontal transport by concrete
        # face-fraction fields, which the HaloTracer cannot follow and
        # which materialize at the full storage halo — so the masked
        # core is halo-trace exempt (extra_halo), reading full-halo
        # fields that align with the fractions (the advection / MaskState
        # precedent). The unimmersed core stays fully traced (None).
        self._immersed: object | None = None
        self._coords: tuple[str, ...] = ()
        # captured at bind: the terrain-following column (mapped, base)
        # of a sigma-coordinate grid, or None off a mapped grid (the
        # byte-identical flat / stretched-only path). On a terrain grid
        # the DIAGNOSE stages J-weight the vertical increment and the
        # baroclinic pressure gradient adopts the slope-corrected
        # (constant-physical-height) horizontal derivative.
        self._column: tuple[str, str] | None = None
        # the masked / terrain DIAGNOSE stages' derived halo substitute
        # (V-N2), computed at bind from the staggered rows they apply;
        # None off a mapped / immersed grid (the flat path stays traced).
        self._extra_halo: HaloSpec | None = None

    def bind(self, table: object) -> None:
        """Capture the immersed / terrain descriptors and coord names.

        Description
        -----------
        Reads the immersed descriptor (IP-D9) and, through
        :func:`~fridom.hydrostatic.modules.terrain.discover_column`,
        the single-base terrain column on the vertical axis (rules
        3.8). The two are mutually exclusive in this iteration: a
        cut-cell mask on top of a tilted sigma column is not modelled
        (its masked continuity would need the Jacobian-weighted face
        fractions), so a terrain + immersed grid is a taught error.
        """
        grid = table.grid
        self._immersed = getattr(grid, "immersed", None)
        self._coords = tuple(grid.names)
        self._column = discover_column(grid, self._vertical)
        if self._column is not None and self._immersed is not None:
            raise NotImplementedError(
                "the hydrostatic model does not support an immersed "
                "(cut-cell) domain on top of a terrain-following sigma "
                "column: the masked continuity would have to weight "
                "the face fractions by the column Jacobian, which is "
                "not built (hydrostatic plan §7). Use a terrain grid "
                "without an immersed mask, or a flat immersed grid")
        self._extra_halo = self._derive_extra_halo(table)

    def _derive_extra_halo(self, table: object) -> HaloSpec | None:
        r"""Derive the masked / terrain stages' ghost width (V-N2).

        Description
        -----------
        Off a mapped / immersed grid the DIAGNOSE stages and pressure
        gradient are plain staggered stencils the halo trace follows,
        so no substitute is declared (``None``). On a **terrain** or
        **immersed** grid they multiply metric / fraction fields the
        tracer cannot materialize, so the module declares its own
        width — derived from the order-2 rows those stages apply, not a
        literal. Two parallel horizontal legs give reach 1 on each
        horizontal coordinate under any boundary: the pressure gradient
        differences the cell pressure onto the faces (centre -> face),
        the flux / masked continuity differences the face transport back
        onto the cell (face -> centre; the direction that carries the
        reach on a bounded axis). A terrain column adds the
        constant-physical-height slope term to the vertical: a column
        derivative (centre -> face ``diff``) re-aligned onto the cell by
        the column interpolation (face -> centre) — the two composing
        two-sided to reach 1 on the bounded vertical (the diff alone
        shrinks at a wall; the interp carries the reach). An immersed
        grid has **no** vertical stencil (the masked continuity's column
        sum is a reduction, reach 0), so its vertical stays 0. A registry
        override of the differences / interpolations moves these values.
        """
        if self._immersed is None and self._column is None:
            return None
        registry = table.grid.dispatch  # type: ignore[attr-defined]
        p_hyd = table["p_hyd"].space  # type: ignore[index]
        u = table["u"].space  # type: ignore[index]
        v = table["v"].space  # type: ignore[index]
        zonal, meridional = self._horizontal
        grad_leg: dict[str, list[tuple[str, object]]] = {
            zonal: [("diff", p_hyd.factor(zonal))],
            meridional: [("diff", p_hyd.factor(meridional))],
        }
        div_leg: dict[str, list[tuple[str, object]]] = {
            zonal: [("diff", u.factor(zonal))],
            meridional: [("diff", v.factor(meridional))],
        }
        if self._column is not None:
            vert = self._vertical
            centre = p_hyd.factor(vert)
            face = registry.resolve("diff", centre)[vert].codomain(centre)
            # the slope gradient reads the column derivative (centre ->
            # face) and re-aligns it onto the cell by the column
            # interpolation (face -> centre; nodal on the nodal-only
            # hydrostatic grid). The pair reaches 1 on the bounded
            # vertical, driven by the interp (the centre -> face diff
            # shrinks at a wall)
            grad_leg[vert] = [("diff", centre), ("interpolate", face)]
        return derive_extra_halo(
            registry, self._coords, [grad_leg, div_leg])

    @property
    def extra_halo(self) -> HaloSpec | None:
        """Exempt the masked / terrain DIAGNOSE stages from the trace.

        Description
        -----------
        On an **immersed** grid the fraction-weighted continuity and the
        fraction lookups run concrete field arithmetic the tracer cannot
        follow. On a **terrain** grid the DIAGNOSE stages and the
        baroclinic pressure gradient multiply ``grid.metric`` coefficient
        fields (the column Jacobian ``J``, the slope-corrected
        ``physical_diff``) that the halo tracer's ``_TracerGrid`` cannot
        materialize — the shared mapped-advection precedent (its own
        ``extra_halo``). Either way the module declares its (order-2
        centered) FD-stencil halo here instead of being traced. Two
        cells per coordinate matches the provisional storage halo the
        materialized metrics carry. Off a mapped / immersed grid this is
        ``None`` — the flat DIAGNOSE stages stay fully halo-traced,
        bitwise unchanged.

        The value is **derived** at :meth:`bind` (see
        :meth:`_derive_extra_halo`) from the staggered ``diff`` rows the
        stages apply, not a literal: 1 per horizontal coordinate on both
        paths, plus 1 on the vertical for a terrain column (the slope
        gradient's vertical difference) and 0 on the vertical for an
        immersed grid (no vertical stencil there).
        """
        return self._extra_halo

    # ================================================================
    #  Field declarations
    # ================================================================
    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """Declare ``u, v`` (velocity), ``w`` (Outer), ``p_hyd``.

        Description
        -----------
        ``u, v`` are the C-grid face velocities (Velocity + ADVECTED,
        PROGNOSTIC). ``w`` is DIAGNOSTIC with a bare ``Velocity``
        role but **not** ADVECTED — the V-H2 case: the hydrostatic
        momentum set has no ``dw/dt``, so ``w`` is not a prognostic
        transported field, yet it carries the vertical velocity role
        (the shared advection's velocity-trio query, ``eigenmodes``).
        ``p_hyd`` is the collocated diagnostic hydrostatic pressure.
        """
        zonal, meridional = self._horizontal
        return (
            fr.model.FieldDeclaration.velocity(
                "u", zonal, space=fr.spatial.Staggered(zonal),
                long_name="Zonal velocity", units="m/s"),
            fr.model.FieldDeclaration.velocity(
                "v", meridional,
                space=fr.spatial.Staggered(meridional),
                long_name="Meridional velocity", units="m/s"),
            fr.model.FieldDeclaration(
                "w",
                space=fr.spatial.SpaceRule(
                    _make_w_space_rule(self._vertical)),
                lifecycle=fr.model.Lifecycle.DIAGNOSTIC,
                roles=(Velocity(self._vertical),),
                long_name="Vertical velocity", units="m/s"),
            fr.model.FieldDeclaration(
                "p_hyd", space=fr.spatial.Collocated(),
                lifecycle=fr.model.Lifecycle.DIAGNOSTIC,
                long_name="Hydrostatic pressure", units="m^2/s^2"),
        )

    field_references = (
        fr.model.FieldReference(
            "b", hint="the hydrostatic pressure integrates buoyancy; "
                      "add a stratification module "
                      "(hy.ConstantStratification)"),
    )

    # ================================================================
    #  Parameters -- csqr and the Rossby number live on the core
    # ================================================================
    parameter_declarations = (
        fr.model.ParameterDeclaration(CSQR, attr="csqr",
                                units="m^2/s^2",
                                doc="squared barotropic phase speed g*H"),
        fr.model.ParameterDeclaration(ROSSBY, attr="rossby", units="1",
                                doc="Rossby number (nonlinear scaling)"),
    )

    # ================================================================
    #  The two DIAGNOSE stages (S1')
    # ================================================================
    @property
    def stages(self) -> tuple[fr.model.Stage, ...]:
        """The pre-tendency diagnostics: ``w`` and ``p_hyd``.

        Description
        -----------
        Both are ``DIAGNOSE`` (S1'), recomputed every substage from
        the current ``(u, v, b)``. Their write sets ``{w}`` and
        ``{p_hyd}`` are disjoint and neither reads the other's
        output, so the schedule places them at equal order without a
        lint collision (order is immaterial).
        """
        return (
            fr.model.Stage(kind=fr.model.StageKind.DIAGNOSE,
                     fn="_diagnose_w", name="diagnose_w"),
            fr.model.Stage(kind=fr.model.StageKind.DIAGNOSE,
                     fn="_diagnose_p_hyd", name="diagnose_p_hyd"),
        )

    def _diagnose_w(
        self, state: State, ctx: StepContext,  # noqa: ARG002
    ) -> dict[str, object]:
        r"""Diagnose ``w`` from continuity (bottom-up face form).

        ``w(z) = -\int_{-H}^{z} (\partial_x u + \partial_y v) dz'`` on
        the both-boundary face set ``Outer``: seeded ``w = 0`` at the
        flat bottom, ``d_z w == -(d_x u + d_y v)`` machine-exactly.

        **Terrain-following column** (a sigma-coordinate grid,
        :meth:`bind` captured ``self._column``): the diagnosed ``w`` is
        the **contravariant vertical volume flux**
        ``J\omega = w_{phys} - u\,Z_x - v\,Z_y`` (``J`` the column
        Jacobian, ``Z_i`` the coordinate-surface slope), built from the
        **flux form** of the physical horizontal divergence,

        .. math::

            J\omega(z) = -\int_{-H}^{z}
                \bigl[\partial_x (J u) + \partial_y (J v)\bigr]\,
                \mathrm{d}z' ,

        with ``J`` on the ``u`` / ``v`` faces. This choice keeps the
        two flat invariants exactly: the fundamental theorem
        ``\partial_z(J\omega) == -[\partial_x(Ju) + \partial_y(Jv)]``
        holds to machine precision (the face-form ``CumulativeIntegral``
        FTC), and the bottom seed ``J\omega = 0`` is the **exact**
        zero-normal-flow bottom boundary condition on the sigma column
        (``\omega = 0`` at the terrain, the natural prognostic-free
        choice — the Cartesian ``w_{phys}`` is *not* zero over a
        slope). The same increment ``\partial_x(Ju) + \partial_y(Jv)``
        is the horizontal leg of the mapped pressure solver's
        J-weighted flux divergence, so the diagnosis is energy-
        consistent with the vertical hydrostatic pairing. With ``J = 1``
        and ``Z = 0`` (a flat grid) it collapses byte-for-byte to the
        Cartesian form above.

        On an immersed (cut-cell) grid this becomes **masked
        continuity** (IP-D9): the horizontal transport divergence is
        fraction-weighted (``(alpha_x u).diff(x) + (alpha_y v).diff(y)``
        with the min-rule face fractions of I0), the running integral
        yields the barotropic **transport** ``alpha_z w`` (with
        ``alpha_z`` on the vertical ``Outer`` faces), and ``w`` is the
        guarded division ``alpha_z w / alpha_z`` (``alpha_z == 0 -> w ==
        0``). The vertical-flux telescoping of the running sum then
        makes the full masked divergence
        ``(alpha_x u).diff(x) + (alpha_y v).diff(y) + (alpha_z w).diff(z)``
        machine-zero on every wet cell (min-rule wet faces read two wet
        cells, so no dry value enters with nonzero weight). On an
        all-wet immersed grid ``alpha == 1`` and the result is byte-
        identical to the unimmersed diagnosis.
        """
        zonal, meridional = self._horizontal
        u, v = state["u"], state["v"]
        cumint = CumulativeIntegral(
            direction="up", target="face")[self._vertical]
        immersed = getattr(u.grid, "immersed", None)
        if immersed is None and self._column is not None:
            # terrain: the flux-form horizontal divergence of the
            # J-weighted transport (J on the u/v faces), so w is the
            # contravariant vertical volume flux Jomega (0 at bottom).
            jname = jacobian_name(self._column)
            grid = u.grid
            ju = u * grid.metric(u.function_space.bare, jname)
            jv = v * grid.metric(v.function_space.bare, jname)
            div_h = ju.diff(zonal) + jv.diff(meridional)
            return {"w": -cumint(div_h)}
        if immersed is None:
            div_h = u.diff(zonal) + v.diff(meridional)
            return {"w": -cumint(div_h)}
        alpha_x = immersed.fraction(u.function_space)
        alpha_y = immersed.fraction(v.function_space)
        div_h = (alpha_x * u).diff(zonal) + (alpha_y * v).diff(meridional)
        transport = -cumint(div_h)  # the barotropic transport alpha_z*w
        alpha_z = self._masked_w_faces(immersed, state)
        az = alpha_z.data
        wet = az > 0.0
        w = jnp.where(wet, transport.data / jnp.where(wet, az, 1.0), 0.0)
        return {"w": transport.with_data(w)}

    def _masked_w_faces(
        self, immersed: object, state: State,
    ) -> object:
        r"""Return ``alpha_z`` on the ``w`` faces (surface = cell fraction).

        Description
        -----------
        The min-rule face fraction on the vertical ``Outer`` faces
        treats the exterior beyond a **physical** boundary as dry, which
        would zero the surface (top) face and destroy the barotropic
        surface DOF ``w(0)`` (the column-divergence carrier under a free
        surface). The physical top boundary is not an immersed dry
        region, so its face fraction is the adjacent (surface) cell
        fraction — the mirror-exterior convention. The physical bottom
        face needs no override: the running sum seeds ``transport == 0``
        there, so ``w`` is zero irrespective of ``alpha_z``.

        The surface override rides the sanctioned boundary machinery
        (``design/plans/active/boundary_trace_plan.md`` §3) rather than
        raw ``.data`` surgery: the surface cell fraction is a
        ``Side.HIGH`` boundary trace of ``theta_cell`` (on the
        ``Center`` cells), relocated onto the vertical ``Outer`` face
        set through the sanctioned Constant bridge (``as_profile`` ->
        ``adopt``) — the explicit cross-node-set relocation the strict
        space algebra otherwise refuses — then ``scatter_set`` overwrites
        the ``Outer`` surface face of ``alpha_z``. The path is a pure
        slice + two retags + a row-scatter (all native VJPs, no
        ``.data``), so the diagnosed ``w`` is bitwise unchanged.

        Parameters
        ----------
        immersed : object
            The grid's immersed descriptor.
        state : State
            The current state (supplies the ``w`` and cell spaces).

        Returns
        -------
        object
            The ``alpha_z`` field with the surface face overridden.
        """
        vertical = self._vertical
        alpha_z = immersed.fraction(state["w"].function_space)
        theta_cell = immersed.fraction(state["p_hyd"].function_space)
        # surface (top) face fraction = surface cell fraction: trace the
        # top Center cell, relocate it onto the Outer face set (the
        # sanctioned Constant bridge), and overwrite the surface face.
        surface = (theta_cell.trace(vertical, Side.HIGH)
                   .as_profile(vertical)
                   .adopt(vertical, NodeSet.OUTER, Side.HIGH))
        return scatter_set(alpha_z, surface)

    def _diagnose_p_hyd(
        self, state: State, ctx: StepContext,  # noqa: ARG002
    ) -> dict[str, object]:
        r"""Diagnose ``p_hyd`` from hydrostatic balance (top-down center).

        ``p_hyd(z) = -\int_{z}^{0} b\, dz'`` on the cell centres
        (co-located with ``b``): the pyOM half-cell form, seeded
        ``p_hyd = 0`` at the surface. The negative sign is the
        integral taken from the upper limit.

        **Terrain-following column**: the vertical increment is taken
        along the *physical* height ``\mathrm{d}z_p = J\,\mathrm{d}z``,
        so the running sum carries the column Jacobian
        (``jacobian=(mapped,)`` — the wired seam, ``jacobian_weight``):
        ``p_hyd(z) = -\int_z^0 b\,J\,\mathrm{d}z'``, the physical
        hydrostatic pressure ``-\int b\,\mathrm{d}z_p`` (converges at
        second order on both uniform-sigma and stretched-sigma
        columns). On a flat grid ``self._column`` is ``None`` and the
        plain (unweighted) form is byte-identical to before.

        On an immersed grid the cumulative sum is **unweighted** (the
        top-down hydrostatic integral of the masked buoyancy, which is
        zero on dry cells through ``MaskState``): the diagnosed
        ``p_hyd`` under the topography is dead, and it never reaches a
        wet momentum tendency because the plain two-point pressure
        gradient on a wet face reads two wet cells (min-rule) while a
        wet/dry face is a closed (``alpha == 0``) face whose spurious
        gradient ``MaskState`` zeros. The partial-bottom-cell pressure-
        gradient refinement (Pacanowski & Gnanadesikan) is designed-for
        (immersed-partial-cells plan §7), not built here.
        """
        jacobian = (None if self._column is None
                    else (self._column[0],))
        p_hyd = -CumulativeIntegral(
            direction="down", target="center",
            jacobian=jacobian)[self._vertical](state["b"])
        return {"p_hyd": p_hyd}

    # ================================================================
    #  Tendency term (linear pressure gradient)
    # ================================================================
    @fr.model.term(advances=("u", "v"), linear=True)
    def pressure_gradient(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""``d_t u = -d_x p_hyd``, ``d_t v = -d_y p_hyd``.

        The **baroclinic** pressure gradient only: a single staggered
        difference of the diagnosed hydrostatic pressure reaching the
        velocity faces, retagged onto the velocities (identity on
        periodic axes; adopts the wall-normal Dirichlet tag on a
        walled axis). The barotropic ``-d_x ps`` / ``-d_y ps`` force
        is owned by the free-surface variant (H3): a linear term in
        ``ExplicitFreeSurface``, the CONSTRAINT-stage projection in
        ``ImplicitFreeSurface``.

        **Terrain-following column**: the horizontal force is the
        gradient at **constant physical height**
        ``-\partial_x p_{hyd}|_{z_p}
        = -(\partial_x p_{hyd}|_z - (Z_x/J)\,\partial_z p_{hyd})``,
        the slope-corrected derivative dispatched through the grid's
        ``physical_diff`` row (the same constant-physical-coordinate
        derivative the mapped advection consumes). Adding the slope
        term is the classic sigma-coordinate pressure-gradient-error
        correction: a stratified fluid at rest over topography stays at
        rest to the scheme's truncation order (the rest-state gate),
        where the plain ``\partial_x p_{hyd}|_z`` alone drives an
        O(1) spurious current. On a flat grid ``self._column`` is
        ``None`` and the plain staggered ``diff`` is byte-identical.
        """
        zonal, meridional = self._horizontal
        u, v = state["u"], state["v"]
        p_hyd = state["p_hyd"]
        if self._column is not None:
            return {
                "u": (-self._slope_gradient(p_hyd, zonal, u)).retag(u),
                "v": (-self._slope_gradient(p_hyd, meridional, v)
                      ).retag(v),
            }
        return {
            "u": (-p_hyd.diff(zonal)).retag(u),
            "v": (-p_hyd.diff(meridional)).retag(v),
        }

    def _slope_gradient(
        self, p_hyd: object, axis: str, target: object,
    ) -> object:
        r"""Return ``\partial_{axis} p_{hyd}|_{z_p}`` on ``target``'s face.

        Description
        -----------
        The horizontal derivative at **constant physical height**,
        ``\partial_i p|_{z_p} = \partial_i p|_z - (Z_i/J)\,
        \partial_z p``, assembled by explicit field arithmetic that
        mirrors the mapped advection's nodal physical divergence
        (``model/modules/advection.py`` ``_mapped_divergence``, the
        same ``grid.metric`` rows) — **not** the ``physical_diff``
        dispatch verb, whose composite reciprocal-Jacobian metric seals
        a never-valid-padding singularity that poisons the reverse pass
        (the differentiability policy). The slope coefficient is the
        ratio ``Z_i/J = d<mapped>_d<axis> / d<mapped>_d<base>`` (both
        finite on a monotone map, so no guard), the column derivative
        ``\partial_z p`` is interpolated from the column faces onto the
        velocity face along both the column and the coupled axis.

        Parameters
        ----------
        p_hyd : object
            The diagnosed hydrostatic pressure (a cell-centre field).
        axis : str
            The horizontal coordinate the gradient is taken along.
        target : object
            The velocity component whose face the gradient lands on
            (fixes the staggering; the retag is applied by the caller).

        Returns
        -------
        object
            The constant-physical-height horizontal derivative on
            ``target``'s face.
        """
        mapped, base = self._column
        grid = p_hyd.grid
        div = p_hyd.diff(axis)
        dcol = p_hyd.diff(base)
        space = dcol.function_space
        slope = grid.metric(space.bare, f"d{mapped}_d{axis}")
        jac = grid.metric(space.bare, f"d{mapped}_d{base}")
        # slope coefficient Z_i / J. Double-`where` guard
        # (differentiability policy): J > 0 on every valid column, but
        # the never-valid column-face padding derives J == 0, where a
        # bare ratio seals the forward value yet leaves the VJP singular
        # and poisons the whole gradient with NaN (physical_diff's
        # composite reciprocal-Jacobian hits the same trap — this is why
        # the gradient is assembled by hand).
        jd = jac.data
        nonzero = jd != 0.0
        coeff = slope.with_data(
            jnp.where(nonzero, slope.data / jnp.where(nonzero, jd, 1.0),
                      0.0))
        corr = coeff * dcol
        registry = grid.dispatch
        for name in (base, axis):
            src = corr.function_space.bare.factor(name)
            dst = div.function_space.bare.factor(name)
            if src is dst or _bc_siblings(src, dst):
                continue
            kind = ("average" if isinstance(dst, AverageSpace)
                    else "interpolate")
            corr = registry.resolve(kind, src)[name](corr)
        return (div - corr.retag(div)).retag(target)
