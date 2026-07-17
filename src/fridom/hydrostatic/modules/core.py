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
from fridom.framework.utils import jaxify, modify_array
from fridom.hydrostatic.diagnostics import DIAGNOSTICS
from fridom.hydrostatic.params import CSQR, ROSSBY
from fridom.hydrostatic.state import State
from fridom.model.roles import Velocity
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.operators.cumulative import CumulativeIntegral
from fridom.spatial.spaces.tensor_product import TensorProductSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.model.context import StepContext
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

    def bind(self, table: object) -> None:
        """Capture the immersed descriptor and coordinate names (IP-D9)."""
        grid = table.grid
        self._immersed = getattr(grid, "immersed", None)
        self._coords = tuple(grid.names)

    @property
    def extra_halo(self) -> HaloSpec | None:
        """Exempt the masked DIAGNOSE stages from the halo trace.

        Description
        -----------
        Only on an immersed grid: the fraction-weighted continuity and
        the fraction lookups run concrete field arithmetic the tracer
        cannot follow, and the materialized fractions carry the grid's
        provisional (per-operator-max) storage halo, so the stages read
        full-halo fields that align with the fractions. Two cells per
        coordinate matches that provisional halo (and the shared FV
        advection / nonhydro2 core), keeping the frozen storage halo
        equal to the fraction cache's halo. Off an immersed grid this is
        ``None`` — the unimmersed DIAGNOSE stages stay fully halo-
        traced, bitwise unchanged.
        """
        if self._immersed is None:
            return None
        return HaloSpec(dict.fromkeys(self._coords, 2))

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
        w_space = state["w"].function_space
        alpha_z = immersed.fraction(w_space)
        theta_cell = immersed.fraction(state["p_hyd"].function_space)
        z_axis = next(
            i for i, f in enumerate(w_space.bare.factors)
            if self._vertical in f.names)
        az = jnp.moveaxis(alpha_z.data, z_axis, 0)
        tc = jnp.moveaxis(theta_cell.data, z_axis, 0)
        az = modify_array(az, -1, tc[-1])  # surface face = surface cell
        return alpha_z.with_data(jnp.moveaxis(az, 0, z_axis))

    def _diagnose_p_hyd(
        self, state: State, ctx: StepContext,  # noqa: ARG002
    ) -> dict[str, object]:
        r"""Diagnose ``p_hyd`` from hydrostatic balance (top-down center).

        ``p_hyd(z) = -\int_{z}^{0} b\, dz'`` on the cell centres
        (co-located with ``b``): the pyOM half-cell form, seeded
        ``p_hyd = 0`` at the surface. The negative sign is the
        integral taken from the upper limit.

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
        p_hyd = -CumulativeIntegral(
            direction="down", target="center")[self._vertical](
                state["b"])
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
        """
        zonal, meridional = self._horizontal
        u, v = state["u"], state["v"]
        p_hyd = state["p_hyd"]
        return {
            "u": (-p_hyd.diff(zonal)).retag(u),
            "v": (-p_hyd.diff(meridional)).retag(v),
        }
