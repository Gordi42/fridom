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

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.hydrostatic.diagnostics import DIAGNOSTICS
from fridom.hydrostatic.params import CSQR, ROSSBY
from fridom.hydrostatic.state import State
from fridom.model.roles import Velocity
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
        """
        zonal, meridional = self._horizontal
        div_h = (state["u"].diff(zonal) + state["v"].diff(meridional))
        w = -CumulativeIntegral(
            direction="up", target="face")[self._vertical](div_h)
        return {"w": w}

    def _diagnose_p_hyd(
        self, state: State, ctx: StepContext,  # noqa: ARG002
    ) -> dict[str, object]:
        r"""Diagnose ``p_hyd`` from hydrostatic balance (top-down center).

        ``p_hyd(z) = -\int_{z}^{0} b\, dz'`` on the cell centres
        (co-located with ``b``): the pyOM half-cell form, seeded
        ``p_hyd = 0`` at the surface. The negative sign is the
        integral taken from the upper limit.
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
