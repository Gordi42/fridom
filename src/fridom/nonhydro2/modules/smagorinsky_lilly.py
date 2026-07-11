r"""
The Smagorinsky-Lilly closure for the nonhydrostatic model.

Description
-----------
The port of the Smagorinsky (1963) / Lilly (1962)
subgrid-scale closure (after the Oceananigans implementation the old
``nonhydro`` module followed). Friction on the velocity trio and
mixing on the tracer targets share one eddy viscosity built from the
strain-rate norm:

.. math::
    \partial_t u_i = \partial_j \tau_{ij}, \qquad
    \partial_t \phi = \nabla \cdot (\kappa_t \nabla \phi),

with :math:`\tau_{ij} = \nu_t \Sigma_{ij}` (the old module's
convention, no factor 2),
:math:`\Sigma_{ij} = \tfrac12(\partial_j u_i + \partial_i u_j)`,

.. math::
    \nu_t = \nu_s + \nu_{\mathrm{bg}}, \qquad
    \kappa_t = \nu_s / \mathrm{Pr} + \kappa_{\mathrm{bg}}, \qquad
    \nu_s = (C_s \, \Delta)^2 \, |\Sigma| \, \Gamma(\mathrm{Ri}),

where :math:`\Delta` is the filter width (cell volume to the power
:math:`1/n`) and the stratification damping is
:math:`\Gamma = \sqrt{1 - \min(\beta \mathrm{Ri}, 1)}` with
:math:`\mathrm{Ri} = N^2 / |\Sigma|^2` and
:math:`N^2 = \max(\partial_z b + N^2_{\mathrm{bg}}, 0)`. The port
folds the damping into the identity
:math:`|\Sigma| \Gamma = \sqrt{\max(|\Sigma|^2 - \beta N^2, 0)}`
(with :math:`\max(x, 0) = (x + |x|)/2`), so the whole closure is pure
field arithmetic — squares, roots, ``.diff``/``.to`` chains — and the
halo stencils are traced normally (no ``extra_halo``, no NaN
plumbing). On the C-grid the strain components land on their natural
positions (diagonals at centers, off-diagonals at edge midpoints) and
the stress divergence lands exactly on each velocity's face; the
scalar fields (:math:`|\Sigma|^2`, :math:`N^2`, :math:`\nu_t`) are
interpolated to the cell centers, replacing the old port's
ignore-the-position raw-array sums.

Targets follow `fr.model.closures.ClosureBase` (D1.4): the **stress** term
always advances the PROGNOSTIC ``Velocity`` family; the **mixing**
term advances the role-resolved tracer targets (default
``fr.model.roles.TRACER``; override with ``fields=`` / ``exclude=``, an
empty resolution simply drops the mixing term). The two terms split
under ``fr.model.term_predicates.advancing(...)``; both are
nonlinear (dropped by ``fr.model.linearize``). The Richardson
damping reads the constant ``stratification.n2`` provide (a
required reference), so a model
without a constant-N^2 stratification module rejects the closure at
assembly.

**Walled grids are future work**: the strain/stress stencils next to
rigid walls are not covered, so ``bind`` rejects walled grids with a
taught error (the CenteredAdvection precedent).
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import numpy as np

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.model.closures.base import ClosureBase
from fridom.model.errors import AssemblyError
from fridom.nonhydro2.params import (
    SMAG_BACKGROUND_KAPPA,
    SMAG_BACKGROUND_NU,
    SMAG_BUOYANCY_MULTIPLIER,
    SMAG_CS,
    SMAG_PRANDTL,
    STRATIFICATION_N2,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable

    from fridom.model.context import StepContext
    from fridom.model.field_table import FieldTable
    from fridom.model.roles import Role
    from fridom.spatial.fields.scalar_field import ScalarField

_VEL_HINT = ("the velocity trio is declared by the dynamical core "
             "(nh.DynamicalCore)")


def _positive_part(field: ScalarField) -> ScalarField:
    """Pointwise ``max(field, 0)`` as pure field arithmetic."""
    return 0.5 * (field + abs(field))


@partial(jaxify, dynamic=(
    "background_viscosity", "background_diffusivity",
    "turbulent_prandtl_number", "smagorinsky_constant",
    "buoyancy_multiplier"))
class SmagorinskyLilly(ClosureBase):

    r"""
    Smagorinsky-Lilly eddy viscosity/diffusivity closure.

    Description
    -----------
    See the module docstring for the physics. All five constants are
    provided parameters (``smagorinsky.*``), swept through
    ``model.update_parameters`` without a recompile.

    Parameters
    ----------
    background_viscosity : float, optional
        Background viscosity :math:`\nu_{\mathrm{bg}}` added to the
        eddy viscosity (default: 1.05e-6).
    background_diffusivity : float, optional
        Background diffusivity :math:`\kappa_{\mathrm{bg}}` added to
        the eddy diffusivity (default: 1.46e-7).
    turbulent_prandtl_number : float, optional
        Turbulent Prandtl number :math:`\mathrm{Pr}` (default: 1.0).
    smagorinsky_constant : float, optional
        The Smagorinsky constant :math:`C_s` (default: 0.16).
    buoyancy_multiplier : float | None, optional
        The Richardson damping multiplier :math:`\beta`; ``None``
        uses :math:`1 / \mathrm{Pr}` (default: None).
    vertical : str, optional
        The vertical coordinate name, along which :math:`N^2` is
        evaluated (default: ``"z"``).
    fields : Role | type[Role] | str | Iterable[str] | None, optional
        Mixing-target override; see `fr.model.closures.ClosureBase`
        (default: None -> every ``fr.model.roles.TRACER`` field).
    exclude : str | Iterable[str], optional
        Mixing targets removed from the resolution (default: ()).
    """

    default_targets = fr.model.roles.TRACER
    # the stress term is target-independent: a friction-only
    # Smagorinsky (all tracers excluded) is legitimate
    _allow_empty_targets = True

    def __init__(
        self,
        background_viscosity: float = 1.05e-6,
        background_diffusivity: float = 1.46e-7,
        turbulent_prandtl_number: float = 1.0,
        smagorinsky_constant: float = 0.16,
        buoyancy_multiplier: float | None = None,
        *,
        vertical: str = "z",
        fields: Role | type[Role] | str | Iterable[str] | None = None,
        exclude: str | Iterable[str] = (),
    ) -> None:
        """Store the closure constants as dynamic leaves."""
        super().__init__(fields=fields, exclude=exclude)
        self.background_viscosity = fr.model.leaf(background_viscosity)
        self.background_diffusivity = fr.model.leaf(background_diffusivity)
        self.turbulent_prandtl_number = fr.model.leaf(
            turbulent_prandtl_number)
        self.smagorinsky_constant = fr.model.leaf(smagorinsky_constant)
        if buoyancy_multiplier is None:
            buoyancy_multiplier = 1.0 / float(turbulent_prandtl_number)
        self.buoyancy_multiplier = fr.model.leaf(buoyancy_multiplier)
        self._vertical = vertical
        self._vel_axes: tuple[tuple[str, str], ...] = ()
        self._target_axes: tuple[tuple[str, tuple[str, ...]], ...] = ()
        self._filter_width: float = 0.0

    # ================================================================
    #  Declarations
    # ================================================================
    field_references = (
        fr.model.FieldReference("u", hint=_VEL_HINT),
        fr.model.FieldReference("v", hint=_VEL_HINT),
        fr.model.FieldReference("w", hint=_VEL_HINT),
        fr.model.FieldReference(
            "b", hint="the Richardson damping reads the buoyancy, "
                      "declared by a stratification module "
                      "(nh.ConstantStratification)"),
    )
    parameter_declarations = (
        fr.model.ParameterDeclaration(
            SMAG_BACKGROUND_NU, attr="background_viscosity",
            units="m^2/s", doc="background viscosity"),
        fr.model.ParameterDeclaration(
            SMAG_BACKGROUND_KAPPA, attr="background_diffusivity",
            units="m^2/s", doc="background diffusivity"),
        fr.model.ParameterDeclaration(
            SMAG_PRANDTL, attr="turbulent_prandtl_number",
            units="1", doc="turbulent Prandtl number"),
        fr.model.ParameterDeclaration(
            SMAG_CS, attr="smagorinsky_constant",
            units="1", doc="Smagorinsky constant"),
        fr.model.ParameterDeclaration(
            SMAG_BUOYANCY_MULTIPLIER, attr="buoyancy_multiplier",
            units="1", doc="Richardson damping multiplier"),
    )
    parameter_references = (
        fr.model.ParameterReference(
            STRATIFICATION_N2,
            hint="the Richardson damping needs the constant "
                 "background N^2 (nh.ConstantStratification)"),
    )

    # ================================================================
    #  Bind: velocity axes, mixing axes, filter width
    # ================================================================
    def bind(self, table: FieldTable) -> None:
        r"""Resolve mixing targets (base), velocities, and the width.

        Raises
        ------
        NotImplementedError
            On a walled grid, on non-uniform mesh factors (no
            constant filter width), or with transverse (slaved)
            velocity components (no directional derivative).
        AssemblyError
            If the vertical coordinate is not a velocity axis (the
            :math:`N^2 = \partial_z b` read needs it).
        """
        super().bind(table)
        owner = type(self).__name__
        factors = getattr(table.grid, "factors", ())
        walled = tuple(
            name for mesh in factors for name in mesh.names
            if not getattr(mesh, "periodic", True))
        if walled:
            raise NotImplementedError(
                f"{owner} does not support walled grids yet "
                f"(bounded coordinates: {walled}); the strain and "
                "stress stencils next to rigid walls are future "
                "work — drop the closure on walled grids")
        selector = table.velocity()
        if selector.transverse:
            raise NotImplementedError(
                f"{owner} needs directional velocity components; "
                f"the transverse (slaved) components "
                f"{selector.transverse} carry no directional "
                "derivative for the strain tensor")
        self._vel_axes = tuple(
            (name, axis) for name, axis in selector.labels
            if name in selector.prognostic)
        if not self._vel_axes:
            raise AssemblyError(
                f"{owner} finds no PROGNOSTIC Velocity-role fields "
                "to apply the stress to; a dynamical core declares "
                "them (nh.DynamicalCore)")
        axes = tuple(axis for _, axis in self._vel_axes)
        if self._vertical not in axes:
            raise AssemblyError(
                f"{owner}: the vertical coordinate "
                f"{self._vertical!r} is not a velocity axis "
                f"({axes}); pass vertical=<coordinate name> for "
                "the N^2 = d(b)/dz read")
        spacings = []
        for mesh in factors:
            dx = getattr(mesh, "dx", None)
            if dx is None:
                raise NotImplementedError(
                    f"{owner} needs a uniform structured grid for "
                    f"its constant filter width; the mesh factor "
                    f"{mesh!r} has no uniform spacing")
            spacings.append(float(dx))
        volume = float(np.prod(spacings))
        self._filter_width = volume ** (1.0 / len(spacings))
        self._target_axes = tuple(
            (name, tuple(table[name].space.names))
            for name in self.targets)

    # ================================================================
    #  Terms (stress always; mixing only with resolved targets)
    # ================================================================
    def tendency_terms(self) -> tuple[fr.model.TendencyTerm, ...]:
        """Return the stress term, plus mixing when targets resolved."""
        terms = (
            fr.model.TendencyTerm(
                name="stress", fn=self._stress,
                treatment=fr.model.Treatment.EXPLICIT,
                advances=tuple(name for name, _ in self._vel_axes)),
        )
        if self.targets:
            terms += (
                fr.model.TendencyTerm(
                    name="mixing", fn=self._mixing,
                    treatment=fr.model.Treatment.EXPLICIT,
                    advances=self.targets),
            )
        return terms

    # ================================================================
    #  The shared eddy viscosity
    # ================================================================
    def _strain(
        self, state: object, i: int, j: int,
    ) -> ScalarField:
        """One strain component on its natural C-grid position."""
        qi, ax_i = self._vel_axes[i]
        qj, ax_j = self._vel_axes[j]
        if i == j:
            return state[qi].diff(ax_i)
        return 0.5 * (state[qi].diff(ax_j) + state[qj].diff(ax_i))

    def _eddy_viscosity(
        self, state: object, ctx: StepContext,
    ) -> ScalarField:
        r"""Compute the Smagorinsky viscosity :math:`\nu_s` at centers.

        Uses the folded damping identity
        :math:`|\Sigma|\,\Gamma = \sqrt{\max(|\Sigma|^2 -
        \beta \max(N^2, 0),\, 0)}` — pure field arithmetic, exactly
        the old :math:`\Gamma(\mathrm{Ri})` including its
        zero-strain (``NaN -> 0``) limit.
        """
        cs = ctx.params[SMAG_CS]
        beta = ctx.params[SMAG_BUOYANCY_MULTIPLIER]
        n2_bg = ctx.params[STRATIFICATION_N2]
        n = len(self._vel_axes)
        diag = [self._strain(state, i, i) for i in range(n)]
        anchor = diag[0]
        sigma2 = diag[0] * diag[0]
        for s in diag[1:]:
            sigma2 = sigma2 + s * s
        for i in range(n):
            for j in range(i + 1, n):
                s = self._strain(state, i, j)
                sigma2 = sigma2 + 2.0 * (s * s).to(anchor)
        n2 = state["b"].diff(self._vertical).to(anchor) + n2_bg
        damped = _positive_part(sigma2 - beta * _positive_part(n2))
        return (cs * self._filter_width) ** 2 * damped ** 0.5

    # ================================================================
    #  The tendency hooks
    # ================================================================
    def _stress(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        r"""``du_i/dt += d_j(nu_t Sigma_ij)`` on each velocity face."""
        nu_t = (self._eddy_viscosity(state, ctx)
                + ctx.params[SMAG_BACKGROUND_NU])
        n = len(self._vel_axes)
        out: dict[str, ScalarField] = {}
        for i, (qi, _) in enumerate(self._vel_axes):
            res = None
            for j in range(n):
                _, ax_j = self._vel_axes[j]
                s = self._strain(state, i, j)
                divergence = (s * nu_t.to(s)).diff(ax_j)
                res = (divergence if res is None
                       else res + divergence)
            out[qi] = res
        return out

    def _mixing(
        self, state: object, ctx: StepContext,
    ) -> dict[str, ScalarField]:
        r"""``dphi/dt += div(kappa_t grad phi)`` on each target."""
        kappa_t = (self._eddy_viscosity(state, ctx)
                   / ctx.params[SMAG_PRANDTL]
                   + ctx.params[SMAG_BACKGROUND_KAPPA])
        out: dict[str, ScalarField] = {}
        for name, axes in self._target_axes:
            q = state[name]
            res = None
            for axis in axes:
                gradient = q.diff(axis)
                flux = gradient * kappa_t.to(gradient)
                contribution = flux.diff(axis)
                res = (contribution if res is None
                       else res + contribution)
            out[name] = res
        return out
