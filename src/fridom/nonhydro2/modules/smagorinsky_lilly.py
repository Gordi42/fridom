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

**Walled grids** (bounded mesh factors, nodal family) are supported
with **free-slip** walls (``slip="free"``, the default). The only
wall-choice-dependent strain is the off-diagonal shear on a
wall-normal edge; at a free-slip wall it is structurally zero, claimed
by retagging the interior ``Inner`` edge factor onto its
``Inner[Dirichlet]`` sibling — the diffusion campaign's flux retag
(``diffusion._dirichlet_face``) applied to the strain and reused in
both :math:`|\Sigma|^2` and the stress; no new spatial operator. The
wall-normal diagonal strain already carries the impermeability
``Inner[Dirichlet]`` tag and closes on its own; the tracer no-flux
mixing wall and the buoyancy-gradient (:math:`N^2`) read take the same
retag. Walled **finite-volume** (``CellAvg``) grids, and immersed /
terrain grids, stay future work (a taught rejection at ``bind``).
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.model.closures.base import ClosureBase
from fridom.model.closures.diffusion import _PERIODIC, _wall_treatment
from fridom.model.errors import AssemblyError
from fridom.nonhydro2.params import (
    SMAG_BACKGROUND_KAPPA,
    SMAG_BACKGROUND_NU,
    SMAG_BUOYANCY_MULTIPLIER,
    SMAG_CS,
    SMAG_PRANDTL,
    STRATIFICATION_N2,
)
from fridom.spatial.bc import BC
from fridom.spatial.spaces.average import CellAvg

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


@jax.custom_jvp
def _sqrt_clipped(x: jax.Array) -> jax.Array:
    r"""Elementwise ``sqrt`` of a non-negative array, AD-safe at zero.

    ``x >= 0`` by construction (a ``max(., 0)`` clip upstream). The
    primal is the plain ``x ** 0.5`` — bitwise identical to an unguarded
    sqrt, so the forward step pays nothing. Only the derivative rule is
    replaced: ``d sqrt/dx = 0.5 * x**-0.5`` is ``inf`` at ``x = 0``, and
    the clip feeds exact zeros into it, so a single clipped cell would
    otherwise turn ``jax.grad`` of the whole field into ``NaN``. The
    tangent rule below guards the ``sqrt`` and pins the (measure-zero)
    subgradient in the clipped cells to ``0`` — the correct value there.

    A ``custom_jvp`` (not ``custom_vjp``) is used deliberately: its
    linear tangent rule is transposed automatically by jax, so both
    forward-mode (``jax.jvp``) and reverse-mode (``jax.grad``) AD flow
    through it; a ``custom_vjp`` would silently break forward mode.
    """
    return x ** 0.5


@_sqrt_clipped.defjvp
def _sqrt_clipped_jvp(
    primals: tuple[jax.Array, ...],
    tangents: tuple[jax.Array, ...],
) -> tuple[jax.Array, jax.Array]:
    """JVP with both sqrt branches guarded (subgradient 0 at ``x = 0``)."""
    (x,), (t,) = primals, tangents
    pos = x > 0
    xs = jnp.where(pos, x, 1.0)
    tangent = jnp.where(pos, 0.5 * t / xs ** 0.5, 0.0)
    return _sqrt_clipped(x), tangent


def _guarded_sqrt(field: ScalarField) -> ScalarField:
    r"""Elementwise ``sqrt`` of a non-negative field, AD-safe at zero.

    The concrete-field path applies :func:`_sqrt_clipped` (a plain
    ``** 0.5`` primal with a guarded tangent rule) on the storage array;
    the forward value is therefore bitwise identical to an unguarded
    sqrt. The halo trace (``HaloTracer``, no storage array and no AD)
    takes the plain ``** 0.5`` and its pointwise stencil unchanged.
    """
    if not isinstance(field, fr.spatial.ScalarField):
        return field ** 0.5
    return field.with_storage(_sqrt_clipped(field.storage))


def _dirichlet_edge(field: ScalarField, axis: str) -> object:
    """Return the Dirichlet BC-sibling of the field's factor on ``axis``.

    Mirrors ``diffusion._dirichlet_face`` for the strain: a staggered
    first difference of a cell field lands on the BC-free nodal
    ``Inner`` face, whose Dirichlet sibling is the same interior face
    carrying the wall-value claim. Retagging a shear strain onto it is
    the free-slip wall (the shear *is* zero there); retagging a
    buoyancy gradient onto it is the tracer no-flux wall.
    """
    factor = field.function_space.bare.factor(axis)
    return factor.mesh.nodal(factor.node_set, bc=BC.DIRICHLET)


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
        # the bounded (walled) coordinate names, frozen at bind; empty
        # on a fully periodic grid (the periodic path stays bit-for-bit)
        self._walled: frozenset[str] = frozenset()

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
        r"""Resolve mixing targets (base), velocities, walls, and width.

        Raises
        ------
        NotImplementedError
            On a walled finite-volume (``CellAvg``) grid, on an
            unsupported wall placement (a fixed-value cell wall), on
            non-uniform mesh factors (no constant filter width), or
            with transverse (slaved) velocity components (no
            directional derivative).
        AssemblyError
            If the vertical coordinate is not a velocity axis (the
            :math:`N^2 = \partial_z b` read needs it).
        """
        super().bind(table)
        owner = type(self).__name__
        factors = getattr(table.grid, "factors", ())
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
        self._walled = self._classify_walls(table, owner)
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

    def _classify_walls(
        self, table: FieldTable, owner: str,
    ) -> frozenset[str]:
        r"""Return the bounded coordinate names; validate placements.

        Each factor the closure touches (the velocity trio, the
        buoyancy ``b`` the :math:`N^2` read needs, and the mixing
        targets) is classified with the diffusion ``_wall_treatment``
        vocabulary: a periodic factor is skipped, a wall-normal
        ``Inner[Dirichlet]`` face or a tangential / no-flux
        ``Center`` cell marks a walled axis, and any other placement (a
        fixed-value cell wall, a bare face) is rejected loudly. Walled
        finite-volume (``CellAvg``) factors are a separate future-work
        rejection — the free-slip strain retag is validated on the
        nodal family only.
        """
        walled: set[str] = set()
        touched = ({name for name, _ in self._vel_axes}
                   | {"b"} | set(self.targets))
        for name in touched:
            space = table[name].space
            for axis in space.names:
                factor = space.bare.factor(axis)
                if _wall_treatment(factor, owner, name, axis) == _PERIODIC:
                    continue
                if isinstance(factor, CellAvg):
                    raise NotImplementedError(
                        f"{owner} does not support walled "
                        f"finite-volume (CellAvg) grids yet (target "
                        f"{name!r} on the bounded axis {axis!r}); the "
                        "free-slip strain retag ships on the nodal "
                        "family only — assemble the model on the nodal "
                        "C-grid, or drop the closure on walled FV grids")
                walled.add(axis)
        return frozenset(walled)

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
        r"""One strain component on its natural C-grid position.

        The diagonal :math:`\Sigma_{aa} = \partial_a u_a` closes on the
        wall-normal velocity's own ``Inner[Dirichlet]`` tag (or is
        interior); the off-diagonal shear
        :math:`\Sigma_{a,t} = \tfrac12(\partial_t u_a + \partial_a u_t)`
        lands on the wall-normal edge, where each walled-axis factor of
        both terms is retagged onto its ``Inner[Dirichlet]`` sibling —
        the free-slip wall claim (the shear *is* zero there), which
        makes the two terms share one space and hands the structural
        zero to the :math:`|\Sigma|^2` and stress consumers.
        """
        qi, ax_i = self._vel_axes[i]
        qj, ax_j = self._vel_axes[j]
        if i == j:
            return state[qi].diff(ax_i)
        s_i = state[qi].diff(ax_j)
        s_j = state[qj].diff(ax_i)
        for ax in (ax_i, ax_j):
            if ax in self._walled:
                s_i = s_i.retag(_dirichlet_edge(s_i, ax))
                s_j = s_j.retag(_dirichlet_edge(s_j, ax))
        return 0.5 * (s_i + s_j)

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
        bz = state["b"].diff(self._vertical)
        if self._vertical in self._walled:
            # the tracer no-flux wall: the buoyancy gradient is zero at
            # a rigid wall, so its interior Inner face retags onto the
            # Dirichlet sibling and the interp to centers grounds
            bz = bz.retag(_dirichlet_edge(bz, self._vertical))
        n2 = bz.to(anchor) + n2_bg
        damped = _positive_part(sigma2 - beta * _positive_part(n2))
        return (cs * self._filter_width) ** 2 * _guarded_sqrt(damped)

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
        for i, (qi, ax_i) in enumerate(self._vel_axes):
            res = None
            for j in range(n):
                _, ax_j = self._vel_axes[j]
                s = self._strain(state, i, j)
                # the off-diagonal flux already carries the free-slip
                # Dirichlet tag on walled ax_j (from _strain), so the
                # closing diff telescopes with the structural zero; the
                # wall-normal diagonal (i == j) lands on the BC-free
                # Inner face and retags back onto the component's own tag
                divergence = (s * nu_t.to(s)).diff(ax_j)
                if i == j and ax_i in self._walled:
                    divergence = divergence.retag(state[qi])
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
                if axis in self._walled:
                    # no-flux tracer wall: retag the interior flux onto
                    # its Dirichlet sibling (structural zero wall flux)
                    flux = flux.retag(_dirichlet_edge(flux, axis))
                contribution = flux.diff(axis)
                res = (contribution if res is None
                       else res + contribution)
            out[name] = res
        return out
