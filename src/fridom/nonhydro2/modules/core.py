"""The nonhydrostatic dynamical core module.

Description
-----------
``DynamicalCore`` is the package's dynamical-core module (D1.3): it
declares the velocity trio ``u, v, w`` (Velocity + ADVECTED roles, on
the C-grid staggered faces) and the diagnostic pressure ``p``; it owns
the core parameters ``nonhydro.dsqr`` and ``scaling.rossby``; and it
owns the pressure-projection **CONSTRAINT** stage (S4). It contributes
**no tendency terms** — Coriolis, buoyancy coupling, and advection are
separate modules — so the minimal core is declarations + a stage
(the decisive D1.3 evidence). It supplies the ``nh.State`` vocabulary
class through ``state_type``.

The ``dsqr`` and ``rossby`` scalars are additionally published as 1-DOF
``ConstantSpace`` AUXILIARY fields so the coupling/advection terms scale
by them through halo-traceable field-times-field products (the tracer
forbids raw ``.data`` scaling outside the ``extra_halo`` exemption);
the scalar provides are the second read surface for analytic consumers
(the pressure eigenvalue, ``nh.eigenmodes``).
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import fridom.framework2 as fr
from fridom.framework.utils import jaxify
from fridom.framework2.grid.decomposition.halo import HaloSpec
from fridom.framework2.grid.fields.vector_field import VectorField
from fridom.framework2.grid.operators.composed import (
    Divergence,
    Gradient,
)
from fridom.nonhydro2.diagnostics import DIAGNOSTICS
from fridom.nonhydro2.modules.pressure import SpectralPressureSolver
from fridom.nonhydro2.params import DSQR, ROSSBY
from fridom.nonhydro2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.model.context import StepContext


@partial(jaxify, dynamic=("dsqr", "rossby"))
class DynamicalCore(fr.Module):

    """Declares u, v, w, p; owns dsqr/rossby and the projection.

    Parameters
    ----------
    dsqr : float | fr.Ramp, optional
        The squared aspect ratio ``(H/L)^2`` (default: 1.0); may be an
        ``fr.Ramp`` for a time-dependent aspect ratio.
    rossby_number : float | fr.Ramp, optional
        The Rossby number scaling the nonlinear terms (default: 1.0);
        may be a ``fr.Ramp`` for a spun-up nonlinearity.
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
    coords : tuple[str, ...], optional
        Grid coordinate names, used to size the projection halo
        exemption (default: ``("x", "y", "z")``).
    """

    state_type = State
    diagnostics = DIAGNOSTICS

    def __init__(
        self,
        dsqr: float | fr.Ramp = 1.0,
        *,
        rossby_number: float | fr.Ramp = 1.0,
        vertical: str = "z",
        coords: tuple[str, ...] = ("x", "y", "z"),
    ) -> None:
        """Store the core parameter leaves and the geometry names."""
        self.dsqr = fr.leaf(dsqr)
        self.rossby = fr.leaf(rossby_number)
        self._vertical = vertical
        self._coords = coords

    # ================================================================
    #  Field declarations
    # ================================================================
    field_declarations = (
        fr.FieldDeclaration.velocity(
            "u", "x", space=fr.Staggered("x"),
            long_name="Zonal velocity", units="m/s"),
        fr.FieldDeclaration.velocity(
            "v", "y", space=fr.Staggered("y"),
            long_name="Meridional velocity", units="m/s"),
        fr.FieldDeclaration.velocity(
            "w", "z", space=fr.Staggered("z"),
            long_name="Vertical velocity", units="m/s"),
        fr.FieldDeclaration(
            "p", space=fr.Collocated(),
            lifecycle=fr.Lifecycle.DIAGNOSTIC,
            long_name="Pressure", units="m^2/s^2"),
    )

    # ================================================================
    #  Parameters -- dsqr and the Rossby number live on the core
    # ================================================================
    parameter_declarations = (
        fr.ParameterDeclaration(DSQR, attr="dsqr", units="1",
                                doc="squared aspect ratio (H/L)^2"),
        fr.ParameterDeclaration(ROSSBY, attr="rossby", units="1",
                                doc="Rossby number (nonlinear scaling)"),
    )

    # ================================================================
    #  The pressure-projection CONSTRAINT stage (S4)
    # ================================================================
    @property
    def extra_halo(self) -> HaloSpec:
        """Exempt the (global, spectral) projection from the halo trace.

        Description
        -----------
        The projection is a whole-domain spectral solve wrapping raw
        arrays (``Fourier``/``.data``); it declares its FD-stencil halo
        here (V-N2) rather than being traced.
        """
        return HaloSpec(dict.fromkeys(self._coords, 2))

    @property
    def stages(self) -> tuple[fr.Stage, ...]:
        """The velocity projection: replace u, v, w and write p."""
        return (
            fr.Stage(kind=fr.StageKind.CONSTRAINT, fn="_project",
                     name="projection"),
        )

    def _project(
        self, state: State, ctx: StepContext,
    ) -> dict[str, object]:
        """Project u, v, w divergence-free; write the pressure p.

        Description
        -----------
        Solve ``lap(p) = div(u*)`` spectrally (the discrete C-grid
        eigenvalue), then subtract ``grad p`` from the provisional
        velocity (the vertical component carries the ``1/dsqr``
        weighting of the nonhydrostatic pressure gradient).
        """
        dsqr = ctx.params[DSQR]
        vel = VectorField({
            "u": state["u"], "v": state["v"], "w": state["w"]})
        div = Divergence()(vel)
        solver = SpectralPressureSolver(
            div.grid, div.function_space, vertical=self._vertical)
        p = solver.solve(div, dsqr=dsqr)
        grad = Gradient()(p)
        return {
            "u": state["u"] - grad["x"],
            "v": state["v"] - grad["y"],
            "w": state["w"] - grad[self._vertical] / dsqr,
            "p": p,
        }
