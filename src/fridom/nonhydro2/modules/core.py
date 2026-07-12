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

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.model.modules.moving_geometry import mapping_params
from fridom.nonhydro2.diagnostics import DIAGNOSTICS
from fridom.nonhydro2.modules.mapped_pressure import (
    MappedPressureSolver,
)
from fridom.nonhydro2.modules.pressure import SpectralPressureSolver
from fridom.nonhydro2.params import DSQR, ROSSBY
from fridom.nonhydro2.state import State
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.operators.composed import (
    Divergence,
    Gradient,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.context import StepContext


@partial(jaxify, dynamic=("dsqr", "rossby"))
class DynamicalCore(fr.model.Module):

    """Declares u, v, w, p; owns dsqr/rossby and the projection.

    Parameters
    ----------
    dsqr : float | fr.model.Ramp, optional
        The squared aspect ratio ``(H/L)^2`` (default: 1.0); may be an
        ``fr.model.Ramp`` for a time-dependent aspect ratio.
    rossby_number : float | fr.model.Ramp, optional
        The Rossby number scaling the nonlinear terms (default: 1.0);
        may be a ``fr.model.Ramp`` for a spun-up nonlinearity.
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
    coords : tuple[str, ...], optional
        Grid coordinate names, used to size the projection halo
        exemption (default: ``("x", "y", "z")``).
    pressure_iterations : int, optional
        The fixed PCG iteration budget of the mapped pressure solve
        (CS-D2); consumed only on a grid whose coordinate mapping
        declares a mapped column — the flat spectral solve is exact
        and iterates nothing (default: 30).
    """

    state_type = State
    diagnostics = DIAGNOSTICS

    def __init__(
        self,
        dsqr: float | fr.model.Ramp = 1.0,
        *,
        rossby_number: float | fr.model.Ramp = 1.0,
        vertical: str = "z",
        coords: tuple[str, ...] = ("x", "y", "z"),
        pressure_iterations: int = 30,
    ) -> None:
        """Store the core parameter leaves and the geometry names."""
        self.dsqr = fr.model.leaf(dsqr)
        self.rossby = fr.model.leaf(rossby_number)
        self._vertical = vertical
        self._coords = coords
        self._pressure_iterations = pressure_iterations

    # ================================================================
    #  Field declarations
    # ================================================================
    field_declarations = (
        fr.model.FieldDeclaration.velocity(
            "u", "x", space=fr.spatial.Staggered("x"),
            long_name="Zonal velocity", units="m/s"),
        fr.model.FieldDeclaration.velocity(
            "v", "y", space=fr.spatial.Staggered("y"),
            long_name="Meridional velocity", units="m/s"),
        fr.model.FieldDeclaration.velocity(
            "w", "z", space=fr.spatial.Staggered("z"),
            long_name="Vertical velocity", units="m/s"),
        fr.model.FieldDeclaration(
            "p", space=fr.spatial.Collocated(),
            lifecycle=fr.model.Lifecycle.DIAGNOSTIC,
            long_name="Pressure", units="m^2/s^2"),
    )

    # ================================================================
    #  Parameters -- dsqr and the Rossby number live on the core
    # ================================================================
    parameter_declarations = (
        fr.model.ParameterDeclaration(DSQR, attr="dsqr", units="1",
                                doc="squared aspect ratio (H/L)^2"),
        fr.model.ParameterDeclaration(ROSSBY, attr="rossby", units="1",
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
    def stages(self) -> tuple[fr.model.Stage, ...]:
        """The velocity projection: replace u, v, w and write p."""
        return (
            fr.model.Stage(kind=fr.model.StageKind.CONSTRAINT, fn="_project",
                     name="projection"),
        )

    def _project(
        self, state: State, ctx: StepContext,
    ) -> dict[str, object]:
        r"""Project u, v, w divergence-free; write the pressure p.

        Description
        -----------
        Solve ``lap(phi) = div(u*)`` spectrally (the discrete C-grid
        eigenvalue), then subtract ``grad phi`` from the provisional
        velocity (the vertical component carries the ``1/dsqr``
        weighting of the nonhydrostatic pressure gradient). The
        stored diagnostic is the **normalized** pressure
        ``p = phi / ctx.stage_dt``: for an explicit one-stage scheme
        (``u* = u + dt * F`` on a divergence-free ``u``) this is
        exactly the potential of the projected *tendency*
        (``lap(p) = div(F)``), i.e. the physical, dt-independent
        pressure of the old stack's project-the-tendency form; on a
        backward leg ``stage_dt`` and ``phi`` flip sign together, so
        ``p`` keeps its physical sign. With a multistep stepper the
        diagnosed ``p`` is the pressure consistent with the stepper's
        weighted tendency combination — slightly time-filtered,
        O(dt^2), inherent to projection methods. The velocity update
        subtracts the gradient of the RAW potential ``phi``; the
        normalization only rescales the stored diagnostic.

        On a grid whose coordinate mapping declares a mapped column
        (terrain-following / boundary-fitted, stage C3) the whole
        stage routes to :meth:`_project_mapped`; a flat/unmapped
        grid takes exactly the code path below (zero behavior
        change).
        """
        grid = state["u"].grid
        mapping = getattr(grid, "mapping", None)
        if mapping is not None and mapping.column_corrections:
            return self._project_mapped(state, ctx)
        dsqr = ctx.params[DSQR]
        vel = VectorField({
            "u": state["u"], "v": state["v"], "w": state["w"]})
        div = Divergence()(vel)
        solver = SpectralPressureSolver(
            div.grid, div.function_space, vertical=self._vertical)
        p = solver.solve(div, dsqr=dsqr)
        grad = Gradient()(p)
        # nodal operator outputs are BC-free; on a walled grid every
        # wall-normal velocity carries the derived Dirichlet tag, so
        # adopt each target's tag — identity on periodic axes
        grad_u = grad["x"].retag(state["u"])
        grad_v = grad["y"].retag(state["v"])
        grad_w = grad[self._vertical].retag(state["w"])
        return {
            "u": state["u"] - grad_u,
            "v": state["v"] - grad_v,
            "w": state["w"] - grad_w / dsqr,
            "p": p / ctx.stage_dt,
        }

    def _project_mapped(
        self, state: State, ctx: StepContext,
    ) -> dict[str, object]:
        r"""Project on a coordinate-mapped grid (stage C3, CS-D2).

        Description
        -----------
        The mapped twin of :meth:`_project`: the divergence, the
        elliptic operator, and the gradient subtraction all come
        from one :class:`MappedPressureSolver` — the J-weighted
        physical divergence in flux form, the SPD flux-form mapped
        Laplacian solved by fixed-iteration PCG (the flat spectral
        inverse at folded coefficients preconditions), and the
        flux-consistent velocity corrections, so the projection
        removes exactly the divergence the operator measures (to
        the CG residual). The stored diagnostic keeps the
        ``p = phi / ctx.stage_dt`` normalization; the vertical
        weight ``1/dsqr`` rides the solver's ``weights`` seam
        keyed by the vertical coordinate name.

        Dynamic geometry (stage C4): the CURRENT mapping-parameter
        fields — module-owned state named after the parameters
        (``MovingGeometry``) — thread through the solver's
        ``params=`` seam, so every metric derivation of the
        operator, the preconditioner means, and the corrections
        reads the substage's geometry; a static mapped grid finds
        no parameter fields in the state and keeps the declaration
        defaults (the exact C3 path).
        """
        dsqr = ctx.params[DSQR]
        grid = state["u"].grid
        vel = {
            "x": state["u"],
            "y": state["v"],
            self._vertical: state["w"],
        }
        solver = MappedPressureSolver(
            grid,
            state["p"].function_space,
            weights={self._vertical: 1.0 / dsqr},
            iterations=self._pressure_iterations,
            params=mapping_params(state, grid))
        p = solver.solve(solver.divergence(vel))
        corr = solver.velocity_correction(p)
        return {
            "u": state["u"] - corr["x"].retag(state["u"]),
            "v": state["v"] - corr["y"].retag(state["v"]),
            "w": state["w"] - corr[self._vertical].retag(state["w"]),
            "p": p / ctx.stage_dt,
        }
