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
from fridom.spatial.operators.flux_diff import (
    FaceDifference,
    FluxDifference,
)
from fridom.spatial.space_patterns import FAMILIES

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.model.context import StepContext
    from fridom.spatial.grid import Grid
    from fridom.spatial.operators.base import Operator
    from fridom.spatial.operators.registry import DispatchKey


# ================================================================
#  The FV C-grid family choice (FV-D3 / stage F3)
# ================================================================
def _fv_capable(grid: Grid) -> bool:
    """
    Whether ``grid`` can carry the periodic FV C-grid (FV-D2 A).

    Description
    -----------
    The finite-volume nonhydro model is periodic-only at 2nd order
    (scoping study §5, FV-D4/F5): every mesh factor must be periodic
    (walls are the open FV-D4 design, stage F4) and the grid must be
    unmapped and unimmersed (mapped/cut-cell FV is stage F5). A grid
    meeting all three seeds the FV C-grid diff profile; anything else
    stays on the validated nodal path.

    Parameters
    ----------
    grid : Grid
        The assembled grid.

    Returns
    -------
    bool
        True iff the grid is fully periodic, unmapped, unimmersed.
    """
    if getattr(grid, "mapping", None) is not None:
        return False
    if getattr(grid, "immersed", None) is not None:
        return False
    return all(getattr(mesh, "periodic", False)
               for mesh in grid.factors)


def _require_fv_capable(grid: Grid) -> None:
    """
    Raise the FV-deferral taught error on a non-capable grid.

    Description
    -----------
    Explicit ``family="fv"`` (or a ``Grid(family="fv")`` default) on a
    grid the FV C-grid cannot yet serve is a taught error, never a
    silent fallback to nodal (the scoping study's FV-D4/F5 deferral).

    Parameters
    ----------
    grid : Grid
        The assembled grid.

    Raises
    ------
    NotImplementedError
        If the grid is walled, mapped, or immersed.
    """
    if _fv_capable(grid):
        return
    reasons = []
    if any(not getattr(mesh, "periodic", False)
           for mesh in grid.factors):
        reasons.append(
            "it has bounded (walled) axes — walled FV is stage F4 "
            "(FV-D4, an open boundary-condition design)")
    if getattr(grid, "mapping", None) is not None:
        reasons.append(
            "it carries a coordinate mapping — mapped / terrain-"
            "following FV is stage F5")
    if getattr(grid, "immersed", None) is not None:
        reasons.append(
            "it carries an immersed domain — cut-cell FV is stage F5")
    raise NotImplementedError(
        "family='fv' is the finite-volume nonhydro model (FV-D2 "
        "option A: scalars on CellAvg, velocities on the C-grid "
        "faces), which currently serves a fully periodic, unmapped, "
        "unimmersed grid only. This grid cannot: "
        + "; ".join(reasons)
        + ". Keep this model family='nodal' (the validated walled / "
        "mapped path), which at 2nd order is bit-identical on the "
        "periodic interior anyway (scoping study §1).")


def fv_cgrid_overrides(
    meshes: tuple,
) -> dict[DispatchKey, Operator]:
    r"""
    Build the FV C-grid ``diff`` override profile (FV-D3, stage F3).

    Description
    -----------
    Re-points the vector-calculus ``("diff", factor)`` resolution so
    that ``grad`` / ``div`` / ``laplacian`` stagger on the average
    family exactly as the nodal C-grid does on the point-value family
    (scoping study §5 FV-D3): the cell-average pressure gradient
    staggers onto the face (``("diff", CellAvg) -> FaceDifference``,
    ``CellAvg -> Right``) and the face-flux divergence lands back on
    the cell (``("diff", Right) -> FluxDifference``, ``Right ->
    CellAvg``). Because the two 2nd-order stencils are bit-identical
    to the nodal ``Center -> Right`` / ``Right -> Center`` numbers
    (scoping study §1), the whole family-agnostic model — the
    ``Div @ Diag @ Grad`` pressure chain included — then runs on the
    FV C-grid unchanged, at bitwise parity with the nodal model.

    The complementary *interpolation* staggering is not overridden
    here: ``("interpolate", CellAvg) -> Right`` is already the seeded
    reconstruct row (G4), and the face-to-cell leg the symbol kit
    needs (``Right -> CellAvg``) is inferred per field by
    ``GridSymbols`` (an average-family field routes an interpolate on
    a nodal-face factor through the ``"average"`` reconstruct kind), so
    the global ``("interpolate", Right)`` row stays the nodal
    ``Right -> Center`` — which the mixed corner (a nodal scalar on an
    FV grid) still relies on for ``.to``.

    The overrides key the periodic face (``Right``); an FV C-grid is
    periodic-only at 2nd order (:func:`_require_fv_capable`), so no
    bounded ``Inner`` row is produced. The nodal ``("diff", Right) ->
    Center`` chain is deliberately *not* touched — it is overridden
    only on the FV-family grid, so a nodal model keeps its resolution.

    Parameters
    ----------
    meshes : tuple
        The grid's mesh factors (``grid.factors``).

    Returns
    -------
    dict[DispatchKey, Operator]
        The ``{("diff", cell_avg): FaceDifference(),
        ("diff", right): FluxDifference()}`` profile, per mesh factor.
    """
    face_diff = FaceDifference()
    flux_diff = FluxDifference()
    overrides: dict[DispatchKey, Operator] = {}
    for mesh in meshes:
        try:
            cell_avg = mesh.cell_avg
            face = mesh.right
        except (AttributeError, ValueError, NotImplementedError):
            continue  # a mesh without the FV / periodic-face family
        overrides[("diff", cell_avg)] = face_diff
        overrides[("diff", face)] = flux_diff
    return overrides


def resolve_model_family(family: str | None, grid: Grid) -> str:
    r"""
    Resolve a model-assembly family choice against the grid (F3).

    Description
    -----------
    The nonhydro model-assembly family (``nh.Model(family=...)`` /
    ``DynamicalCore(family=...)``, scoping study §8): ``None`` is the
    **auto** default — it follows the grid's own ``default_family``,
    and *promotes* the ``"nodal"`` grid default to ``"fv"`` whenever
    the grid can carry the periodic FV C-grid (:func:`_fv_capable`).
    This is the flip: a plain periodic nonhydro model is finite-volume
    by default, safe because the 2nd-order stencils are bit-identical
    to nodal (scoping study §1). A walled or mapped grid stays
    ``"nodal"`` (the validated path). An explicit ``"fv"`` on a
    non-capable grid is a taught error, never a silent fallback.

    Parameters
    ----------
    family : str | None
        The requested family, or None for the auto default.
    grid : Grid
        The assembled grid the model runs on.

    Returns
    -------
    str
        The resolved concrete family (``"nodal"`` or ``"fv"``).

    Raises
    ------
    ValueError
        If ``family`` is neither None nor a known family name.
    NotImplementedError
        If the resolved family is ``"fv"`` on a non-capable grid.
    """
    if family is None:
        family = getattr(grid, "default_family", "nodal")
        if family == "nodal" and _fv_capable(grid):
            family = "fv"
    if family not in FAMILIES:
        raise ValueError(
            f"the model family must be one of {FAMILIES} or None "
            f"(auto), got {family!r}")
    if family == "fv":
        _require_fv_capable(grid)
    return family


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
    single_precision_solve : bool, optional
        Run the spectral machinery of the pressure projection in
        single precision while the velocity state stays ``float64``
        — a performance option. On a flat grid this is the whole
        solve (the ``rfftn`` / spectral divide / ``irfftn``
        pipeline, forwarded to :class:`SpectralPressureSolver`); on
        a mapped grid it is the PCG *preconditioner* (forwarded to
        :class:`MappedPressureSolver` — mixed-precision PCG: the
        iterates, the operator and the inner products stay
        ``float64``). Static (a treedef aux, part of the module
        fingerprint), so a given value never retraces. Off by
        default (bitwise identical projection); on, the result
        carries the reduced round-off of the affected pipeline, an
        opt-in accuracy trade (default: False).
    pressure_iterations : int, optional
        The fixed PCG iteration budget of the mapped pressure solve
        (CS-D2); consumed only on a grid whose coordinate mapping
        declares a mapped column — the flat spectral solve is exact
        and iterates nothing (default: 30).
    family : str | None, optional
        The discretization family of the whole core state (FV-D3,
        stage F3): ``"fv"`` declares ``u, v, w, p`` on the
        finite-volume C-grid (scalars on ``CellAvg``, velocities on
        the point-value faces — FV-D2 option A) and seeds the FV
        C-grid ``diff`` profile so the pressure chain staggers on the
        average family; ``"nodal"`` is the point-value C-grid. ``None``
        defers to the grid-level default (``grid.default_family``), so
        an explicitly assembled core follows the grid. The
        ``nh.Model`` factory resolves the flip (a periodic grid
        promotes ``None`` to ``"fv"``); an explicit ``"fv"`` on a
        walled or mapped grid is a taught error (default: None).
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
        single_precision_solve: bool = False,
        pressure_iterations: int = 30,
        family: str | None = None,
    ) -> None:
        """Store the core parameter leaves and the geometry names."""
        if family is not None and family not in FAMILIES:
            raise ValueError(
                f"family must be one of {FAMILIES} or None, got "
                f"{family!r}")
        self.dsqr = fr.model.leaf(dsqr)
        self.rossby = fr.model.leaf(rossby_number)
        self._vertical = vertical
        self._coords = coords
        self._single_precision_solve = bool(single_precision_solve)
        self._pressure_iterations = pressure_iterations
        self._family = family

    # ================================================================
    #  Field declarations
    # ================================================================
    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """Declare ``u, v, w, p`` on the requested family (FV-D3).

        Description
        -----------
        The velocity trio stays on the C-grid faces on both families
        (``Staggered`` -> ``Right`` under nodal and FV alike, FV-D2
        option A); ``p`` is the collocated cell scalar (``Center``
        nodal, ``CellAvg`` under ``family="fv"``). ``self._family`` is
        None (defer to the grid default) unless the factory / user
        pinned it.
        """
        family = self._family
        return (
            fr.model.FieldDeclaration.velocity(
                "u", "x", space=fr.spatial.Staggered("x", family=family),
                long_name="Zonal velocity", units="m/s"),
            fr.model.FieldDeclaration.velocity(
                "v", "y", space=fr.spatial.Staggered("y", family=family),
                long_name="Meridional velocity", units="m/s"),
            fr.model.FieldDeclaration.velocity(
                "w", "z", space=fr.spatial.Staggered("z", family=family),
                long_name="Vertical velocity", units="m/s"),
            fr.model.FieldDeclaration(
                "p", space=fr.spatial.Collocated(family=family),
                lifecycle=fr.model.Lifecycle.DIAGNOSTIC,
                long_name="Pressure", units="m^2/s^2"),
        )

    # ================================================================
    #  The FV C-grid diff profile (grid-aware dispatch hook, F3)
    # ================================================================
    def grid_dispatch_overrides(
        self, grid: Grid,
    ) -> Mapping[DispatchKey, Operator]:
        r"""
        Contribute the FV C-grid ``diff`` profile when family='fv'.

        Description
        -----------
        The grid-aware twin of the static ``dispatch`` attribute
        (consumed at assembly step 3, so it applies to preset *and*
        explicit assembly): when the resolved family is ``"fv"`` it
        merges the per-mesh-factor ``("diff", CellAvg) ->
        FaceDifference`` / ``("diff", Right) -> FluxDifference``
        overrides (:func:`fv_cgrid_diff_overrides`), so ``grad`` /
        ``div`` / ``laplacian`` — the pressure chain — stagger on the
        average family. Keyed on per-factor spaces (which a
        ``SpacePattern`` cannot express), it needs the grid; the
        static ``dispatch`` attribute cannot build it.

        The family resolves as the declarations do —
        ``self._family`` or the grid default — so the profile is on
        exactly the grids whose ``u, v, w, p`` landed on ``CellAvg``.
        An FV family on a non-capable (walled / mapped / immersed)
        grid is a taught error here, never a silent nodal fallback.

        Parameters
        ----------
        grid : Grid
            The assembled grid the model runs on.

        Returns
        -------
        Mapping[DispatchKey, Operator]
            The FV C-grid diff overrides (empty on a nodal model).
        """
        family = (self._family if self._family is not None
                  else getattr(grid, "default_family", "nodal"))
        if family != "fv":
            return {}
        _require_fv_capable(grid)
        return fv_cgrid_overrides(grid.factors)

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
            div.grid, div.function_space, vertical=self._vertical,
            single_precision=self._single_precision_solve)
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
            single_precision=self._single_precision_solve,
            params=mapping_params(state, grid))
        # one metric derivation for the whole projection: divergence,
        # solve and correction share the solver's per-solve memo (it
        # dies with the call, so the next step re-derives at the new
        # geometry — MappedPressureSolver.project)
        p, corr = solver.project(vel)
        return {
            "u": state["u"] - corr["x"].retag(state["u"]),
            "v": state["v"] - corr["y"].retag(state["v"]),
            "w": state["w"] - corr[self._vertical].retag(state["w"]),
            "p": p / ctx.stage_dt,
        }
