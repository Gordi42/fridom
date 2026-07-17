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
from fridom.nonhydro2.modules.immersed_pressure import (
    ImmersedPressureSolver,
)
from fridom.nonhydro2.modules.mapped_pressure import (
    MappedPressureSolver,
)
from fridom.nonhydro2.modules.pressure import SpectralPressureSolver
from fridom.nonhydro2.params import DSQR, ROSSBY
from fridom.nonhydro2.state import State
from fridom.spatial.bc import BC
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
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.model.context import StepContext
    from fridom.spatial.grid import Grid
    from fridom.spatial.operators.base import Operator
    from fridom.spatial.operators.registry import DispatchKey


# ================================================================
#  The FV C-grid family choice (FV-D3 / stage F3)
# ================================================================
def _fv_capable(grid: Grid, *, dynamic_geometry: bool = False) -> bool:
    """
    Whether ``grid`` carries the FV C-grid auto default (FV-D2 A).

    Description
    -----------
    The finite-volume nonhydro model serves periodic, walled, mapped
    terrain-following (stage F5) *and* immersed cut-cell (stage I2,
    IP-D7) grids at 2nd order: periodic axes stagger on the ``Right``
    face, walled axes on the Neumann-``CellAvg`` / Dirichlet-``Inner``
    origins, a mapped column runs the family-aware
    :class:`MappedPressureSolver` (a ``CellAvg`` tracer transporting
    in conservative J-weighted flux form), and an immersed grid runs
    the masked :class:`ImmersedPressureSolver` (fractions as
    volume/area weights — immersed physics *is* finite-volume). The
    **auto** default is FV on all of them (owner ruling 2026-07-17:
    FV wherever capable, no surprising family changes by grid type;
    on immersed grids nodal would also be the *wrong* default — the
    nodal path ignores the mask, taught error at
    :func:`_require_nodal_capable`).

    One carve-out keeps the auto default on the validated nodal path:
    **dynamically driven** mappings (``dynamic_geometry=True``) — a
    grid whose mapping parameters move in time (a
    :class:`~fridom.model.modules.moving_geometry.MovingGeometry`
    module, stage C4) needs the ALE mesh-velocity correction for
    correct physics, and that correction is **nodal-only** — its
    column derivative reduces onto the nodal ``Center`` family and
    cannot retag onto ``CellAvg`` (the F5-style family-awareness gap
    was never done for :class:`MeshVelocityCorrection`). So a moving
    geometry auto-defaults to nodal, where ALE works, and the default
    path never hits that gap. Time-dependence is a *model* property
    (which modules are assembled), not a *grid* property — the grid
    cannot self-report it — so the caller (the ``nh.Model`` factory)
    supplies ``dynamic_geometry``; an explicit ``family="fv"`` with a
    ``MeshVelocityCorrection`` is a taught error at the module's
    ``bind`` (naming the gap), never a silent nodal fallback.

    A grid declaring **both** a mapped column and an immersed domain
    routes to FV as well, so the *specific* composition taught error
    fires (:func:`_require_fv_capable`), never the generic nodal one.

    Parameters
    ----------
    grid : Grid
        The assembled grid.
    dynamic_geometry : bool, optional
        Whether the model drives the grid's mapping parameters in time
        (a ``MovingGeometry`` module is assembled). ``True`` keeps the
        auto default nodal on a mapped grid, since the ALE correction
        is nodal-only (default: False).

    Returns
    -------
    bool
        True iff the grid is immersed, or its mapping (if any) is
        static — every grid the FV C-grid auto default is served on;
        only a dynamically driven mapping stays nodal (ALE).
    """
    # immersed physics is finite-volume (IP-D7): the nodal path
    # ignores the mask, and mapped + immersed must reach the specific
    # composition error in _require_fv_capable
    if getattr(grid, "immersed", None) is not None:
        return True
    # a mapping driven in time: the ALE correction is nodal-only, so
    # auto stays nodal (dynamic_geometry implies a mapped grid — a plain
    # grid carries no mapping parameters to drive)
    return not dynamic_geometry


def _require_fv_capable(grid: Grid) -> None:
    """
    Raise the FV-deferral taught error on a mapped + immersed grid.

    Description
    -----------
    Explicit ``family="fv"`` (or a ``Grid(family="fv")`` default) is
    served on periodic, walled (stage F4), mapped terrain-following
    (stage F5) *and* immersed cut-cell (stage I2) grids — the FV C-grid
    covers every geometry iteration 2 supports. The one combination it
    does not is a grid that declares **both** a mapped column and an
    immersed domain (plan §6, a designed-for composition): a taught
    error, never a silent unmasked or unmapped run.

    Parameters
    ----------
    grid : Grid
        The assembled grid.

    Raises
    ------
    NotImplementedError
        If the grid declares both a mapped column and an immersed
        domain.
    """
    mapping = getattr(grid, "mapping", None)
    mapped_column = mapping is not None and bool(
        getattr(mapping, "column_corrections", None))
    if mapped_column and getattr(grid, "immersed", None) is not None:
        raise NotImplementedError(
            "family='fv' on a grid that declares both a terrain-"
            "following mapped column and an immersed domain is a "
            "designed-for composition (immersed-partial-cells plan §6): "
            "the mapped PCG and the masked PCG are not yet composed. "
            "Use one geometry or the other.")


def _require_nodal_capable(grid: Grid) -> None:
    """
    Raise the nodal-deferral taught error on an immersed grid (IP-D7).

    Description
    -----------
    Immersed physics is finite-volume: the cut-cell fractions are the
    volume/area weights of the masked divergence, operator and flux
    forms, and the nodal (point-value) path never consults them — a
    nodal immersed run is silently **unmasked**. So explicit
    ``family="nodal"`` on an immersed grid is a taught error (strictly
    better than the pre-I2 silent unmasked nodal run), never a fallback.

    Parameters
    ----------
    grid : Grid
        The assembled grid.

    Raises
    ------
    NotImplementedError
        If the grid carries an immersed domain.
    """
    if getattr(grid, "immersed", None) is None:
        return
    raise NotImplementedError(
        "family='nodal' on an immersed grid ignores the mask: immersed "
        "physics is finite-volume — the cut-cell fractions are the "
        "volume/area weights of the masked pressure solve and the "
        "flux-form advection, which the nodal path never consults, so a "
        "nodal immersed run is silently unmasked. Use family='fv' (the "
        "auto default on an immersed grid, stage I2) or drop the "
        "immersed domain.")


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

    On a **periodic** mesh factor the overrides key the periodic face
    (``Right``). On a **walled** (bounded) mesh factor (stage F4) the
    face family is ``Inner`` and the pressure / velocity carry the
    walled parity tags: the pressure gradient staggers the
    Neumann-tagged ``CellAvg`` (the solve space's DCT-II origin) onto
    the interior faces, and the flux divergence staggers the
    Dirichlet-tagged interior faces (the wall-normal velocity's
    no-normal-flow parity) back onto the cells — so both the
    ``Div @ Diag @ Grad`` pressure chain (on the tagged origins) and
    the physical divergence / gradient (on the BC-free faces) resolve.
    The nodal ``("diff", ...) -> Center`` chains are deliberately
    *not* touched — the overrides ride only the FV-family grid, so a
    nodal model keeps its resolution.

    Parameters
    ----------
    meshes : tuple
        The grid's mesh factors (``grid.factors``).

    Returns
    -------
    dict[DispatchKey, Operator]
        The per-mesh-factor ``diff`` profile: ``CellAvg -> face``
        (``FaceDifference``) and ``face -> CellAvg``
        (``FluxDifference``), on the BC-free and (walled) tagged
        origins alike.
    """
    face_diff = FaceDifference()
    flux_diff = FluxDifference()
    overrides: dict[DispatchKey, Operator] = {}
    for mesh in meshes:
        try:
            cell_avg = mesh.cell_avg
        except (AttributeError, ValueError, NotImplementedError):
            continue  # a mesh without the FV / cell-average family
        overrides[("diff", cell_avg)] = face_diff
        if getattr(mesh, "periodic", False):
            overrides[("diff", mesh.right)] = flux_diff
        else:
            # walled (F4): the tagged pressure / velocity origins plus
            # the BC-free interior face
            overrides[("diff", mesh.average(CellAvg, bc=BC.NEUMANN))] = (
                face_diff)
            overrides[("diff", mesh.inner)] = flux_diff
            overrides[("diff", mesh.nodal(
                NodeSet.INNER, bc=BC.DIRICHLET))] = flux_diff
    return overrides


def resolve_model_family(
    family: str | None, grid: Grid, *, dynamic_geometry: bool = False,
) -> str:
    r"""
    Resolve a model-assembly family choice against the grid (F3).

    Description
    -----------
    The nonhydro model-assembly family (``nh.Model(family=...)`` /
    ``DynamicalCore(family=...)``, scoping study §8): ``None`` is the
    **auto** default — it follows the grid's own ``default_family``,
    and *promotes* the ``"nodal"`` grid default to ``"fv"`` whenever
    the grid can carry the FV C-grid (:func:`_fv_capable`). This is
    the flip: a plain periodic, walled, **static mapped** or
    **immersed** nonhydro model is finite-volume by default — safe
    because the 2nd-order stencils are bit-identical to nodal on
    flat/walled grids (scoping study §1; the walled solve is
    eager-bitwise, ≤1.2e-14 jitted, §11), the mapped FV pressure
    operator is likewise bit-identical to nodal (the mapped column
    rides a uniform *computational* mesh, §13), and on an immersed
    grid FV is the only *correct* choice (immersed physics is
    finite-volume, stage I2; the nodal path silently ignores the
    mask) — the owner ruling of 2026-07-17 (FV wherever capable).
    The auto default stays ``"nodal"`` only on a **dynamically
    driven** mapping (``dynamic_geometry=True``: the ALE correction
    is nodal-only, :func:`_fv_capable`). An **explicit** ``"fv"`` is
    served on periodic, walled, mapped terrain-following *and*
    immersed cut-cell grids (stages F4, F5, I2); only a grid that
    declares **both** a mapped column and an immersed domain rejects
    it (the mapped and masked PCGs are not yet composed, plan §6),
    and an explicit ``"fv"`` with a moving-geometry ALE module is
    rejected at that module's ``bind`` (dynamism is not visible on
    the grid). An explicit ``"nodal"`` on an immersed grid is
    likewise a taught error (the mask is FV-only, IP-D7).

    Parameters
    ----------
    family : str | None
        The requested family, or None for the auto default.
    grid : Grid
        The assembled grid the model runs on.
    dynamic_geometry : bool, optional
        Whether the model drives the grid's mapping parameters in time
        (a ``MovingGeometry`` module is assembled). Consulted only for
        the auto default, where it keeps a moving mapping on the nodal
        path (the ALE correction is nodal-only); an explicit family
        ignores it (default: False).

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
        if family == "nodal" and _fv_capable(
                grid, dynamic_geometry=dynamic_geometry):
            family = "fv"
    if family not in FAMILIES:
        raise ValueError(
            f"the model family must be one of {FAMILIES} or None "
            f"(auto), got {family!r}")
    if family == "fv":
        _require_fv_capable(grid)
    else:
        _require_nodal_capable(grid)
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
        The fixed PCG iteration budget of the fixed-iteration pressure
        solve (CS-D2); consumed on a grid whose coordinate mapping
        declares a mapped column *and* on an immersed (cut-cell) grid
        — both run the fixed-iteration PCG. The flat spectral solve is
        exact and iterates nothing (default: 30).
    pressure_tolerance : float | None, optional
        An optional PCG convergence break forwarded to the mapped and
        immersed pressure solvers (the measure-weighted true relative
        residual; masked scan, exact gradient — see
        :class:`ConjugateGradient`). ``pressure_iterations`` becomes the
        maximum budget. ``None`` runs the fixed count; the tolerance
        should sit above the residual floor (~1e-14) or it never fires
        (default: None).
    pressure_preconditioner : str, optional
        The PCG preconditioner of the fixed-iteration pressure solve
        (B4): ``"spectral"`` (the flat separable spectral inverse) or
        ``"multigrid"`` (the semicoarsened geometric-multigrid V-cycle).
        Consumed on a mapped or immersed grid; the flat spectral solve
        is exact and ignores it (a flat grid never raises on the knob).
        Static (a treedef aux, part of the module fingerprint), like
        ``single_precision_solve`` (default: ``"spectral"``).
    multigrid_levels : int, optional
        The maximum multigrid level count when
        ``pressure_preconditioner="multigrid"`` (the builder floors on
        small grids); ignored otherwise. Static in the fingerprint
        (default: 5).
    family : str | None, optional
        The discretization family of the whole core state (FV-D3,
        stage F3): ``"fv"`` declares ``u, v, w, p`` on the
        finite-volume C-grid (scalars on ``CellAvg``, velocities on
        the point-value faces — FV-D2 option A) and seeds the FV
        C-grid ``diff`` profile so the pressure chain staggers on the
        average family; ``"nodal"`` is the point-value C-grid. ``None``
        defers to the grid-level default (``grid.default_family``), so
        an explicitly assembled core follows the grid. The
        ``nh.Model`` factory resolves the flip (any periodic, walled,
        static-mapped or immersed grid promotes ``None`` to ``"fv"``;
        a moving-geometry mapping stays nodal); an explicit ``"fv"``
        on a grid with both a mapped column and an immersed domain,
        or an explicit ``"nodal"`` on an immersed grid, is a taught
        error (default: None).
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
        pressure_tolerance: float | None = None,
        pressure_preconditioner: str = "spectral",
        multigrid_levels: int = 5,
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
        self._pressure_tolerance = pressure_tolerance
        self._pressure_preconditioner = pressure_preconditioner
        self._multigrid_levels = multigrid_levels
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
        An FV family on a non-capable grid — one that declares both a
        mapped column and an immersed domain — is a taught error here,
        never a silent nodal fallback; walled, mapped and immersed
        grids are served (stages F4, F5, I2).

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
        stage routes to :meth:`_project_mapped`; on an immersed
        (cut-cell) grid it routes to :meth:`_project_immersed` (stage
        I2); a flat/unmapped/unimmersed grid takes exactly the code
        path below (zero behavior change).
        """
        grid = state["u"].grid
        mapping = getattr(grid, "mapping", None)
        if mapping is not None and mapping.column_corrections:
            return self._project_mapped(state, ctx)
        if getattr(grid, "immersed", None) is not None:
            return self._project_immersed(state, ctx)
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
            tolerance=self._pressure_tolerance,
            single_precision=self._single_precision_solve,
            preconditioner=self._pressure_preconditioner,
            multigrid_levels=self._multigrid_levels,
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

    def _project_immersed(
        self, state: State, ctx: StepContext,
    ) -> dict[str, object]:
        r"""Project on an immersed (cut-cell) grid (stage I2, IP-D6).

        Description
        -----------
        The masked twin of :meth:`_project`: the divergence, the
        cut-cell elliptic operator, and the gradient subtraction all
        come from one :class:`ImmersedPressureSolver` — the
        full-volume-scaled masked divergence, the SPD open-area-weighted
        Poisson operator solved by fixed-iteration PCG (the wet-masked
        flat spectral inverse preconditions, projected onto the
        wet-region-constant nullspace), and the boolean-masked velocity
        corrections, so the projection removes exactly the masked
        divergence the operator measures (to the CG residual). The
        stored diagnostic keeps the ``p = phi / ctx.stage_dt``
        normalization and is masked to zero under the topography; the
        vertical weight ``1/dsqr`` rides the solver's own vertical leg.
        A ``MaskState`` CONSTRAINT stage (added by the factory) keeps
        the dry velocity DOFs dead against the other tendency modules.
        """
        dsqr = ctx.params[DSQR]
        grid = state["u"].grid
        vel = {
            "x": state["u"],
            "y": state["v"],
            self._vertical: state["w"],
        }
        solver = ImmersedPressureSolver(
            grid,
            state["p"].function_space,
            vertical=self._vertical,
            dsqr=dsqr,
            iterations=self._pressure_iterations,
            tolerance=self._pressure_tolerance,
            single_precision=self._single_precision_solve,
            preconditioner=self._pressure_preconditioner,
            multigrid_levels=self._multigrid_levels)
        p, corr = solver.project(vel)
        return {
            "u": state["u"] - corr["x"].retag(state["u"]),
            "v": state["v"] - corr["y"].retag(state["v"]),
            "w": state["w"] - corr[self._vertical].retag(state["w"]),
            "p": p / ctx.stage_dt,
        }
