r"""The nonhydrostatic dynamical core module.

Description
-----------
``Core`` is the package's dynamical-core module (D1.3): it declares
the velocity trio ``u, v, w`` (Velocity + ADVECTED roles, on the
C-grid staggered faces) and the diagnostic pressure ``p``; it owns
the aspect ratio :math:`\delta` (``nonhydro.aspect_ratio``, squared
at every use site); and it owns the pressure-projection
**CONSTRAINT** stage (S4). It contributes **no tendency terms** —
Coriolis, buoyancy coupling, and advection are separate modules — so
the minimal core is declarations + a stage (the decisive D1.3
evidence). It supplies the ``nh.State`` vocabulary class through
``state_type``.

The core is **scaling-neutral** (``fr.scaling``): the aspect ratio
:math:`\delta = H/L` is a pure geometry number present in every
assembly — dimensional and nondimensional alike — and the core
carries no nonlinearity leaf of its own (the scaling variant is fixed
by the Coriolis / stratification module kwarg sets and the
``Model(scaling=)`` policy). The projection's vertical weight is the
live :math:`1/\delta^2`, squared at the use site from the provided
:math:`\delta` (``ctx.params``, stage time — Ramp-able).
"""
from __future__ import annotations

import numbers
from functools import partial
from typing import TYPE_CHECKING

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.model.halo_demand import derive_extra_halo
from fridom.model.modules.moving_geometry import mapping_params
from fridom.nonhydro2.diagnostics import DIAGNOSTICS
from fridom.nonhydro2.modules.composed_pressure import (
    ComposedPressureSolver,
)
from fridom.nonhydro2.modules.immersed_pressure import (
    ImmersedPressureSolver,
)
from fridom.nonhydro2.modules.mapped_pressure import (
    MappedPressureSolver,
)
from fridom.nonhydro2.modules.pressure import SpectralPressureSolver
from fridom.nonhydro2.params import ASPECT_RATIO
from fridom.nonhydro2.state import State
from fridom.nonhydro2.units import (
    COMPONENT_FACTORS,
    DERIVED_FACTORS,
    coordinate_factors,
)
from fridom.spatial.bc import BC
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.operators.banded import validate_tridiagonal_method
from fridom.spatial.operators.composed import (
    Divergence,
    Gradient,
)
from fridom.spatial.operators.flux_diff import (
    FaceDifference,
    FluxDifference,
)
from fridom.spatial.operators.multigrid_hierarchy import (
    validate_agglomerate,
)
from fridom.spatial.operators.staggering import mapped_mesh
from fridom.spatial.space_patterns import FAMILIES
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.model.context import StepContext
    from fridom.spatial.decomposition.halo import HaloSpec
    from fridom.spatial.grid import Grid
    from fridom.spatial.operators.base import Operator
    from fridom.spatial.operators.registry import DispatchKey


# ================================================================
#  Pressure-projection routing predicates (S4)
# ================================================================
def _stretched_column(grid: Grid) -> bool:
    r"""
    Whether ``grid`` carries a stretched (mapped) mesh factor.

    Description
    -----------
    The second half of the mapped-solve routing predicate
    (:meth:`Core._project`). A
    :class:`~fridom.spatial.meshes.mapped_interval.MappedIntervalMesh`
    factor — a stretched vertical, the usual boundary-layer refinement
    — sets **no** ``mapping.column_corrections``: it declares no
    analytic map, only a non-uniform measure. But it is not a flat
    grid either: a coordinate-mapped mesh deliberately carries no
    spectral basis (``MappedIntervalMesh.cosine()`` raises), so the
    separable :class:`SpectralPressureSolver` has no transform to build
    on it. Such a grid is the **degenerate mapped column** (identity
    map, ``J == 1``, the stretch entirely in ``grid.measure``), which
    the mapped PCG's stretched-base path (N2/N3) already serves
    exactly — so it routes there.

    Parameters
    ----------
    grid : Grid
        The assembled grid the model runs on.

    Returns
    -------
    bool
        True iff at least one mesh factor carries a coordinate map.
    """
    return any(mapped_mesh(mesh) for mesh in grid.factors)


def _no_spectral_on_stretched(route: str) -> str:
    """
    Build the taught error of an explicit spectral fold on a stretch.

    Description
    -----------
    Every spectral leg — the flat separable solve and the masked /
    mapped spectral *preconditioner* alike — needs a per-axis
    ``transform`` row, and a
    :class:`~fridom.spatial.meshes.mapped_interval.MappedIntervalMesh`
    carries no spectral basis (``cosine()`` deliberately raises), so
    the row is missing and the dispatch fails deep inside the solve
    with a bare ``DispatchError``. The auto default already avoids
    this (:meth:`Core._resolved_preconditioner` resolves a stretched
    grid to ``"multigrid"``), so only an **explicit**
    ``pressure_preconditioner="spectral"`` reaches here.

    Parameters
    ----------
    route : str
        The projection route the message names (``"immersed"``).

    Returns
    -------
    str
        The taught error message.
    """
    return (
        f"nh.Core(pressure_preconditioner='spectral') on a stretched "
        f"grid: the {route} spectral preconditioner needs a per-axis "
        "spectral transform, which a stretched MappedIntervalMesh "
        "factor does not supply (a coordinate-mapped mesh carries no "
        "spectral basis). Use pressure_preconditioner='multigrid' — "
        "the V-cycle reads the same grid.measure widths and serves a "
        "stretched column — or drop the argument, which is the auto "
        "default on a stretched grid")


def _no_staggered_face(axis: str, coords: tuple[str, ...]) -> str:
    """
    Build the taught error of a coordinate with no velocity face.

    Description
    -----------
    :meth:`Core.bind` derives the projection's halo from the C-grid
    ``div`` / ``grad`` rows, which needs the *staggered* face factor of
    each coordinate — the one the velocity trio lives on. The trio is
    declared on the **fixed** coordinates ``"x"``, ``"y"``, ``"z"``
    (:attr:`Core.field_declarations`), which ``vertical=`` does not
    move: that argument only re-keys the projection legs of the
    already-declared ``w``. So a ``coords=`` naming anything else
    leaves that coordinate collocated in every component and the face
    lookup finds nothing. Before this message that lookup was a bare
    ``next()`` and the failure a naked ``StopIteration``.

    Parameters
    ----------
    axis : str
        The coordinate with no staggered velocity face.
    coords : tuple[str, ...]
        The core's declared coordinate names.

    Returns
    -------
    str
        The taught error message.
    """
    return (
        f"nh.Core: no velocity component is staggered along "
        f"{axis!r}, so the pressure projection has no divergence leg "
        f"to difference there. The core declares u, v, w on the "
        f"fixed coordinates 'x', 'y', 'z' (the fr.spatial.Staggered "
        f"spaces of Core.field_declarations), so every name in coords= "
        f"must be one of those; coords={coords} names {axis!r}. "
        f"Renaming the vertical does not move that declaration — "
        f"vertical= only re-keys the legs of the already-declared w — "
        f"so a differently named vertical mesh is not (yet) supported: "
        f"name the grid's vertical mesh 'z' and keep the defaults "
        f"vertical='z', coords=('x', 'y', 'z')")


def _no_chart_grid(chart: tuple[str, ...]) -> str:
    r"""
    Build the taught refusal of a chart-coupled grid.

    Description
    -----------
    The nonhydrostatic model is **Cartesian-only**. Every stage the
    core owns is metric-blind: the pressure projection differences the
    C-grid ``div`` / ``grad`` legs with no :math:`\sqrt{g}` volume and
    no :math:`g_{ij}` raising, and the flux-form advection it composes
    with transports along the coordinate directions. On a grid carrying
    an embedding chart (:attr:`fridom.spatial.Grid.chart_coords`) those
    expressions are not merely less accurate, they are the wrong
    equations — so the refusal is a taught error rather than a silent
    run.

    The refusal is raised **before** the coordinate loop of
    :meth:`Core.bind`, which is what a chart grid used to die in: the
    default ``coords=("x", "y", "z")`` raised a bare ``KeyError`` out
    of the pressure space's ``factor("x")`` lookup, and a
    ``coords=("lon", "lat", "z")`` reached :func:`_no_staggered_face`,
    whose advice (rename the vertical mesh to ``"z"``) is wrong here —
    the vertical *is* named ``"z"``; it is the horizontal pair that is
    charted. The two messages therefore compose: chart-ness is decided
    first, and :func:`_no_staggered_face` keeps the flat-grid
    vertical-naming case it was written for.

    Parameters
    ----------
    chart : tuple[str, ...]
        The grid's chart-coupled base coordinates.

    Returns
    -------
    str
        The taught error message.
    """
    return (
        f"nh.Core: this grid carries an embedding chart on {chart}, "
        f"which the nonhydrostatic model does not support. The core "
        f"declares u, v, w on the fixed Cartesian coordinates 'x', "
        f"'y', 'z', and every stage it owns is metric-blind — the "
        f"pressure projection differences the C-grid legs with no "
        f"sqrt(g) volume and no g_ij raising, and flux-form advection "
        f"transports along the coordinate directions — so a chart run "
        f"would be wrong physics, not merely a less accurate one. "
        f"Supported instead: a Cartesian grid, optionally carrying a "
        f"fr.spatial.CoordinateMapping(maps=...) terrain-following or "
        f"stretched column (a per-coordinate map is not a chart, and "
        f"leaves chart_coords None). For a model ON a chart use the "
        f"shallow-water package — sw.Model on "
        f"fr.spatial.spherical.Grid — which carries the metric-aware "
        f"chart forms (fr.model.modules.RotationCoriolis included)")


# ================================================================
#  Pressure-solver knob validation (construction-time, CS-D2)
# ================================================================
def validate_pressure_iterations(iterations: int) -> int:
    """
    Validate the fixed-iteration PCG budget at construction.

    Description
    -----------
    :class:`~fridom.spatial.operators.krylov.ConjugateGradient`
    validates the same budget, but it is built at *stage* time, inside
    the jit trace — so a bad value surfaced as a
    ``TermEvaluationError`` wrapping the real message, and on a flat
    (spectral) grid, which iterates nothing, never surfaced at all.
    Checking on the core makes it a construction error on every route.

    Parameters
    ----------
    iterations : int
        The requested ``pressure_iterations``.

    Returns
    -------
    int
        The validated budget.

    Raises
    ------
    TypeError
        If ``iterations`` is not an int.
    ValueError
        If ``iterations`` is below 1.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError(
            "nh.Core pressure_iterations must be an int (the "
            f"fixed-iteration PCG budget), got {iterations!r}")
    if iterations < 1:
        raise ValueError(
            "nh.Core pressure_iterations must be >= 1 (the "
            f"fixed-iteration PCG budget), got {iterations}")
    return iterations


def validate_pressure_tolerance(
    tolerance: float | None,
) -> float | None:
    """
    Validate the PCG convergence break at construction.

    Description
    -----------
    The construction-time twin of :func:`validate_pressure_iterations`
    for the ``tolerance`` seam (same reason: the solver's own check
    runs inside the trace, or never on a flat grid).

    Parameters
    ----------
    tolerance : float | None
        The requested ``pressure_tolerance``.

    Returns
    -------
    float | None
        The validated tolerance (None = the fixed-count opt-out).

    Raises
    ------
    TypeError
        If ``tolerance`` is neither a real number nor None.
    ValueError
        If ``tolerance`` is not positive.
    """
    if tolerance is None:
        return None
    if isinstance(tolerance, bool) or not isinstance(
            tolerance, int | float):
        raise TypeError(
            "nh.Core pressure_tolerance must be a float, or None to "
            "run the fixed pressure_iterations count, got "
            f"{tolerance!r}")
    if tolerance <= 0:
        raise ValueError(
            "nh.Core pressure_tolerance must be > 0 (or None to run "
            f"the fixed pressure_iterations count), got {tolerance}")
    return float(tolerance)


# ================================================================
#  The FV C-grid family choice (FV-D3 / stage F3)
# ================================================================
def _fv_capable(grid: Grid) -> bool:  # noqa: ARG001
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

    Since the **ALE-on-FV closure** (2026-07-17) a dynamically driven
    mapping is FV-capable too: the ALE mesh-velocity correction
    (:class:`~fridom.model.modules.moving_geometry.MeshVelocityCorrection`)
    is now family-aware — the conservative flux form on the ``CellAvg``
    column factors, the advective form on the point-valued
    wall-normal velocity — so a moving-geometry model runs on FV
    wherever a static mapped one does. The old ``dynamic_geometry``
    carve-out (which kept moving geometry nodal-auto) is gone. So the
    predicate is now unconditionally True: every grid the model runs
    on is FV-capable. The one unserved *composition* — a grid
    declaring both a mapped column and an immersed domain — is a
    taught error in :func:`_require_fv_capable`, reached after the auto
    flip, not gated here.

    Parameters
    ----------
    grid : Grid
        The assembled grid (unused: capability no longer depends on the
        grid type; kept for the predicate's call surface).

    Returns
    -------
    bool
        True — every grid the FV C-grid auto default is served on.
    """
    return True


def _require_fv_capable(grid: Grid) -> None:
    """
    FV capability check for the resolved family (now unconditional).

    Description
    -----------
    Explicit ``family="fv"`` (or a ``Grid(family="fv")`` default) is
    served on periodic, walled (stage F4), mapped terrain-following
    (stage F5), immersed cut-cell (stage I2) *and* — since the
    mapped + immersed composition (plan stage M2, decisions MI-D2/D4) —
    a grid that declares **both** a mapped column and an immersed
    domain: the composed cut-cell metric PCG
    (:class:`~fridom.nonhydro2.modules.composed_pressure.ComposedPressureSolver`)
    serves it, so the taught error the composition previously raised is
    lifted. The FV C-grid now covers every geometry iteration 2
    supports; nothing is rejected here.

    Parameters
    ----------
    grid : Grid
        The assembled grid (unused: FV serves every grid).
    """


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


def resolve_model_family(family: str | None, grid: Grid) -> str:
    r"""
    Resolve a model-assembly family choice against the grid (F3).

    Description
    -----------
    The nonhydro model-assembly family (``nh.Model(family=...)`` /
    ``Core(family=...)``, scoping study §8): ``None`` is the
    **auto** default — it follows the grid's own ``default_family``,
    and *promotes* the ``"nodal"`` grid default to ``"fv"`` whenever
    the grid can carry the FV C-grid (:func:`_fv_capable`, now every
    grid). This is the flip: a plain periodic, walled, **mapped**
    (static *or* dynamically driven) or **immersed** nonhydro model is
    finite-volume by default — safe because the 2nd-order stencils are
    bit-identical to nodal on flat/walled grids (scoping study §1; the
    walled solve is eager-bitwise, ≤1.2e-14 jitted, §11), the mapped FV
    pressure operator is likewise bit-identical to nodal (the mapped
    column rides a uniform *computational* mesh, §13), the ALE
    mesh-velocity correction is family-aware (the ALE-on-FV closure,
    2026-07-17: conservative flux form on the ``CellAvg`` column
    factors), and on an immersed grid FV is the only *correct* choice
    (immersed physics is finite-volume, stage I2; the nodal path
    silently ignores the mask) — the owner ruling of 2026-07-17 (FV
    wherever capable). An **explicit** ``"fv"`` is served on periodic,
    walled, mapped terrain-following, immersed cut-cell *and* composed
    mapped + immersed grids (stages F4, F5, I2, M2) — nothing is
    rejected (:func:`_require_fv_capable`; the composition's taught
    error was lifted when the composed cut-cell metric PCG landed). An
    explicit ``"nodal"`` on an immersed grid *is* a taught error (the
    mask is FV-only, IP-D7).

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
        If the resolved family is ``"nodal"`` on an immersed grid.
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
    else:
        _require_nodal_capable(grid)
    return family


@partial(jaxify, dynamic=("aspect_ratio",))
class Core(fr.model.Module):

    r"""Declares u, v, w, p; owns the aspect ratio and the projection.

    Parameters
    ----------
    aspect_ratio : float | fr.model.Ramp, optional
        The aspect ratio :math:`\delta = H/L` (default: 1.0);
        published as ``nonhydro.aspect_ratio`` and **squared at the
        use sites** (the projection's vertical weight
        :math:`1/\delta^2`, the coupling terms' ``delta**2``). May
        be an ``fr.model.Ramp`` for a time-dependent aspect ratio;
        must be nonzero (the projection divides by its square).
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
        The PCG iteration budget of the fixed-iteration pressure
        solve (CS-D2); consumed on a grid whose coordinate mapping
        declares a mapped column *and* on an immersed (cut-cell) grid
        — both run the fixed-iteration PCG. Under the default
        ``pressure_tolerance`` this is the **maximum** budget, not the
        work actually done: the convergence break stops the recurrence
        earlier and the remaining scanned steps are branch-skipped
        no-ops. The flat spectral solve is exact and iterates nothing
        (default: 30).
    pressure_tolerance : float | None, optional
        The PCG convergence break forwarded to the mapped and
        immersed pressure solvers (the measure-weighted true relative
        residual; masked scan, exact gradient — see
        :class:`ConjugateGradient`). The default ``1e-8`` makes
        ``pressure_iterations`` the maximum budget and sits well above
        the residual floor (~1e-14); ``None`` is the opt-out that runs
        the fixed count (default: 1e-8).

        The break is **real and it fires**: the scan keeps its static
        length, but every step past convergence is skipped through a
        ``lax.cond`` that survives XLA optimization as a genuine
        branch. Measured achieved counts on immersed routes are 12-25
        against the default budget of 30, so the *typical* solve is
        well inside the budget and raising ``pressure_iterations``
        alone does not refine the answer — it only raises the ceiling.
        Turn on ``pressure_report`` to see the achieved count instead
        of guessing it.
    pressure_report : bool, optional
        Emit a host-side convergence report — the achieved PCG
        iteration count, the budget, and the relative residual — once
        per pressure solve, through ``jax.debug.print``. This is the
        evidence for sizing ``pressure_iterations``: a report reading
        ``k=13/30`` says the budget is generous, and one reading
        ``k=30/30`` with a relative residual above ``pressure_tolerance``
        says the projection ran out of budget and left the velocity
        measurably divergent (which is otherwise entirely silent). Off
        by default, and with it off the compiled step is byte-identical
        — the flag is static (a treedef aux). Consumed on the mapped /
        stretched / immersed / composed routes; the flat spectral solve
        is exact and iterates nothing, so it reports nothing
        (default: False).
    pressure_preconditioner : str | None, optional
        The PCG preconditioner of the fixed-iteration pressure solve
        (B4): ``"spectral"`` (the flat separable spectral inverse),
        ``"multigrid"`` (the geometric-multigrid V-cycle) or ``"none"``.
        ``None`` (the default) is **auto**: a uniform mapped or immersed
        grid resolves to ``"spectral"`` (byte-identical to the previous
        explicit default), a composed mapped + immersed grid resolves to
        ``"multigrid"`` (MI-D3: the masked spectral fold does not
        converge in the default budget on a genuine cut chart, the
        multigrid V-cycle does), and so does a **stretched** grid (one
        carrying a ``MappedIntervalMesh`` factor): the separable
        spectral inverse has no transform to build on a coordinate-
        mapped mesh — it is rejected at construction, N1 — while the
        V-cycle builds its vertical bands from the same ``grid.measure``
        widths and serves the stretched column (N3). An explicit string
        is honoured on every route unchanged. Consumed on a mapped /
        stretched / immersed / composed grid; the flat spectral solve is
        exact and ignores it. Static (a treedef aux, part of the module
        fingerprint), like ``single_precision_solve`` (default: None).
    multigrid_levels : int | None, optional
        The multigrid depth when ``pressure_preconditioner="multigrid"``;
        ignored otherwise. ``None`` (the default) coarsens to the
        four-cell horizontal floor (floor-limited depth, h-independent
        iteration counts at every size); an ``int`` is a maximum cap as
        before (the builder floors on small grids either way). Static in
        the fingerprint (default: None).
    multigrid_tridiagonal_method : str, optional
        The vertical-line tridiagonal kernel of the multigrid smoother
        (``"auto"`` / ``"cusparse"`` / ``"pcr"`` / ``"scan"``),
        forwarded to the mapped and immersed solvers and validated at
        construction; consumed only for
        ``pressure_preconditioner="multigrid"``. Static in the
        fingerprint (default: ``"auto"``).
    multigrid_coarsen_vertical : bool, optional
        Whether the mapped multigrid V-cycle coarsens the vertical
        column too (full 3-D coarsening), forwarded to the
        :class:`MappedPressureSolver` (only — the immersed solver keeps
        semicoarsening, out of GM-D9's scope). ``True`` — the
        owner-ratified default (GM-D9, 2026-07-18) — coarsens the
        vertical alongside the horizontals wherever the vertical mesh
        supports it (identical 10-iteration convergence, -6..-11% per CG
        iteration at 128/256/512^3 on the GB-2 mapped protocol),
        degrading to horizontal semicoarsening automatically where it
        cannot (a Chebyshev vertical, an indivisible ``n_z``); ``False``
        restores pure semicoarsening. Consumed only for a mapped grid
        with ``pressure_preconditioner="multigrid"``. Static in the
        fingerprint (default: True).
    multigrid_agglomerate : int | None, optional
        The coarse-grid agglomeration threshold ``tau`` in planes
        (MG-D10), forwarded to the mapped and immersed solvers. From
        the first coarse level whose shortest would-be per-shard extent
        falls below ``tau`` (and that is small enough to replicate),
        that level and every level below it are built fully replicated,
        so the redundant coarse compute runs collective-free instead of
        paying a ring halo exchange to shard one or two planes. ``None``
        (the default) disables agglomeration; a no-op on one device.
        Consumed only for ``pressure_preconditioner="multigrid"``.
        Static in the fingerprint (default: None).
    family : str | None, optional
        The discretization family of the whole core state (FV-D3,
        stage F3): ``"fv"`` declares ``u, v, w, p`` on the
        finite-volume C-grid (scalars on ``CellAvg``, velocities on
        the point-value faces — FV-D2 option A) and seeds the FV
        C-grid ``diff`` profile so the pressure chain staggers on the
        average family; ``"nodal"`` is the point-value C-grid. ``None``
        defers to the grid-level default (``grid.default_family``), so
        an explicitly assembled core follows the grid. The
        ``nh.Model`` factory resolves the flip (every grid promotes
        ``None`` to ``"fv"`` — periodic, walled, mapped static or
        dynamically driven, immersed, and composed mapped + immersed);
        an explicit ``"nodal"`` on an immersed grid is a taught error
        (default: None).
    """

    state_type = State
    diagnostics = DIAGNOSTICS

    def __init__(
        self,
        *,
        aspect_ratio: float | fr.model.Ramp = 1.0,
        vertical: str = "z",
        coords: tuple[str, ...] = ("x", "y", "z"),
        single_precision_solve: bool = False,
        pressure_iterations: int = 30,
        pressure_tolerance: float | None = 1e-8,
        pressure_report: bool = False,
        pressure_preconditioner: str | None = None,
        multigrid_levels: int | None = None,
        multigrid_tridiagonal_method: str = "auto",
        multigrid_coarsen_vertical: bool = True,
        multigrid_agglomerate: int | None = None,
        family: str | None = None,
    ) -> None:
        """Store the core parameter leaves and the geometry names.

        Raises
        ------
        ValueError
            On an unknown ``family``, a ``pressure_iterations`` below
            1, or a non-positive ``pressure_tolerance``.
        TypeError
            On ``aspect_ratio=0`` (the projection's vertical weight
            divides by its square, so an exact zero poisons the run
            far from here), or a non-int ``pressure_iterations`` /
            non-real ``pressure_tolerance``.
        """
        if family is not None and family not in FAMILIES:
            raise ValueError(
                f"family must be one of {FAMILIES} or None, got "
                f"{family!r}")
        if (isinstance(aspect_ratio, numbers.Number)
                and float(aspect_ratio) == 0.0):
            raise TypeError(
                "nh.Core aspect_ratio=0 is refused: the pressure "
                "projection divides by its square, so an exact zero "
                "poisons the run (and its VJP) far from here; pass a "
                "nonzero aspect ratio")
        self.aspect_ratio = fr.model.leaf(aspect_ratio)
        self._vertical = vertical
        self._coords = coords
        self._single_precision_solve = bool(single_precision_solve)
        self._pressure_iterations = validate_pressure_iterations(
            pressure_iterations)
        self._pressure_tolerance = validate_pressure_tolerance(
            pressure_tolerance)
        self._pressure_report = bool(pressure_report)
        self._pressure_preconditioner = pressure_preconditioner
        self._multigrid_levels = multigrid_levels
        self._multigrid_tridiagonal_method = validate_tridiagonal_method(
            multigrid_tridiagonal_method)
        self._multigrid_coarsen_vertical = bool(multigrid_coarsen_vertical)
        self._multigrid_agglomerate = validate_agglomerate(
            multigrid_agglomerate)
        self._family = family
        # the projection's derived halo substitute (V-N2), computed
        # once at bind from the C-grid ``div`` / ``grad`` rows the stage
        # applies; None until bound (assembly reads it only post-bind).
        self._extra_halo: HaloSpec | None = None

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
    #  Bind (step 4): derive the projection's halo substitute
    # ================================================================
    def bind(self, table: object) -> None:
        r"""Derive the pressure projection's ``extra_halo`` (V-N2).

        Description
        -----------
        The projection is a whole-domain spectral / CG solve wrapping
        raw arrays the halo trace cannot follow, so the module declares
        its own ghost width. That width is not a literal: it is the
        two-sided reach of the very C-grid ``div`` / ``grad`` rows the
        stage applies (:meth:`_project`), composed across the global
        transform **barrier** by per-side max (never sum). ``div`` and
        ``grad`` are order-2 staggered differences (reach 1) on opposite
        sides of the transform, so the derived width is ``max(1, 1) =
        1`` — half the old hardcoded 2 (``pressure_solver_halo.md`` §3,
        §5). A registry ``diff`` override moves the value automatically,
        in both the declaration and the discrete eigenvalue (one source
        of truth). Runs once at bind (the merged registry is visible);
        the value is read at assembly steps 5 / 7.

        Bind is also where the model's **geometry refusals** land: it
        is the first hook that sees the assembled grid together with
        the resolved field spaces. A chart-coupled grid is refused
        outright (:func:`_no_chart_grid`) *before* the coordinate loop
        below, which is what a chart grid used to die in — a bare
        ``KeyError`` out of ``p.factor("x")`` on the default
        ``coords=("x", "y", "z")``, and the wrong advice from
        :func:`_no_staggered_face` (rename the vertical) on
        ``coords=("lon", "lat", "z")``. Deciding chart-ness first
        leaves :func:`_no_staggered_face` the flat-grid
        vertical-naming case it was written for.

        Raises
        ------
        NotImplementedError
            If the grid carries an embedding chart — the
            nonhydrostatic model is Cartesian-only.
        ValueError
            If a name in ``coords`` carries no staggered velocity face
            (:func:`_no_staggered_face`).
        """
        grid = table.grid  # type: ignore[attr-defined]
        chart = getattr(grid, "chart_coords", None)
        if chart is not None:
            raise NotImplementedError(_no_chart_grid(chart))
        registry = grid.dispatch
        p = table["p"].space  # type: ignore[index]
        vel = tuple(table[name].space  # type: ignore[index]
                    for name in ("u", "v", "w"))
        div_leg: dict[str, list[tuple[str, object]]] = {}
        grad_leg: dict[str, list[tuple[str, object]]] = {}
        for axis in self._coords:
            centre = p.factor(axis)
            # ``grad`` differences the pressure centre onto the face;
            # ``div`` differences the face-normal velocity back — the
            # two legs the transform separates.
            grad_leg[axis] = [("diff", centre)]
            face = next((vs.factor(axis) for vs in vel
                         if vs.factor(axis) is not centre), None)
            if face is None:
                raise ValueError(_no_staggered_face(axis, self._coords))
            div_leg[axis] = [("diff", face)]
        self._extra_halo = derive_extra_halo(
            registry, self._coords, [div_leg, grad_leg])

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
        Every grid is FV-capable (:func:`_require_fv_capable`, kept as
        the capability seam), walled, mapped, immersed and composed
        mapped + immersed alike (stages F4, F5, I2, M2).

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
    #  Parameters -- the aspect ratio lives on the core
    # ================================================================
    parameter_declarations = (
        fr.model.ParameterDeclaration(
            ASPECT_RATIO, attr="aspect_ratio", units="1",
            doc="aspect ratio H/L (squared at the use sites)"),
    )

    @property
    def unit_factors(self) -> dict[str, fr.model.UnitFactor]:
        """Dimensional-factor rows (``model.units``, §D).

        The nonhydrostatic amplitude table
        (:mod:`fridom.nonhydro2.units`) plus the coordinate rows and
        the diagnosed quantities' rows — an instance property
        because ``coords=`` / ``vertical=`` rename the coordinate
        keys (the vertical row is ``delta*L``).
        """
        return {**coordinate_factors(self._coords, self._vertical),
                **COMPONENT_FACTORS, **DERIVED_FACTORS}

    @property
    def family(self) -> str | None:
        """The requested discretization family (None = grid default).

        The ``nh.Model`` preset reads this to resolve the auto flip
        against the grid (``resolve_model_family``) before assembly.
        """
        return self._family

    def time_dependent_linear_parameters(self) -> tuple[str, ...]:
        """Report a ramped aspect ratio feeding the frozen linear operator.

        The aspect ratio enters ``L`` through the pressure projection,
        which is a CONSTRAINT stage (S4) rather than a ``linear=True``
        term, so the structural term sweep (TDF-D4) cannot see it: a
        model assembled without stratification would otherwise slip a
        ramped aspect ratio past a frozen-``L`` (exponential) stepper
        silently. The core owns the leaf, so it reports it here
        directly (and closes the cross-module hole where a
        stratification term consumes but does not own it).
        """
        names = list(super().time_dependent_linear_parameters())
        if isinstance(self.aspect_ratio, fr.model.TimeDependent):
            names.append(str(ASPECT_RATIO))
        return tuple(names)

    # ================================================================
    #  The pressure-projection CONSTRAINT stage (S4)
    # ================================================================
    @property
    def extra_halo(self) -> HaloSpec | None:
        """Exempt the (global, spectral) projection from the halo trace.

        Description
        -----------
        The projection is a whole-domain spectral solve wrapping raw
        arrays (``Fourier``/``.data``); it declares its FD-stencil halo
        here (V-N2) rather than being traced. The value is **derived**
        at :meth:`bind` from the ``div`` / ``grad`` rows the stage
        applies (per-side max across the transform barrier — width 1 on
        the default C-grid), not a literal. ``None`` before bind
        (assembly reads it only post-bind).
        """
        return self._extra_halo

    @property
    def stages(self) -> tuple[fr.model.Stage, ...]:
        """The velocity projection: replace u, v, w and write p."""
        return (
            fr.model.Stage(kind=fr.model.StageKind.CONSTRAINT, fn="_project",
                     name="projection"),
        )

    def _resolved_preconditioner(
        self, *, composed: bool, stretched: bool = False,
    ) -> str:
        """Resolve the ``None`` = auto PCG preconditioner per route.

        Description
        -----------
        An explicit ``pressure_preconditioner`` string is honoured
        unchanged on every route. ``None`` (the default) is auto: a
        composed mapped + immersed grid resolves to ``"multigrid"`` (the
        MI-D3 ratified default, the masked spectral fold not converging
        in the default budget on a genuine cut chart), a **stretched**
        column likewise (N1/N3: the separable spectral inverse has no
        transform to build on a ``MappedIntervalMesh``, and rejects the
        column at construction; the V-cycle builds its vertical bands
        from the same ``grid.measure`` widths and serves it), every
        other route (uniform mapped, immersed) to ``"spectral"`` —
        byte-identical to the previous explicit default.

        Parameters
        ----------
        composed : bool
            Whether the grid declares both a mapped column and an
            immersed domain (the composed route).
        stretched : bool, optional
            Whether the grid carries a stretched (``MappedIntervalMesh``)
            mesh factor (default: False).

        Returns
        -------
        str
            The concrete preconditioner name for the solver.
        """
        if self._pressure_preconditioner is not None:
            return self._pressure_preconditioner
        return "multigrid" if composed or stretched else "spectral"

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
        I2, which serves a stretched vertical itself); on a grid that
        declares **both** (mapped column *and* immersed domain) it
        routes to :meth:`_project_composed` (the composed cut-cell
        metric solve, stage M2); on a grid that declares no analytic
        map but carries a **stretched** ``MappedIntervalMesh`` factor
        (:func:`_stretched_column`) it likewise routes to
        :meth:`_project_mapped` — the degenerate identity column, which
        the spectral solve below cannot serve (a coordinate-mapped mesh
        carries no spectral basis, so its transform row is missing).
        A flat/unmapped/unimmersed grid takes exactly the code path
        below (zero behavior change).
        """
        grid = state["u"].grid
        mapping = getattr(grid, "mapping", None)
        mapped_column = bool(
            mapping is not None and mapping.column_corrections)
        immersed = getattr(grid, "immersed", None) is not None
        if mapped_column and immersed:
            return self._project_composed(state, ctx)
        if mapped_column:
            return self._project_mapped(state, ctx)
        if immersed:
            # the masked solve serves a stretched vertical itself
            # (its bands read the physical widths, plan §6)
            return self._project_immersed(state, ctx)
        if _stretched_column(grid):
            # a bare stretched mesh factor declares no column
            # correction, but it is a mapped column all the same (the
            # degenerate identity map) and the spectral solve has no
            # transform to build on it
            return self._project_mapped(state, ctx)
        delta = ctx.params[ASPECT_RATIO]
        dsqr = delta * delta
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

        Warm start (Phase E): the PCG is seeded with the previous
        step's potential ``x0 = state["p"] * ctx.stage_dt``. The
        stored diagnostic is ``p = phi / stage_dt``, so multiplying by
        the CURRENT ``stage_dt`` reconstructs the previous solve
        variable ``phi`` at this stage's increment; the RHS-relative
        stopping test then saves the iterations a good guess makes
        unnecessary. The first step's zero-initialized ``p`` seeds a
        zero guess, and the mean gauge is enforced start-independently
        inside the solve, so the result is unchanged.
        """
        delta = ctx.params[ASPECT_RATIO]
        dsqr = delta * delta
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
            report=self._pressure_report,
            single_precision=self._single_precision_solve,
            preconditioner=self._resolved_preconditioner(
                composed=False, stretched=_stretched_column(grid)),
            multigrid_levels=self._multigrid_levels,
            multigrid_tridiagonal_method=(
                self._multigrid_tridiagonal_method),
            multigrid_coarsen_vertical=self._multigrid_coarsen_vertical,
            multigrid_agglomerate=self._multigrid_agglomerate,
            params=mapping_params(state, grid))
        # one metric derivation for the whole projection: divergence,
        # solve and correction share the solver's per-solve memo (it
        # dies with the call, so the next step re-derives at the new
        # geometry — MappedPressureSolver.project)
        p, corr = solver.project(vel, x0=state["p"] * ctx.stage_dt)
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

        On a **stretched** grid (a ``MappedIntervalMesh`` factor, A1)
        the masked operator itself is already stretch-aware (its
        analytic diagonal and vertical bands read the physical
        ``grid.measure`` widths, plan §6), but the wet-masked
        *spectral* preconditioner is not: the stretched factor carries
        no spectral basis, so the auto preconditioner resolves to the
        V-cycle and an explicit ``"spectral"`` is a taught error
        (:func:`_no_spectral_on_stretched`) rather than the bare
        ``DispatchError`` it used to be.

        Warm start (Phase E): the PCG is seeded with the previous
        step's potential ``x0 = state["p"] * ctx.stage_dt`` (the
        stored ``p = phi / stage_dt`` rescaled back to this stage's
        increment); the wet-mean gauge is enforced start-independently
        inside the solve, so a good guess only saves iterations. The
        first step's zero ``p`` seeds a zero guess (unchanged).
        """
        delta = ctx.params[ASPECT_RATIO]
        dsqr = delta * delta
        grid = state["u"].grid
        vel = {
            "x": state["u"],
            "y": state["v"],
            self._vertical: state["w"],
        }
        stretched = _stretched_column(grid)
        preconditioner = self._resolved_preconditioner(
            composed=False, stretched=stretched)
        if stretched and preconditioner == "spectral":
            # only an explicit pressure_preconditioner="spectral"
            # reaches here; the auto default already picks the V-cycle
            raise NotImplementedError(_no_spectral_on_stretched(
                "masked (cut-cell)"))
        solver = ImmersedPressureSolver(
            grid,
            state["p"].function_space,
            vertical=self._vertical,
            dsqr=dsqr,
            iterations=self._pressure_iterations,
            tolerance=self._pressure_tolerance,
            report=self._pressure_report,
            single_precision=self._single_precision_solve,
            preconditioner=preconditioner,
            multigrid_levels=self._multigrid_levels,
            multigrid_tridiagonal_method=(
                self._multigrid_tridiagonal_method),
            multigrid_agglomerate=self._multigrid_agglomerate)
        p, corr = solver.project(vel, x0=state["p"] * ctx.stage_dt)
        return {
            "u": state["u"] - corr["x"].retag(state["u"]),
            "v": state["v"] - corr["y"].retag(state["v"]),
            "w": state["w"] - corr[self._vertical].retag(state["w"]),
            "p": p / ctx.stage_dt,
        }

    def _project_composed(
        self, state: State, ctx: StepContext,
    ) -> dict[str, object]:
        r"""Project on a composed mapped + immersed grid (stage M2).

        Description
        -----------
        The composed twin of :meth:`_project_mapped` and
        :meth:`_project_immersed`: the divergence, the elliptic
        operator, and the gradient subtraction all come from one
        :class:`~fridom.nonhydro2.modules.composed_pressure.ComposedPressureSolver`
        — the mapped J-weighted metric operator with the immersed
        open-area fractions inserted as diagonal face / corner weights
        (MI-D2), solved by fixed-iteration PCG (the fraction-weighted
        multigrid V-cycle or the wet-masked spectral inverse
        preconditions, projected onto the wet-region-constant
        nullspace), and the boolean-masked, flux-consistent velocity
        corrections, so the projection removes exactly the masked
        divergence the operator measures (to the CG residual). The
        stored diagnostic keeps the ``p = phi / ctx.stage_dt``
        normalization and is masked to zero under the topography; the
        vertical weight ``1/dsqr`` rides the solver's ``weights`` seam.

        Dynamic geometry threads the CURRENT mapping-parameter fields
        through ``params=`` (as :meth:`_project_mapped`); the immersed
        wet region is static (plan §6). Warm start (Phase E): the PCG
        is seeded with the previous step's potential
        ``x0 = state["p"] * ctx.stage_dt``.
        """
        delta = ctx.params[ASPECT_RATIO]
        dsqr = delta * delta
        grid = state["u"].grid
        vel = {
            "x": state["u"],
            "y": state["v"],
            self._vertical: state["w"],
        }
        solver = ComposedPressureSolver(
            grid,
            state["p"].function_space,
            weights={self._vertical: 1.0 / dsqr},
            iterations=self._pressure_iterations,
            tolerance=self._pressure_tolerance,
            report=self._pressure_report,
            single_precision=self._single_precision_solve,
            preconditioner=self._resolved_preconditioner(composed=True),
            multigrid_levels=self._multigrid_levels,
            multigrid_tridiagonal_method=(
                self._multigrid_tridiagonal_method),
            multigrid_coarsen_vertical=self._multigrid_coarsen_vertical,
            params=mapping_params(state, grid))
        p, corr = solver.project(vel, x0=state["p"] * ctx.stage_dt)
        return {
            "u": state["u"] - corr["x"].retag(state["u"]),
            "v": state["v"] - corr["y"].retag(state["v"]),
            "w": state["w"] - corr[self._vertical].retag(state["w"]),
            "p": p / ctx.stage_dt,
        }
