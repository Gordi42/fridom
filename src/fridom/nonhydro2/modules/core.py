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
from fridom.nonhydro2.params import DSQR, ROSSBY
from fridom.nonhydro2.state import State
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
    ``DynamicalCore(family=...)``, scoping study §8): ``None`` is the
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
    walled, mapped terrain-following *and* immersed cut-cell grids
    (stages F4, F5, I2); only a grid that declares **both** a mapped
    column and an immersed domain rejects it (the mapped and masked
    PCGs are not yet composed, plan §6). An explicit ``"nodal"`` on an
    immersed grid is likewise a taught error (the mask is FV-only,
    IP-D7).

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
        The PCG convergence break forwarded to the mapped and
        immersed pressure solvers (the measure-weighted true relative
        residual; masked scan, exact gradient — see
        :class:`ConjugateGradient`). The default ``1e-8`` makes
        ``pressure_iterations`` the maximum budget and sits well above
        the residual floor (~1e-14); ``None`` is the opt-out that runs
        the fixed count (default: 1e-8).
    pressure_preconditioner : str | None, optional
        The PCG preconditioner of the fixed-iteration pressure solve
        (B4): ``"spectral"`` (the flat separable spectral inverse),
        ``"multigrid"`` (the geometric-multigrid V-cycle) or ``"none"``.
        ``None`` (the default) is **auto**: a mapped or immersed grid
        resolves to ``"spectral"`` (byte-identical to the previous
        explicit default), a composed mapped + immersed grid resolves to
        ``"multigrid"`` (MI-D3: the masked spectral fold does not
        converge in the default budget on a genuine cut chart, the
        multigrid V-cycle does). An explicit string is honoured on every
        route unchanged. Consumed on a mapped / immersed / composed grid;
        the flat spectral solve is exact and ignores it. Static (a
        treedef aux, part of the module fingerprint), like
        ``single_precision_solve`` (default: None).
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
        dynamically driven, and immersed); an explicit ``"fv"`` on a
        grid with both a mapped column and an immersed domain, or an
        explicit ``"nodal"`` on an immersed grid, is a taught error
        (default: None).
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
        pressure_tolerance: float | None = 1e-8,
        pressure_preconditioner: str | None = None,
        multigrid_levels: int | None = None,
        multigrid_tridiagonal_method: str = "auto",
        multigrid_coarsen_vertical: bool = True,
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
        self._multigrid_tridiagonal_method = validate_tridiagonal_method(
            multigrid_tridiagonal_method)
        self._multigrid_coarsen_vertical = bool(multigrid_coarsen_vertical)
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
        """
        grid = table.grid  # type: ignore[attr-defined]
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
            face = next(vs.factor(axis) for vs in vel
                        if vs.factor(axis) is not centre)
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

    def time_dependent_linear_parameters(self) -> tuple[str, ...]:
        """Report a ramped ``dsqr`` feeding the frozen linear operator.

        ``dsqr`` enters ``L`` through the pressure projection, which is
        a CONSTRAINT stage (S4) rather than a ``linear=True`` term, so
        the structural term sweep (TDF-D4) cannot see it: a model
        assembled without stratification would otherwise slip a ramped
        ``dsqr`` past a frozen-``L`` (exponential) stepper silently. The
        core owns the leaf, so it reports it here directly (and closes
        the cross-module hole where a stratification term consumes but
        does not own ``dsqr``).
        """
        names = list(super().time_dependent_linear_parameters())
        if isinstance(self.dsqr, fr.model.TimeDependent):
            names.append(str(DSQR))
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

    def _resolved_preconditioner(self, *, composed: bool) -> str:
        """Resolve the ``None`` = auto PCG preconditioner per route.

        Description
        -----------
        An explicit ``pressure_preconditioner`` string is honoured
        unchanged on every route. ``None`` (the default) is auto: a
        composed mapped + immersed grid resolves to ``"multigrid"`` (the
        MI-D3 ratified default, the masked spectral fold not converging
        in the default budget on a genuine cut chart), every other route
        (mapped, immersed) to ``"spectral"`` — byte-identical to the
        previous explicit default.

        Parameters
        ----------
        composed : bool
            Whether the grid declares both a mapped column and an
            immersed domain (the composed route).

        Returns
        -------
        str
            The concrete preconditioner name for the solver.
        """
        if self._pressure_preconditioner is not None:
            return self._pressure_preconditioner
        return "multigrid" if composed else "spectral"

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
        I2); on a grid that declares **both** (mapped column *and*
        immersed domain) it routes to :meth:`_project_composed` (the
        composed cut-cell metric solve, stage M2); a
        flat/unmapped/unimmersed grid takes exactly the code path below
        (zero behavior change).
        """
        grid = state["u"].grid
        mapping = getattr(grid, "mapping", None)
        mapped_column = mapping is not None and mapping.column_corrections
        immersed = getattr(grid, "immersed", None) is not None
        if mapped_column and immersed:
            return self._project_composed(state, ctx)
        if mapped_column:
            return self._project_mapped(state, ctx)
        if immersed:
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
            preconditioner=self._resolved_preconditioner(composed=False),
            multigrid_levels=self._multigrid_levels,
            multigrid_tridiagonal_method=(
                self._multigrid_tridiagonal_method),
            multigrid_coarsen_vertical=self._multigrid_coarsen_vertical,
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

        Warm start (Phase E): the PCG is seeded with the previous
        step's potential ``x0 = state["p"] * ctx.stage_dt`` (the
        stored ``p = phi / stage_dt`` rescaled back to this stage's
        increment); the wet-mean gauge is enforced start-independently
        inside the solve, so a good guess only saves iterations. The
        first step's zero ``p`` seeds a zero guess (unchanged).
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
            preconditioner=self._resolved_preconditioner(composed=False),
            multigrid_levels=self._multigrid_levels,
            multigrid_tridiagonal_method=(
                self._multigrid_tridiagonal_method))
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
        dsqr = ctx.params[DSQR]
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
