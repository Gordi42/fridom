r"""The hydrostatic dynamical core module.

Description
-----------
``Core`` is the package's dynamical-core module (D1.3): it declares
the horizontal velocities ``u, v`` (Velocity + ADVECTED, on the
C-grid faces), the **diagnosed** vertical velocity ``w`` and the
**diagnosed** hydrostatic pressure ``p_hyd``; it owns the two
pre-tendency **DIAGNOSE** stages (S1'); and it contributes the
single linear pressure-gradient term. It supplies the ``hy.State``
vocabulary class through ``state_type``.

**Gravity-first (the nondimensionalization refactor):** the physical
constant of the hydrostatic model is the gravitational acceleration,
centralized on the core — ``hy.Core(gravity=...)`` provides
``hydrostatic.gravity``, which the free-surface family (and any
future EOS module) references. The core's own terms and stages are
scale-free (they read no physics constant), so the two variants are:

- **dimensional** (``gravity=``): provides ``hydrostatic.gravity``;
- **nondimensional** (no kwarg): provides nothing — the free-surface
  module carries the external Froude number instead, under a
  nondimensional ``fr.scaling`` policy.

The retired ``hydrostatic.csqr`` / ``scaling.nonlinearity`` provides
are gone: there is no reference-depth parameter in the step path
(every column depth is genuine geometry) and the nonlinearity number
lives on the scaling mechanism's own module.

The two DIAGNOSE stages (recomputed from the current state every
substage, so a restart / ``set_state`` sees them fresh before any
term reads — the S1' placement of ``03_time_stepping.md`` §5.2):

.. math::

    w(z)      = -\int_{-H}^{z} (\partial_x u + \partial_y v)\, dz', \\
    p_{hyd}(z) = -\int_{z}^{0} b \, dz'.

The ``b`` field is **optional**: a buoyancy module
(``hy.ConstantStratification`` / ``hy.BuoyancyTracer``) declares it and
the ``p_hyd`` DIAGNOSE integrates it. With no buoyancy module the core
carries no ``b``, diagnoses ``p_hyd = 0``, and the baroclinic gradient
``-grad p_hyd`` vanishes — a **constant-density**, barotropic flow
driven by the surface pressure alone.

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

The stored ``w`` is the **physical** vertical velocity on every grid
(``physical_state_components.md`` ruling (b)): the equation above is
the flat/stretched case, where the continuity ``w`` already *is*
physical. On a terrain (sigma) column :meth:`_diagnose_w` adds the
slope terms ``u Z_x + v Z_y`` on top of the FTC-exact contravariant
flux ``J\omega``, so the stored ``w`` is nonzero at the bed over a
slope; the flux ``J\omega`` (the FTC-exact, zero-at-the-terrain
working quantity) is then the read-only ``State.chart["w"]``.

**Embedding charts (spherical-models plan S2).** On an orthogonal
two-coordinate chart extruded along a flat vertical — the thin-shell
``(lon, lat, z)`` sphere of ``fr.spatial.spherical.Grid(...,
vertical=)``, or the torus — the stored ``u`` / ``v`` stay the
**physical** components along the chart coordinates (pass
``horizontal=("lon", "lat")``), the continuity DIAGNOSE becomes the
area-weighted metric divergence
``(1/sqrt_g)[d_1(h_2 u) + d_2(h_1 v)]`` (``h_i = sqrt(g_ii)``), and the
pressure gradient the physical gradient ``d_i p / h_i``
(:mod:`fridom.model.chart_seams`). ``sqrt_g`` is independent of the
vertical (shallow atmosphere), so both vertical integrals are the flat
ones. On the identity chart every factor is exactly 1.0 and the arm
reduces to the flat core bitwise. Chart + ``maps=`` terrain and the
finite-volume family on a chart are taught refusals.

The linear pressure-gradient term reads the **baroclinic** pressure
``p_hyd`` only:

.. math::

    \partial_t u = -\partial_x p_{hyd}, \qquad
    \partial_t v = -\partial_y p_{hyd}.

**The discretization family (FV-D3, stage F3).** ``hy.Core(family=)``
chooses between the nodal point-value C-grid (the default) and the
**finite-volume** (cell-average) one: under ``family="fv"`` the cell
scalars ``p_hyd`` / ``b`` land on ``CellAvg`` and the velocities keep
their point value along their own axis while going cell-average
transversely (``u`` on ``Right(x) ⊗ CellAvg(y) ⊗ CellAvg(z)``, FV-D2
option A); the diagnosed ``w`` is the transversely averaged
``Outer(z)`` face. The core then contributes the FV C-grid ``diff``
profile (:func:`fv_cgrid_overrides`) through the grid-aware dispatch
hook, which re-points ``("diff", CellAvg) -> FaceDifference`` and
``("diff", face) -> FluxDifference`` per mesh factor — so every
staggered difference of the package (the continuity divergence, both
pressure gradients, the transport divergences of the free-surface
family and ``ZStarGeometry``, the barotropic ``Div @ Diag @ Grad``
chain) becomes the exact discrete Gauss / face-difference row without
a single family branch in the physics. On a flat periodic box the two
families' 2nd-order stencils are bit-identical, so an FV model is
bitwise the nodal one there.

There is **no auto flip** (:func:`resolve_model_family`): ``None``
follows ``grid.default_family``, so every pre-existing assembly stays
nodal. The ``hy.Model`` factory adopts the resolved family as the
grid default, which is what makes ``b``, ``ps``, the split-explicit
``U`` / ``V`` and a z* ``eta`` follow the core. ``family="fv"`` on an
immersed (cut-cell) grid is a taught refusal — the hydrostatic
cut-cell path is nodal with explicit fractions.

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

import numbers
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.hydrostatic.diagnostics import DIAGNOSTICS
from fridom.hydrostatic.modules.terrain import (
    discover_column,
    jacobian_name,
    masked_w_faces,
    require_chart_immersed_order,
    slope_velocity_on_w,
)
from fridom.hydrostatic.params import GRAVITY
from fridom.hydrostatic.state import State
from fridom.hydrostatic.units import (
    COMPONENT_FACTORS,
    DERIVED_FACTORS,
    buoyancy_factor,
    coordinate_factors,
    vertical_extent,
    vertical_velocity_factor,
)
from fridom.model.chart_seams import (
    chart_gradient,
    edge_scale,
    sealed_metric_divide,
    thin_shell_chart,
    volume_scale,
)
from fridom.model.halo_demand import derive_extra_halo
from fridom.model.modules.moving_geometry import mapping_params
from fridom.model.roles import Velocity
from fridom.spatial.bc import BC
from fridom.spatial.fields.scalar_field import _bc_siblings
from fridom.spatial.operators.cumulative import CumulativeIntegral
from fridom.spatial.operators.flux_diff import (
    FaceDifference,
    FluxDifference,
)
from fridom.spatial.space_patterns import FAMILIES
from fridom.spatial.spaces.average import AverageSpace, CellAvg
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
from fridom.spatial.spaces.tensor_product import TensorProductSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.model.context import StepContext
    from fridom.spatial.decomposition.halo import HaloSpec
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid
    from fridom.spatial.operators.base import Operator
    from fridom.spatial.operators.registry import DispatchKey
    from fridom.spatial.spaces.function_space import FunctionSpace


# ================================================================
#  The discretization family (FV-D3 / stage F3, hydrostatic)
# ================================================================
def effective_family(family: str | None, grid: Grid) -> str:
    """Return the family a declaration resolves into on ``grid``.

    Description
    -----------
    The pattern-resolution rule of
    :meth:`~fridom.spatial.space_patterns.SpacePattern.resolve`,
    lifted so the module-side spellings (the ``w`` space rule, the
    dispatch profile) agree with the declarations exactly: an explicit
    ``family`` wins, ``None`` defers to ``grid.default_family``.

    Parameters
    ----------
    family : str | None
        The requested family (None = defer to the grid).
    grid : Grid
        The assembled grid.

    Returns
    -------
    str
        ``"nodal"`` or ``"fv"``.
    """
    if family is not None:
        return family
    return getattr(grid, "default_family", "nodal")


def _require_fv_capable(grid: Grid) -> None:
    """Refuse the FV family on an immersed (cut-cell) grid.

    Description
    -----------
    The hydrostatic immersed path is **nodal with explicit fractions**:
    the masked continuity, the wet barotropic reductions and the
    open-face gates all multiply concrete ``alpha`` fields onto nodal
    point values (IP-D9). None of that machinery is written against the
    average family — the min-rule face fractions live on the nodal
    velocity faces and the cut-cell centroid geometry on ``Center``
    cells — so an FV immersed run would silently mix a cell-average
    state with fraction weights derived for point values. It is a
    taught refusal, never a silent fallback; the nodal immersed path is
    unchanged and remains the supported cut-cell spelling.

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
        "hy.Core(family='fv') on an immersed grid is not supported: "
        "the hydrostatic cut-cell path is nodal with explicit "
        "fractions (the masked continuity, the wet barotropic "
        "reductions and the open-face gates multiply min-rule "
        "alpha fields onto point-valued C-grid faces, IP-D9), and "
        "none of that machinery is written against the CellAvg "
        "family — an FV immersed run would weight cell averages with "
        "point-value fractions. Use family='nodal' (the immersed "
        "default) or drop the immersed domain.")


def resolve_model_family(family: str | None, grid: Grid) -> str:
    r"""Resolve a hydrostatic model-assembly family choice (stage F3).

    Description
    -----------
    The hydrostatic twin of ``nonhydro2``'s
    ``resolve_model_family``, with **no auto flip**: ``None`` follows
    the grid's own ``default_family`` verbatim, so a plain
    ``fr.spatial.Grid(...)`` keeps the nodal point-value C-grid and
    every pre-existing hydrostatic assembly is bitwise unchanged (the
    nodal-bitwise gate). The finite-volume family is reached by asking
    for it — ``hy.Core(family="fv")`` or a grid that declares
    ``grid.set_default_family("fv")``.

    This deliberately differs from the nonhydrostatic package, which
    promotes ``None -> "fv"`` on every grid: there the two families'
    2nd-order stencils are bit-identical on the pressure chain, while
    the hydrostatic column carries a cumulative integral, a barotropic
    2-D solve and (on a cut-cell grid) an explicitly nodal masked
    path — so the flip is opt-in until each of those has been walked
    through on FV.

    Parameters
    ----------
    family : str | None
        The requested family, or None to follow the grid.
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
        If the resolved family is ``"fv"`` on an immersed grid.
    """
    resolved = effective_family(family, grid)
    if resolved not in FAMILIES:
        raise ValueError(
            f"the model family must be one of {FAMILIES} or None "
            f"(follow the grid default), got {resolved!r}")
    if resolved == "fv":
        _require_fv_capable(grid)
    return resolved


def fv_cgrid_overrides(
    meshes: tuple, vertical: str,
) -> dict[DispatchKey, Operator]:
    r"""Build the hydrostatic FV C-grid ``diff`` override profile.

    Description
    -----------
    Re-points the vector-calculus ``("diff", factor)`` resolution so
    that every horizontal C-grid difference staggers on the average
    family exactly as the nodal C-grid does on the point-value family
    (FV-D3): the cell-average pressure gradient staggers onto the face
    (``("diff", CellAvg) -> FaceDifference``) and the face-flux
    divergence lands back on the cell (``("diff", face) ->
    FluxDifference``, the exact discrete Gauss row). With the profile
    merged, the family-agnostic spellings of the whole package —
    ``fu.diff(x) + fv.diff(y)`` in the continuity DIAGNOSE, the
    transport divergences of the free-surface family and
    ``ZStarGeometry``, ``-grad p_hyd`` and ``-grad ps`` — are FV-correct
    without a single family branch, and (on a periodic flat box) carry
    bit-identical numbers to the nodal C-grid.

    The **vertical** picks up one row the nonhydrostatic profile does
    not need: the diagnosed ``w`` lives on the both-boundary face set
    ``Outer`` (the face-form ``CumulativeIntegral`` codomain), so
    ``("diff", Outer)`` is re-pointed to ``FluxDifference`` as well
    (``Outer -> CellAvg``, the exact Gauss row that makes
    ``d_z (J omega) == -(d_x (Ju) + d_y (Jv))`` land on the buoyancy
    cell to machine precision). Its nodal counterpart is
    ``Outer -> Center`` through the seeded ``FiniteDifference``, which
    is the same two-point stencil.

    Re-pointing the column's ``("diff", CellAvg)`` to
    ``FaceDifference`` also **replaces** the seeded ``FVDerivative``
    chain (``reconstruct`` onto the interior faces, then
    ``flux_diff`` back), whose zero-padded wall faces make the
    co-located column derivative O(1/dz) wrong in the boundary cells.
    The face derivative itself is wall-free (``Inner`` excludes the
    walls); the *return* hop onto the cell is where the FV family has
    no consistent closure, and
    :meth:`Core._slope_gradient` takes it through the nodal
    co-located sibling instead (see there).

    On a **periodic** mesh factor the overrides key the periodic face
    (``Right``); on a **bounded** factor the interior face (``Inner``)
    and, for the pressure, the Neumann-tagged ``CellAvg`` origin the
    walled barotropic solve expands on. The nodal ``("diff", ...)``
    chains are untouched — the profile rides only an FV assembly.

    Parameters
    ----------
    meshes : tuple
        The grid's mesh factors (``grid.factors``).
    vertical : str
        The vertical coordinate name (the axis carrying ``Outer``).

    Returns
    -------
    dict[DispatchKey, Operator]
        The per-mesh-factor ``diff`` profile.
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
            continue
        # bounded: the tagged pressure origin, the interior face (both
        # BC-free and the wall-normal Dirichlet claim) and -- on the
        # vertical -- the both-boundary w faces
        overrides[("diff", mesh.average(CellAvg, bc=BC.NEUMANN))] = (
            face_diff)
        overrides[("diff", mesh.inner)] = flux_diff
        overrides[("diff", mesh.nodal(
            NodeSet.INNER, bc=BC.DIRICHLET))] = flux_diff
        if vertical in mesh.names:
            overrides[("diff", mesh.outer)] = flux_diff
    return overrides


def _make_w_space_rule(
    vertical: str,
    family: str | None = None,
) -> Callable[[Grid], FunctionSpace]:
    """Build the ``w`` space rule: collocated horizontal, Outer vertical.

    Description
    -----------
    The diagnosed vertical velocity lives at the C-grid w-points: the
    cell scalars in the horizontal (co-located with ``p_hyd`` / ``b``)
    and the **both-boundary** face set ``Outer`` along ``vertical``
    (n + 1 faces, including the flat bottom ``w = 0`` and the free
    surface). No ``Dof`` tag resolves to ``Outer`` (``Staggered`` is
    the interior ``Inner`` on a bounded axis), so ``w`` is declared
    through this ``SpaceRule`` escape hatch — the space the face-form
    ``CumulativeIntegral`` lands on, so the DIAGNOSE write is exact.

    The rule is **family-aware** (FV-D2 option A): the horizontal
    factors are ``Center`` on the nodal family and ``CellAvg`` on the
    finite-volume one — the transversely cell-averaged twin of the
    same face — while the vertical stays the point-valued ``Outer``
    face on both. The face-form ``CumulativeIntegral`` carries
    ``CellAvg -> Outer`` exactly as it carries ``Center -> Outer``, so
    the DIAGNOSE write is exact on either family.

    Parameters
    ----------
    vertical : str
        The vertical coordinate name.
    family : str | None, optional
        The requested discretization family; ``None`` defers to the
        grid-level default (default: None).

    Returns
    -------
    Callable[[Grid], FunctionSpace]
        The pure per-grid space rule.
    """
    def rule(grid: Grid) -> FunctionSpace:
        fv = effective_family(family, grid) == "fv"
        factors = []
        for mesh in grid.factors:
            if vertical in mesh.names:
                factors.append(mesh.outer)
            else:
                factors.append(mesh.cell_avg if fv else mesh.center)
        return TensorProductSpace.of(*factors)

    return rule


@partial(jaxify, dynamic=("gravity",))
class Core(fr.model.Module):

    r"""Declares u, v, w, p_hyd; owns gravity and the DIAGNOSE.

    Parameters
    ----------
    gravity : float | fr.model.Ramp | None, optional
        The gravitational acceleration :math:`g` [m/s^2]
        (DIMENSIONAL variant); published as ``hydrostatic.gravity``
        and referenced by the free-surface family. ``None`` is the
        NONDIMENSIONAL variant (no physical constant at all — the
        free surface carries the external Froude number, under a
        nondimensional ``fr.scaling`` policy). Must be nonzero when
        given (the energy weight divides by it) (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
    horizontal : tuple[str, str], optional
        The (zonal, meridional) coordinate names of the two velocity
        components (default: ``("x", "y")``).
    family : str | None, optional
        The discretization family of the core state (FV-D3, stage F3):
        ``"fv"`` declares the finite-volume (cell-average) C-grid —
        ``p_hyd`` on ``CellAvg(x) ⊗ CellAvg(y) ⊗ CellAvg(z)``, ``u`` /
        ``v`` point-valued along their own axis and cell-averaged
        transversely (``Right(x) ⊗ CellAvg(y) ⊗ CellAvg(z)``, FV-D2
        option A), the diagnosed ``w`` on ``CellAvg(x) ⊗ CellAvg(y) ⊗
        Outer(z)`` — and seeds the FV C-grid ``diff`` profile so every
        staggered difference of the package staggers on the average
        family. ``"nodal"`` is the point-value C-grid. ``None`` — the
        default — defers to the grid-level default
        (``grid.default_family``), which is ``"nodal"`` unless the grid
        or the ``hy.Model`` factory says otherwise; there is **no** auto
        flip, so every existing assembly stays bitwise nodal
        (:func:`resolve_model_family`). ``family="fv"`` on an immersed
        (cut-cell) grid is a taught error — the hydrostatic cut-cell
        path is nodal with explicit fractions (default: None).
    """

    state_type = State
    diagnostics = DIAGNOSTICS

    def __init__(
        self,
        *,
        gravity: float | fr.model.Ramp | None = None,
        vertical: str = "z",
        horizontal: tuple[str, str] = ("x", "y"),
        family: str | None = None,
    ) -> None:
        """Store the gravity leaf, the geometry names and the family.

        Raises
        ------
        TypeError
            On malformed ``horizontal`` names, or ``gravity=0`` (the
            energy weight and analytic consumers divide by it).
        ValueError
            On an unknown ``family``.
        """
        if family is not None and family not in FAMILIES:
            raise ValueError(
                f"hy.Core family must be one of {FAMILIES} or None "
                f"(follow the grid default), got {family!r}")
        horizontal = tuple(horizontal)
        if (len(horizontal) != 2  # noqa: PLR2004 — zonal + meridional
                or not all(isinstance(name, str) for name in horizontal)
                or horizontal[0] == horizontal[1]):
            raise TypeError(
                "horizontal names the (zonal, meridional) coordinates: "
                f"two distinct strings, got {horizontal!r}")
        if (isinstance(gravity, numbers.Number)
                and float(gravity) == 0.0):
            raise TypeError(
                "hy.Core gravity=0 is refused: the barotropic energy "
                "weight and the analytic consumers divide by it, so "
                "an exact zero poisons the run far from here; pass a "
                "nonzero gravity")
        self.gravity = (None if gravity is None
                        else fr.model.leaf(gravity))
        self._nondim: bool = gravity is None
        self._vertical = vertical
        self._horizontal = horizontal
        # the requested discretization family (None = follow the grid
        # default). The hy.Model factory resolves it against the grid
        # and adopts it as the grid default, so every family=None
        # declaration of the model (b, ps, U, V, eta) follows.
        self._family: str | None = family
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
        # the vertical mesh extent H (the flat-column depth), captured
        # at bind — the model.units vertical scale (fridom.hydrostatic
        # .units: the hydrostatic model has no aspect ratio, so the
        # vertical rows close on this one sanctioned geometry read)
        self._vertical_extent: float = 1.0
        # captured at bind: the terrain-following column (mapped, base)
        # of a sigma-coordinate grid, or None off a mapped grid (the
        # byte-identical flat / stretched-only path). On a terrain grid
        # the DIAGNOSE stages J-weight the vertical increment and the
        # baroclinic pressure gradient adopts the slope-corrected
        # (constant-physical-height) horizontal derivative.
        self._column: tuple[str, str] | None = None
        # captured at bind: the orthogonal thin-shell chart pair of a
        # spherical (lon, lat, z) grid, or None off a chart grid (the
        # byte-identical flat / terrain paths). On a chart the
        # continuity is the area-weighted metric divergence and the
        # pressure gradient the physical gradient d_i p / h_i
        # (spherical-models plan S2).
        self._chart: tuple[str, str] | None = None
        # the masked / terrain DIAGNOSE stages' derived halo substitute
        # (V-N2), computed at bind from the staggered rows they apply;
        # None off a mapped / immersed grid (the flat path stays traced).
        self._extra_halo: HaloSpec | None = None
        # whether the partial-bottom pressure-gradient correction (PB-D2)
        # fires: True only on a flat immersed grid carrying genuine bottom
        # cuts. False off it (unimmersed, terrain-chart, all-wet /
        # staircase / lateral-only), so the plain diff runs byte-identical.
        self._pb_active: bool = False
        # the family the declarations resolved into, captured at bind
        # (the requested self._family resolved against the grid).
        self._resolved_family: str = "nodal"
        # whether a buoyancy module declared ``b`` (captured at bind).
        # With no buoyancy (constant density) the hydrostatic pressure is
        # identically zero and the DIAGNOSE skips the integral — the flow
        # is barotropic, driven by the surface pressure alone.
        self._has_buoyancy: bool = True

    def bind(self, table: object) -> None:
        """Capture the immersed / terrain descriptors and coord names.

        Description
        -----------
        Reads the immersed descriptor (IP-D9) and, through
        :func:`~fridom.hydrostatic.modules.terrain.discover_column`,
        the single-base terrain column on the vertical axis (rules
        3.8). A **terrain + immersed** grid is the composed masked
        contravariant continuity (stage M5): the masked-continuity
        divergence weights the ``J``-weighted horizontal transport by
        the min-rule face fractions (``_diagnose_w``). It requires the
        Jacobian-weighted chart fractions
        (:func:`~fridom.hydrostatic.modules.terrain.require_chart_immersed_order`);
        a collocation-order mask on a chart is a taught error.

        Bind is also where the **family** is resolved against the grid
        (:func:`resolve_model_family` — ``family='fv'`` on an immersed
        grid is refused here) and where a half-FV assembly is caught:
        the family the core declared on must be the family every other
        3-D field of the model landed on, or the mismatch surfaces far
        away as a ``SpaceMismatchError`` inside a tendency
        (:meth:`_require_uniform_family`).
        """
        grid = table.grid
        self._immersed = getattr(grid, "immersed", None)
        self._coords = tuple(grid.names)
        self._has_buoyancy = "b" in table.names
        self._vertical_extent = vertical_extent(grid, self._vertical)
        self._column = discover_column(
            grid, self._vertical, chart_ok=True)
        self._chart = thin_shell_chart(grid, "hy.Core")
        self._resolved_family = resolve_model_family(self._family, grid)
        self._require_chart_names()
        require_chart_immersed_order(grid, self._column)
        self._require_uniform_family(table)
        self._extra_halo = self._derive_extra_halo(table)
        self._pb_active = self._derive_pb_active(table)

    def _require_chart_names(self) -> None:
        """Refuse a chart whose coordinates are not ``horizontal=``.

        Raises
        ------
        ValueError
            If the chart pair differs from the core's ``horizontal``
            names (the velocity components must be the chart's own
            physical components).
        NotImplementedError
            For the finite-volume family on a chart (untested).
        """
        if self._chart is None:
            return
        if tuple(self._horizontal) != tuple(self._chart):
            raise ValueError(
                f"hy.Core(horizontal={self._horizontal!r}) does not "
                f"match the grid's chart coordinates {self._chart!r}: "
                "on a chart grid u / v are the physical components "
                "along the chart's own coordinates — pass "
                f"horizontal={self._chart!r} (to the core and the "
                "free surface)")
        if self._resolved_family == "fv":
            raise NotImplementedError(
                "hy.Core(family='fv') on an embedding chart is not "
                "supported: the thin-shell chart arm is validated on "
                "the nodal point-value C-grid only. Use family='nodal'")

    def _require_uniform_family(self, table: object) -> None:
        """Refuse a half-FV assembly with a taught error (stage F3).

        Description
        -----------
        The family reaches sibling modules through the **grid**
        default: ``hy.Model`` resolves the core's ``family=`` and calls
        ``grid.set_default_family(...)``, so every ``family=None``
        declaration of the model (``b``, ``ps``, the split-explicit
        ``U``/``V``, a z* ``eta``) follows uniformly. Assembling
        ``fr.model.Model`` by hand with ``hy.Core(family="fv")`` on a
        nodal grid skips that step and leaves the core on ``CellAvg``
        beside a ``Center`` buoyancy — which fails deep inside a
        tendency as a bare space mismatch. Catch it here instead.
        Only ``family=None`` declarations are held to this: a field
        whose pattern pins its family is the per-field mixed-model
        override (a passive ``CellAvg`` tracer advected beside the
        nodal core) and is served as before the family existed.

        Parameters
        ----------
        table : object
            The binding table (carries the resolved field spaces).

        Raises
        ------
        ValueError
            If a 3-D sibling field resolved onto the other family.
        """
        want_fv = self._resolved_family == "fv"
        offenders = []
        for record in table:
            space = getattr(record, "space", None)
            if space is None or record.name in ("u", "v", "w", "p_hyd"):
                continue
            if getattr(record.pattern, "family", None) is not None:
                # an explicit per-field ``family=`` is the mixed-model
                # override of SpacePattern (FV-D1b), e.g. a passive
                # CellAvg tracer beside the nodal core — a choice, not
                # a grid default that failed to reach the field
                continue
            cells = [factor for factor in space.bare.factors
                     if not isinstance(factor, ConstantSpace)]
            if not cells:
                continue  # a scalar / all-constant parameter field
            # the collocated (cell) factors of the sibling: a velocity
            # face is nodal on both families, so only cell factors say
            # which family the field landed on
            fv = any(isinstance(factor, AverageSpace) for factor in cells)
            nodal_cell = any(
                isinstance(factor, NodalSpace)
                and factor.node_set is NodeSet.CENTER for factor in cells)
            if ((want_fv and nodal_cell and not fv)
                    or (not want_fv and fv)):
                offenders.append(record.name)
        if not offenders:
            return
        raise ValueError(
            f"hy.Core(family={self._resolved_family!r}) is assembled "
            f"beside the fields {tuple(offenders)}, which resolved onto "
            "the other discretization family: the core's cells would "
            "mix CellAvg with Center. The family reaches sibling "
            "modules through the grid default, which the hy.Model "
            "factory sets from core.family — assemble through "
            "hy.Model(...), or call "
            f"grid.set_default_family({self._resolved_family!r}) before "
            "an explicit fr.model.Model assembly.")

    def _derive_pb_active(self, table: object) -> bool:
        r"""Whether the partial-bottom correction fires (PB-D2 / PB-D5).

        Description
        -----------
        True only on a **flat immersed** grid whose wet-centroid offsets
        carry a genuine bottom cut (``delta > 0`` somewhere). Off it —
        unimmersed, a terrain chart (deferred, PB-D3), or an immersed
        grid with no bottom cut (all-wet, collocation staircase,
        lateral-cut-only) — the correction is a proven no-op, so the flat
        pressure gradient runs byte-identical (G1 / G2). The static
        offset field is concrete (memoized geometry), so the presence
        test is a host-side bool resolved once at bind.
        """
        if (self._immersed is None or self._column is not None
                or not self._has_buoyancy):
            return False
        delta = self._immersed.centroid_offset(
            table["b"].space, self._vertical)  # type: ignore[index]
        return bool(jnp.any(delta.data > 0.0))

    def _derive_extra_halo(self, table: object) -> HaloSpec | None:
        r"""Derive the masked / terrain stages' ghost width (V-N2).

        Description
        -----------
        Off a mapped / immersed grid the DIAGNOSE stages and pressure
        gradient are plain staggered stencils the halo trace follows,
        so no substitute is declared (``None``). On a **terrain** or
        **immersed** grid they multiply metric / fraction fields the
        tracer cannot materialize, so the module declares its own
        width — derived from the order-2 rows those stages apply, not a
        literal. Two parallel horizontal legs give reach 1 on each
        horizontal coordinate under any boundary: the pressure gradient
        differences the cell pressure onto the faces (centre -> face),
        the flux / masked continuity differences the face transport back
        onto the cell (face -> centre; the direction that carries the
        reach on a bounded axis). A terrain column adds the
        constant-physical-height slope term to the vertical: a column
        derivative (centre -> face ``diff``) re-aligned onto the cell by
        the column interpolation (face -> centre) — the two composing
        two-sided to reach 1 on the bounded vertical (the diff alone
        shrinks at a wall; the interp carries the reach). An immersed
        grid has **no** vertical stencil (the masked continuity's column
        sum is a reduction, reach 0), so its vertical stays 0. A registry
        override of the differences / interpolations moves these values.
        """
        if (self._immersed is None and self._column is None
                and self._chart is None):
            return None
        registry = table.grid.dispatch  # type: ignore[attr-defined]
        p_hyd = table["p_hyd"].space  # type: ignore[index]
        u = table["u"].space  # type: ignore[index]
        v = table["v"].space  # type: ignore[index]
        zonal, meridional = self._horizontal
        grad_leg: dict[str, list[tuple[str, object]]] = {
            zonal: [("diff", p_hyd.factor(zonal))],
            meridional: [("diff", p_hyd.factor(meridional))],
        }
        div_leg: dict[str, list[tuple[str, object]]] = {
            zonal: [("diff", u.factor(zonal))],
            meridional: [("diff", v.factor(meridional))],
        }
        legs = [grad_leg, div_leg]
        if self._column is not None:
            vert = self._vertical
            centre = p_hyd.factor(vert)
            face = registry.resolve("diff", centre)[vert].codomain(centre)
            # the slope gradient reads the column derivative (centre ->
            # face) and re-aligns it onto the cell by the column
            # interpolation (face -> centre; nodal on the nodal-only
            # hydrostatic grid). The pair reaches 1 on the bounded
            # vertical, driven by the interp (the centre -> face diff
            # shrinks at a wall)
            grad_leg[vert] = [("diff", centre), ("interpolate", face)]
            # the physical-w slope terms interpolate u / v onto the w
            # faces: a horizontal (Right -> Center) average per coupled
            # coordinate and a vertical Center -> Outer lift, each a
            # single staggered row (reach 1). A separate barrier-max leg,
            # so it does not deepen the horizontal reach the flux /
            # gradient legs already carry.
            slope_leg: dict[str, list[tuple[str, object]]] = {
                zonal: [("interpolate", u.factor(zonal))],
                meridional: [("interpolate", v.factor(meridional))],
                vert: [("interpolate", centre)],
            }
            legs.append(slope_leg)
        return derive_extra_halo(registry, self._coords, legs)

    @property
    def extra_halo(self) -> HaloSpec | None:
        """Exempt the masked / terrain DIAGNOSE stages from the trace.

        Description
        -----------
        On an **immersed** grid the fraction-weighted continuity and the
        fraction lookups run concrete field arithmetic the tracer cannot
        follow. On a **terrain** grid the DIAGNOSE stages and the
        baroclinic pressure gradient multiply ``grid.metric`` coefficient
        fields (the column Jacobian ``J``, the slope-corrected
        ``physical_diff``) that the halo tracer's ``_TracerGrid`` cannot
        materialize — the shared mapped-advection precedent (its own
        ``extra_halo``). Either way the module declares its (order-2
        centered) FD-stencil halo here instead of being traced. Two
        cells per coordinate matches the provisional storage halo the
        materialized metrics carry. Off a mapped / immersed grid this is
        ``None`` — the flat DIAGNOSE stages stay fully halo-traced,
        bitwise unchanged.

        The value is **derived** at :meth:`bind` (see
        :meth:`_derive_extra_halo`) from the staggered ``diff`` rows the
        stages apply, not a literal: 1 per horizontal coordinate on both
        paths, plus 1 on the vertical for a terrain column (the slope
        gradient's vertical difference) and 0 on the vertical for an
        immersed grid (no vertical stencil there).
        """
        return self._extra_halo

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

        Every space carries the core's ``family=`` (FV-D3): under
        ``"fv"`` the collocated coordinates land on ``CellAvg`` and
        the staggered ones stay the point-value face, so ``u`` is
        ``Right(x) ⊗ CellAvg(y) ⊗ CellAvg(z)`` (option A), ``p_hyd``
        is ``CellAvg`` throughout and ``w`` is the transversely
        cell-averaged ``Outer(z)`` face. ``None`` defers to the
        grid-level default.
        """
        zonal, meridional = self._horizontal
        family = self._family
        return (
            fr.model.FieldDeclaration.velocity(
                "u", zonal,
                space=fr.spatial.Staggered(zonal, family=family),
                long_name="Zonal velocity", units="m/s"),
            fr.model.FieldDeclaration.velocity(
                "v", meridional,
                space=fr.spatial.Staggered(meridional, family=family),
                long_name="Meridional velocity", units="m/s"),
            fr.model.FieldDeclaration(
                "w",
                space=fr.spatial.SpaceRule(
                    _make_w_space_rule(self._vertical, family)),
                lifecycle=fr.model.Lifecycle.DIAGNOSTIC,
                roles=(Velocity(self._vertical),),
                long_name="Vertical velocity", units="m/s"),
            fr.model.FieldDeclaration(
                "p_hyd", space=fr.spatial.Collocated(family=family),
                lifecycle=fr.model.Lifecycle.DIAGNOSTIC,
                long_name="Hydrostatic pressure", units="m^2/s^2"),
        )

    #: no hard reference on ``b``: the buoyancy field is **optional**.
    #: a buoyancy module (hy.ConstantStratification / hy.BuoyancyTracer)
    #: declares it and the DIAGNOSE integrates it into the hydrostatic
    #: pressure; with no buoyancy module the model is constant-density
    #: (p_hyd == 0, a barotropic flow), so ``b`` is not required.
    field_references = ()

    # ================================================================
    #  The FV C-grid diff profile (grid-aware dispatch hook, F3)
    # ================================================================
    def grid_dispatch_overrides(
        self, grid: Grid,
    ) -> Mapping[DispatchKey, Operator]:
        r"""Contribute the FV C-grid ``diff`` profile when family='fv'.

        Description
        -----------
        The grid-aware twin of the static ``dispatch`` attribute
        (consumed at assembly step 3, so it applies to preset *and*
        explicit assembly): when the resolved family is ``"fv"`` it
        merges the per-mesh-factor ``("diff", CellAvg) ->
        FaceDifference`` / ``("diff", face) -> FluxDifference``
        overrides (:func:`fv_cgrid_overrides`), so every staggered
        difference of the hydrostatic package — the continuity
        DIAGNOSE, the baroclinic and barotropic pressure gradients,
        the transport divergences, the barotropic ``Div @ Diag @ Grad``
        chain — staggers on the average family. Keyed on per-factor
        spaces (which a ``SpacePattern`` cannot express), it needs the
        grid; the static ``dispatch`` attribute cannot build it.

        The family resolves as the declarations do —
        ``self._family`` or the grid default — so the profile rides
        exactly the grids whose ``u, v, w, p_hyd`` landed on the
        average family.

        Parameters
        ----------
        grid : Grid
            The assembled grid the model runs on.

        Returns
        -------
        Mapping[DispatchKey, Operator]
            The FV C-grid diff overrides (empty on a nodal model).
        """
        if effective_family(self._family, grid) != "fv":
            return {}
        _require_fv_capable(grid)
        return fv_cgrid_overrides(grid.factors, self._vertical)

    @property
    def family(self) -> str | None:
        """The requested discretization family (None = grid default).

        The ``hy.Model`` preset reads this to resolve the family
        against the grid (:func:`resolve_model_family`) and adopt it as
        the grid-level default before assembly.
        """
        return self._family

    # ================================================================
    #  Parameters -- gravity centralizes on the core (dimensional)
    # ================================================================
    @property
    def scaling_variant(self) -> str:
        """The constructor-fixed variant (``fr.scaling`` seam)."""
        return "nondimensional" if self._nondim else "dimensional"

    @property
    def unit_factors(self) -> dict[str, fr.model.UnitFactor]:
        """Dimensional-factor rows (``model.units``, §D).

        The hydrostatic amplitude table
        (:mod:`fridom.hydrostatic.units`): the coordinate rows (an
        instance property because ``horizontal=`` / ``vertical=``
        rename the keys), ``u`` / ``v`` / ``p_hyd``, and the
        ``H``-closed ``w`` / ``b`` rows (``H`` the bind-captured
        vertical mesh extent — the flat-only vertical convention).
        """
        height = self._vertical_extent
        return {
            **coordinate_factors(self._horizontal, self._vertical,
                                 height),
            **COMPONENT_FACTORS,
            **DERIVED_FACTORS,
            "w": vertical_velocity_factor(height),
            "b": buoyancy_factor(height),
        }

    @property
    def parameter_declarations(
        self,
    ) -> tuple[fr.model.ParameterDeclaration, ...]:
        """``hydrostatic.gravity`` (dimensional variant only)."""
        if self._nondim:
            return ()
        return (fr.model.ParameterDeclaration(
            GRAVITY, attr="gravity", units="m/s^2",
            doc="gravitational acceleration"),)

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

    def _geometry_params(
        self, state: State,
    ) -> Mapping[str, ScalarField] | None:
        """Return the CURRENT mapping-parameter fields, or None.

        Description
        -----------
        The stage-C4 discovery convention
        (:func:`~fridom.model.modules.moving_geometry.mapping_params`):
        dynamic geometry parameters are state fields named exactly
        after the mapping parameters. Every terrain metric read on the
        step path threads the result through ``grid.metric(...,
        params=)`` / the ``with_params`` reduction seam, so a moving
        column (a ``MovingGeometry`` ``H(t)``, a z* free surface's
        ``eta``) is visible to ``w``, ``p_hyd`` and the slope-corrected
        pressure gradient. Off a mapped grid — and on a mapped grid
        whose parameters do not ride the state — this is ``None``, the
        exact static path (byte-identical to before).

        Parameters
        ----------
        state : State
            The current model state.

        Returns
        -------
        Mapping[str, ScalarField] | None
            The current parameter fields by name, or None.
        """
        if self._column is None:
            return None
        return mapping_params(state, state["u"].grid)

    def _diagnose_w(
        self, state: State, ctx: StepContext,  # noqa: ARG002
    ) -> dict[str, object]:
        r"""Diagnose the **physical** vertical velocity ``w`` (S1').

        ``w(z) = -\int_{-H}^{z} (\partial_x u + \partial_y v) dz'`` on
        the both-boundary face set ``Outer``: seeded ``w = 0`` at the
        flat bottom, ``d_z w == -(d_x u + d_y v)`` machine-exactly. On a
        flat column the diagnosed ``w`` **is** the physical vertical
        velocity, and this path is byte-for-byte unchanged.

        **Terrain-following column** (a sigma-coordinate grid,
        :meth:`bind` captured ``self._column``): the stored ``w`` is the
        **physical** vertical velocity

        .. math::

            w = J\omega + u\,Z_x + v\,Z_y ,

        the sum of the **contravariant volume flux** ``J\omega`` (``J``
        the column Jacobian, ``Z_i`` the coordinate-surface slope) and
        the slope-advection terms the tilted sigma surfaces carry
        (``physical_state_components.md`` ruling (b)). The flux is built
        first, from the **flux form** of the physical horizontal
        divergence,

        .. math::

            J\omega(z) = -\int_{-H}^{z}
                \bigl[\partial_x (J u) + \partial_y (J v)\bigr]\,
                \mathrm{d}z' ,

        with ``J`` on the ``u`` / ``v`` faces — the exact-telescoping
        working quantity: the fundamental theorem
        ``\partial_z(J\omega) == -[\partial_x(Ju) + \partial_y(Jv)]``
        holds to machine precision (the face-form ``CumulativeIntegral``
        FTC), and the bottom seed ``J\omega = 0`` is the **exact**
        zero-normal-flow bottom boundary condition on the sigma column
        (``\omega = 0`` at the terrain). The slope terms
        (:func:`~fridom.hydrostatic.modules.terrain.slope_velocity_on_w`)
        are then **added on top** — so the stored ``w`` is nonzero over a
        slope at the bed (the *flux*, not physical ``w``, vanishes at the
        terrain, which is correct physics: fluid follows the tilted
        surface). ``State.chart["w"]`` subtracts the same slope terms to
        recover the flux ``J\omega`` on demand. With ``J = 1`` and
        ``Z = 0`` (a flat grid) the slope vanishes and ``w`` collapses
        byte-for-byte to the Cartesian form above.

        On an immersed (cut-cell) grid this becomes **masked
        continuity** (IP-D9): the horizontal transport divergence is
        fraction-weighted (``(alpha_x u).diff(x) + (alpha_y v).diff(y)``
        with the min-rule face fractions of I0), the running integral
        yields the barotropic **transport** ``alpha_z w`` (with
        ``alpha_z`` on the vertical ``Outer`` faces), and the flux is the
        guarded division ``alpha_z w / alpha_z`` (``alpha_z == 0 -> 0``).
        The vertical-flux telescoping of the running sum then makes the
        full masked divergence machine-zero on every wet cell (min-rule
        wet faces read two wet cells, so no dry value enters with nonzero
        weight). On a **flat** immersed grid (``self._column is None``)
        that flux is already the physical ``w`` and is stored as-is,
        byte-identical to before.

        On a **terrain + immersed** grid (stage M5) the two compose: the
        fraction weights the ``J``-weighted transport at the face
        (``(alpha_x J u).diff(x) + (alpha_y J v).diff(y)`` — ``alpha`` on
        the *metric-weighted* flux, never the field), the guarded
        division yields the masked contravariant flux ``alpha_z J\omega /
        alpha_z`` (``0`` at the terrain and on every closed face), and
        the slope terms are **added on the wet faces only** — the same
        ``jnp.where(wet, ...)`` gate the flux division rides, so a dry
        face stays ``w = 0`` (no flux, no slope) while a wet face carries
        the physical ``w = J\omega + u Z_x + v Z_y``. The slope terms
        read the min-rule-consistent velocities (dead DOFs zeroed by
        ``MaskState``), so they respect the mask without a second gate.
        """
        zonal, meridional = self._horizontal
        u, v = state["u"], state["v"]
        params = self._geometry_params(state)
        cumint = CumulativeIntegral(
            direction="up", target="face")[self._vertical]
        immersed = getattr(u.grid, "immersed", None)
        # the horizontal transport whose divergence continuity integrates:
        # J-weighted on a terrain column (so the running integral is the
        # contravariant volume flux Jomega, 0 at the sigma bottom), the
        # plain velocity on a flat column. The immersed fraction then
        # weights this metric-weighted transport at the face (never the
        # field).
        if self._column is not None:
            jname = jacobian_name(self._column)
            grid = u.grid
            fu = u * grid.metric(u.function_space.bare, jname,
                                 params=params)
            fv = v * grid.metric(v.function_space.bare, jname,
                                 params=params)
        elif self._chart is not None:
            # thin-shell chart: the area-weighted transports h_j U_i
            # (the transverse edge lengths on the u / v faces); the
            # divergence is closed by the cell area below
            fu = u * edge_scale(u, zonal, self._chart)
            fv = v * edge_scale(v, meridional, self._chart)
        else:
            fu, fv = u, v
        if immersed is None:
            div_h = self._chart_area(fu.diff(zonal) + fv.diff(meridional))
            flux = -cumint(div_h)
            if self._column is None:
                return {"w": flux}  # flat: the flux is already physical w
            # terrain: add the slope terms so the stored w is physical
            slope = slope_velocity_on_w(
                u, v, flux, self._column, self._horizontal,
                self._vertical, params)
            return {"w": flux + slope.retag(flux)}
        alpha_x = immersed.fraction(u.function_space)
        alpha_y = immersed.fraction(v.function_space)
        div_h = self._chart_area(
            (alpha_x * fu).diff(zonal) + (alpha_y * fv).diff(meridional))
        transport = -cumint(div_h)  # the barotropic transport alpha_z*Jomega
        if self._column is None:
            # flat immersed: the masked flux is already the physical w
            # (surface-only override; the bottom flux is a hard 0 seed).
            alpha_z = masked_w_faces(immersed, state, self._vertical)
            az = alpha_z.data
            wet = az > 0.0
            flux = jnp.where(
                wet, transport.data / jnp.where(wet, az, 1.0), 0.0)
            return {"w": transport.with_data(flux)}
        # terrain + immersed: the stored w is physical. The wet mask
        # overrides BOTH physical boundaries (include_bottom) so the bed
        # slope is not zeroed on the physical bottom face the min-rule
        # marked exterior-dry; interior cut faces stay dry (w = 0). The
        # flux (0 at the bottom seed) is unchanged by the bottom override.
        alpha_z = masked_w_faces(
            immersed, state, self._vertical, include_bottom=True)
        az = alpha_z.data
        wet = az > 0.0
        flux = jnp.where(wet, transport.data / jnp.where(wet, az, 1.0), 0.0)
        slope = slope_velocity_on_w(
            u, v, transport, self._column, self._horizontal,
            self._vertical, params)
        w = jnp.where(wet, flux + slope.retag(transport).data, 0.0)
        return {"w": transport.with_data(w)}

    def _chart_area(self, div_h: ScalarField) -> ScalarField:
        r"""Close a chart transport divergence by the cell area.

        Description
        -----------
        On a thin-shell chart the horizontal continuity is
        :math:`(1/\sqrt g)[\partial_1(h_2 U_1) + \partial_2(h_1 U_2)]`:
        the summed difference of the area-weighted transports divided
        by the cell area :math:`\sqrt g` on its own (cell) space — the
        VJP-sealed metric divide (exact-zero root in the never-valid
        padding). Dividing **before** the running integral keeps the
        fundamental theorem ``d_z w == -div_h`` machine-exact.
        :math:`\sqrt g` does not depend on the vertical (shallow
        atmosphere), so the vertical leg is the flat one. Off a chart
        this is the identity (no op is traced — byte-identical).
        """
        if self._chart is None:
            return div_h
        return sealed_metric_divide(div_h, volume_scale(div_h))

    def _masked_w_faces(self, immersed: object, state: State) -> object:
        """Return ``alpha_z`` on the ``w`` faces (surface override).

        Description
        -----------
        A thin delegate to
        :func:`~fridom.hydrostatic.modules.terrain.masked_w_faces` (the
        surface-only wet-face indicator the flux division rides), kept as
        a method so the mirrored core tests reach the machinery through
        the module. The physical-``w`` terrain path calls the free
        function with ``include_bottom=True`` directly.
        """
        return masked_w_faces(immersed, state, self._vertical)

    def _diagnose_p_hyd(
        self, state: State, ctx: StepContext,  # noqa: ARG002
    ) -> dict[str, object]:
        r"""Diagnose ``p_hyd`` from hydrostatic balance (top-down center).

        ``p_hyd(z) = -\int_{z}^{0} b\, dz'`` on the cell centres
        (co-located with ``b``): the pyOM half-cell form, seeded
        ``p_hyd = 0`` at the surface. The negative sign is the
        integral taken from the upper limit.

        **Terrain-following column**: the vertical increment is taken
        along the *physical* height ``\mathrm{d}z_p = J\,\mathrm{d}z``,
        so the running sum carries the column Jacobian
        (``jacobian=(mapped,)`` — the wired seam, ``jacobian_weight``):
        ``p_hyd(z) = -\int_z^0 b\,J\,\mathrm{d}z'``, the physical
        hydrostatic pressure ``-\int b\,\mathrm{d}z_p`` (converges at
        second order on both uniform-sigma and stretched-sigma
        columns). On a flat grid ``self._column`` is ``None`` and the
        plain (unweighted) form is byte-identical to before.

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
        if not self._has_buoyancy:
            # constant density: no ``b`` field, so the hydrostatic
            # pressure is identically zero and the baroclinic gradient
            # ``-grad p_hyd`` vanishes (a barotropic flow driven by the
            # surface pressure alone). The ``* 0.0`` keeps the p_hyd
            # space and stays a plain traced op (no field materialization,
            # so the halo trace follows it).
            return {"p_hyd": state["p_hyd"] * 0.0}
        jacobian = (None if self._column is None
                    else (self._column[0],))
        cumint = CumulativeIntegral(
            direction="down", target="center", jacobian=jacobian)
        p_hyd = -cumint.with_params(
            self._geometry_params(state))[self._vertical](state["b"])
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

        **Terrain-following column**: the horizontal force is the
        gradient at **constant physical height**
        ``-\partial_x p_{hyd}|_{z_p}
        = -(\partial_x p_{hyd}|_z - (Z_x/J)\,\partial_z p_{hyd})``,
        the slope-corrected derivative dispatched through the grid's
        ``physical_diff`` row (the same constant-physical-coordinate
        derivative the mapped advection consumes). Adding the slope
        term is the classic sigma-coordinate pressure-gradient-error
        correction: a stratified fluid at rest over topography stays at
        rest to the scheme's truncation order (the rest-state gate),
        where the plain ``\partial_x p_{hyd}|_z`` alone drives an
        O(1) spurious current. On a flat grid ``self._column`` is
        ``None`` and the plain staggered ``diff`` is byte-identical.

        **Immersed partial bottom cells** (a cut z-level grid,
        ``self._pb_active``, a flat immersed grid with a bottom cut): the
        full-cell ``p_hyd`` integral labels each cell's pressure at the
        cell centre, but a partial bottom cell's wet volume sits above
        it (the wet-centroid offset ``delta``, PB-D1), so neighbouring
        columns cut at different depths difference pressures at
        mismatched heights — the Pacanowski & Gnanadesikan (1998)
        partial-cell pressure-gradient error. The correction reconstructs
        each cell's pressure to a common height with the cell-local
        buoyancy as the hydrostatic slope (PB-D2), an added cell-centred
        term ``S * (b_up - b)`` with the static weight
        ``S = delta * dz / (2 (zeta_up - zeta))`` (``zeta = z_c + delta``
        the wet-centroid height) and ``b_up - b`` the upward vertical
        buoyancy increment; a resting column with ``b`` sampled at the
        wet-centroid heights (PB-D4) then stays at rest to machine
        precision on flat / stretched z-levels. The term is exactly zero
        off a partial bottom cell (``delta <= 0``: full, dry, lateral-cut
        and surface-cut cells),
        so an all-wet or staircase grid is byte-identical to the plain
        ``diff`` above (G1 / G2). Terrain-chart cut cells are a recorded
        follow-up (PB-D3), handled by the ``self._column`` branch above
        (no partial-bottom correction).
        """
        zonal, meridional = self._horizontal
        u, v = state["u"], state["v"]
        p_hyd = state["p_hyd"]
        if self._column is not None:
            params = self._geometry_params(state)
            return {
                "u": (-self._slope_gradient(
                    p_hyd, zonal, u, params)).retag(u),
                "v": (-self._slope_gradient(
                    p_hyd, meridional, v, params)).retag(v),
            }
        if self._pb_active:
            p_hyd = self._partial_bottom_pressure(p_hyd, state)
        if self._chart is not None:
            # thin-shell chart: the physical gradient d_i p / h_i on
            # the velocity's own face (sealed metric divide); the
            # identity chart divides by exactly 1.0 (bitwise the flat
            # diff below)
            return {
                "u": (-chart_gradient(p_hyd, zonal)).retag(u),
                "v": (-chart_gradient(p_hyd, meridional)).retag(v),
            }
        return {
            "u": (-p_hyd.diff(zonal)).retag(u),
            "v": (-p_hyd.diff(meridional)).retag(v),
        }

    def _partial_bottom_pressure(
        self, p_hyd: object, state: State,
    ) -> object:
        r"""Add the partial-bottom well-balancing correction (PB-D2).

        Description
        -----------
        Returns ``p_hyd`` plus the cell-centred correction
        ``S * (b_up - b)``, with ``S`` the static well-balancing weight
        ``delta * dz / (2 (zeta_up - zeta))`` built here from the
        wet-centroid offsets (``delta``, ``zeta = z_c + delta``) and
        ``b_up - b`` the upward vertical buoyancy increment
        (:meth:`_upward_increment`). The correction
        reconstructs each partial bottom cell's pressure to a common
        height with the cell-local buoyancy as the hydrostatic slope
        (PB-D2), cancelling the half-cell height mismatch of the
        full-cell ``p_hyd`` integral. It is exactly zero off a partial
        bottom cell (``S == 0`` there), so an all-wet / staircase grid is
        byte-identical to the plain ``diff`` (G1 / G2). ``S`` is static
        geometry; the only traced factor is ``b_up - b``, linear in ``b``
        with a clean transpose VJP (no step-path divide — PB-D6).

        Parameters
        ----------
        p_hyd : object
            The diagnosed hydrostatic pressure (a cell-centre field).
        state : State
            The current state (supplies ``b``).

        Returns
        -------
        object
            The corrected cell-centre pressure for the horizontal
            gradient (the diagnosed ``p_hyd`` field is untouched).
        """
        vertical = self._vertical
        b = state["b"]
        grid = b.grid
        # the descriptor from the field's grid materializes the static
        # geometry at the module's (extra-halo-widened) storage frame,
        # the fraction-weighting precedent (``_diagnose_w``); memoized
        # concrete-only, so this is a cache hit after the first trace.
        immersed = grid.immersed
        space = b.function_space
        delta = immersed.centroid_offset(space, vertical)
        z_center = grid.evaluation_nodes(space, vertical)
        dz = grid.measure(space.bare, name=vertical)
        # wet-centroid heights and their upward spacing (static). The
        # spacing is the physical distance between the neighbouring wet
        # centroids, so with b sampled there the increment ratio is the
        # exact local buoyancy slope on a linear stratification (G3).
        zeta = delta.with_data(z_center.data + delta.data)
        spacing = self._upward_increment(zeta, vertical).data
        # the well-balancing weight, nonzero only on a bottom cut
        # (delta > 0); the sealed divide never sees a live 0/0.
        dd = delta.data
        bottom = dd > 0.0
        weight = jnp.where(
            bottom, dd * dz.data / (2.0 * jnp.where(bottom, spacing, 1.0)),
            0.0)
        increment = self._upward_increment(b, vertical)
        return p_hyd + increment * delta.with_data(weight)

    def _upward_increment(
        self, field: object, vertical: str,
    ) -> object:
        r"""Return ``f_{k+1} - f_k`` as a cell field, z-shard-safe.

        Description
        -----------
        The one-cell-up vertical increment behind the partial-bottom
        correction — the dynamic buoyancy increment ``b_up - b`` and the
        static wet-centroid spacing ``zeta_up - zeta``. The upward
        difference is one-sided (a bottom cut's lower neighbour is dry,
        so a centred difference would read the masked ``b = 0`` below),
        so it is a whole-column shift rather than a staggered ``diff``.
        When the negotiation shards the vertical the operand is resharded
        onto the axis-local layout (the ``CumulativeIntegral`` contract
        that already keeps ``p_hyd`` axis-local) so the shift sees the
        whole column, then resharded back; a no-op when the vertical is
        already local (the flat-z / horizontally-sharded common case, no
        cost).

        Parameters
        ----------
        field : object
            The cell-centred operand (``b`` or the wet-centroid
            heights), on its layout.
        vertical : str
            The vertical coordinate name.

        Returns
        -------
        object
            ``field_up - field`` on the operand's space and layout (the
            top cell reads a zero above, masked to ``0`` by the
            ``S == 0`` weight).
        """
        grid = field.grid
        layout = field.function_space.layout
        reshard = (grid.decomposition.device_count > 1
                   and layout is not None
                   and not layout.is_local(vertical))
        if reshard:
            from fridom.spatial.operators.movement import (  # noqa: PLC0415 — keep movement off the module import path
                Reshard,
            )
            local = grid.decomposition.layout_for((vertical,))
            local_field = Reshard(grid, local)(field)
            out = _up_shift_local(local_field, vertical)
            return Reshard(grid, layout)(out)
        return _up_shift_local(field, vertical)

    def _slope_gradient(
        self, p_hyd: object, axis: str, target: object,
        params: Mapping[str, ScalarField] | None = None,
    ) -> object:
        r"""Return ``\partial_{axis} p_{hyd}|_{z_p}`` on ``target``'s face.

        Description
        -----------
        The horizontal derivative at **constant physical height**,
        ``\partial_i p|_{z_p} = \partial_i p|_z - (Z_i/J)\,
        \partial_z p``, assembled by explicit field arithmetic that
        mirrors the mapped advection's nodal physical divergence
        (``model/modules/advection.py`` ``_mapped_divergence``, the
        same ``grid.metric`` rows) — **not** the ``physical_diff``
        dispatch verb, whose composite reciprocal-Jacobian metric seals
        a never-valid-padding singularity that poisons the reverse pass
        (the differentiability policy). The slope coefficient is the
        ratio ``Z_i/J = d<mapped>_d<axis> / d<mapped>_d<base>`` (both
        finite on a monotone map, so no guard), the column derivative
        ``\partial_z p`` is interpolated from the column faces onto the
        velocity face along both the column and the coupled axis.

        Parameters
        ----------
        p_hyd : object
            The diagnosed hydrostatic pressure (a cell-centre field).
        axis : str
            The horizontal coordinate the gradient is taken along.
        target : object
            The velocity component whose face the gradient lands on
            (fixes the staggering; the retag is applied by the caller).
        params : Mapping[str, ScalarField] | None, optional
            The CURRENT mapping-parameter fields the slope and the
            Jacobian derive from (stage C4,
            :meth:`_geometry_params`); None reads the static
            declaration defaults (default: None).

        Returns
        -------
        object
            The constant-physical-height horizontal derivative on
            ``target``'s face.
        """
        mapped, base = self._column
        grid = p_hyd.grid
        div = p_hyd.diff(axis)
        dcol = p_hyd.diff(base)
        space = dcol.function_space
        slope = grid.metric(space.bare, f"d{mapped}_d{axis}",
                            params=params)
        jac = grid.metric(space.bare, f"d{mapped}_d{base}",
                          params=params)
        # slope coefficient Z_i / J. Double-`where` guard
        # (differentiability policy): J > 0 on every valid column, but
        # the never-valid column-face padding derives J == 0, where a
        # bare ratio seals the forward value yet leaves the VJP singular
        # and poisons the whole gradient with NaN (physical_diff's
        # composite reciprocal-Jacobian hits the same trap — this is why
        # the gradient is assembled by hand).
        jd = jac.data
        nonzero = jd != 0.0
        coeff = slope.with_data(
            jnp.where(nonzero, slope.data / jnp.where(nonzero, jd, 1.0),
                      0.0))
        corr = coeff * dcol
        registry = grid.dispatch
        for name in (base, axis):
            src = corr.function_space.bare.factor(name)
            dst = div.function_space.bare.factor(name)
            if src is dst or _bc_siblings(src, dst):
                continue
            corr = self._hop(corr, registry, name, src, dst)
        return (div - corr.retag(div)).retag(target)

    @staticmethod
    def _hop(
        corr: object, registry: object, name: str,
        src: FunctionSpace, dst: FunctionSpace,
    ) -> object:
        r"""Carry ``corr``'s ``name`` factor from ``src`` onto ``dst``.

        Description
        -----------
        The two re-alignments :meth:`_slope_gradient` needs, plus the
        one seam where the FV family has no consistent row:

        - **cell -> face** (``CellAvg -> Right|Inner`` on the
          horizontal axis under FV, ``Center -> Right`` under nodal):
          the seeded ``("interpolate", src)`` row (G4);
        - **face -> cell on the column**: under nodal this is the
          one-sided ``("interpolate", Inner) -> Center`` interpolation,
          which extrapolates the *derivative* into the boundary cells
          and stays 2nd order there. Its FV twin
          ``("average", Inner) -> CellAvg`` is a
          ``LinearReconstruction`` with the ``"closed"`` wall closure:
          it zero-pads the wall faces, leaving an O(1) error in the
          boundary cells that never converges (and the seeded
          ``FVDerivative`` chain an O(1/dz) one) — the sigma
          pressure-gradient-error gate then fails outright on FV. There
          is no one-sided ``Inner -> CellAvg`` reconstruction in the
          spatial layer to ask for instead.

          So the FV column hop is routed through the **co-located nodal
          sibling**: the same one-sided ``Inner -> Center``
          interpolation the nodal column takes, then the co-located
          ``("deconvolve", Center) -> CellAvg`` crossing (the 2nd-order
          identity of G3 — ``Center`` and ``CellAvg`` sample the same
          cell midpoints). The FV slope correction is then **bitwise**
          the nodal one, relabelled onto the average family, which is
          exactly the parity the flat-box gate asserts elsewhere.

        Parameters
        ----------
        corr : object
            The slope correction on its current space.
        registry : object
            The grid's operator registry.
        name : str
            The coordinate being re-aligned.
        src : FunctionSpace
            ``corr``'s current factor along ``name``.
        dst : FunctionSpace
            The target factor along ``name``.

        Returns
        -------
        object
            ``corr`` with its ``name`` factor on ``dst``.
        """
        if isinstance(dst, AverageSpace) and isinstance(src, NodalSpace):
            # the FV column's face -> cell hop, through the nodal
            # co-located sibling (see the Description)
            corr = registry.resolve("interpolate", src)[name](corr)
            mid = corr.function_space.bare.factor(name)
            if mid is not dst:
                corr = registry.resolve("deconvolve", mid)[name](corr)
            return corr
        kind = ("average" if isinstance(dst, AverageSpace)
                else "interpolate")
        return registry.resolve(kind, src)[name](corr)


# ================================================================
#  Partial-bottom correction: one-cell-up buoyancy shift (kernel)
# ================================================================
def _up_shift_local(b: object, vertical: str) -> object:
    r"""Return ``b_{k+1} - b_k`` on an axis-local operand.

    Description
    -----------
    The pure whole-column upward shift behind
    :meth:`Core._upward_increment` (the operand is
    undistributed along ``vertical``, so the shift never crosses a shard
    boundary): the top cell reads a zero above, which the ``delta == 0``
    weight masks. A slice + concatenate, so the reverse pass is the exact
    transpose (no divide — PB-D6).

    Parameters
    ----------
    b : object
        The buoyancy field (axis-local along ``vertical``).
    vertical : str
        The vertical coordinate name.

    Returns
    -------
    object
        ``b_up - b`` on ``b``'s space.
    """
    data = b.data
    axis = b.function_space.bare.names.index(vertical)
    upper: list[object] = [slice(None)] * data.ndim
    upper[axis] = slice(1, None)
    zero_shape = list(data.shape)
    zero_shape[axis] = 1
    zero = jnp.zeros(tuple(zero_shape), dtype=data.dtype)
    b_up = jnp.concatenate([data[tuple(upper)], zero], axis=axis)
    return b.with_data(b_up - data)
