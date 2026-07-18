r"""Coarsened multigrid level hierarchy for grid-based solvers.

Description
-----------
Owning design: ``design/plans/active/multigrid_pathway_plan.md`` §B3
(decisions MG-D4/D6) and ``multigrid_generalization_plan.md`` (GM-D5:
promotion to ``spatial`` + coarsening-axis freedom). A grid/space-generic
structural builder: it assembles the geometric-multigrid level chain a
solver turns into
:class:`~fridom.spatial.operators.multigrid.MultigridLevel`\ s by
re-instantiating itself on each coarse grid (re-discretization, MG-D6).
Its two model consumers are the nonhydro2 pressure solvers
(:class:`~fridom.nonhydro2.modules.mapped_pressure.MappedPressureSolver`,
:class:`~fridom.nonhydro2.modules.immersed_pressure.ImmersedPressureSolver`)
and the hydrostatic barotropic free-surface solver; the builder itself
knows nothing model-specific — a caller-supplied ``rediscretize``
callback carries any per-coarse-grid dispatch re-establishment (e.g. the
FV C-grid ``diff`` profile) back on the model side.

The chain is a finest-first list of ``(grid, space, transfer)`` triples.

Coarsening (MG-D4 / GM-D5)
--------------------------
The **horizontal** axes coarsen — by a factor of 2 per level — down to
the :data:`MIN_COARSE_CELLS` floor, stopping at indivisibility. The
vertical axis is excluded by default (``vertical`` name; MG-D4
semicoarsening, paired with the vertical line smoother). ``vertical=None``
excludes no axis on that ground; ``coarsen_vertical=True`` lets the
vertical coarsen under the same floor as the horizontal axes (GM-D9 —
the door MG-D4 recorded; the mapped/immersed solvers keep the
semicoarsening default, the flip lands at the solver layer later). A mesh
whose family cannot coarsen (``Mesh.coarsenable`` is ``False``, e.g.
a ``ChebyshevMesh`` vertical) silently stays full — graceful degradation,
never an error. ``multigrid_levels`` (``max_levels``) defaults to
``None`` — floor-limited depth: the builder coarsens all the way to the
floor, which restores h-independent iteration counts at every size; an
``int`` caps the count as a **maximum**. A grid too small for any
coarsening yields a one-level, smoothing-only hierarchy (which the
V-cycle runs as fixed coarse sweeps, never an error).

Re-discretization on the coarse grid (MG-D6)
--------------------------------------------
``Grid.coarsened`` rebuilds the coarse sibling from re-derived geometry
but drops the model's dispatch overrides (they are model-specific). The
optional ``rediscretize`` callback is invoked on each freshly-built
coarse grid so the model can re-establish those overrides (on the FV
C-grid, the pressure ``diff`` staggering profile; a nodal-family grid
needs nothing). The callback is expected to be idempotent — a memoized
coarse grid (``Grid.coarsened`` caches per factors/devices) may be handed
back over several traces that rebuild the hierarchy.
"""
# Multigrid pathway plan, phase B (B3); GM-D5 promotion + coarsen_vertical
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.meshes.structured_1d import StructuredMesh1D
from fridom.spatial.operators.transfer import GridTransfer

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.tensor_product import SpaceLike

#: the per-axis coarsening ratio of one level (MG-D4)
HORIZONTAL_FACTOR = 2
#: the coarse-cell floor per axis: no axis coarsens below it
MIN_COARSE_CELLS = 4
#: the conservative replicate-size guard for agglomeration (MG-D10):
#: a still-shardable coarse level whose replicated per-device footprint
#: would exceed this is kept sharded, so a huge device-count run never
#: replicates a level that still holds real work. ~4 MiB of float64
#: (e.g. 32 x 32 x 512 cells) — deep coarse levels sit far below it.
_AGGLOMERATE_MAX_BYTES = 4 * 1024 * 1024
#: float64 itemsize used for the conservative replicate-size guard
#: (an over-estimate for a single-precision solve, i.e. more
#: conservative about replicating — never under-counts the footprint)
_REPLICATE_ITEMSIZE = 8


# ================================================================
#  Coarse-axis selection
# ================================================================
def _coarsenable_factors(
    grid: Grid, vertical: str | None, *, coarsen_vertical: bool = False,
) -> dict[str, int]:
    """
    Select the axes that coarsen at the next level.

    Description
    -----------
    A single-name :class:`StructuredMesh1D` factor whose family can
    coarsen (:attr:`Mesh.coarsenable`) coarsens by
    :data:`HORIZONTAL_FACTOR` iff its cell count stays integral **and**
    at or above :data:`MIN_COARSE_CELLS` after the division — the floor
    and the indivisibility stop. The ``vertical`` axis is skipped unless
    ``coarsen_vertical`` is set (MG-D4 semicoarsening default; GM-D9
    flip); ``vertical=None`` skips no axis on that ground. A mesh that
    cannot coarsen (a ``ChebyshevMesh`` vertical) silently stays full —
    graceful degradation. An empty result means no further coarsening is
    possible (the caller ends the hierarchy).

    Parameters
    ----------
    grid : Grid
        The current (finer) level's grid.
    vertical : str | None
        The vertical coordinate name, excluded unless
        ``coarsen_vertical`` is set; ``None`` designates no vertical.
    coarsen_vertical : bool, optional
        Whether the vertical coarsens under the same floor as the
        horizontal axes (default: False, MG-D4 semicoarsening).

    Returns
    -------
    dict[str, int]
        The per-name factor-2 divisors for the coarsenable axes (empty
        when none qualify).
    """
    factors: dict[str, int] = {}
    for mesh in grid.factors:
        if len(mesh.names) != 1 or not isinstance(mesh, StructuredMesh1D):
            continue
        name = mesh.names[0]
        if name == vertical and not coarsen_vertical:
            continue
        if not mesh.coarsenable:
            continue
        n_cells = mesh.n_cells
        if (n_cells % HORIZONTAL_FACTOR == 0
                and n_cells // HORIZONTAL_FACTOR >= MIN_COARSE_CELLS):
            factors[name] = HORIZONTAL_FACTOR
    return factors


# ================================================================
#  Agglomeration switch (MG-D10)
# ================================================================
def validate_agglomerate(agglomerate: int | None) -> int | None:
    """
    Validate the agglomeration threshold knob.

    Description
    -----------
    ``None`` (agglomeration off) passes through; otherwise the
    threshold must be a positive integer plane count (booleans
    rejected). ``tau = 1`` is valid but effectively off — a per-shard
    extent is always at least one plane, so the switch never fires
    below it.

    Parameters
    ----------
    agglomerate : int | None
        The user knob (``None`` or the plane threshold ``tau``).

    Returns
    -------
    int | None
        The validated knob (unchanged).

    Raises
    ------
    ValueError
        If ``agglomerate`` is not ``None`` and not a positive integer.
    """
    if agglomerate is None:
        return None
    if isinstance(agglomerate, bool) or not isinstance(agglomerate, int):
        raise TypeError(
            "multigrid_agglomerate must be None or a positive integer "
            f"plane threshold, got {agglomerate!r}")
    if agglomerate < 1:
        raise ValueError(
            "multigrid_agglomerate must be a positive integer plane "
            f"threshold (or None to disable), got {agglomerate}")
    return agglomerate


def _should_agglomerate(coarse: Grid, tau: int) -> bool:
    r"""
    Whether ``coarse`` (and every level below it) should replicate.

    Description
    -----------
    The coarse-grid agglomeration switch (MG-D10): a candidate coarse
    level crosses the switch when its shortest would-be per-shard
    extent along any sharded axis is **below** ``tau`` planes **and**
    its replicated per-device footprint stays under
    :data:`_AGGLOMERATE_MAX_BYTES`. A level the ordinary negotiation
    already replicates (no sharded axis) or a single-device
    decomposition crosses too (``Layout({})`` either way, so
    replicating from here down is a no-op there). A still-shardable
    level whose extent clears ``tau``, or one small-per-shard but too
    large to replicate (the bytes guard), stays sharded — so a huge
    device-count run never replicates a level that still holds real
    work, and coarsening may still reach the switch at a deeper,
    smaller level.

    Parameters
    ----------
    coarse : Grid
        The candidate coarse grid (negotiated with its ordinary
        shard-or-replicate layout).
    tau : int
        The per-shard extent threshold (planes); the switch fires
        below it.

    Returns
    -------
    bool
        Whether to replicate this level and every level below it.
    """
    decomposition = coarse.decomposition
    device_count = decomposition.device_count
    if device_count <= 1:
        return True
    axes = decomposition.default_layout.device_axes
    if not axes:
        return True  # already replicated by the ordinary negotiation
    sizes = {name: getattr(mesh, "n_cells", None)
             for mesh in coarse.factors for name in mesh.names}
    extents = [-(-sizes[name] // device_count)
               for name, _ in axes if sizes.get(name) is not None]
    if not extents or min(extents) >= tau:
        return False
    cells = 1
    for mesh in coarse.factors:
        cells *= getattr(mesh, "n_cells", 1) or 1
    return cells * _REPLICATE_ITEMSIZE <= _AGGLOMERATE_MAX_BYTES


# ================================================================
#  The hierarchy builder
# ================================================================
def coarsen_levels(
    fine_grid: Grid,
    fine_space: SpaceLike,
    *,
    vertical: str | None,
    coarsen_vertical: bool = False,
    max_levels: int | None = None,
    order: int = 2,
    rediscretize: Callable[[Grid], None] | None = None,
    agglomerate: int | None = None,
) -> list[tuple[Grid, SpaceLike, GridTransfer | None]]:
    r"""
    Assemble the coarsened ``(grid, space, transfer)`` chain.

    Description
    -----------
    Finest-first (index 0 is ``fine_grid`` / ``fine_space``); ``None``
    coarsens to the floor, an ``int`` caps the chain at that many
    entries. Each step coarsens the qualifying axes
    (:func:`_coarsenable_factors`) by 2, invokes ``rediscretize`` on the
    fresh coarse grid (the model re-establishes its dispatch overrides,
    MG-D6), builds the down-transfer with
    :class:`~fridom.spatial.operators.transfer.GridTransfer`, and derives
    the next level's space as ``transfer.restrict(probe).function_space``
    — so the restricted residual the V-cycle hands down lands **exactly**
    on the coarse operator's space (B2 API contract). Coarsening stops
    when no axis qualifies (the floor / indivisibility / a
    non-coarsenable mesh), so a grid too small for any coarsening returns
    a single ``(grid, space, None)`` level (smoothing-only, MG-D4).

    Parameters
    ----------
    fine_grid : Grid
        The finest level's grid.
    fine_space : SpaceLike
        The finest level's (cell) solve space.
    vertical : str | None
        The vertical coordinate name, kept at full resolution unless
        ``coarsen_vertical`` is set; ``None`` designates no vertical.
    coarsen_vertical : bool, optional
        Whether the vertical coarsens under the same floor as the
        horizontal axes (default: False, MG-D4 semicoarsening default;
        the GM-D9 full-coarsening flip is a solver-layer default change,
        not made here).
    max_levels : int | None, optional
        ``None`` coarsens all the way to the floor (floor-limited
        depth); an ``int`` (>= 1) caps the number of levels, in which
        case the realized count may still be smaller when the grid
        floors first (default: None).
    order : int, optional
        The transfer-pair order forwarded to ``GridTransfer``
        (default: 2).
    rediscretize : Callable[[Grid], None] | None, optional
        Idempotent per-coarse-grid callback re-establishing the model's
        dispatch overrides (default: None, no re-discretization).
    agglomerate : int | None, optional
        The coarse-grid agglomeration threshold ``tau`` in planes
        (MG-D10): from the first coarse level whose shortest would-be
        per-shard extent falls **below** ``tau`` (and that is small
        enough to replicate, :data:`_AGGLOMERATE_MAX_BYTES`), that
        level and every level below it are built **fully replicated**
        (``Grid.coarsened(replicated=True)``) so the redundant coarse
        compute runs collective-free instead of paying a ring halo
        exchange to shard one or two planes. ``None`` disables
        agglomeration (the ordinary shard-or-replicate negotiation at
        every level); a no-op on one device (default: None).

    Returns
    -------
    list[tuple[Grid, SpaceLike, GridTransfer | None]]
        The finest-first level chain; only the last entry's transfer is
        ``None``.
    """
    levels: list[tuple[Grid, SpaceLike, GridTransfer | None]] = []
    grid, space = fine_grid, fine_space
    agglomerated = False
    while max_levels is None or len(levels) + 1 < max_levels:
        factors = _coarsenable_factors(
            grid, vertical, coarsen_vertical=coarsen_vertical)
        if not factors:
            break
        if agglomerate is not None and not agglomerated:
            # peek at the ordinary (shard-or-replicate) coarse level to
            # decide whether the switch fires here; below the switch
            # every level is built replicated (MG-D10)
            agglomerated = _should_agglomerate(
                grid.coarsened(factors), agglomerate)
        coarse = grid.coarsened(factors, replicated=agglomerated)
        if rediscretize is not None:
            rediscretize(coarse)
        transfer = GridTransfer(grid, coarse, order=order)
        probe = grid.create_field(space)
        coarse_space = transfer.restrict(probe).function_space
        levels.append((grid, space, transfer))
        grid, space = coarse, coarse_space
    levels.append((grid, space, None))
    return levels
