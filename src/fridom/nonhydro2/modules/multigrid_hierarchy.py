r"""Semicoarsened multigrid level hierarchy for the pressure solvers.

Description
-----------
Owning design: ``design/plans/active/multigrid_pathway_plan.md`` §B3
(decisions MG-D4/D6). The shared structural builder both
:class:`~fridom.nonhydro2.modules.mapped_pressure.MappedPressureSolver`
and :class:`~fridom.nonhydro2.modules.immersed_pressure.ImmersedPressureSolver`
call to assemble their geometric-multigrid level chain: a finest-first
list of ``(grid, space, transfer)`` triples that each solver turns into
:class:`~fridom.spatial.operators.multigrid.MultigridLevel`\ s by
re-instantiating itself on the coarse grid (re-discretization, MG-D6).

Semicoarsening (MG-D4)
----------------------
Only the **horizontal** axes coarsen — by a factor of 2 per level — and
the vertical mesh stays at full resolution, paired with the vertical
line smoother. ``multigrid_levels`` is a **maximum**: the builder floors
every horizontal axis at :data:`MIN_COARSE_CELLS` cells and stops at
indivisibility, so it degrades gracefully on a small grid (a grid too
small for any coarsening yields a one-level, smoothing-only hierarchy —
which the V-cycle runs as fixed coarse sweeps, never an error).

Re-discretization on the coarse grid (MG-D6)
--------------------------------------------
``Grid.coarsened`` rebuilds the coarse sibling from re-derived geometry
but drops the model's dispatch overrides (they are model-specific). On
the finite-volume C-grid the pressure ``diff`` staggering is exactly
such an override, so the builder re-merges the FV profile onto each
coarse grid — **once**, gated on ``grid.override_keys`` so a memoized
coarse grid (``Grid.coarsened`` caches per factors/devices) is
re-discretized a single time however many traces rebuild the hierarchy.
A nodal-family grid needs no such merge (its ``diff`` rows are the
registry defaults).
"""
# Multigrid pathway plan, phase B (B3): the semicoarsened hierarchy
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.meshes.structured_1d import StructuredMesh1D
from fridom.spatial.operators.registry import check_override_key
from fridom.spatial.operators.transfer import GridTransfer
from fridom.spatial.spaces.average import AverageSpace

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.tensor_product import SpaceLike

#: the per-axis coarsening ratio of a semicoarsening level (MG-D4)
HORIZONTAL_FACTOR = 2
#: the coarse-cell floor per horizontal axis: no axis coarsens below it
MIN_COARSE_CELLS = 4


# ================================================================
#  Coarse-axis selection and re-discretization
# ================================================================
def _is_fv(space: SpaceLike) -> bool:
    """Whether the pressure space is finite-volume (a ``CellAvg`` cell)."""
    return any(isinstance(factor, AverageSpace)
               for factor in space.factors)


def _coarsenable_factors(grid: Grid, vertical: str) -> dict[str, int]:
    """
    Select the horizontal axes that coarsen at the next level.

    Description
    -----------
    A horizontal axis (a single-name :class:`StructuredMesh1D` factor
    that is not the ``vertical``) coarsens by :data:`HORIZONTAL_FACTOR`
    iff its cell count stays integral **and** at or above
    :data:`MIN_COARSE_CELLS` after the division — the floor and the
    indivisibility stop. An empty result means no further coarsening is
    possible (the caller ends the hierarchy).

    Parameters
    ----------
    grid : Grid
        The current (finer) level's grid.
    vertical : str
        The vertical coordinate name (never coarsened, MG-D4).

    Returns
    -------
    dict[str, int]
        The per-name factor-2 divisors for the coarsenable horizontal
        axes (empty when none qualify).
    """
    factors: dict[str, int] = {}
    for mesh in grid.factors:
        if len(mesh.names) != 1 or not isinstance(mesh, StructuredMesh1D):
            continue
        name = mesh.names[0]
        if name == vertical:
            continue
        n_cells = mesh.n_cells
        if (n_cells % HORIZONTAL_FACTOR == 0
                and n_cells // HORIZONTAL_FACTOR >= MIN_COARSE_CELLS):
            factors[name] = HORIZONTAL_FACTOR
    return factors


def _rediscretize(grid: Grid, *, fv: bool) -> None:
    """
    Re-establish the FV ``diff`` profile on a coarse grid (idempotent).

    Description
    -----------
    ``Grid.coarsened`` drops model dispatch overrides, so an FV coarse
    grid resolves the pressure ``diff`` on the registry defaults (the
    wrong face family). This re-merges the FV C-grid profile — but only
    once per grid, gated on :attr:`Grid.override_keys`, so the memoized
    coarse grid is re-discretized a single time even when the hierarchy
    is rebuilt across traces. A nodal grid needs nothing.

    Parameters
    ----------
    grid : Grid
        The coarse grid to re-discretize.
    fv : bool
        Whether the pressure space is finite-volume.
    """
    if not fv:
        return
    # deferred import: ``core`` imports the solver modules, which import
    # this module — the function-local import breaks the load cycle
    from fridom.nonhydro2.modules.core import (  # noqa: PLC0415
        fv_cgrid_overrides,
    )
    overrides = fv_cgrid_overrides(grid.factors)
    wanted = {check_override_key(key) for key in overrides}
    if not wanted <= grid.override_keys:
        grid.merge_overrides(overrides)


# ================================================================
#  The hierarchy builder
# ================================================================
def coarsen_levels(
    fine_grid: Grid,
    fine_space: SpaceLike,
    *,
    vertical: str,
    max_levels: int,
    order: int = 2,
) -> list[tuple[Grid, SpaceLike, GridTransfer | None]]:
    r"""
    Assemble the semicoarsened ``(grid, space, transfer)`` chain.

    Description
    -----------
    Finest-first (index 0 is ``fine_grid`` / ``fine_space``), at most
    ``max_levels`` entries. Each step coarsens the qualifying horizontal
    axes (:func:`_coarsenable_factors`) by 2, builds the down-transfer
    with :class:`~fridom.spatial.operators.transfer.GridTransfer`, and
    derives the next level's pressure space as
    ``transfer.restrict(probe).function_space`` — so the restricted
    residual the V-cycle hands down lands **exactly** on the coarse
    operator's space (B2 API contract). Coarsening stops when no
    horizontal axis qualifies (the floor / indivisibility), so a grid
    too small for any coarsening returns a single ``(grid, space,
    None)`` level (smoothing-only, MG-D4).

    Parameters
    ----------
    fine_grid : Grid
        The finest level's grid.
    fine_space : SpaceLike
        The finest level's (cell) pressure space.
    vertical : str
        The vertical coordinate name (kept at full resolution).
    max_levels : int
        The maximum number of levels (>= 1); the realized count may be
        smaller when the grid floors first.
    order : int, optional
        The transfer-pair order forwarded to ``GridTransfer``
        (default: 2).

    Returns
    -------
    list[tuple[Grid, SpaceLike, GridTransfer | None]]
        The finest-first level chain; only the last entry's transfer is
        ``None``.
    """
    fv = _is_fv(fine_space)
    levels: list[tuple[Grid, SpaceLike, GridTransfer | None]] = []
    grid, space = fine_grid, fine_space
    while len(levels) + 1 < max_levels:
        factors = _coarsenable_factors(grid, vertical)
        if not factors:
            break
        coarse = grid.coarsened(factors)
        _rediscretize(coarse, fv=fv)
        transfer = GridTransfer(grid, coarse, order=order)
        probe = grid.create_field(space)
        coarse_space = transfer.restrict(probe).function_space
        levels.append((grid, space, transfer))
        grid, space = coarse, coarse_space
    levels.append((grid, space, None))
    return levels
