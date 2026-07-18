r"""Derive a module's ``extra_halo`` from the operators its stage applies.

Description
-----------
"Structure declared, numbers derived"
(``design/research/pressure_solver_halo.md`` §5, option 1). A module
whose stage drops to raw arrays (a global spectral / CG solve the halo
trace cannot follow) declares its own halo substitute
(:attr:`fridom.model.Module.extra_halo`). Historically that value was a
hardcoded literal; this helper derives it instead from the **bound
registry rows the stage actually applies**, so the declaration tracks
an operator override in both value and stencil from one source of
truth.

The accounting has two rules, both two-sided (per side, in the
factor's index space, so a staggered stencil's bias survives — the
interval accounting of ``design/research/storage_halo_width.md`` §1):

- **within a leg** the reaches *Minkowski-sum* (``HaloSpec.grow``): an
  un-synced chain of applications accumulates ghost depth per side.
  Opposite-biased staggered pairs (a forward ``diff`` re-aligned by a
  backward ``interpolate``) telescope back to the single-stencil
  reach — the composed window is ``[0, +1] ⊕ [-1, 0] = [-1, +1]``,
  width 1, not 2.
- **across legs** separated by a *barrier* (a global transform / solve
  whose output carries no ghost validity and is synced once regardless
  of width) the demands take the per-side *max* (``HaloSpec.merge_max``)
  — never the sum. Independent parallel terms merge the same way. This
  is the §3 accounting result: the pressure projection's ``div`` and
  ``grad`` sit on opposite sides of the spectral transform, so the
  declaration is ``max(1, 1) = 1``, not ``1 + 1 = 2``.

The two-sided demand collapses to the symmetric storage width the
negotiation seam consumes.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.decomposition.halo import HaloSpec

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping, Sequence

    from fridom.spatial.operators.registry import OperatorRegistry
    from fridom.spatial.spaces.tensor_product import SpaceLike

    #: One registry lookup a leg applies on a coordinate: the dispatch
    #: ``kind`` and the (bare) 1-D factor space it resolves against —
    #: the exact row the stage's block expansion consumes
    #: (``composed._diff_entry`` / ``composed._interp_onto`` resolve
    #: rows this way).
    Row = tuple[str, SpaceLike]

    #: A barrier-separated dataflow leg: for each coordinate name the
    #: ordered rows the leg applies there (empty / absent — reach 0).
    Leg = Mapping[str, Sequence[Row]]


def row_reach(
    registry: OperatorRegistry,
    kind: str,
    factor: SpaceLike,
    axis: str,
) -> tuple[int, int]:
    r"""
    Two-sided ghost reach of one registry row.

    Description
    -----------
    Resolve the dispatch ``kind`` row on ``factor``, bind it to
    ``axis`` (the ``registry.resolve(kind, factor)[axis]`` spelling the
    vector-calculus block expansion uses), and read its two-sided
    ``(below, above)`` reach straight off the bound operator. The
    number therefore comes from the same operator object the stage
    applies — a registry override moves the value automatically.

    Parameters
    ----------
    registry : OperatorRegistry
        The merged dispatch registry (as visible after assembly
        step 3).
    kind : str
        The dispatch kind (e.g. ``"diff"``, ``"interpolate"``).
    factor : SpaceLike
        The bare 1-D factor space the row is resolved on.
    axis : str
        The coordinate name the row is bound to.

    Returns
    -------
    tuple[int, int]
        The ``(below, above)`` reach in the factor's index space.
    """
    op = registry.resolve(kind, factor)[axis]
    return op.requirements(factor).reach


def derive_extra_halo(
    registry: OperatorRegistry,
    names: tuple[str, ...],
    legs: Sequence[Leg],
) -> HaloSpec:
    r"""
    Derive a symmetric per-coordinate ``extra_halo`` from the legs.

    Description
    -----------
    Compose the barrier-separated dataflow ``legs`` per the two rules
    of the module docstring — Minkowski-sum the two-sided reaches
    within each leg, take the per-side max across legs — and collapse
    the result to the symmetric storage width. Every reach is derived
    from a live registry row through :func:`row_reach`, so the returned
    spec is the tight demand of exactly the operators the stage runs.

    A coordinate touched by no row in any leg carries reach 0 (a
    ``HaloSpec`` entry of ``0`` still exempts the module from the halo
    trace — the exemption is any non-``None`` spec, unchanged here).

    Parameters
    ----------
    registry : OperatorRegistry
        The merged dispatch registry the stage resolves rows against.
    names : tuple[str, ...]
        The coordinate names the declaration covers (the grid's names).
    legs : Sequence[Leg]
        The barrier-separated legs; each maps a coordinate name to the
        ordered ``(kind, factor)`` rows the leg applies on that axis.

    Returns
    -------
    HaloSpec
        The symmetric per-coordinate ghost width, over ``names``.
    """
    combined = HaloSpec.zero(names)
    for leg in legs:
        leg_spec = HaloSpec.zero(names)
        for coord, rows in leg.items():
            for kind, factor in rows:
                leg_spec = leg_spec.grow(
                    coord, row_reach(registry, kind, factor, coord))
        combined = combined.merge_max(leg_spec)
    return combined.symmetric().over(names)


def require_solver_halo(
    grid: object,
    axes: tuple[str, ...],
    *,
    solver: str,
) -> None:
    r"""
    Assert the negotiated halo suffices for a hand-rolled diagonal.

    Description
    -----------
    The CG pressure solvers assemble their elliptic diagonal with
    hand-rolled 2-point spellings (``jnp.roll`` / concatenate) that
    bypass the registry stencil-consumption guard
    (``staggering.apply_staggered``), so an under-provisioned halo would
    read out of bounds *silently* rather than raise the taught guard
    error. This restores the check at solver build: every solved axis
    must carry at least one negotiated ghost layer — which the pressure
    core's derived :attr:`~fridom.model.Module.extra_halo` (>= 1)
    guarantees, but which is asserted here explicitly rather than relied
    on through the physics floor (``pressure_solver_halo.md`` §7.2).

    Parameters
    ----------
    grid : object
        The negotiated grid the solve runs on.
    axes : tuple[str, ...]
        The solved coordinate axes.
    solver : str
        A short name of the solver, for the error message.

    Raises
    ------
    ValueError
        If any solved axis carries a negotiated halo width below 1.
    """
    decomposition = getattr(grid, "decomposition", None)
    halo = getattr(decomposition, "halo", None)
    if halo is None:
        return
    for axis in axes:
        if axis in halo and halo[axis] < 1:
            raise ValueError(
                f"{solver}: the hand-rolled 2-point diagonal on "
                f"{axis!r} needs at least one negotiated ghost layer, "
                f"but the halo width there is {halo[axis]}. This "
                "stencil bypasses the registry consumption guard, so "
                "the pressure core must declare extra_halo >= 1 on "
                "every solved axis")
