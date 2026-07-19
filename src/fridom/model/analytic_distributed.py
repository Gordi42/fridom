r"""
Distributed router for the analytic (plain-Fourier) eigenmode route.

Description
-----------
The analytic sibling of the numeric channel engine's
``resolve_distributed_contraction``
(``spatial.operators.distributed_contract``): when a fully periodic
(plain-Fourier) analytic eigenmode consumer -- a projection, an
``f(L)`` operator, a balance operator, a random / single-mode synthesis
-- runs on a grid whose default layout **shards** a transform axis, the
naive per-component ``forward`` / ``backward`` hits the Tier-1 taught
error (the GSPMD distributed-FFT fault). This module resolves a
per-component ``DistributedTransform``
(``spatial.operators.distributed_transform``) sharing one transpose
geometry and exposes the fused ``apply_matrix`` / ``synthesize`` region
as an :class:`AnalyticDistributedRoute`; the matrix (or coefficient
columns)
are built frame-locally on the transpose engine's **internal**
coefficient frame (``dt.coeff.bare``, the half axis re-designated).

The route declines (returns ``None`` -- the caller keeps the existing,
bit-identical single-device / replicated path or the taught error) on a
single device, a non-plain-Fourier component (the walled-vertical
``ComposedTransform`` the transpose engine itself declines -- Wave B), a
padded / non-1-D layout, or a layout whose per-component transpose
geometries disagree (never on a fully periodic grid, but a defensive
guard). The operand check is separate: even when a route resolves, a
**replicated** operand (a deliberately gathered field) keeps the plain
path -- only a genuinely sharded stage axis is rerouted.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.operators.distributed_transform import (
    resolve_distributed_transform,
)
from fridom.spatial.operators.mixed import resolve_transform

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.spatial.operators.base import FieldLike
    from fridom.spatial.operators.distributed_transform import (
        DistributedTransform,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike


class AnalyticDistributedRoute:

    r"""
    A resolved distributed route for one analytic eigenmode set.

    Description
    -----------
    Holds one representative
    :class:`~fridom.spatial.operators.distributed_transform.DistributedTransform`
    (its geometry transposes every component -- they coincide on a
    fully periodic grid) and the per-component internal coefficient
    frame ``coeff_of`` the matrix / column builder threads its symbols
    on. Application delegates to the transform's fused
    ``apply_matrix`` / ``synthesize`` regions.

    Parameters
    ----------
    transform : DistributedTransform
        The representative fused transform (shared transpose geometry).
    coeff_of : Mapping[str, SpaceLike]
        Per-component internal coefficient (bare) frame -- the frame
        hook the eigenmode symbol kit rebuilds on.
    a_name : str
        The sharded coordinate name (the operand-layout check reads it).
    """

    def __init__(
        self,
        transform: DistributedTransform,
        coeff_of: Mapping[str, SpaceLike],
        a_name: str,
    ) -> None:
        """Store the transform, the frame map and the sharded axis."""
        self._transform: DistributedTransform = transform
        self._coeff_of: dict[str, SpaceLike] = dict(coeff_of)
        self._a_name: str = a_name

    def coeff_of(self, name: str) -> SpaceLike:
        """Return the internal coefficient frame of a component."""
        return self._coeff_of[name]

    def shards(self, field: FieldLike) -> bool:
        """
        Whether the operand actually shards the transform (stage) axis.

        Description
        -----------
        The route is only taken when the operand's own layout shards the
        sharded coordinate ``a`` (a replicated / gathered operand keeps
        the plain path, which needs no reshard and stays bit-identical).
        """
        layout = field.function_space.layout
        return layout is not None and not layout.is_local(self._a_name)

    def apply_matrix(
        self,
        fields: dict[str, FieldLike],
        matrix: jax.Array,
    ) -> dict[str, FieldLike]:
        """Apply a per-mode ``D x D`` matrix (the fused region)."""
        return self._transform.apply_matrix(fields, matrix)

    def synthesize(
        self,
        coeffs: dict[str, jax.Array],
        templates: dict[str, FieldLike],
    ) -> dict[str, FieldLike]:
        """Synthesize internal-frame coefficient columns to fields."""
        return self._transform.synthesize(coeffs, templates)


def resolve_route(
    grid: object,
    analysis_spaces: Mapping[str, SpaceLike],
    prognostic: tuple[str, ...],
) -> AnalyticDistributedRoute | None:
    r"""
    Resolve the fused distributed route, or ``None`` when ineligible.

    Description
    -----------
    Resolves a
    :class:`~fridom.spatial.operators.distributed_transform.DistributedTransform`
    for every analysis component; declines (``None``) when any component
    is unserved (single device, walled ``ComposedTransform``, padded /
    non-1-D layout) or when the per-component transpose geometries
    disagree (a defensive guard -- they coincide on a fully periodic
    grid, verified for staggered ``u`` / ``v`` / ``w`` and collocated
    ``b`` / ``p``). The representative transform is a prognostic
    component's (all share the geometry).

    Parameters
    ----------
    grid : object
        The grid carrying the decomposition.
    analysis_spaces : Mapping[str, SpaceLike]
        The per-component (bare) analysis spaces the kit threads.
    prognostic : tuple[str, ...]
        The prognostic component order (the matrix stacks these).

    Returns
    -------
    AnalyticDistributedRoute | None
        The resolved route, or ``None`` when ineligible.
    """
    transforms: dict[str, DistributedTransform] = {}
    for name, space in analysis_spaces.items():
        bare = space.bare
        transform = resolve_transform(grid, bare)
        dt = resolve_distributed_transform(transform, grid, bare)
        if dt is None:
            return None
        transforms[name] = dt
    geometries = {dt.geometry for dt in transforms.values()}
    if len(geometries) != 1:
        return None
    device_axes = grid.decomposition.default_layout.device_axes
    if len(device_axes) != 1:
        return None
    (a_name, _), = device_axes
    coeff_of = {name: dt.coeff.bare for name, dt in transforms.items()}
    return AnalyticDistributedRoute(
        transforms[prognostic[0]], coeff_of, a_name)


def analytic_route(
    em: object,
    fields: object,
) -> AnalyticDistributedRoute | None:
    r"""
    Return the route for ``em`` when ``fields`` shard a stage axis.

    Description
    -----------
    The consumer front door: resolves the route for the eigenmode set
    (``em._analysis`` / ``em._components``) and returns it only when the
    operand actually shards the transform axis. ``None`` keeps the
    caller's existing (bit-identical single-device / replicated) path or
    its taught error.

    Parameters
    ----------
    em : object
        The analytic eigenmodes (``_grid`` / ``_analysis`` /
        ``_components``).
    fields : object
        The operand state / vector field (indexed by component name).

    Returns
    -------
    AnalyticDistributedRoute | None
        The route, or ``None`` when the plain path applies.
    """
    route = resolve_route(
        em.grid,
        em._analysis,  # noqa: SLF001 — eigenmode-set internals
        em._components)  # noqa: SLF001 — eigenmode-set internals
    if route is None:
        return None
    probe = fields[em._components[0]]  # noqa: SLF001
    if not route.shards(probe):
        return None
    return route
