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

import jax.numpy as jnp

from fridom.spatial.fields.storage import factor_axes
from fridom.spatial.operators.distributed_transform import (
    resolve_distributed_transform,
)
from fridom.spatial.operators.mixed import resolve_transform
from fridom.spatial.operators.transform import axis_slice
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.coefficient import FourierSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.spatial.operators.base import FieldLike
    from fridom.spatial.operators.distributed_transform import (
        DistributedTransform,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike


# ================================================================
#  Hermitian half-axis re-expression (single-device -> internal)
# ================================================================
def _half_spectrum_axis(space: SpaceLike) -> tuple[int | None, int]:
    r"""
    Locate a coefficient frame's real-origin (half-spectrum) axis.

    Description
    -----------
    Scans the (bare) product frame for its real-origin Fourier factor
    -- the Hermitian half axis, stored on ``0..n//2``. Returns the
    array axis and the factor's **nodal** extent ``n``; a frame with no
    real-origin Fourier factor (the 2-D internal frame, which carries
    the full complex spectrum on both axes) returns ``(None, 0)``.

    Parameters
    ----------
    space : SpaceLike
        The (bare or laid-out) coefficient frame.

    Returns
    -------
    tuple[int | None, int]
        The half axis (or ``None``) and its nodal extent.
    """
    for factor, axis in factor_axes(space.bare):
        if isinstance(factor, FourierSpace) and factor.scalars is Scalars.REAL:
            return axis, factor.origin.shape[0]
    return None, 0


def _mirror_full_axes(arr: jax.Array, keep: int) -> jax.Array:
    r"""
    Map index ``i -> (n - i) % n`` on every axis except ``keep``.

    Description
    -----------
    The multi-axis spectral reflection ``k -> -k`` (negative
    wavenumbers modulo the extent) on the full-spectrum axes: index 0
    is fixed and ``1..n-1`` reverse (``roll(flip)``). The ``keep`` axis
    (the stored half axis) is left untouched.

    Parameters
    ----------
    arr : jax.Array
        The coefficient array.
    keep : int
        The array axis to leave unreflected (the half axis).

    Returns
    -------
    jax.Array
        The reflected array.
    """
    for axis in range(arr.ndim):
        if axis == keep:
            continue
        arr = jnp.roll(jnp.flip(arr, axis=axis), 1, axis=axis)
    return arr


def hermitian_reframe(
    column: jax.Array,
    source: SpaceLike,
    target: SpaceLike,
) -> jax.Array:
    r"""
    Re-express a half-spectrum column onto the re-designated frame.

    Description
    -----------
    The device-independent gain columns are built on the single-device
    coefficient frame ``source`` (the Hermitian half spectrum on one
    axis ``h``, full spectra elsewhere). When the sharded axis is that
    half axis, the fused distributed transform re-designates the half to
    another axis, so its internal frame ``target`` cannot consume the
    ``source`` column directly. This is the pure **frame-local**
    re-expression bridging the two: the synthesized field is real, so
    the full spectrum obeys :math:`c(-k) = \overline{c(k)}` (negative
    indices modulo the extent per axis). For every ``target`` lattice
    point ``(k_h, ...)`` the coefficient is

    - the stored gain at ``(k_h, ...)`` when ``k_h`` lies in the
      ``source`` half ``0..n_h//2`` (an interior half-axis mode);
    - :math:`\overline{\text{gain}(-k)}` -- the stored gain at the
      reflected multi-axis index -- when ``k_h`` lies in the missing
      half ``n_h//2+1..n_h-1`` (its reflection ``n_h - k_h`` is stored).

    On the self-conjugate half-axis planes (``k_h = 0`` and, at even
    ``n_h``, the Nyquist ``k_h = n_h//2``) the stored value and its
    reflected conjugate are averaged: the single-device backward
    (``irfft`` on the half axis) keeps only the Hermitian part of those
    planes -- the anti-Hermitian half is discarded as it does not
    survive to the real field -- and this average reproduces that
    projection exactly, so the re-expressed columns synthesize to the
    same field the single-device / replicated backward produces. The
    reflected columns are then sliced to ``target``'s own half spectrum
    (or kept full when ``target`` carries no half axis). Every operation
    is a per-index reflection / gather (no reduction, no collective), so
    a downstream ``shard_map`` shards the result without any all-gather.

    Parameters
    ----------
    column : jax.Array
        The gain column on ``source`` (the half axis at ``n_h//2+1``,
        other axes full), complex.
    source : SpaceLike
        The single-device coefficient frame (half spectrum on axis
        ``h``).
    target : SpaceLike
        The internal (re-designated) coefficient frame the fused
        synthesize consumes.

    Returns
    -------
    jax.Array
        The re-expressed column on ``target``.
    """
    src_axis, src_n = _half_spectrum_axis(source)
    tgt_axis, tgt_n = _half_spectrum_axis(target)
    kh = jnp.arange(src_n)
    direct = kh <= src_n // 2
    src = jnp.where(direct, kh, src_n - kh)
    self_conj = kh == 0
    if src_n % 2 == 0:
        self_conj = self_conj | (kh == src_n // 2)
    stored = jnp.take(column, src, axis=src_axis)
    mirror = jnp.take(
        jnp.conj(_mirror_full_axes(column, src_axis)), src, axis=src_axis)
    shape = [1] * column.ndim
    shape[src_axis] = src_n
    is_direct = jnp.reshape(direct, shape)
    is_self = jnp.reshape(self_conj, shape)
    full = jnp.where(
        is_direct, jnp.where(is_self, 0.5 * (stored + mirror), stored),
        mirror)
    if tgt_axis is None:
        return full
    return axis_slice(full, tgt_axis, 0, tgt_n // 2 + 1)


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
