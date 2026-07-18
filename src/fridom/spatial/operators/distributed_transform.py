r"""
General distributed transform apply (slab-decomposed transpose FFTs).

Description
-----------
The reusable core of the GSPMD transform-illegality campaign phase 3
(``design/plans/active/gspmd_transform_illegality_plan.md``): a
standalone, **layout-preserving** distributed realization of
``Transform.forward`` / ``Transform.backward`` on a 1-D device mesh,
consuming the transform's ``distributed_forward_plan`` /
``distributed_backward_plan``. It generalizes the fused spectral solve
(:class:`~fridom.spatial.operators.distributed_solve.SlabPlan`) beyond
the pressure solve: the solve keeps its coefficients on the transpose
partner (one ``all_to_all``, valid because a diagonal is separable), but
a general apply must hand the coefficients back on the operand's own
negotiated layout, so it transposes **back** after the sharded-axis
stage (two ``all_to_all`` moves per direction).

The transpose engine is a pure per-shard body run inside one
``jax.shard_map`` region:

1. the local stages (the Hermitian ``rfft`` on the half axis first, then
   the full-spectrum stages, **including** the transpose partner ``b``)
   run per shard while the sharded axis ``a`` is still distributed;
2. one ``jax.lax.all_to_all`` transposes the decomposition (``a``
   becomes local, ``b`` becomes sharded);
3. the ``a`` stage runs on the now-local axis (a full ``fft``, or a
   Hermitian ``rfft`` when ``a`` is itself the half axis — the 2-D
   channel);
4. a second ``all_to_all`` transposes back (``a`` becomes sharded on its
   coefficient extent, ``b`` becomes local), so the coefficient array
   leaves sharded on the **same** coordinate the nodal operand entered
   sharded on -- layout-preserving from the outside.

The backward direction mirrors it exactly. Indivisible split axes ride
the padded balanced ``all_to_all`` (extents rounded up to a multiple of
the device count, sliced back after each transpose); on a divisible axis
the pad equals the true extent and the pad/slice statically elide.

This body is deliberately shared: :class:`TransposeGeometry` plus
:func:`transpose_forward` / :func:`transpose_backward` are the primitive
the 2-D channel eigenbasis contraction
(``spatial.operators.distributed_contract``) is the first production
consumer of -- it wraps the same transpose around a per-``kx`` column
contraction. The general :class:`DistributedTransform` here is the
field-level apply the transform layer will route through once its wider
consumers (state transforms, exponential stepper, Krylov derivative) are
converted (an explicitly separate follow-up).

Scope: a plain :class:`~fridom.spatial.operators.fourier.Fourier`
transform on a 1-D device mesh whose default layout shards one of its
axes; the resolver declines (returns None -- the caller keeps the
replicated / taught-error path) on a single device, a non-1-D mesh, a
padded (dealiasing) transform, a non-Fourier family (the trig / mixed
stage kernels are the follow-up), or an operand the transform's own
distributed planner declines.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.spatial.decomposition.layout import Layout
from fridom.spatial.fields.storage import storage_dtype
from fridom.spatial.operators.fourier import Fourier
from fridom.spatial.operators.transform import axis_slice

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.spatial.operators.base import FieldLike
    from fridom.spatial.operators.transform import Transform
    from fridom.spatial.spaces.tensor_product import SpaceLike


def _ceil_mult(n: int, shards: int) -> int:
    """Round ``n`` up to the nearest multiple of ``shards``."""
    return -(-n // shards) * shards


def _tail_pad(arr: jax.Array, axis: int, count: int) -> jax.Array:
    """Zero-pad ``count`` slots at the tail of ``axis`` (``count>=0``)."""
    pads = [(0, 0)] * arr.ndim
    pads[axis] = (0, count)
    return jnp.pad(arr, pads)


# ================================================================
#  The transpose geometry and its per-shard bodies (shared core)
# ================================================================
@dataclass(frozen=True)
class TransposeGeometry:

    r"""
    Static geometry of one layout-preserving transpose transform.

    Description
    -----------
    Everything :func:`transpose_forward` / :func:`transpose_backward`
    need to run the two-``all_to_all`` pipeline per shard, resolved once
    per ``(grid, transform, bare)``. The sharded coordinate ``a`` is
    localized for its own stage and re-sharded on its coefficient
    extent; the transpose partner ``b`` parks the shardedness while
    ``a`` transforms; the local stages (Hermitian half first) run on
    permanently-local axes.

    Parameters
    ----------
    axis_name : str
        The 1-D device-mesh axis name.
    a : int
        The sharded coordinate's array axis (nodal frame).
    b : int
        The transpose partner's array axis (parks the shardedness).
    a_half : bool
        Whether the ``a`` stage is the Hermitian ``rfft`` (``a`` is the
        half axis -- the 2-D channel); otherwise a full ``fft``.
    a_n : int
        ``a``'s nodal extent (the ``irfft`` length when ``a_half``).
    a_spec_n : int
        ``a``'s coefficient extent (``a_n // 2 + 1`` when ``a_half``,
        else ``a_n``).
    b_n : int
        The transpose partner's extent (unchanged across the pipeline;
        the full spectrum for a transformed partner, the nodal extent
        for a passive one).
    pad_a : int
        ``a``'s nodal padded-even extent (``ceil_mult(a_n)``).
    pad_a_spec : int
        ``a``'s coefficient padded extent (``ceil_mult(a_spec_n)``).
    pad_b : int
        The partner's padded split extent (``ceil_mult(b_n)``).
    local_stages : tuple[tuple[int, bool, int], ...]
        The permanently-local stages as ``(array axis, half, nodal
        extent)``, in forward execution order (the Hermitian half stage
        first). Empty for the 2-D channel.
    real : bool
        Whether the nodal domain is real (synthesis takes the real
        part).
    """

    axis_name: str
    a: int
    b: int
    a_half: bool
    a_n: int
    a_spec_n: int
    b_n: int
    pad_a: int
    pad_a_spec: int
    pad_b: int
    local_stages: tuple[tuple[int, bool, int], ...]
    real: bool

    @property
    def padded(self) -> bool:
        """Whether a split axis needs the padded balanced all-to-all."""
        return (self.pad_a != self.a_n or self.pad_b != self.b_n
                or self.pad_a_spec != self.a_spec_n)


def _stage_forward(c: jax.Array, axis: int, *, half: bool) -> jax.Array:
    """Run one forward Fourier stage (``rfft`` if half, else ``fft``)."""
    if half:
        return jnp.fft.rfft(c, axis=axis, norm="forward")
    return jnp.fft.fft(c, axis=axis, norm="forward")


def _stage_backward(c: jax.Array, axis: int, *, half: bool,
                    n: int) -> jax.Array:
    """Run one inverse Fourier stage (``irfft`` if half, else ``ifft``)."""
    if half:
        return jnp.fft.irfft(c, n=n, axis=axis, norm="forward")
    return jnp.fft.ifft(c, axis=axis, norm="forward")


def transpose_forward(piece: jax.Array,
                      geom: TransposeGeometry) -> jax.Array:
    r"""
    Analyze one nodal shard into the layout-preserving spectral frame.

    Description
    -----------
    The per-shard body of the forward pipeline (run inside a
    ``jax.shard_map``): local stages, ``all_to_all`` (``a`` local /
    ``b`` sharded), the ``a`` stage, ``all_to_all`` back (``a`` sharded
    on its coefficient extent / ``b`` local). The input ``a`` axis
    arrives on the padded-even nodal frame and the output ``a`` axis
    leaves on the padded coefficient frame; the partner ``b`` is padded
    for each transpose and sliced back.

    Parameters
    ----------
    piece : jax.Array
        The nodal shard (``a`` on ``pad_a``, other axes true).
    geom : TransposeGeometry
        The static transpose geometry.

    Returns
    -------
    jax.Array
        The coefficient shard (``a`` sharded on ``pad_a_spec``, ``b``
        local on ``b_n``).
    """
    c = piece
    for axis, half, _n in geom.local_stages:
        c = _stage_forward(c, axis, half=half)
    if geom.pad_b != geom.b_n:
        c = _tail_pad(c, geom.b, geom.pad_b - geom.b_n)
    c = jax.lax.all_to_all(
        c, geom.axis_name, split_axis=geom.b, concat_axis=geom.a,
        tiled=True)
    if geom.pad_a != geom.a_n:
        c = axis_slice(c, geom.a, 0, geom.a_n)
    c = _stage_forward(c, geom.a, half=geom.a_half)
    if geom.pad_a_spec != geom.a_spec_n:
        c = _tail_pad(c, geom.a, geom.pad_a_spec - geom.a_spec_n)
    c = jax.lax.all_to_all(
        c, geom.axis_name, split_axis=geom.a, concat_axis=geom.b,
        tiled=True)
    if geom.pad_b != geom.b_n:
        c = axis_slice(c, geom.b, 0, geom.b_n)
    return c


def transpose_backward(c: jax.Array,
                       geom: TransposeGeometry) -> jax.Array:
    r"""
    Synthesize one coefficient shard back to the nodal frame.

    Description
    -----------
    The exact mirror of :func:`transpose_forward`: ``all_to_all``
    (``a`` local / ``b`` sharded), the inverse ``a`` stage, ``all_to_all``
    back (``a`` sharded on its nodal extent / ``b`` local), the inverse
    local stages in reverse order (the Hermitian half stage last, landing
    real). The output ``a`` axis leaves on the padded-even nodal frame.

    Parameters
    ----------
    c : jax.Array
        The coefficient shard (``a`` on ``pad_a_spec``, ``b`` on
        ``b_n``).
    geom : TransposeGeometry
        The static transpose geometry.

    Returns
    -------
    jax.Array
        The nodal shard (``a`` sharded on ``pad_a``, other axes true).
    """
    if geom.pad_b != geom.b_n:
        c = _tail_pad(c, geom.b, geom.pad_b - geom.b_n)
    c = jax.lax.all_to_all(
        c, geom.axis_name, split_axis=geom.b, concat_axis=geom.a,
        tiled=True)
    if geom.pad_a_spec != geom.a_spec_n:
        c = axis_slice(c, geom.a, 0, geom.a_spec_n)
    c = _stage_backward(c, geom.a, half=geom.a_half, n=geom.a_n)
    if geom.pad_a != geom.a_n:
        c = _tail_pad(c, geom.a, geom.pad_a - geom.a_n)
    c = jax.lax.all_to_all(
        c, geom.axis_name, split_axis=geom.a, concat_axis=geom.b,
        tiled=True)
    if geom.pad_b != geom.b_n:
        c = axis_slice(c, geom.b, 0, geom.b_n)
    for axis, half, n in reversed(geom.local_stages):
        c = _stage_backward(c, axis, half=half, n=n)
    if geom.real and jnp.iscomplexobj(c):
        return c.real
    return c


# ================================================================
#  The general fused field-level apply (layout-preserving)
# ================================================================
class DistributedTransform:

    r"""
    Layout-preserving fused distributed transform apply.

    Description
    -----------
    Wraps :func:`transpose_forward` / :func:`transpose_backward` in a
    cached jit-wrapped ``jax.shard_map`` region and pairs it with field
    plumbing. :meth:`apply` runs ``backward(middle(forward(.)))`` in one
    region -- a nodal operand enters on its negotiated (``a``-sharded)
    layout, the optional pointwise ``middle`` acts on the internal
    ``a``-sharded coefficient frame (half spectrum on the local half
    axis, full spectra elsewhere), and the nodal result leaves on the
    **same** layout, so the operator is sharding-neutral from the
    outside. The coefficient frame stays **internal** deliberately: the
    iteration-1 storage contract replicates coefficient spaces
    (``decomposition.sharding``), so a coefficient array committed to a
    field would re-gather -- the fused region never materializes one, the
    same choice the spectral solve and channel contraction make.

    Build through :func:`resolve_distributed_transform`, not this
    plumbing constructor.

    Parameters
    ----------
    geom : TransposeGeometry
        The static transpose geometry.
    mesh : jax.sharding.Mesh
        The decomposition's 1-D device mesh.
    nodal : SpaceLike
        The bare nodal domain space.
    coeff : SpaceLike
        The bare internal coefficient space (half spectrum on the local
        half axis, full spectra elsewhere).
    """

    def __init__(
        self,
        geom: TransposeGeometry,
        mesh: jax.sharding.Mesh,
        nodal: SpaceLike,
        coeff: SpaceLike,
    ) -> None:
        """Store the geometry and build the shard_map region."""
        self._geom: TransposeGeometry = geom
        self._mesh: jax.sharding.Mesh = mesh
        self._nodal: SpaceLike = nodal
        self._coeff: SpaceLike = coeff

        nodal_spec: list[str | None] = [None] * len(nodal.shape)
        nodal_spec[geom.a] = geom.axis_name
        self._spec = jax.sharding.PartitionSpec(*nodal_spec)
        #: raw per-shard transpose regions (analysis / synthesis), the
        #: coefficient side sharded on ``a`` -- exposed for fused
        #: consumers and for the standalone forward-correctness tests
        self.forward_region: Callable[[jax.Array], jax.Array] = jax.jit(
            jax.shard_map(
                lambda p: transpose_forward(p, geom), mesh=mesh,
                in_specs=self._spec, out_specs=self._spec))
        self.backward_region: Callable[[jax.Array], jax.Array] = jax.jit(
            jax.shard_map(
                lambda c: transpose_backward(c, geom), mesh=mesh,
                in_specs=self._spec, out_specs=self._spec))
        self._roundtrip: Callable[[jax.Array], jax.Array] = jax.jit(
            jax.shard_map(
                lambda p: transpose_backward(
                    transpose_forward(p, geom), geom),
                mesh=mesh, in_specs=self._spec, out_specs=self._spec))

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def nodal(self) -> SpaceLike:
        """The bare nodal domain space."""
        return self._nodal

    @property
    def coeff(self) -> SpaceLike:
        """The bare internal coefficient space (``a``-sharded frame)."""
        return self._coeff

    @property
    def geometry(self) -> TransposeGeometry:
        """The static transpose geometry."""
        return self._geom

    # ================================================================
    #  Application
    # ================================================================
    def apply(
        self,
        f: FieldLike,
        middle: Callable[[jax.Array], jax.Array] | None = None,
    ) -> FieldLike:
        r"""
        Run ``backward(middle(forward(f)))`` in one region (no gather).

        Description
        -----------
        The fused transform apply: analysis, an optional pointwise
        ``middle`` on the internal ``a``-sharded coefficient frame, and
        synthesis -- one ``shard_map`` region, so the coefficient array
        only ever exists as per-device slabs. ``middle`` is a pure
        function of the coefficient shard (its own ``a`` rows); None runs
        the plain forward/backward round trip.

        Parameters
        ----------
        f : FieldLike
            The nodal operand, sharded on ``a``.
        middle : Callable[[jax.Array], jax.Array] | None, optional
            A pointwise map on the internal coefficient shard, or None
            for the identity round trip (default: None).

        Returns
        -------
        FieldLike
            The nodal result on the operand's own layout.
        """
        geom = self._geom
        if middle is None:
            region = self._roundtrip
        else:
            region = jax.jit(jax.shard_map(
                lambda p: transpose_backward(
                    middle(transpose_forward(p, geom)), geom),
                mesh=self._mesh, in_specs=self._spec,
                out_specs=self._spec))
        space = f.function_space
        if geom.padded:
            decomposition = f.grid.decomposition
            piece = decomposition.unpad_even(f.storage, space)
            out = region(piece)
            return f.with_storage(decomposition.pad_even(out, space))
        return f.with_data(region(jnp.asarray(f.data)))


# ================================================================
#  Resolution (from the transform's layout plan; memoized per grid)
# ================================================================
#: per-grid memo (the ``distributed_solve`` ``WeakKeyDictionary`` idiom:
#: a dropped grid auto-evicts its memo)
_TRANSFORMS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def build_distributed_transform(
    transform: Transform, grid: object, bare: SpaceLike,
) -> DistributedTransform | None:
    r"""
    Build the layout-preserving apply from the transform's plan, or None.

    Description
    -----------
    Derives the transpose geometry from
    ``transform.distributed_forward_plan(bare)`` (and validates the
    mirror ``distributed_backward_plan``): the sharded axis ``a`` is the
    plan's last stage, the transpose partner ``b`` the coefficient
    layout's sharded axis, the local stages the remaining stages (the
    Hermitian half first). Returns None (the caller keeps the replicated
    / taught-error path) on a single device, a non-1-D mesh, a padded
    transform, a non-Fourier family, or an operand the planner declines.

    Parameters
    ----------
    transform : Transform
        The transform resolved for ``bare``.
    grid : object
        The grid carrying the decomposition.
    bare : SpaceLike
        The bare nodal operand space.

    Returns
    -------
    DistributedTransform | None
        The reusable apply, or None when ineligible.
    """
    if not isinstance(transform, Fourier) or transform.pad is not None:
        return None
    forward = transform.distributed_forward_plan(bare)
    if forward is None:
        return None
    if transform.distributed_backward_plan(forward.codomain) is None:
        return None
    decomposition = grid.decomposition
    mesh = decomposition.device_mesh
    if len(mesh.axis_names) != 1:
        return None
    axis_name = mesh.axis_names[0]
    shards = int(mesh.shape[axis_name])
    names = bare.names
    a_stage = forward.stages[-1]
    a = a_stage.index
    a_name = a_stage.axis
    ((name_b, _),) = forward.codomain.layout.device_axes
    b = names.index(name_b)
    a_n = bare.factor(a_name).shape[0]
    b_n = bare.factor(name_b).shape[0]
    local = tuple(
        (s.index, s.half, bare.factor(s.axis).shape[0])
        for s in forward.stages[:-1])
    geom = TransposeGeometry(
        axis_name=axis_name, a=a, b=b, a_half=a_stage.half,
        a_n=a_n, a_spec_n=a_n, b_n=b_n,
        pad_a=_ceil_mult(a_n, shards),
        pad_a_spec=_ceil_mult(a_n, shards),
        pad_b=_ceil_mult(b_n, shards),
        local_stages=local,
        real=not jnp.issubdtype(storage_dtype(bare),
                                jnp.complexfloating))
    coeff = forward.codomain.bare.with_layout(
        Layout({a_name: axis_name}))
    return DistributedTransform(geom, mesh, bare, coeff)


def resolve_distributed_transform(
    transform: Transform, grid: object, bare: SpaceLike,
) -> DistributedTransform | None:
    """
    Resolve (and memoize) the distributed apply of ``bare``, or None.

    Parameters
    ----------
    transform : Transform
        The transform resolved for ``bare``.
    grid : object
        The grid carrying the decomposition.
    bare : SpaceLike
        The bare nodal operand space.

    Returns
    -------
    DistributedTransform | None
        The memoized apply, or None when ineligible.
    """
    memo = _TRANSFORMS.setdefault(grid, {})
    if bare not in memo:
        memo[bare] = build_distributed_transform(transform, grid, bare)
    return memo[bare]
