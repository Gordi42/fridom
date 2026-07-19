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
        #: per-(component order) fused matrix / project regions, built
        #: lazily and cached (stable identity: eager re-application adds
        #: zero compiles, the ``ContractPlan`` idiom)
        self._matrix_regions: dict[tuple[str, ...], Callable] = {}
        self._project_regions: dict[tuple[str, ...], Callable] = {}

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
        #: the diagonal-middle region: ``backward(diag * forward(.))``
        #: in one shard_map, the diagonal threaded through ``in_specs``
        #: (sharded on ``a``) so each shard scales its own ``a`` modes
        self._diagonal_region: Callable[
            [jax.Array, jax.Array], jax.Array] = jax.jit(
            jax.shard_map(
                lambda p, d: transpose_backward(
                    transpose_forward(p, geom) * d, geom),
                mesh=mesh, in_specs=(self._spec, self._spec),
                out_specs=self._spec))

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

    def apply_diagonal(
        self,
        f: FieldLike,
        diagonal: jax.Array,
    ) -> FieldLike:
        r"""
        Run ``backward(diagonal * forward(f))`` in one region (no gather).

        Description
        -----------
        The fused *diagonal-middle* apply: analysis, an elementwise
        per-mode multiply by ``diagonal`` on the internal ``a``-sharded
        coefficient frame, and synthesis -- one ``shard_map`` region, so
        the coefficient array only ever exists as per-device slabs and
        the nodal result leaves on the operand's own layout. Unlike the
        closure ``middle`` of :meth:`apply` (a shard-agnostic pointwise
        map), ``diagonal`` may vary along the **sharded** coordinate
        ``a``: it is threaded through the region's ``in_specs`` (sharded
        on ``a``, like the channel contraction's basis), so each device
        multiplies its own ``a`` modes. This serves an operator symbol
        (``op.eigenvalues`` on :attr:`coeff`) as a distributed fused
        forward/backward -- the phase-3 spectral-derivative consumer.

        The diagonal is the symbol's per-mode array on the internal
        coefficient frame (:attr:`coeff`); it must be **endo** (its
        codomain equals its domain, so backward returns to the operand's
        own nodal space -- a retagging symbol has no layout-preserving
        distributed form).

        Parameters
        ----------
        f : FieldLike
            The nodal operand, sharded on ``a``.
        diagonal : jax.Array
            The per-mode diagonal on the internal coefficient frame
            (broadcast over :attr:`coeff`; the ``a`` axis at its full
            coefficient extent).

        Returns
        -------
        FieldLike
            The nodal result on the operand's own layout.
        """
        geom = self._geom
        diag = jnp.broadcast_to(
            jnp.asarray(diagonal), self._coeff.bare.shape)
        if geom.pad_a_spec != geom.a_spec_n:
            diag = _tail_pad(
                diag, geom.a, geom.pad_a_spec - geom.a_spec_n)
        region = self._diagonal_region
        space = f.function_space
        if geom.padded:
            decomposition = f.grid.decomposition
            piece = decomposition.unpad_even(f.storage, space)
            out = region(piece, diag)
            return f.with_storage(decomposition.pad_even(out, space))
        return f.with_data(region(jnp.asarray(f.data), diag))

    # ================================================================
    #  Fused per-mode matrix apply (multi-component)
    # ================================================================
    def _matrix_region(
        self, names: tuple[str, ...],
    ) -> Callable[[dict[str, jax.Array], jax.Array], dict[str, jax.Array]]:
        """Build (once) the fused matrix region for a component order."""
        cached = self._matrix_regions.get(names)
        if cached is not None:
            return cached
        geom = self._geom
        comp_specs = dict.fromkeys(names, self._spec)
        mat_spec = self._matrix_spec(names, tail=2)

        def body(
            comps: dict[str, jax.Array], matrix: jax.Array,
        ) -> dict[str, jax.Array]:
            """Forward, per-mode ``D x D`` matrix, backward (one shard)."""
            z = jnp.stack(
                [transpose_forward(comps[name], geom) for name in names],
                axis=-1)
            out = jnp.einsum("...jd,...d->...j", matrix, z)
            return {name: transpose_backward(out[..., i], geom)
                    for i, name in enumerate(names)}

        region = jax.jit(jax.shard_map(
            body, mesh=self._mesh,
            in_specs=(comp_specs, mat_spec), out_specs=comp_specs))
        self._matrix_regions[names] = region
        return region

    def _matrix_spec(
        self, names: tuple[str, ...], *, tail: int,  # noqa: ARG002
    ) -> jax.sharding.PartitionSpec:
        """Sharded ``in_spec`` for a ``(*coeff, *tail)`` mode array."""
        spec: list[str | None] = [None] * (len(self._coeff.shape) + tail)
        spec[self._geom.a] = self._geom.axis_name
        return jax.sharding.PartitionSpec(*spec)

    def _pad_a(self, arr: jax.Array) -> jax.Array:
        """Zero-pad an internal-frame array's ``a`` axis to ``pad_a_spec``."""
        geom = self._geom
        if geom.pad_a_spec == geom.a_spec_n:
            return arr
        return _tail_pad(arr, geom.a, geom.pad_a_spec - geom.a_spec_n)

    def apply_matrix(
        self,
        fields: dict[str, FieldLike],
        matrix: jax.Array,
    ) -> dict[str, FieldLike]:
        r"""
        Apply a per-mode ``D x D`` matrix on ``D`` components (no gather).

        Description
        -----------
        The fused multi-component analogue of :meth:`apply_diagonal`:
        transposes-forward all ``D`` component fields, stacks them on a
        trailing axis, contracts the per-mode ``D x D`` ``matrix`` with
        one ``einsum`` (``"...jd,...d->...j"`` -- the contracted ``d`` axis
        is the local component axis, so no collective), transposes-backward
        every output column and takes the real part -- one ``shard_map``
        region, so the coefficient frame stays internal and no axis is
        gathered. The ``D`` components may live on **different** function
        spaces (staggered ``u`` / ``v`` / ``w`` and collocated ``b``); on a
        fully periodic grid their transpose geometries coincide, so one
        :attr:`geometry` transposes them all -- each field's own space
        still drives its (un)pad-even framing. Analytic eigenmode
        projections, ``f(L)`` and the balance operators are all this call
        with different pre-assembled matrices.

        Parameters
        ----------
        fields : dict[str, FieldLike]
            The ``D`` nodal operands (ordered), each sharded on ``a``.
        matrix : jax.Array
            The per-mode matrix, shape ``(*coeff.bare, D, D)`` on the
            internal coefficient frame (the component axes ordered as
            ``fields``); the ``a`` axis at its coefficient extent,
            threaded sharded on ``a`` per shard.

        Returns
        -------
        dict[str, FieldLike]
            The contracted real component fields, on the operands' layouts.
        """
        geom = self._geom
        names = tuple(fields)
        mat = self._pad_a(jnp.asarray(matrix))
        region = self._matrix_region(names)
        if geom.padded:
            decomposition = next(iter(fields.values())).grid.decomposition
            pieces = {
                name: decomposition.unpad_even(f.storage, f.function_space)
                for name, f in fields.items()}
            out = region(pieces, mat)
            return {
                name: fields[name].with_storage(decomposition.pad_even(
                    out[name], fields[name].function_space))
                for name in names}
        pieces = {name: jnp.asarray(fields[name].data) for name in names}
        out = region(pieces, mat)
        return {name: fields[name].with_data(out[name]) for name in names}

    def project(
        self,
        fields: dict[str, FieldLike],
        rows: jax.Array,
    ) -> jax.Array:
        r"""
        Analyze ``D`` components into ``J`` modal amplitudes (no gather).

        Description
        -----------
        The forward (analysis + contraction) half of :meth:`apply_matrix`:
        transposes-forward all components, stacks them, and contracts the
        ``(*coeff, J, D)`` ``rows`` into ``J`` raw amplitude arrays on the
        internal coefficient frame (``a`` sharded), **without** the
        backward synthesis. Paired with :meth:`synthesize` it is the split
        the exponential stepper's per-stage arithmetic rides between (the
        amplitudes never leave the sharded frame -- no storage-contract
        violation).

        Parameters
        ----------
        fields : dict[str, FieldLike]
            The ``D`` nodal operands (ordered), sharded on ``a``.
        rows : jax.Array
            The projection rows, shape ``(*coeff.bare, J, D)`` on the
            internal coefficient frame.

        Returns
        -------
        jax.Array
            The ``J`` modal amplitudes, shape ``(*coeff (padded a), J)``,
            sharded on ``a`` (the internal frame).
        """
        geom = self._geom
        names = tuple(fields)
        rws = self._pad_a(jnp.asarray(rows))
        cached = self._project_regions.get(names)
        if cached is None:
            comp_specs = dict.fromkeys(names, self._spec)
            row_spec = self._matrix_spec(names, tail=2)
            amp_spec = self._matrix_spec(names, tail=1)

            def body(
                comps: dict[str, jax.Array], rows_: jax.Array,
            ) -> jax.Array:
                """Forward all components, contract to amplitudes."""
                z = jnp.stack(
                    [transpose_forward(comps[name], geom)
                     for name in names], axis=-1)
                return jnp.einsum("...jd,...d->...j", rows_, z)

            cached = jax.jit(jax.shard_map(
                body, mesh=self._mesh,
                in_specs=(comp_specs, row_spec), out_specs=amp_spec))
            self._project_regions[names] = cached
        if geom.padded:
            decomposition = next(iter(fields.values())).grid.decomposition
            pieces = {
                name: decomposition.unpad_even(f.storage, f.function_space)
                for name, f in fields.items()}
            return cached(pieces, rws)
        pieces = {name: jnp.asarray(fields[name].data) for name in names}
        return cached(pieces, rws)

    def synthesize(
        self,
        coeffs: dict[str, jax.Array],
        templates: dict[str, FieldLike],
    ) -> dict[str, FieldLike]:
        r"""
        Synthesize internal-frame coefficient columns to real fields.

        Description
        -----------
        The backward (synthesis) half: each component's coefficient column
        is already built (host-side, frame-locally) on the internal
        coefficient frame -- the exact frame the forward transpose
        produces -- so only the mirrored inverse transpose runs, no forward
        and no contraction. The ``a`` axis is zero-padded to the balanced
        extent and the region shards it, and the output lands on the
        templates' own nodal spaces (the grid's default layout; the
        sharded axis is never gathered). The analytic random-state (and
        ``mode()``) synthesis rides this half.

        Parameters
        ----------
        coeffs : dict[str, jax.Array]
            Per-component coefficient columns on :attr:`coeff` (the ``a``
            axis at its coefficient extent), complex.
        templates : dict[str, FieldLike]
            Template physical nodal fields (the components' spaces), for
            the output layout / wrapping.

        Returns
        -------
        dict[str, FieldLike]
            The synthesized real component fields.
        """
        geom = self._geom
        region = self.backward_region
        out = {}
        for name, arr in coeffs.items():
            padded = self._pad_a(jnp.broadcast_to(
                jnp.asarray(arr), self._coeff.bare.shape))
            nodal = region(padded)
            template = templates[name]
            if geom.padded:
                decomposition = template.grid.decomposition
                out[name] = template.with_storage(decomposition.pad_even(
                    nodal, template.function_space))
            else:
                out[name] = template.with_data(nodal)
        return out


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


# ================================================================
#  The walled-vertical (mixed Fourier x Fourier x trig) fused region
# ================================================================
class WalledVerticalTransform:

    r"""
    Fused Fourier-transpose + local-trig eigenmode matrix apply.

    Description
    -----------
    The multi-component analogue of :meth:`DistributedTransform.apply_matrix`
    for the **walled-vertical analytic tier** (the ``ComposedTransform``
    the plain :func:`build_distributed_transform` declines): the two
    periodic (Fourier) axes ride the transpose pipeline
    (:func:`transpose_forward` / :func:`transpose_backward`) while the
    bounded (trig ``z``) axis rides **local** inside the fused region --
    never sharded across it, the ``ContractPlan`` idiom. Per shard, per
    component: the fully-complex Fourier transpose (the sharded axis
    localizes for its own ``fft``, the partner parks the shardedness), a
    **local** DST/DCT on the bounded axis, and a
    :meth:`~fridom.spatial.symbols.ModeChart.embed` onto the shared
    ``0..n`` union mode lattice; the ``D`` components stack, the per-mode
    ``D x D`` matrix contracts (the union axis a pointwise batch, the
    component axis local -- no collective), and the mirror path (restrict,
    inverse trig, inverse Fourier transpose, real part) lands each
    component back on its own nodal space. Only ``all_to_all``
    collectives ever run -- never an ``all_gather`` / ``all_reduce``.

    The per-mode matrix is built frame-locally on the internal
    coefficient frame (:meth:`coeff_of`) through the Wave-A frame hook
    (``Eigenmodes.operator_matrix`` ->
    :func:`~fridom.model.eigenstates.assemble_walled_operator_matrix`),
    on the same union lattice the region embeds onto, so the two compose
    exactly. Build through :func:`resolve_walled_vertical_transform`, not
    this plumbing constructor.

    Parameters
    ----------
    mesh : jax.sharding.Mesh
        The decomposition's 1-D device mesh.
    geom : TransposeGeometry
        The shared Fourier transpose geometry (the bounded axis is a
        passive local axis, carried through the two ``all_to_all``
        moves untouched). Fully complex: with only two periodic axes
        there is no third local Fourier axis for a Hermitian half, so
        both ride full spectra and the real domain is recovered by the
        backward's ``.real``.
    components : tuple[str, ...]
        The prognostic component order (the region's stacked column).
    coeff_of : Mapping[str, SpaceLike]
        Per-component internal coefficient (bare) frame (the trig
        ``z`` factor on each component's own lattice); covers every
        analysis component (the symbol kit reads the auxiliary ``p``
        frame too), not only the prognostic ones.
    trig_forward : Mapping[str, Callable]
        Per-prognostic-component local forward DST/DCT on the bounded
        axis (a pure array map).
    trig_backward : Mapping[str, Callable]
        Per-prognostic-component local inverse DST/DCT on the bounded
        axis.
    chart : ModeChart
        The union-lattice chart aligning the components' trig lattices.
    """

    def __init__(
        self,
        mesh: jax.sharding.Mesh,
        geom: TransposeGeometry,
        components: tuple[str, ...],
        coeff_of: dict,
        trig_forward: dict,
        trig_backward: dict,
        chart: object,
    ) -> None:
        """Store the geometry / trig kit and build the shard_map region."""
        self._mesh: jax.sharding.Mesh = mesh
        self._geom: TransposeGeometry = geom
        self._components: tuple[str, ...] = components
        self._coeff_of: dict = coeff_of
        self._trig_forward: dict = trig_forward
        self._trig_backward: dict = trig_backward
        self._chart: object = chart

        ndim = len(coeff_of[components[0]].shape)
        nodal_spec: list[str | None] = [None] * ndim
        nodal_spec[geom.a] = geom.axis_name
        self._spec = jax.sharding.PartitionSpec(*nodal_spec)
        mat_spec_list: list[str | None] = [None] * (ndim + 2)
        mat_spec_list[geom.a] = geom.axis_name
        mat_spec = jax.sharding.PartitionSpec(*mat_spec_list)
        comp_specs = dict.fromkeys(components, self._spec)
        self._region: Callable = jax.jit(jax.shard_map(
            self._body, mesh=mesh,
            in_specs=(comp_specs, mat_spec), out_specs=comp_specs))

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def geometry(self) -> TransposeGeometry:
        """The shared Fourier transpose geometry."""
        return self._geom

    def coeff_of(self, name: str) -> SpaceLike:
        """Return the internal coefficient frame of a component."""
        return self._coeff_of[name]

    # ================================================================
    #  The per-shard region (runs under jax.shard_map)
    # ================================================================
    def _body(self, comps: dict, matrix: jax.Array) -> dict:
        """Transpose + trig + union matrix + inverse (one shard)."""
        geom = self._geom
        chart = self._chart
        z = jnp.stack([
            chart.embed(
                self._trig_forward[name](transpose_forward(
                    comps[name], geom)),
                self._coeff_of[name])
            for name in self._components], axis=-1)
        out = jnp.einsum("...jd,...d->...j", matrix, z)
        result = {}
        for i, name in enumerate(self._components):
            column = chart.restrict(out[..., i], self._coeff_of[name])
            result[name] = transpose_backward(
                self._trig_backward[name](column), geom)
        return result

    def _pad_a(self, arr: jax.Array) -> jax.Array:
        """Zero-pad the matrix's sharded axis to the balanced extent."""
        geom = self._geom
        if geom.pad_a_spec == geom.a_spec_n:
            return arr
        return _tail_pad(arr, geom.a, geom.pad_a_spec - geom.a_spec_n)

    # ================================================================
    #  Application
    # ================================================================
    def apply_matrix(
        self,
        fields: dict,
        matrix: jax.Array,
    ) -> dict:
        r"""
        Apply a per-mode ``D x D`` union-lattice matrix (no gather).

        Description
        -----------
        The walled-vertical analogue of
        :meth:`DistributedTransform.apply_matrix`: transposes-forward the
        two periodic axes of every component, runs the local trig on the
        bounded axis, embeds onto the union lattice, contracts the
        per-mode matrix, and mirrors back to real nodal fields on the
        operands' own layouts -- one ``shard_map`` region, no axis
        gathered.

        Parameters
        ----------
        fields : dict
            The ``D`` nodal operands (ordered), each sharded on the
            periodic axis ``geom.a``.
        matrix : jax.Array
            The per-mode matrix, shape ``(*union, D, D)`` on the internal
            union lattice (the sharded periodic axis at its coefficient
            extent), threaded sharded on ``geom.a`` per shard.

        Returns
        -------
        dict
            The contracted real component fields, on the operands'
            layouts.
        """
        geom = self._geom
        mat = self._pad_a(jnp.asarray(matrix))
        if geom.padded:
            decomposition = next(iter(fields.values())).grid.decomposition
            pieces = {
                name: decomposition.unpad_even(
                    fields[name].storage, fields[name].function_space)
                for name in self._components}
            out = self._region(pieces, mat)
            return {
                name: fields[name].with_storage(decomposition.pad_even(
                    out[name], fields[name].function_space))
                for name in self._components}
        pieces = {name: jnp.asarray(fields[name].data)
                  for name in self._components}
        out = self._region(pieces, mat)
        return {name: fields[name].with_data(out[name])
                for name in self._components}

    def synthesize(self, coeffs: dict, templates: dict) -> dict:
        """Synthesis is unreachable for the walled route (frames differ).

        The internal frame here re-designates the single-device
        Hermitian half axis to the full spectrum (both periodic axes run
        fully complex), so it never coincides with the single-device
        coefficient frame the random-phase columns are drawn on -- the
        random-state consumer keeps its replicated (gathered, still
        device-count invariant) backward. This method exists only for
        the
        :class:`~fridom.model.analytic_distributed.AnalyticDistributedRoute`
        interface and is never called on a walled route.
        """
        raise NotImplementedError(
            "the walled-vertical distributed route has no fused "
            "synthesis: its internal frame never coincides with the "
            "single-device random-phase frame, so random-state "
            "synthesis stays on the replicated backward")


def _trig_kernel(part: object, stage: object, *,
                 forward: bool) -> Callable:
    """Bind a per-axis trig transform stage as a pure array map."""
    if forward:
        return lambda data: part._forward_kernel(data, stage)  # noqa: SLF001 — planner seam
    return lambda data: part._backward_kernel(data, stage)  # noqa: SLF001 — planner seam


#: per-grid memo of resolved walled-vertical transforms (the
#: ``_TRANSFORMS`` idiom: a dropped grid auto-evicts its memo)
_WALLED_TRANSFORMS: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def build_walled_vertical_transform(  # noqa: PLR0911 — many decline guards
    grid: object,
    analysis_spaces: dict,
    prognostic: tuple[str, ...],
) -> WalledVerticalTransform | None:
    r"""
    Build the walled-vertical fused region, or None when ineligible.

    Description
    -----------
    Serves the mixed ``Fourier x Fourier x trig`` analytic eigenmode
    components on a grid whose default layout shards one of the two
    periodic (Fourier) axes: the Fourier part rides the transpose
    pipeline (built through :func:`build_distributed_transform` on the
    ``ComposedTransform``'s Fourier part alone -- with two periodic axes
    and no third local Fourier axis, that geometry is fully complex, the
    real domain recovered by the backward's ``.real``), the trig part
    runs local on the bounded axis, and every component's internal frame
    (:meth:`WalledVerticalTransform.coeff_of`) is the Fourier codomain
    threaded through the trig codomain.

    Returns None (the caller keeps the replicated / taught-error path) on
    a single device, a non-1-D mesh, any component that is not a
    ``ComposedTransform`` of exactly one Fourier part and one trig part,
    a layout that shards the bounded (trig) axis (its Fourier part's
    distributed plan then declines), or a layout whose per-component
    Fourier transpose geometries disagree.

    Parameters
    ----------
    grid : object
        The grid carrying the decomposition.
    analysis_spaces : dict
        The per-component (bare) analysis spaces the kit threads (every
        analysis component, prognostic and auxiliary).
    prognostic : tuple[str, ...]
        The prognostic component order (the region's stacked column).

    Returns
    -------
    WalledVerticalTransform | None
        The reusable fused region, or None when ineligible.
    """
    from fridom.spatial.operators.mixed import (  # noqa: PLC0415 — deferred: avoid an import cycle at module load
        ComposedTransform,
        resolve_transform,
    )
    from fridom.spatial.symbols import ModeChart  # noqa: PLC0415 — deferred

    decomposition = grid.decomposition
    if getattr(decomposition, "device_count", 1) <= 1:
        return None
    mesh = decomposition.device_mesh
    if len(mesh.axis_names) != 1:
        return None
    coeff_of: dict = {}
    trig_forward: dict = {}
    trig_backward: dict = {}
    geometries = set()
    for name, space in analysis_spaces.items():
        bare = space.bare
        transform = resolve_transform(grid, bare)
        if not isinstance(transform, ComposedTransform):
            return None
        fourier = [p for p in transform.parts if p._hermitian]  # noqa: SLF001 — family classvar
        trig = [p for p in transform.parts if not p._hermitian]  # noqa: SLF001 — family classvar
        if len(fourier) != 1 or len(trig) != 1:
            return None
        fourier_part, trig_part = fourier[0], trig[0]
        fdt = build_distributed_transform(fourier_part, grid, bare)
        if fdt is None:
            return None
        geometries.add(fdt.geometry)
        cframe = trig_part.codomain(fdt.coeff.bare)
        coeff_of[name] = cframe
        if name in prognostic:
            fwd = trig_part.forward_plan(bare).stages
            bwd = trig_part.backward_plan(cframe).stages
            if len(fwd) != 1 or len(bwd) != 1:
                return None
            trig_forward[name] = _trig_kernel(
                trig_part, fwd[0], forward=True)
            trig_backward[name] = _trig_kernel(
                trig_part, bwd[0], forward=False)
    if len(geometries) != 1:
        return None
    return WalledVerticalTransform(
        mesh, next(iter(geometries)), tuple(prognostic),
        coeff_of, trig_forward, trig_backward, ModeChart(grid))


def resolve_walled_vertical_transform(
    grid: object,
    analysis_spaces: dict,
    prognostic: tuple[str, ...],
) -> WalledVerticalTransform | None:
    """
    Resolve (and memoize) the walled-vertical fused region, or None.

    Parameters
    ----------
    grid : object
        The grid carrying the decomposition.
    analysis_spaces : dict
        The per-component (bare) analysis spaces.
    prognostic : tuple[str, ...]
        The prognostic component order.

    Returns
    -------
    WalledVerticalTransform | None
        The memoized region, or None when ineligible.
    """
    key = (tuple(sorted(
        (name, id(space.bare)) for name, space in analysis_spaces.items())),
        tuple(prognostic))
    memo = _WALLED_TRANSFORMS.setdefault(grid, {})
    if key not in memo:
        memo[key] = build_walled_vertical_transform(
            grid, analysis_spaces, prognostic)
    return memo[key]
