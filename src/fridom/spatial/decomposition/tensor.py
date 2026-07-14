"""
The jax-sharding decomposition of tensor-product grids.

Description
-----------
Owning class doc: ``design/specs/grid/classes/decomposition.md``.
One code path for any device count: single-device runs use a
one-device ``jax.make_mesh`` (no separate ``SingleDecomposition``;
``jax.sharding`` degrades gracefully). The iteration-1 multi-device
realization is a **1-D device mesh**: every negotiated layout shards
at most one coordinate name over one device axis (the first
GHOST-capable factor in the default layout, transpose pencils via
``layout_for``).

Storage blocking (multi-device): a sharded ("blocked") axis stores
``n_shards`` uniform blocks of ``cells_per_shard + 1 + 2 * width``
slots — per-shard true data behind a leading ghost region, with the
trailing side absorbing ghosts plus the stagger padding that makes
staggered pairs (n vs n + 1 DOFs) shard to one uniform storage shape
(the sanctioned mitigation of ``04_decomposition.md`` section 5).
Blocks are aligned by **cells**, so every space of a mesh places its
per-shard first true DOF at global DOF index ``s * cells_per_shard``
and staggered domain/codomain pairs share one block frame.
``ConstantSpace`` and coefficient factors are never blocked: they are
replicated (constants) or device-local (coefficients, iteration-1
LOCAL preference) in every layout.

``sync`` fills every ghost slot: periodic wrap on periodic mesh
factors, the **BC-structured homogeneous fill** on bounded ones (odd
extension for Dirichlet-structured spaces, even/mirror for
Neumann-structured, one-sided extrapolation for BC-free; blanket
zero-fill is rejected by the halo/storage contract). On one shard
this is the local fill; across shards it is a ``jax.shard_map`` +
``jax.lax.ppermute`` halo exchange with the local fill as the
boundary-edge path.
"""
# Wave 1: trivial single-device TensorDecomposition --
#    Wave 2C: single-device halo fill --
#    Wave 3: multi-device (blocking, ppermute exchange, redistribute)
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from fridom.spatial.bc import BC
from fridom.spatial.decomposition.decomposition import (
    Decomposition,
)
from fridom.spatial.spaces.average import CellAvg, FaceAvg
from fridom.spatial.spaces.coefficient import CoefficientSpace
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.spatial.decomposition.decomposition import (
        SpaceLike,
    )
    from fridom.spatial.decomposition.halo import HaloSpec
    from fridom.spatial.decomposition.layout import Layout
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )


class _ReblockPlan(NamedTuple):

    """
    The shard-local re-blocking plan of one (space, layout) pair.

    Description
    -----------
    Built once per key by ``TensorDecomposition._build_local_reblock``
    and cached: ``scatter``/``gather`` are the jit-wrapped
    ``jax.shard_map`` callables ``pad``/``unpad`` apply (stable
    function identity, so repeated eager calls hit jax's tracing
    cache); ``pspec``, ``pad_widths`` and ``true_slices`` are the
    static structure they close over (per-shard values).
    """

    pspec: jax.sharding.PartitionSpec
    pad_widths: tuple[tuple[int, int], ...]
    true_slices: tuple[slice, ...]
    scatter: Callable[[jax.Array], jax.Array]
    gather: Callable[[jax.Array], jax.Array]


class TensorDecomposition(Decomposition):

    """
    jax.sharding-based decomposition of tensor-product grids.

    Description
    -----------
    Normally constructed by ``negotiate``; the constructor is also
    the plain single-device path a grid calls directly. It implements
    every ``Decomposition`` abstract method and adds no public
    surface — solver code programs against the ABC.

    One sanctioned special case: ``sync`` branches statically on the
    per-axis shard count and skips the ``ppermute`` exchange entirely
    on unsharded axes (a Python-level branch on static structure);
    trait and layout structure are unchanged by the short-circuit.

    Parameters
    ----------
    meshes : tuple[object, ...]
        The grid's mesh factors (consumed through their ``periodic``
        / ``n_cells`` descriptors by the halo fill and the blocking).
    names : tuple[str, ...]
        The grid's coordinate names, concatenated across meshes.
    halo : HaloSpec
        The negotiated per-name ghost widths.
    layouts : tuple[Layout, ...]
        The closed layout vocabulary; the first entry is the default
        layout. Every coordinate name a layout shards must be one of
        `names`. With more than one device, all layouts must share
        one device-mesh axis (the iteration-1 1-D device mesh).
    device_ids : tuple[int, ...] | None, optional
        Indices into ``jax.devices()``; None selects the first
        device (default: None).
    """

    def __init__(
        self,
        meshes: tuple[object, ...],
        names: tuple[str, ...],
        halo: HaloSpec,
        layouts: tuple[Layout, ...],
        device_ids: tuple[int, ...] | None = None,
    ) -> None:
        """Build the device mesh and shardings (called by negotiate)."""
        self._meshes: tuple[object, ...] = tuple(meshes)
        self._names: tuple[str, ...] = tuple(names)
        self._halo: HaloSpec = halo
        self._layouts: tuple[Layout, ...] = tuple(layouts)
        self._validate_layouts()
        self._device_mesh: jax.sharding.Mesh = self._build_device_mesh(
            device_ids)
        # per-(space, layout) caches of the derived static structure
        # (spaces are interned and layouts are hashable values, so
        # structurally-equal queries hit the same entry): the axis
        # geometry drives pad/unpad/sync/storage_shape on every
        # operator application — computing it once per key removes
        # the repeated per-factor Python loops from the hot path.
        self._geometry_cache: dict[
            tuple[object, Layout],
            tuple[tuple[str, int, object, int, int, int, int], ...],
        ] = {}
        self._sharding_cache: dict[
            tuple[object, Layout], jax.sharding.Sharding] = {}
        self._reblock_cache: dict[
            tuple[object, Layout], _ReblockPlan | None] = {}

    def _validate_layouts(self) -> None:
        """Check the layout vocabulary against the grid names."""
        if not self._layouts:
            raise ValueError("layouts must contain at least one Layout")
        for layout in self._layouts:
            for name, _ in layout.device_axes:
                if name not in self._names:
                    raise ValueError(
                        f"layout shards unknown coordinate name "
                        f"{name!r}; grid names are {self._names}")

    def _build_device_mesh(
        self,
        device_ids: tuple[int, ...] | None,
    ) -> jax.sharding.Mesh:
        """Build the (iteration-1: 1-D) ``jax.sharding.Mesh``."""
        devices = jax.devices()
        if device_ids is None:
            selected = (devices[0],)
        else:
            if len(set(device_ids)) != len(device_ids):
                raise ValueError(
                    f"duplicate device ids: {tuple(device_ids)}")
            selected = tuple(devices[i] for i in device_ids)

        # one named axis per device-mesh axis referenced by the
        # layouts; a placeholder axis when no layout shards anything.
        axis_names: list[str] = []
        for layout in self._layouts:
            for _, axis in layout.device_axes:
                if axis not in axis_names:
                    axis_names.append(axis)
        if not axis_names:
            axis_names = ["devices"]
        # Auto axis types: global jnp operations on sharded storage
        # stay legal (the partitioner chooses); only the sync's
        # shard_map region is explicitly manual.
        if len(selected) == 1:
            # single device: every axis has size 1 (degenerate mesh)
            return jax.make_mesh(
                (1,) * len(axis_names), tuple(axis_names),
                devices=selected,
                axis_types=(jax.sharding.AxisType.Auto,)
                * len(axis_names))
        if len(axis_names) != 1:
            raise NotImplementedError(
                "iteration 1 realizes a 1-D device mesh: all layouts "
                f"must share one device axis, got {tuple(axis_names)}")
        return jax.make_mesh(
            (len(selected),), tuple(axis_names), devices=selected,
            axis_types=(jax.sharding.AxisType.Auto,))

    # ================================================================
    #  Negotiated structure
    # ================================================================

    @property
    def halo(self) -> HaloSpec:
        """The negotiated per-name ghost widths."""
        return self._halo

    @property
    def device_count(self) -> int:
        """Number of devices in the device mesh."""
        return int(self._device_mesh.devices.size)

    @property
    def device_mesh(self) -> jax.sharding.Mesh:
        """
        The ``jax.sharding.Mesh`` backing the shardings.

        Description
        -----------
        Exposed for the transpose-based distributed machinery (the
        slab FFT pipeline builds its ``jax.shard_map`` regions over
        this mesh); iteration 1 realizes a 1-D mesh.
        """
        return self._device_mesh

    @property
    def default_layout(self) -> Layout:
        """The layout attached to bare spaces at field creation."""
        return self._layouts[0]

    @property
    def layouts(self) -> tuple[Layout, ...]:
        """All negotiated layouts (default + transform pencils)."""
        return self._layouts

    # ================================================================
    #  Internal helpers
    # ================================================================

    def _resolve_layout(
        self,
        space: SpaceLike,
        layout: Layout | None,
    ) -> Layout:
        """Resolve None to the space's or default layout; validate."""
        if layout is None:
            layout = space.layout
        if layout is None:
            layout = self._layouts[0]
        if layout not in self._layouts:
            raise ValueError(
                f"{layout} is not in the negotiated layout "
                f"vocabulary of this decomposition")
        return layout

    def _axis_entries(
        self, space: SpaceLike,
    ) -> tuple[tuple[str, int, object], ...]:
        """Pair every storage axis with its (name, true n, factor)."""
        entries = []
        for factor in space.factors:
            for name, n in zip(factor.names, factor.shape,
                               strict=True):
                entries.append((name, n, factor))
        return tuple(entries)

    def _width(self, name: str, factor: object) -> int:
        """
        Ghost width of one factor axis.

        Description
        -----------
        ``ConstantSpace`` and coefficient factors structurally carry
        width 0 (their halo exchange is skipped by construction);
        other factors read the negotiated per-name width, with names
        outside the spec carrying 0.
        """
        if isinstance(factor, ConstantSpace | CoefficientSpace):
            return 0
        try:
            return self._halo[name]
        except KeyError:
            return 0

    def _n_shards(self, name: str, factor: object,
                  layout: Layout) -> int:
        """
        Return the number of shards of one factor axis in `layout`.

        Description
        -----------
        1 unless the name is layout-mapped to a multi-device axis
        and the factor is ghost-shardable: ``ConstantSpace`` factors
        are replicated in every layout and coefficient factors stay
        device-local (iteration-1 LOCAL preference), so both are
        never blocked.
        """
        axis = dict(layout.device_axes).get(name)
        if axis is None:
            return 1
        if isinstance(factor, ConstantSpace | CoefficientSpace):
            return 1
        return int(self._device_mesh.shape[axis])

    @staticmethod
    def _cells_per_shard(factor: object, shards: int) -> int:
        r"""
        Uniform per-shard cell capacity of a blocked factor axis.

        Description
        -----------
        ``ceil(n_cells / shards)`` — every block carries this many cell
        slots, so a cell count that does not divide the device count
        pads to a uniform storage shape (04_decomposition.md section 5).
        Blocks are cell-aligned, so the last shard is the short one,
        holding ``n_cells - (shards - 1) * cells`` true cells with its
        surplus slots inert like ghosts; this generalizes the staggered
        pair's ``cells + 1`` capacity to arbitrary surplus. **Heavy**
        padding that would empty a trailing shard
        (``(shards - 1) * cells >= n_cells``) is rejected — negotiation
        (``_shardable_names``) precludes it, and a hand-built
        decomposition fails loudly here rather than corrupting a
        zero-length shard.
        """
        n_cells = getattr(factor.mesh, "n_cells", None)
        if n_cells is None:
            raise ValueError(
                f"cannot block {factor!r} over {shards} devices: the "
                "mesh cell count must exist "
                "(negotiation is expected to preclude this)")
        cells = -(-n_cells // shards)
        if (shards - 1) * cells >= n_cells:
            raise ValueError(
                f"cannot block {factor!r} over {shards} devices: the "
                f"padding is too heavy ({n_cells} cells over {shards} "
                "shards empties a trailing shard; negotiation is "
                "expected to preclude this)")
        return cells

    @staticmethod
    def _block_bounds(n: int, shards: int,
                      cells: int) -> tuple[int, ...]:
        """
        Global true-DOF block boundaries of a blocked axis.

        Description
        -----------
        Shard ``s`` owns DOFs ``[bounds[s], bounds[s + 1])``: blocks
        are aligned by cells (``s * cells``), so staggered pairs
        share one frame; the last shard absorbs the staggered
        surplus/deficit (n vs cells * shards).
        """
        bounds = [min(s * cells, n) for s in range(shards)]
        bounds.append(n)
        return tuple(bounds)

    def _geometry(
        self, space: SpaceLike, layout: Layout,
    ) -> tuple[tuple[str, int, object, int, int, int, int], ...]:
        """
        Return the cached axis geometry of `space` under `layout`.

        Description
        -----------
        One record per storage axis:
        ``(name, n, factor, shards, width, block, total)`` — the
        ``_axis_entries`` pairing joined with ``_axis_storage``.
        Cached on the interned ``(space, layout)`` key, so the
        per-factor Python loops run once per static structure
        instead of once per operator application.

        Parameters
        ----------
        space : SpaceLike
            The (product) space.
        layout : Layout
            The resolved layout (a member of ``self.layouts``).

        Returns
        -------
        tuple[tuple[str, int, object, int, int, int, int], ...]
            The per-axis geometry records.
        """
        key = (space, layout)
        cached = self._geometry_cache.get(key)
        if cached is None:
            cached = tuple(
                (name, n, factor,
                 *self._axis_storage(name, n, factor, layout))
                for name, n, factor in self._axis_entries(space))
            self._geometry_cache[key] = cached
        return cached

    def _axis_storage(
        self, name: str, n: int, factor: object, layout: Layout,
    ) -> tuple[int, int, int, int]:
        """
        Storage geometry of one factor axis under `layout`.

        Returns
        -------
        tuple[int, int, int, int]
            ``(shards, width, block, total)``: shard count, ghost
            width, per-shard block length, global storage length.
            An unblocked axis is one block of ``n + 2 * width``.
        """
        width = self._width(name, factor)
        shards = self._n_shards(name, factor, layout)
        if shards == 1:
            return 1, width, n + 2 * width, n + 2 * width
        cells = self._cells_per_shard(factor, shards)
        # capacity cells + 1 covers the staggered n + 1 spaces so
        # every space of the mesh shares one uniform block length
        block = cells + 1 + 2 * width
        return shards, width, block, shards * block

    def _local_reblock(
        self,
        space: SpaceLike,
        layout: Layout,
    ) -> _ReblockPlan | None:
        """
        Return the cached shard-local re-blocking plan, or None.

        Description
        -----------
        The plan drives the collective-free ``pad``/``unpad`` paths:
        semantically, block ``s`` of a blocked axis holds exactly the
        true DOFs ``[s * cells, (s + 1) * cells)``, so re-blocking is
        device-local — but expressed as global ``jnp`` slicing the
        SPMD partitioner cannot see that and inserts all-to-alls.
        The plan makes the locality explicit for ``jax.shard_map``:
        the blocked-axes ``PartitionSpec``, the **per-shard**
        ``jnp.pad`` widths (true piece -> block) and interior slices
        (block -> true piece) of every storage axis, and the two
        shard-mapped callables realizing them (``_ReblockPlan``).
        The callables are built once per plan and jit-wrapped, so
        eager ``pad``/``unpad`` calls hit jax's tracing cache instead
        of re-tracing a fresh closure on every call (compile-count
        stability: a warmed model re-run must add zero compiles).

        ``shard_map`` requires equal per-shard shapes, so the plan
        runs it on a uniform frame: for a **divisible** blocked axis
        the true shape already is uniform (``n == shards * cells``);
        for a **non-divisible** (padded-even) axis the shard_map runs
        on the padded-true frame ``shards * cells`` (evenly split, the
        last shard's surplus cells inert), and the plan's callables
        wrap a trailing trim (``true <-> padded-true``) around it. The
        divisible **staggered** surplus/deficit spaces
        (``n == shards * cells +- 1`` on an evenly-divisible cell
        count) still return None — the ``+1`` capacity cannot be
        expressed as a uniform per-shard slice — and, with fully
        unblocked geometries, fall back to the global re-blocking
        path.

        Parameters
        ----------
        space : SpaceLike
            The (product) space.
        layout : Layout
            The resolved layout (a member of ``self.layouts``).

        Returns
        -------
        _ReblockPlan | None
            The plan, or None when the shard-local path does not
            apply.
        """
        key = (space, layout)
        if key in self._reblock_cache:
            return self._reblock_cache[key]
        plan = self._build_local_reblock(space, layout)
        self._reblock_cache[key] = plan
        return plan

    def _build_local_reblock(
        self,
        space: SpaceLike,
        layout: Layout,
    ) -> _ReblockPlan | None:
        """Compute the ``_local_reblock`` plan (uncached)."""
        axes = dict(layout.device_axes)
        geometry = self._geometry(space, layout)
        spec: list[str | None] = [None] * len(geometry)
        widths: list[tuple[int, int]] = []
        slices: list[slice] = []
        outer_pad: list[tuple[int, int]] = []
        outer_slice: list[slice] = []
        blocked = False
        cell_padded = False
        for axis, (name, n, factor, shards, width, block,
                   _) in enumerate(geometry):
            if shards == 1:
                # per-shard == global on an unblocked axis; the
                # trailing side absorbs the stagger padding
                widths.append((width, block - n - width))
                slices.append(slice(width, width + n))
                outer_pad.append((0, 0))
                outer_slice.append(slice(0, n))
                continue
            cells = block - 1 - 2 * width
            n_cells = getattr(factor.mesh, "n_cells", None)
            divisible = n_cells is not None and n_cells % shards == 0
            # A divisible axis is shard-local only in the uniform case
            # (``n == shards * cells``); a non-divisible mild axis runs
            # the shard_map on the padded-true frame ``shards * cells``
            # (the last shard's surplus cells inert) and wraps a
            # trailing trim. The divisible staggered surplus
            # (``n == shards * cells + 1``) needs ``n > shards * cells``
            # and falls back to the global path. Heavy padding is
            # already excluded upstream (_cells_per_shard, negotiation).
            if n > shards * cells or (divisible and n != shards * cells):
                return None
            surplus = shards * cells - n
            cell_padded = cell_padded or bool(surplus)
            blocked = True
            spec[axis] = axes[name]
            widths.append((width, block - width - cells))
            slices.append(slice(width, width + cells))
            outer_pad.append((0, surplus))
            outer_slice.append(slice(0, n))
        if not blocked:
            return None
        pspec = jax.sharding.PartitionSpec(*spec)
        pad_widths = tuple(widths)
        true_slices = tuple(slices)

        def scatter_local(piece: jax.Array) -> jax.Array:
            return jnp.pad(piece, pad_widths)

        def gather_local(block: jax.Array) -> jax.Array:
            return block[true_slices]

        scatter = jax.jit(jax.shard_map(
            scatter_local, mesh=self._device_mesh,
            in_specs=pspec, out_specs=pspec))
        gather = jax.jit(jax.shard_map(
            gather_local, mesh=self._device_mesh,
            in_specs=pspec, out_specs=pspec))
        if not cell_padded:
            # divisible geometry: no trailing trim, so pad/unpad drive
            # the shard_map callables directly (byte-for-byte identical
            # to the pre-padding path)
            return _ReblockPlan(pspec, pad_widths, true_slices,
                                scatter, gather)
        # padded-even: wrap the shard_map callables with the trailing
        # trim (true -> padded-true before scatter; padded-true -> true
        # after gather). The wrappers are built once and jit-cached, so
        # eager pad/unpad stays compile-count stable.
        pre_pad = tuple(outer_pad)
        post_slice = tuple(outer_slice)

        def scatter_padded(piece: jax.Array) -> jax.Array:
            return scatter(jnp.pad(piece, pre_pad))

        def gather_padded(block: jax.Array) -> jax.Array:
            return gather(block)[post_slice]

        return _ReblockPlan(pspec, pad_widths, true_slices,
                            jax.jit(scatter_padded),
                            jax.jit(gather_padded))

    # ================================================================
    #  Shapes and shardings
    # ================================================================

    def sharding(
        self,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.sharding.Sharding:
        """Return the sharding of `space`'s storage under `layout`."""
        layout = self._resolve_layout(space, layout)
        key = (space, layout)
        cached = self._sharding_cache.get(key)
        if cached is not None:
            return cached
        axes = dict(layout.device_axes)
        spec = []
        for name, _, factor in self._axis_entries(space):
            if isinstance(factor, ConstantSpace | CoefficientSpace):
                spec.append(None)
            else:
                spec.append(axes.get(name))
        sharding = jax.sharding.NamedSharding(
            self._device_mesh, jax.sharding.PartitionSpec(*spec))
        self._sharding_cache[key] = sharding
        return sharding

    def local_slice(
        self,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> tuple[slice, ...]:
        """
        Return the global true-DOF index range of the local shard.

        Description
        -----------
        Single-controller jax addresses the global array from every
        process, so the "local" shard is the whole true extent for
        any device count (a multi-host backend would return the
        per-host range). Per-DOF keying over these indices is
        therefore device-count invariant by construction.
        """
        self._resolve_layout(space, layout)
        return tuple(slice(0, n) for n in space.shape)

    def storage_shape(
        self,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> tuple[int, ...]:
        """Return the global storage shape (true + halo + padding)."""
        layout = self._resolve_layout(space, layout)
        return tuple(
            total for *_, total in self._geometry(space, layout))

    # ================================================================
    #  Storage construction and views
    # ================================================================

    def zeros(
        self,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.Array:
        """Return a zero-filled, sharded, storage-shaped array."""
        layout = self._resolve_layout(space, layout)
        arr = jnp.zeros(self.storage_shape(space, layout))
        return jax.device_put(arr, self.sharding(space, layout))

    def pad(
        self,
        arr: jax.Array,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.Array:
        """
        Map true-shape data to halo/stagger-padded storage.

        Description
        -----------
        With a shard-local plan (``_local_reblock``, covering the
        divisible-uniform and the non-divisible padded-even axes) this
        is one ``jax.shard_map`` region padding every shard's true
        piece into its block locally — zero collectives. Unblocked
        geometries pad globally; the divisible staggered spaces fall
        back to the global per-block re-assembly.
        """
        layout = self._resolve_layout(space, layout)
        if tuple(arr.shape) != tuple(space.shape):
            raise ValueError(
                f"pad expects a true-shape array {space.shape}, "
                f"got {tuple(arr.shape)}")
        plan = self._local_reblock(space, layout)
        if plan is not None:
            return jax.device_put(
                plan.scatter(arr), self.sharding(space, layout))
        widths = []
        blocked = []
        for axis, (_name, n, factor, shards, width, block,
                   _) in enumerate(self._geometry(space, layout)):
            if shards == 1:
                # trailing side absorbs the stagger padding (zero on
                # a one-shard axis)
                widths.append((width, block - n - width))
            else:
                widths.append((0, 0))
                blocked.append((axis, n, factor, shards, width, block))
        padded = jnp.pad(arr, widths)
        for axis, n, factor, shards, width, block in blocked:
            padded = self._scatter_axis(
                padded, axis, n, factor, shards, width, block)
        return jax.device_put(padded, self.sharding(space, layout))

    def _scatter_axis(
        self,
        arr: jax.Array,
        axis: int,
        n: int,
        factor: object,
        shards: int,
        width: int,
        block: int,
    ) -> jax.Array:
        """Map a true axis to a blocked one (ghost slots zeroed)."""
        cells = self._cells_per_shard(factor, shards)
        bounds = self._block_bounds(n, shards, cells)
        pieces = []
        for s in range(shards):
            piece = _take(arr, axis, slice(bounds[s], bounds[s + 1]))
            pads = [(0, 0)] * arr.ndim
            pads[axis] = (width,
                          block - width - (bounds[s + 1] - bounds[s]))
            pieces.append(jnp.pad(piece, pads))
        return jnp.concatenate(pieces, axis=axis)

    def _gather_axis(
        self,
        arr: jax.Array,
        axis: int,
        n: int,
        factor: object,
        shards: int,
        width: int,
        block: int,
    ) -> jax.Array:
        """Blocked storage axis -> true axis (pads dropped)."""
        cells = self._cells_per_shard(factor, shards)
        bounds = self._block_bounds(n, shards, cells)
        pieces = []
        for s in range(shards):
            start = s * block + width
            pieces.append(_take(
                arr, axis,
                slice(start, start + bounds[s + 1] - bounds[s])))
        return jnp.concatenate(pieces, axis=axis)

    def unpad(
        self,
        arr: jax.Array,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.Array:
        """
        Map padded storage to true-shape data (pads dropped).

        Description
        -----------
        With a shard-local plan (``_local_reblock``, covering the
        divisible-uniform and the non-divisible padded-even axes) this
        is one ``jax.shard_map`` region slicing every block's interior
        locally — zero collectives on the perf-critical (center /
        outer) spaces; the divisible output is evenly sharded on the
        blocked axes (shard ``s`` holds exactly its block's true DOFs,
        so a following ``pad`` stays local too). Unblocked geometries
        slice globally; the divisible staggered spaces fall back to
        the global per-block gather.
        """
        layout = self._resolve_layout(space, layout)
        storage = self.storage_shape(space, layout)
        if tuple(arr.shape) != storage:
            raise ValueError(
                f"unpad expects a storage-shaped array {storage}, "
                f"got {tuple(arr.shape)}")
        plan = self._local_reblock(space, layout)
        if plan is not None:
            return plan.gather(arr)
        slices = []
        for axis, (_name, n, factor, shards, width, block,
                   _) in enumerate(self._geometry(space, layout)):
            if shards == 1:
                slices.append(slice(width, width + n))
            else:
                arr = self._gather_axis(
                    arr, axis, n, factor, shards, width, block)
                slices.append(slice(0, n))
        return arr[tuple(slices)]

    # ================================================================
    #  Data movement
    # ================================================================

    def sync(
        self,
        arr: jax.Array,
        space: SpaceLike,
        *,
        layout: Layout | None = None,
        fills: Mapping[str, jax.Array] | None = None,
    ) -> jax.Array:
        """
        Fill halos (see ``Decomposition.sync``).

        Description
        -----------
        Unsharded axes are filled locally (the sanctioned static
        branch skips the exchange entirely) — periodic wrap on
        periodic mesh factors, the BC-structured homogeneous fill on
        bounded ones. Sharded axes exchange interior shard edges via
        ``jax.shard_map`` + ``jax.lax.ppermute``, with the same local
        fill as the physical-boundary edge path. Width-0 axes are
        skipped structurally; an all-width-0 space is returned
        unchanged. Local fills run first so the exchanged edges carry
        valid corner ghosts.
        """
        layout = self._resolve_layout(space, layout)
        if fills is not None:
            raise NotImplementedError(
                "inhomogeneous ghost fill is designed-for; "
                "iteration 1 is homogeneous only")
        exchanged = []
        for axis, (name, n, factor, shards, width, _,
                   _) in enumerate(self._geometry(space, layout)):
            if not width:
                continue
            if shards == 1:
                arr = _fill_axis(arr, axis, n, width, factor)
            else:
                exchanged.append((axis, name, n, factor, width))
        for axis, name, n, factor, width in exchanged:
            arr = self._exchange_axis(
                arr, axis, name, n, factor, width, layout)
        return arr

    def _exchange_axis(
        self,
        arr: jax.Array,
        axis: int,
        name: str,
        n: int,
        factor: object,
        width: int,
        layout: Layout,
    ) -> jax.Array:
        """
        Halo exchange along one sharded axis (multi-device).

        Description
        -----------
        A ``jax.shard_map`` region over the 1-D device mesh: interior
        shard edges exchange via ``jax.lax.ppermute`` (a ring on
        periodic meshes, a chain on bounded ones); the physical
        boundary blocks of a bounded mesh apply the BC-structured
        local fill, computed SPMD on every shard and masked in by
        ``jax.lax.axis_index``.
        """
        axis_name = dict(layout.device_axes)[name]
        shards = int(self._device_mesh.shape[axis_name])
        cells = self._cells_per_shard(factor, shards)
        spec = [None] * arr.ndim
        spec[axis] = axis_name
        pspec = jax.sharding.PartitionSpec(*spec)

        def exchange(block: jax.Array) -> jax.Array:
            return _exchange_block(
                block, axis, n, width, factor,
                shards=shards, cells=cells, axis_name=axis_name)

        return jax.shard_map(
            exchange, mesh=self._device_mesh,
            in_specs=pspec, out_specs=pspec)(arr)

    def patch_physical_ends(
        self,
        out_arr: jax.Array,
        in_arr: jax.Array,
        out_space: SpaceLike,
        in_space: SpaceLike,
        axis: str,
        patch: Callable[..., jax.Array],
        *,
        layout: Layout | None = None,
    ) -> jax.Array:
        """
        Overwrite the physical-wall ends of a reconstructed axis.

        Description
        -----------
        See ``Decomposition.patch_physical_ends``. ``shards == 1`` is a
        static branch: the callback runs directly on both walls of the
        single block with ``t = n`` (bitwise-identical to the
        undistributed operator). ``shards >= 2`` runs one
        ``jax.shard_map`` co-sharding ``in_arr`` and ``out_arr`` on the
        axis's device-mesh axis; ``s = jax.lax.axis_index`` locates the
        shard, the last shard absorbs the staggered true-count deficit
        (the ``_exchange_block`` idiom), both wall patches are computed
        on every shard, and each is masked onto its boundary shard.
        """
        layout = self._resolve_layout(out_space, layout)
        out_axis = out_space.names.index(axis)
        in_axis = in_space.names.index(axis)
        _, n_out, _, shards, width_out, _, _ = self._geometry(
            out_space, layout)[out_axis]
        _, n_in, factor, _, width_in, _, _ = self._geometry(
            in_space, layout)[in_axis]

        if shards == 1:
            out_arr = patch(in_arr, out_arr, 0, width_in, n_in,
                            width_out, n_out)
            return patch(in_arr, out_arr, 1, width_in, n_in,
                         width_out, n_out)

        axis_name = dict(layout.device_axes)[axis]
        cells = self._cells_per_shard(factor, shards)
        # the right wall lives on the last shard; if a wall space's last
        # shard holds no true DOF (an n_cells-1 space whose padded last
        # shard empties) the masked right patch would be silently
        # dropped -- fail loudly instead. Negotiation precludes this
        # (its last-shard >= width + 1 check keeps every wall shard
        # non-empty); only a hand-built decomposition can reach here.
        if min(n_out, n_in) - (shards - 1) * cells < 1:
            raise ValueError(
                f"cannot patch physical ends of {axis!r} over {shards} "
                f"shards: a wall space's last shard holds no true DOF "
                f"(n_out={n_out}, n_in={n_in}, cells={cells}); "
                "negotiation is expected to preclude this")
        in_spec = [None] * in_arr.ndim
        in_spec[in_axis] = axis_name
        out_spec = [None] * out_arr.ndim
        out_spec[out_axis] = axis_name
        in_pspec = jax.sharding.PartitionSpec(*in_spec)
        out_pspec = jax.sharding.PartitionSpec(*out_spec)

        def body(in_block: jax.Array,
                 out_block: jax.Array) -> jax.Array:
            s = jax.lax.axis_index(axis_name)
            t_in = jnp.where(s == shards - 1,
                             n_in - (shards - 1) * cells, cells)
            t_out = jnp.where(s == shards - 1,
                              n_out - (shards - 1) * cells, cells)
            left = patch(in_block, out_block, 0, width_in, t_in,
                         width_out, t_out)
            right = patch(in_block, out_block, 1, width_in, t_in,
                          width_out, t_out)
            out_block = jnp.where(s == 0, left, out_block)
            return jnp.where(s == shards - 1, right, out_block)

        return jax.shard_map(
            body, mesh=self._device_mesh,
            in_specs=(in_pspec, out_pspec), out_specs=out_pspec)(
                in_arr, out_arr)

    def layout_for(
        self,
        local_names: tuple[str, ...],
    ) -> Layout:
        """Return a negotiated layout keeping `local_names` local."""
        for layout in self._layouts:
            if all(layout.is_local(name) for name in local_names):
                return layout
        raise ValueError(
            f"no negotiated layout keeps {local_names} device-local; "
            f"the vocabulary is {self._layouts}")

    def _blocking(
        self, space: SpaceLike, layout: Layout,
    ) -> tuple[tuple[int, int, int], ...]:
        """Return the static blocking signature (redistribute)."""
        return tuple(
            (shards, width, block)
            for _, _, _, shards, width, block, _
            in self._geometry(space, layout))

    def redistribute(
        self,
        arr: jax.Array,
        space: SpaceLike,
        src: Layout,
        dst: Layout,
    ) -> jax.Array:
        """
        Transpose an array between two negotiated layouts.

        Description
        -----------
        When the two layouts block the storage identically, this is
        a resharding ``jax.device_put`` (XLA lowers it to
        all-to-all). Otherwise the array is re-blocked through the
        true shape — ghost slots of the result are zero until the
        next sync (``Reshard``'s post-kernel sync refills them; raw
        consumers are the width-0 transform pencils).
        """
        src = self._resolve_layout(space, src)
        dst = self._resolve_layout(space, dst)
        src_storage = self.storage_shape(space, src)
        if tuple(arr.shape) != src_storage:
            raise ValueError(
                f"redistribute expects a storage-shaped array "
                f"{src_storage} under src, got {tuple(arr.shape)}")
        if self._blocking(space, src) == self._blocking(space, dst):
            # a resharding device_put; XLA lowers it to all-to-all.
            return jax.device_put(arr, self.sharding(space, dst))
        return self.pad(self.unpad(arr, space, src), space, dst)

    def gather(
        self,
        arr: jax.Array,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.Array:
        """Gather the global true-shape array (I/O, diagnostics)."""
        layout = self._resolve_layout(space, layout)
        true = self.unpad(arr, space, layout)
        replicated = jax.sharding.NamedSharding(
            self._device_mesh,
            jax.sharding.PartitionSpec(*([None] * true.ndim)))
        return jax.device_put(true, replicated)


# ================================================================
#  Single-shard halo fill (halo/storage contract)
# ================================================================
# distance (in cell widths) from the (left, right) boundary to the
# nearest node of the set, before BC drops
_BOUNDARY_DISTANCE: dict[NodeSet, tuple[float, float]] = {
    NodeSet.CENTER: (0.5, 0.5),
    NodeSet.LEFT: (0.0, 1.0),
    NodeSet.RIGHT: (1.0, 0.0),
    NodeSet.OUTER: (0.0, 0.0),
    NodeSet.INNER: (1.0, 1.0),
}

# whether the (left, right) boundary DOF is a member of the node set
# (a Dirichlet-constrained member DOF is dropped from the space's
# shape, which moves the nearest true DOF one cell inward; Neumann
# never drops — spaces.md shape note, owner decision 2026-07-07)
_BOUNDARY_MEMBERSHIP: dict[NodeSet, tuple[bool, bool]] = {
    NodeSet.CENTER: (False, False),
    NodeSet.LEFT: (True, False),
    NodeSet.RIGHT: (False, True),
    NodeSet.OUTER: (True, True),
    NodeSet.INNER: (False, False),
}

# nearest-true-DOF distance classes of the bounded fill
_OFFSET = 0.5   # boundary between ghost and first DOF (center-like)
_VACANT = 1.0   # a lattice node sits on the boundary, ghost slot
_MEMBER = 0.0   # the boundary node is a true DOF of the space


def _take(arr: jax.Array, axis: int, index: slice) -> jax.Array:
    """Slice ``arr`` with ``index`` along ``axis``."""
    slices: list[slice] = [slice(None)] * arr.ndim
    slices[axis] = index
    return arr[tuple(slices)]


def _set(arr: jax.Array, axis: int, start: int | jax.Array,
         values: jax.Array) -> jax.Array:
    """
    Write ``values`` into ``arr`` at ``start`` along ``axis``.

    Description
    -----------
    Spelled as a ``dynamic_update_slice`` rather than ``arr.at[...]``
    (which jax lowers to a ``scatter``, and which reaches XLA's
    in-place emitter only through a rewrite that is not guaranteed).

    Used by the multi-device exchange, where the written slab is
    *received* data (a ``ppermute`` result), so the write genuinely
    delivers new values into the block. The single-shard fill does
    NOT write: its slabs are a remap of the array's own contents, and
    it spells that as an index map — see ``_fill_axis``.
    """
    return jax.lax.dynamic_update_slice_in_dim(arr, values, start, axis)


def _boundary_geometry(
    factor: FunctionSpace, side: int,
) -> tuple[BC, float]:
    """
    Classify one bounded boundary of a factor space.

    Description
    -----------
    Returns the boundary component's BC kind and the distance (in
    cell widths) from the boundary to the nearest **true** DOF —
    the quantity the BC-structured fill is keyed on. Spaces outside
    the iteration-1 families raise.

    Parameters
    ----------
    factor : FunctionSpace
        A factor space on a bounded mesh.
    side : int
        0 for the left boundary, 1 for the right.

    Returns
    -------
    tuple[BC, float]
        The (bc kind, nearest-true-DOF distance) pair.
    """
    kind = factor.bc.components[side]
    if isinstance(factor, NodalSpace):
        node_set = factor.node_set
        if node_set not in _BOUNDARY_DISTANCE:
            raise NotImplementedError(
                f"no bounded halo fill for {factor!r}")
        distance = _BOUNDARY_DISTANCE[node_set][side]
        if (_BOUNDARY_MEMBERSHIP[node_set][side]
                and kind is BC.DIRICHLET):
            # the Dirichlet-constrained boundary DOF is dropped from
            # the space (Neumann keeps it: it stays a true DOF)
            distance += 1.0
        return kind, distance
    if isinstance(factor, CellAvg):
        return kind, _OFFSET
    if isinstance(factor, FaceAvg):
        return kind, _VACANT
    raise NotImplementedError(
        f"no bounded halo fill for {factor!r}")


def _bounded_ghosts(
    true: jax.Array,
    axis: int,
    n: int,
    width: int,
    factor: FunctionSpace,
    side: int,
) -> jax.Array | None:
    """
    Ghost values of one bounded boundary (BC-structured fill).

    Description
    -----------
    Realizes the halo/storage contract's bounded-edge fill:
    Dirichlet-structured spaces get the odd (zero-value) extension,
    Neumann-structured spaces the even (mirror) extension, and
    BC-free sides return **None** — exterior values are undefined
    (R1, boundary_plan.md) and the caller leaves the slots
    untouched. Blanket zero-fill is rejected by design. Ghost slot
    ``k`` counts outward from the true region (k = 1 is adjacent to
    the first true DOF).

    Parameters
    ----------
    true : jax.Array
        The true-extent array (ghosts stripped) along ``axis``.
    axis : int
        The storage axis being filled.
    n : int
        The true DOF count along ``axis``.
    width : int
        The ghost width to fill.
    factor : FunctionSpace
        The factor space owning the axis.
    side : int
        0 for the left boundary, 1 for the right.

    Returns
    -------
    jax.Array
        The ``width`` ghost slots in storage order (outermost
        first on the left side, innermost first on the right).
    """
    kind, distance = _boundary_geometry(factor, side)
    if kind is BC.NONE:
        # R1 (boundary_plan.md 2c): a BC-free bounded side defines
        # no exterior values — nothing is filled, nothing may read
        # the slots (exterior-needing operator rows do not exist)
        return None

    def dof(k: int) -> jax.Array:
        """Return the k-th true DOF from this side (k=1 nearest)."""
        if k > n:
            raise NotImplementedError(
                f"the bounded halo fill of width {width} reaches "
                f"deeper than the {n} DOFs along the axis")
        index = k - 1 if side == 0 else n - k
        return _take(true, axis, slice(index, index + 1))

    ghosts = _ghost_values(kind, distance, dof, width, factor)
    if side == 0:
        ghosts.reverse()
    return jnp.concatenate(ghosts, axis=axis)


def _ghost_slot(
    kind: BC,
    distance: float,
    k: int,
    factor: FunctionSpace,
) -> tuple[int, int]:
    """
    Return the (true-DOF rank, sign) one ghost slot reads.

    Description
    -----------
    The single source of truth for the BC-structured fill: ghost slot
    ``k`` (k = 1 adjacent to the true region, counting outward) takes
    ``sign`` times the ``rank``-th true DOF counted from the same
    side. Dirichlet-structured sides get the odd (zero-value)
    extension (``sign = -1``), Neumann-structured sides the even
    (mirror) extension (``sign = +1``). ``sign = 0`` marks the slot
    the Dirichlet ``_VACANT`` geometry sets to exact zero — the ghost
    slot *is* the constrained boundary DOF — and its rank is then
    meaningless. BC-free sides never reach this helper (the callers
    return early: R1, exterior values are undefined). ``BC.ROBIN``
    fills are data-parameterized and raise, pointing at the stage-2e
    ``("ghost_fill", space)`` data path.

    Both the single-shard fill (``_axis_map``) and the multi-device
    physical-boundary fill (``_ghost_values`` under ``_exchange_block``)
    read the table from here, so the two paths cannot drift.

    Parameters
    ----------
    kind : BC
        The boundary component's BC kind.
    distance : float
        The nearest-true-DOF distance class (``_boundary_geometry``).
    k : int
        The ghost slot, counting outward from the true region (k = 1
        is adjacent to the first true DOF).
    factor : FunctionSpace
        The factor space owning the axis (error messages only).

    Returns
    -------
    tuple[int, int]
        The (rank, sign) the slot reads; sign 0 means "exact zero".
    """
    if kind is BC.NONE:  # pragma: no cover — guarded by the callers
        raise NotImplementedError(
            "BC-free bounded sides have no ghost fill (R1, "
            "boundary_plan.md): exterior values are undefined")
    if kind is BC.DIRICHLET and distance == _OFFSET:
        return k, -1
    if kind is BC.DIRICHLET:
        # _VACANT: the ghost slot k = 1 IS the (zero) boundary DOF;
        # deeper slots odd-reflect about it
        return (1, 0) if k == 1 else (k - 1, -1)
    if kind is BC.NEUMANN and distance == _OFFSET:
        return k, 1
    if kind is BC.NEUMANN and distance == _MEMBER:
        # the boundary node is a true DOF (Neumann keeps it): the
        # even/mirror extension reflects about that node, which is
        # excluded from the reflection — ghost slot k mirrors the
        # interior node k cells inside, i.e. dof(k + 1)
        # (decomposition.md even-extension contract)
        return k + 1, 1
    if kind is BC.ROBIN:
        raise NotImplementedError(
            "Robin ghost fills are data-parameterized (alpha, g are "
            "dynamic) and arrive with the ('ghost_fill', space) "
            "data path — boundary_plan.md stage 2e; Robin "
            "derivatives are supported flux-form")
    raise NotImplementedError(
        "the Neumann (even) fill is grounded for node sets whose "
        "nearest DOF is boundary-offset or on the boundary; got "
        f"{factor!r} with a vacant lattice node on the boundary")


def _ghost_values(
    kind: BC,
    distance: float,
    dof: Callable[[int], jax.Array],
    width: int,
    factor: FunctionSpace,
) -> list[jax.Array]:
    """
    Return one boundary's ghost values, innermost slot first.

    Description
    -----------
    The value form of the ``_ghost_slot`` table, used by the
    multi-device physical-boundary fill (a shard's block is filled
    from received/local slabs, not by remapping an axis).

    Parameters
    ----------
    kind : BC
        The boundary component's BC kind.
    distance : float
        The nearest-true-DOF distance class (``_boundary_geometry``).
    dof : Callable[[int], jax.Array]
        Accessor for the k-th true DOF from this side (k=1 nearest).
    width : int
        The ghost width to fill.
    factor : FunctionSpace
        The factor space owning the axis (error messages only).

    Returns
    -------
    list[jax.Array]
        The ``width`` ghost slices, adjacent-to-true first.
    """
    values = []
    for k in range(1, width + 1):
        rank, sign = _ghost_slot(kind, distance, k, factor)
        if not sign:
            values.append(jnp.zeros_like(dof(rank)))
        elif sign < 0:
            values.append(-dof(rank))
        else:
            values.append(dof(rank))
    return values


def _axis_map(
    size: int,
    n: int,
    width: int,
    factor: FunctionSpace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return the ghost fill of one storage axis as a static index map.

    Description
    -----------
    Returns ``(src, neg, zero)``, each of length ``size``: storage
    slot ``i`` of the filled axis takes ``arr[src[i]]``, negated where
    ``neg[i]``, replaced by exact zero where ``zero[i]``. True slots
    map to themselves, and so do the ghost slots of a BC-free side
    (R1: exterior values are undefined, the slot keeps what it holds).

    Pure host-side structure — the arrays are jit constants.

    Parameters
    ----------
    size : int
        The storage extent of the axis (ghosts + true + padding).
    n : int
        The true DOF count along the axis.
    width : int
        The negotiated ghost width along the axis.
    factor : FunctionSpace
        The factor space owning the axis (mesh topology, BC structure).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        The (source index, negate mask, zero mask) of the axis.
    """
    src = np.arange(size, dtype=np.int32)
    neg = np.zeros(size, dtype=bool)
    zero = np.zeros(size, dtype=bool)
    trail = size - n - width
    if factor.mesh.periodic:
        if width > n or trail > n:
            raise NotImplementedError(
                f"periodic wrap with halo {width} wider than the "
                f"axis length {n} is not supported")
        src[:width] = np.arange(n, n + width)
        src[width + n:] = np.arange(width, width + trail)
        return src, neg, zero
    for side, depth in ((0, width), (1, trail)):
        if not depth:
            continue
        kind, distance = _boundary_geometry(factor, side)
        if kind is BC.NONE:  # R1: nothing is filled (identity)
            continue
        for k in range(1, depth + 1):
            slot = width - k if side == 0 else width + n + k - 1
            rank, sign = _ghost_slot(kind, distance, k, factor)
            if not sign:
                zero[slot] = True
                continue
            if rank > n:
                raise NotImplementedError(
                    f"the bounded halo fill of width {depth} reaches "
                    f"deeper than the {n} DOFs along the axis")
            src[slot] = width + (rank - 1 if side == 0 else n - rank)
            neg[slot] = sign < 0
    return src, neg, zero


def _fill_axis(
    arr: jax.Array,
    axis: int,
    n: int,
    width: int,
    factor: FunctionSpace,
) -> jax.Array:
    """
    Fill the ghost slots of one storage axis (single shard).

    Description
    -----------
    Periodic mesh factors wrap; bounded ones get the BC-structured
    fill per side. The trailing storage side also absorbs the
    stagger padding, which on a single shard equals the ghost
    width, so every non-true slot is (re)written.

    The fill is spelled as an **index map** (``_axis_map`` + one
    ``take``), because the ghost fill is one: every filled slot reads
    exactly one slot of the same array, up to a sign. That is the
    whole reason for the spelling, and it is a performance contract,
    not a stylistic one — a ghost fill is almost never *executed* on
    its own. XLA fuses it into whatever kernel consumes the field, so
    what the fill costs is what it costs to **re-derive one filled
    element inside the consumer's loop**; a gather is one indexed load
    and stays cheap under fusion.

    The two spellings this replaced both lose, and they lose in
    opposite regimes (ms/step, 1 A100, flat nonhydro 256^3):

    - ``concatenate`` rebuilds the array. XLA makes it a fusion ROOT,
      so where a consumer does not absorb it the rebuild materializes
      a fresh O(field) buffer — every synced field round-trips through
      HBM. Linear step 7.46, advective 10.61.
    - ``dynamic_update_slice`` writes the ghosts in place, which is
      near-free *when it is materialized*. But XLA does not materialize
      it: it absorbs the write into the consuming kernel, and a chain
      of 2 * ndim in-place writes is expensive to **re-derive**, since
      each DUS is an index-conditional read of the previous one and a
      consumer re-evaluates that nest at every offset it reads. On the
      linear step, whose consumers are narrow, this wins (6.04); on the
      advective step, whose stencils are wide, two tendency fusions
      swallow 42 DUS and cost 2.7 ms each — 11.03, worse than the
      concatenate it replaced. It is not a launch-count effect: the DUS
      build launches FEWER kernels (47 vs 81 top-level ops per step).

    The map is the only spelling that is cheap in both regimes — it is
    absorbed like the DUS (no HBM round-trip: linear 6.03) and costs
    one indexed load to re-derive like the concatenate (advective
    9.36). So do not "restore" the in-place write: it is a local
    optimum that is only cheap while nothing fuses it, and what fuses
    it is not a property this function can see.

    Parameters
    ----------
    arr : jax.Array
        The storage-shaped array.
    axis : int
        The storage axis to fill.
    n : int
        The true DOF count along ``axis``.
    width : int
        The negotiated ghost width along ``axis``.
    factor : FunctionSpace
        The factor space owning the axis (supplies mesh topology
        and BC structure).
    """
    size = arr.shape[axis]
    src, neg, zero = _axis_map(size, n, width, factor)
    if not (neg.any() or zero.any()
            or (src != np.arange(size, dtype=np.int32)).any()):
        return arr  # identity map: both sides are BC-free (R1)

    # "clip" only to keep jnp.take from emitting the out-of-bounds
    # select of its default "fill" mode; the map is in bounds by
    # construction, so the clamp folds into the (constant) indices
    out = jnp.take(arr, jnp.asarray(src), axis=axis, mode="clip")
    shape = [1] * arr.ndim
    shape[axis] = size
    if neg.any():
        out = jnp.where(jnp.asarray(neg).reshape(shape), -out, out)
    if zero.any():
        # a select, not a multiply by 0.0: the Dirichlet _VACANT slot
        # is exactly +0.0, and 0.0 * x is -0.0 for negative x
        out = jnp.where(jnp.asarray(zero).reshape(shape),
                        jnp.zeros((), out.dtype), out)
    return out


# ================================================================
#  Multi-device halo exchange (one shard_map region per axis)
# ================================================================
def _exchange_block(
    block: jax.Array,
    axis: int,
    n: int,
    width: int,
    factor: FunctionSpace,
    *,
    shards: int,
    cells: int,
    axis_name: str,
) -> jax.Array:
    """
    Per-shard body of the sharded-axis halo exchange.

    Description
    -----------
    Runs SPMD under ``jax.shard_map``. The local block along ``axis``
    is ``[width ghosts | t true DOFs | trailing ghosts + stagger
    padding]`` with ``t`` shard-dependent (the last shard absorbs the
    staggered surplus/deficit, located via ``jax.lax.axis_index``).
    Interior edges: every shard sends its first/last ``width`` true
    values to its neighbors via ``jax.lax.ppermute``. Physical
    boundaries of a bounded mesh: the BC-structured fill is computed
    from an edge buffer on every shard (static shapes; the buffer may
    reach into the already-exchanged left ghosts on a short last
    shard) and masked onto the boundary shards.

    Parameters
    ----------
    block : jax.Array
        The local shard of the storage array.
    axis : int
        The storage axis being exchanged.
    n : int
        The global true DOF count along ``axis``.
    width : int
        The negotiated ghost width along ``axis``.
    factor : FunctionSpace
        The factor space owning the axis.
    shards : int
        The device count along the mesh axis.
    cells : int
        Mesh cells per shard (blocks are cell-aligned).
    axis_name : str
        The device-mesh axis name.

    Returns
    -------
    jax.Array
        The local block with valid ghost slots.
    """
    s = jax.lax.axis_index(axis_name)
    # per-shard true count: cells except on the last (staggered) shard
    t = jnp.where(s == shards - 1, n - (shards - 1) * cells, cells)
    periodic = bool(factor.mesh.periodic)

    # ---- interior edges: ppermute exchange ------------------------
    left_send = _take(block, axis, slice(width, 2 * width))
    right_send = jax.lax.dynamic_slice_in_dim(block, t, width, axis)
    if periodic:
        fwd = [(i, (i + 1) % shards) for i in range(shards)]
        bwd = [(i, (i - 1) % shards) for i in range(shards)]
    else:
        fwd = [(i, i + 1) for i in range(shards - 1)]
        bwd = [(i, i - 1) for i in range(1, shards)]
    from_left = jax.lax.ppermute(right_send, axis_name, fwd)
    from_right = jax.lax.ppermute(left_send, axis_name, bwd)
    block = _set(block, axis, 0, from_left)
    block = _set(block, axis, width + t, from_right)

    # ---- physical boundaries: BC-structured local fill ------------
    if not periodic:
        # the boundary-member Neumann mirror reaches one node past
        # the width (dof(width + 1))
        depth = width + 1
        lead = _take(block, axis, slice(width, width + depth))
        left_fill = _bounded_ghosts(lead, axis, depth, width,
                                    factor, 0)
        if left_fill is not None:  # BC-free side: slots stay (R1)
            # the shard mask selects on the ghost slab, not on the
            # whole block: masking the block would build a fresh
            # O(block) buffer on every shard, boundary or not, and
            # would put a second reader on the buffer the write wants
            # in place (_set)
            keep = _take(block, axis, slice(0, width))
            block = _set(block, axis, 0,
                         jnp.where(s == 0, left_fill, keep))
        trail_buf = jax.lax.dynamic_slice_in_dim(
            block, width + t - depth, depth, axis)
        right_fill = _bounded_ghosts(trail_buf, axis, depth, width,
                                     factor, 1)
        if right_fill is not None:
            keep = jax.lax.dynamic_slice_in_dim(
                block, width + t, width, axis)
            block = _set(block, axis, width + t,
                         jnp.where(s == shards - 1, right_fill, keep))
    return block
