"""
The jax-sharding decomposition of tensor-product grids.

Description
-----------
Owning class doc: ``notes/framework2/classes/decomposition.md``.
Iteration 1 covers the **single-device** case: one code path with a
one-device ``jax.make_mesh`` (no separate ``SingleDecomposition``;
``jax.sharding`` degrades gracefully). Multi-device layouts, the
``shard_map`` + ``ppermute`` halo exchange, and negotiation arrive
in Wave 3.
"""
# Wave 1: trivial single-device TensorDecomposition --
#    Wave 3: multi-device
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.framework2.grid.decomposition.decomposition import (
    Decomposition,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.framework2.grid.decomposition.decomposition import (
        SpaceLike,
    )
    from fridom.framework2.grid.decomposition.halo import HaloSpec
    from fridom.framework2.grid.decomposition.layout import Layout


class TensorDecomposition(Decomposition):

    """
    jax.sharding-based decomposition of tensor-product grids.

    Description
    -----------
    Normally constructed by ``negotiate`` (Wave 3); the constructor
    is also the plain single-device path a grid calls directly. It
    implements every ``Decomposition`` abstract method and adds no
    public surface — solver code programs against the ABC.

    One sanctioned special case: ``sync`` branches statically on
    ``n_devices == 1`` and skips the exchange entirely (a
    Python-level branch on static structure); trait and layout
    structure are unchanged by the short-circuit.

    Parameters
    ----------
    meshes : tuple[object, ...]
        The grid's mesh factors (opaque here in iteration 1; the
        BC-structured bounded-edge fill of Wave 3 consumes them).
    names : tuple[str, ...]
        The grid's coordinate names, concatenated across meshes.
    halo : HaloSpec
        The negotiated per-name ghost widths.
    layouts : tuple[Layout, ...]
        The closed layout vocabulary; the first entry is the default
        layout. Every coordinate name a layout shards must be one of
        `names`.
    device_ids : tuple[int, ...] | None, optional
        Indices into ``jax.devices()``; None selects the first
        device (default: None). Iteration 1 supports exactly one
        device.
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
        """Build the one-device ``jax.sharding.Mesh``."""
        devices = jax.devices()
        if device_ids is None:
            selected = (devices[0],)
        else:
            selected = tuple(devices[i] for i in device_ids)
        if len(selected) != 1:
            raise NotImplementedError(
                "multi-device TensorDecomposition arrives with "
                "negotiate() in Wave 3; iteration 1 is single-device")

        # one named axis per device-mesh axis referenced by the
        # layouts (all of size 1 on the one-device mesh); a
        # placeholder axis when no layout shards anything.
        axis_names: list[str] = []
        for layout in self._layouts:
            for _, axis in layout.device_axes:
                if axis not in axis_names:
                    axis_names.append(axis)
        if not axis_names:
            axis_names = ["devices"]
        return jax.make_mesh(
            (1,) * len(axis_names), tuple(axis_names), devices=selected)

    # ================================================================
    #  Negotiated structure
    # ================================================================

    @property
    def halo(self) -> HaloSpec:
        """The negotiated per-name ghost widths."""
        return self._halo

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

    def _width(self, name: str) -> int:
        """Ghost width along `name`; names outside the spec carry 0."""
        try:
            return self._halo[name]
        except KeyError:
            return 0

    def _n_shards(self, name: str, layout: Layout) -> int:
        """Return the number of shards along `name` under `layout`."""
        axis = dict(layout.device_axes).get(name)
        if axis is None:
            return 1
        return self._device_mesh.shape[axis]

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
        axes = dict(layout.device_axes)
        spec = jax.sharding.PartitionSpec(
            *(axes.get(name) for name in space.names))
        return jax.sharding.NamedSharding(self._device_mesh, spec)

    def local_slice(
        self,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> tuple[slice, ...]:
        """Return the global true-DOF index range of the shard."""
        self._resolve_layout(space, layout)
        # single device: the local shard is the whole true extent.
        return tuple(slice(0, n) for n in space.shape)

    def storage_shape(
        self,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> tuple[int, ...]:
        """Return the global storage shape (true + halo + padding)."""
        layout = self._resolve_layout(space, layout)
        shape = []
        for name, n in zip(space.names, space.shape, strict=True):
            shards = self._n_shards(name, layout)
            # per-shard true extent, padded up so staggered pairs
            # (n vs n + 1) shard to a uniform storage shape; on one
            # shard this reduces to n + 2 * width.
            local = -(-n // shards)
            shape.append(shards * (local + 2 * self._width(name)))
        return tuple(shape)

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
        """Map true-shape data to halo/stagger-padded storage."""
        layout = self._resolve_layout(space, layout)
        if tuple(arr.shape) != tuple(space.shape):
            raise ValueError(
                f"pad expects a true-shape array {space.shape}, "
                f"got {tuple(arr.shape)}")
        storage = self.storage_shape(space, layout)
        widths = []
        for name, n, stored in zip(
                space.names, space.shape, storage, strict=True):
            w = self._width(name)
            # trailing side absorbs the stagger padding (zero on a
            # one-shard axis).
            widths.append((w, stored - n - w))
        padded = jnp.pad(arr, widths)
        return jax.device_put(padded, self.sharding(space, layout))

    def unpad(
        self,
        arr: jax.Array,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.Array:
        """Map padded storage to true-shape data (pads dropped)."""
        layout = self._resolve_layout(space, layout)
        storage = self.storage_shape(space, layout)
        if tuple(arr.shape) != storage:
            raise ValueError(
                f"unpad expects a storage-shaped array {storage}, "
                f"got {tuple(arr.shape)}")
        slices = tuple(
            slice(self._width(name), self._width(name) + n)
            for name, n in zip(space.names, space.shape, strict=True))
        return arr[slices]

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
        """Exchange halos (see ``Decomposition.sync``)."""
        self._resolve_layout(space, layout)
        if fills is not None:
            raise NotImplementedError(
                "inhomogeneous ghost fill is designed-for; "
                "iteration 1 is homogeneous only")
        # sanctioned static branch: on one device there is nothing
        # to exchange, so the shard_map + ppermute region is skipped
        # entirely (Python-level branch on static structure).
        if self._device_mesh.size == 1:
            return arr
        raise NotImplementedError(  # pragma: no cover
            "the multi-device halo exchange arrives in Wave 3")

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

    def redistribute(
        self,
        arr: jax.Array,
        space: SpaceLike,
        src: Layout,
        dst: Layout,
    ) -> jax.Array:
        """Transpose an array between two negotiated layouts."""
        src = self._resolve_layout(space, src)
        dst = self._resolve_layout(space, dst)
        src_storage = self.storage_shape(space, src)
        if tuple(arr.shape) != src_storage:
            raise ValueError(
                f"redistribute expects a storage-shaped array "
                f"{src_storage} under src, got {tuple(arr.shape)}")
        if self.storage_shape(space, dst) != src_storage:
            raise NotImplementedError(  # pragma: no cover
                "repadding between layouts with different storage "
                "shapes arrives in Wave 3")
        # a resharding device_put; XLA lowers it to all-to-all.
        return jax.device_put(arr, self.sharding(space, dst))

    def gather(
        self,
        arr: jax.Array,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.Array:
        """Gather the global true-shape array (I/O, diagnostics)."""
        layout = self._resolve_layout(space, layout)
        # single device: dropping the pads yields the global array.
        return self.unpad(arr, space, layout)
