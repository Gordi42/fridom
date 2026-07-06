"""
The jax-sharding decomposition of tensor-product grids.

Description
-----------
Owning class doc: ``notes/framework2/classes/decomposition.md``.
Iteration 1 covers the **single-device** case: one code path with a
one-device ``jax.make_mesh`` (no separate ``SingleDecomposition``;
``jax.sharding`` degrades gracefully). ``sync`` performs the local
halo fill — periodic wrap on periodic mesh factors, the
**BC-structured homogeneous fill** on bounded ones (odd extension
for Dirichlet-structured spaces, even/mirror for Neumann-structured,
one-sided extrapolation for BC-free; blanket zero-fill is rejected
by the halo/storage contract). Multi-device layouts, the
``shard_map`` + ``ppermute`` halo exchange, and negotiation arrive
in Wave 3.
"""
# Wave 1: trivial single-device TensorDecomposition --
#    Wave 2C: single-device halo fill -- Wave 3: multi-device
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.decomposition.decomposition import (
    Decomposition,
)
from fridom.framework2.grid.spaces.average import CellAvg, FaceAvg
from fridom.framework2.grid.spaces.coefficient import CoefficientSpace
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    from fridom.framework2.grid.decomposition.decomposition import (
        SpaceLike,
    )
    from fridom.framework2.grid.decomposition.halo import HaloSpec
    from fridom.framework2.grid.decomposition.layout import Layout
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )


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
        for name, n, factor in self._axis_entries(space):
            shards = self._n_shards(name, layout)
            # per-shard true extent, padded up so staggered pairs
            # (n vs n + 1) shard to a uniform storage shape; on one
            # shard this reduces to n + 2 * width.
            local = -(-n // shards)
            shape.append(
                shards * (local + 2 * self._width(name, factor)))
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
        for (name, n, factor), stored in zip(
                self._axis_entries(space), storage, strict=True):
            w = self._width(name, factor)
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
            slice(self._width(name, factor),
                  self._width(name, factor) + n)
            for name, n, factor in self._axis_entries(space))
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
        """
        Fill halos (see ``Decomposition.sync``).

        Description
        -----------
        Single-device iteration 1: no cross-shard exchange happens
        (the sanctioned static branch skips the ``shard_map`` +
        ``ppermute`` region entirely), but every ghost slot is
        filled locally — periodic wrap on periodic mesh factors, the
        BC-structured homogeneous fill on bounded ones. Width-0 axes
        are skipped structurally; an all-width-0 space is returned
        unchanged.
        """
        self._resolve_layout(space, layout)
        if fills is not None:
            raise NotImplementedError(
                "inhomogeneous ghost fill is designed-for; "
                "iteration 1 is homogeneous only")
        if self._device_mesh.size != 1:
            raise NotImplementedError(  # pragma: no cover
                "the multi-device halo exchange arrives in Wave 3")
        for axis, (name, n, factor) in enumerate(
                self._axis_entries(space)):
            width = self._width(name, factor)
            if width:
                arr = _fill_axis(arr, axis, n, width, factor)
        return arr

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


# ================================================================
#  Single-device halo fill (halo/storage contract)
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
# (a constrained member DOF is dropped from the space's shape, which
# moves the nearest true DOF one cell inward)
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
                and kind is not BC.NONE):
            # the constrained boundary DOF is dropped from the space
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
) -> jax.Array:
    """
    Ghost values of one bounded boundary (BC-structured fill).

    Description
    -----------
    Realizes the halo/storage contract's bounded-edge fill:
    Dirichlet-structured spaces get the odd (zero-value) extension,
    Neumann-structured spaces the even (mirror) extension, and
    BC-free spaces a one-sided linear extrapolation (consistent with
    the iteration-1 second-order stencils). Blanket zero-fill is
    rejected by design. Ghost slot ``k`` counts outward from the
    true region (k = 1 is adjacent to the first true DOF).

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

    def dof(k: int) -> jax.Array:
        """Return the k-th true DOF from this side (k=1 nearest)."""
        if k > n:
            raise NotImplementedError(
                f"the bounded halo fill of width {width} reaches "
                f"deeper than the {n} DOFs along the axis")
        index = k - 1 if side == 0 else n - k
        return _take(true, axis, slice(index, index + 1))

    if kind is BC.NONE:
        if n < 2:  # noqa: PLR2004 — two-point extrapolation
            raise NotImplementedError(
                "the BC-free one-sided extrapolation needs at "
                f"least two DOFs along the axis, got {n}")
        ghosts = [(1.0 + k) * dof(1) - float(k) * dof(2)
                  for k in range(1, width + 1)]
    elif kind is BC.DIRICHLET and distance == _OFFSET:
        ghosts = [-dof(k) for k in range(1, width + 1)]
    elif kind is BC.DIRICHLET:
        # _VACANT: the ghost slot k = 1 IS the (zero) boundary DOF;
        # deeper slots odd-reflect about it
        ghosts = [jnp.zeros_like(dof(1)) if k == 1 else -dof(k - 1)
                  for k in range(1, width + 1)]
    elif kind is BC.NEUMANN and distance == _OFFSET:
        ghosts = [dof(k) for k in range(1, width + 1)]
    else:
        raise NotImplementedError(
            "the Neumann (even) fill is grounded for "
            "boundary-offset node sets only in iteration 1; got "
            f"{factor!r} with a lattice node on the boundary")
    if side == 0:
        ghosts.reverse()
    return jnp.concatenate(ghosts, axis=axis)


def _fill_axis(
    arr: jax.Array,
    axis: int,
    n: int,
    width: int,
    factor: FunctionSpace,
) -> jax.Array:
    """
    Fill the ghost slots of one storage axis (single device).

    Description
    -----------
    Periodic mesh factors wrap; bounded ones get the BC-structured
    fill per side. The trailing storage side also absorbs the
    stagger padding, which on a single device equals the ghost
    width, so every non-true slot is (re)written.

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
    true = _take(arr, axis, slice(width, width + n))
    trail = arr.shape[axis] - n - width
    if factor.mesh.periodic:
        if width > n or trail > n:
            raise NotImplementedError(
                f"periodic wrap with halo {width} wider than the "
                f"axis length {n} is not supported")
        left = _take(true, axis, slice(n - width, n))
        right = _take(true, axis, slice(0, trail))
    else:
        left = _bounded_ghosts(true, axis, n, width, factor, 0)
        right = _bounded_ghosts(true, axis, n, trail, factor, 1)
    return jnp.concatenate([left, true, right], axis=axis)
