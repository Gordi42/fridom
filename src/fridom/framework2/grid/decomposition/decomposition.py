"""
The ``Decomposition`` ABC and the resharding report.

Description
-----------
Owning class doc: ``notes/framework2/classes/decomposition.md``. The
grid owns exactly one ``Decomposition``; fields and operators reach
it only through the grid. Backends implement the abstract surface;
``TensorDecomposition`` is the iteration-1 jax-sharding backend and
``GraphDecomposition`` the designed-for unstructured one.

The methods here take spaces through the structural ``SpaceLike``
protocol so this cluster stays import-independent of the space
classes: any object exposing the product protocol (``shape``,
``names``, ``factors``, ``factor(name)``, ``layout``) qualifies —
in particular ``FunctionSpace`` and ``TensorProductSpace``.
"""
# Wave 1: Decomposition (ABC), ReshardingReport, SpaceLike --
#    Wave 3: negotiate()
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import jax

from fridom.framework2.grid.decomposition.halo import HaloSpec, trace_halo
from fridom.framework2.grid.decomposition.layout import Layout
from fridom.framework2.grid.decomposition.traits import HaloStrategy

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping


@runtime_checkable
class SpaceLike(Protocol):

    """
    Structural stand-in for a (product of) function space(s).

    Description
    -----------
    The duck-typing seam between the decomposition and the space
    clusters: the shared product protocol of
    ``notes/framework2/classes/product_spaces.md``, restricted to
    the members the decomposition consumes. ``FunctionSpace`` and
    ``TensorProductSpace`` satisfy it structurally; the decomposition
    never imports them.
    """

    @property
    def shape(self) -> tuple[int, ...]:
        """True global DOF shape: concatenated factor shapes."""
        ...

    @property
    def names(self) -> tuple[str, ...]:
        """All coordinate names, concatenated across factors."""
        ...

    @property
    def factors(self) -> tuple[object, ...]:
        """The per-mesh factor spaces, flat, in coordinate order."""
        ...

    @property
    def layout(self) -> Layout | None:
        """Negotiated device layout, or None for a bare space."""
        ...

    def factor(self, name: str) -> object:
        """Return the factor space contributing coordinate `name`."""
        ...


@dataclass(frozen=True)
class ReshardingReport:

    """
    What a renegotiation changed (returned by ``grid.negotiate``).

    Description
    -----------
    The model uses this report to re-``device_put`` its state once
    after a pre-freeze renegotiation.

    Parameters
    ----------
    old : Layout
        The default layout before renegotiation.
    new : Layout
        The default layout after renegotiation.
    changed : bool
        Whether the two layouts differ.
    """

    old: Layout
    new: Layout
    changed: bool


class Decomposition(ABC):

    """
    Distribution of a grid's DOFs across devices (grid-owned).

    Description
    -----------
    The abstract surface every backend implements. Storage shapes
    (halo plus stagger padding) are owned here and invisible above:
    ``pad``/``unpad`` map between true-shape data and storage, and
    ``local_slice`` is expressed in global true-DOF indices. The
    negotiated layouts form a closed vocabulary: every ``layout``
    argument must be one of ``self.layouts``.
    """

    # ================================================================
    #  Negotiated structure
    # ================================================================

    @property
    @abstractmethod
    def halo(self) -> HaloSpec:
        """The negotiated per-name ghost widths."""
        ...

    @property
    @abstractmethod
    def default_layout(self) -> Layout:
        """
        The layout attached to bare spaces at field creation.

        Description
        -----------
        State fields are stepped in it. Transform outputs legally
        stay in their pencils (section 5.1).
        """
        ...

    @property
    @abstractmethod
    def layouts(self) -> tuple[Layout, ...]:
        """All negotiated layouts (default + transform pencils)."""
        ...

    # ================================================================
    #  Shapes and shardings
    # ================================================================

    @abstractmethod
    def sharding(
        self,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.sharding.Sharding:
        """
        Return the jax sharding of `space`'s storage under `layout`.

        Parameters
        ----------
        space : SpaceLike
            The (product) space whose storage is placed.
        layout : Layout | None, optional
            A negotiated layout; None resolves to the space's layout
            if set, else the default layout (default: None).

        Returns
        -------
        jax.sharding.Sharding
            The sharding of the storage-shaped array.
        """
        ...

    @abstractmethod
    def local_slice(
        self,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> tuple[slice, ...]:
        """
        Return the global true-DOF index range of the local shard.

        Description
        -----------
        The anchor for the random factory's per-shard keying and for
        materializing coordinate shards; expressed in global
        true-DOF indices (never in storage indices).

        Parameters
        ----------
        space : SpaceLike
            The (product) space.
        layout : Layout | None, optional
            A negotiated layout; None resolves as in ``sharding``
            (default: None).

        Returns
        -------
        tuple[slice, ...]
            One slice per array axis.
        """
        ...

    @abstractmethod
    def storage_shape(
        self,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> tuple[int, ...]:
        """
        Return the global storage shape of `space` under `layout`.

        Description
        -----------
        True shape plus halo plus stagger padding; storage padding
        for unevenly sharding staggered pairs lives here and is
        invisible above this layer.

        Parameters
        ----------
        space : SpaceLike
            The (product) space.
        layout : Layout | None, optional
            A negotiated layout; None resolves as in ``sharding``
            (default: None).

        Returns
        -------
        tuple[int, ...]
            The global storage shape.
        """
        ...

    # ================================================================
    #  Storage construction and views
    # ================================================================

    @abstractmethod
    def zeros(
        self,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.Array:
        """
        Return a zero-filled, sharded, storage-shaped array.

        Parameters
        ----------
        space : SpaceLike
            The (product) space.
        layout : Layout | None, optional
            A negotiated layout; None resolves as in ``sharding``
            (default: None).

        Returns
        -------
        jax.Array
            Zeros of ``storage_shape(space, layout)``, committed to
            ``sharding(space, layout)``.
        """
        ...

    @abstractmethod
    def pad(
        self,
        arr: jax.Array,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.Array:
        """
        Map true-shape data to halo/stagger-padded storage.

        Parameters
        ----------
        arr : jax.Array
            A true-shape array (``space.shape``).
        space : SpaceLike
            The (product) space.
        layout : Layout | None, optional
            A negotiated layout; None resolves as in ``sharding``
            (default: None).

        Returns
        -------
        jax.Array
            The storage-shaped array (pad slots zero-filled).
        """
        ...

    @abstractmethod
    def unpad(
        self,
        arr: jax.Array,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.Array:
        """
        Map padded storage to true-shape data (pads dropped).

        Parameters
        ----------
        arr : jax.Array
            A storage-shaped array.
        space : SpaceLike
            The (product) space.
        layout : Layout | None, optional
            A negotiated layout; None resolves as in ``sharding``
            (default: None).

        Returns
        -------
        jax.Array
            The true-shape array (``space.shape``).
        """
        ...

    # ================================================================
    #  Data movement
    # ================================================================

    @abstractmethod
    def sync(
        self,
        arr: jax.Array,
        space: SpaceLike,
        *,
        layout: Layout | None = None,
        fills: Mapping[str, jax.Array] | None = None,
    ) -> jax.Array:
        """
        Exchange halos; fill bounded edges.

        Description
        -----------
        `fills` carries per-name ghost values for bounded axes: None
        means periodic wrap (periodic mesh) or the space's
        BC-structured homogeneous fill (bounded mesh); a supplied
        array is inhomogeneous ghost-fill data resolved by
        ``grid.sync`` (designed-for). Names with width 0 are skipped
        structurally.

        Parameters
        ----------
        arr : jax.Array
            A storage-shaped array.
        space : SpaceLike
            The (product) space.
        layout : Layout | None, optional
            A negotiated layout; None resolves as in ``sharding``
            (default: None).
        fills : Mapping[str, jax.Array] | None, optional
            Per-name inhomogeneous ghost data (default: None).

        Returns
        -------
        jax.Array
            The storage-shaped array with valid halos.
        """
        ...

    @abstractmethod
    def layout_for(
        self,
        local_names: tuple[str, ...],
    ) -> Layout:
        """
        Return a negotiated layout keeping the named factors local.

        Parameters
        ----------
        local_names : tuple[str, ...]
            Coordinate names that must be device-local.

        Returns
        -------
        Layout
            A member of ``self.layouts`` in which every name of
            `local_names` is local.
        """
        ...

    @abstractmethod
    def redistribute(
        self,
        arr: jax.Array,
        space: SpaceLike,
        src: Layout,
        dst: Layout,
    ) -> jax.Array:
        """
        Transpose an array between two negotiated layouts.

        Parameters
        ----------
        arr : jax.Array
            A storage-shaped array laid out per `src`.
        space : SpaceLike
            The (product) space.
        src : Layout
            The current layout (a member of ``self.layouts``).
        dst : Layout
            The target layout (a member of ``self.layouts``).

        Returns
        -------
        jax.Array
            The array resharded to `dst`.
        """
        ...

    @abstractmethod
    def gather(
        self,
        arr: jax.Array,
        space: SpaceLike,
        layout: Layout | None = None,
    ) -> jax.Array:
        """
        Gather the global true-shape array (I/O, diagnostics).

        Parameters
        ----------
        arr : jax.Array
            A storage-shaped array.
        space : SpaceLike
            The (product) space.
        layout : Layout | None, optional
            A negotiated layout; None resolves as in ``sharding``
            (default: None).

        Returns
        -------
        jax.Array
            The global true-shape array.
        """
        ...


# ================================================================
#  Negotiation (the per-mesh entry point, doc 04 section 5)
# ================================================================
#: the single device-mesh axis of the iteration-1 1-D realization
_DEVICE_AXIS = "devices"

#: mesh factory attributes of the ghost-shardable space families
_GHOST_FAMILY = ("center", "left", "right", "outer", "inner",
                 "cell_avg", "face_avg")


def negotiate(
    grid: object,
    registry: object,
    *,
    state_spaces: tuple[SpaceLike, ...] | None = None,
    tendency: Callable[..., object] | None = None,
    halo: HaloSpec | None = None,
    device_ids: tuple[int, ...] | None = None,
) -> Decomposition:
    """
    Choose a backend and layouts from mesh traits + operator demands.

    Description
    -----------
    Collects, per factor space in play, the mesh's declared
    ``mesh.decomposition_traits(space)`` and the registry operators'
    ``requirements`` (scoped to `state_spaces` when given). The halo
    comes from ``trace_halo`` when a `tendency` is supplied, else
    from the per-operator maximum over the registry (the provisional
    path — exact under the iteration-1 sync-after-every-operator
    contract), else from the explicit `halo=` override.

    Iteration-1 layout realization: a 1-D device mesh over all
    requested devices; the default layout shards the first
    GHOST-capable factor whose cell count divides the device count
    and whose per-shard extent respects ``min_local_size`` and the
    negotiated halo; every further such factor contributes a
    transpose pencil (plus the replicated layout as the last-resort
    pencil). When nothing is shardable, auto-selected devices fall
    back to a single device; explicitly requested ones raise.

    Parameters
    ----------
    grid : object
        The grid being negotiated (supplies ``names``/``factors``).
    registry : object
        The (duck-typed) operator dispatch registry.
    state_spaces : tuple[SpaceLike, ...] | None, optional
        The model's state-field spaces; scopes the operator demands
        and feeds the trace (default: None).
    tendency : Callable[..., object] | None, optional
        The tendency to halo-trace over `state_spaces`
        (default: None).
    halo : HaloSpec | None, optional
        Explicit per-name halo override (default: None).
    device_ids : tuple[int, ...] | None, optional
        Indices into ``jax.devices()``; None uses all available
        devices (default: None).

    Returns
    -------
    Decomposition
        The negotiated backend (iteration 1:
        ``TensorDecomposition``; ``GraphDecomposition`` is
        designed-for).
    """
    names = tuple(grid.names)
    meshes = tuple(grid.factors)
    spec = _negotiated_halo(names, registry, state_spaces=state_spaces,
                            tendency=tendency, halo=halo)

    if device_ids is None:
        ids = tuple(range(len(jax.devices())))
        explicit = False
    else:
        ids = tuple(device_ids)
        explicit = True

    layouts: tuple[Layout, ...] = (Layout({}),)
    if len(ids) > 1:
        shardable = _shardable_names(meshes, spec, len(ids))
        if shardable:
            layouts = (*(Layout({name: _DEVICE_AXIS})
                         for name in shardable), Layout({}))
        elif explicit:
            raise ValueError(
                f"no factor of {names} is GHOST-shardable over "
                f"{len(ids)} devices (cell counts must divide the "
                "device count and per-shard extents must cover "
                "min_local_size and halo + 1)")
        else:
            ids = ids[:1]  # auto-selection falls back to one device

    from fridom.framework2.grid.decomposition.tensor import (  # noqa: PLC0415 — tensor imports this module
        TensorDecomposition,
    )
    return TensorDecomposition(
        meshes=meshes, names=names, halo=spec, layouts=layouts,
        device_ids=ids)


def _negotiated_halo(
    names: tuple[str, ...],
    registry: object,
    *,
    state_spaces: tuple[SpaceLike, ...] | None,
    tendency: Callable[..., object] | None,
    halo: HaloSpec | None,
) -> HaloSpec:
    """Resolve the halo source: explicit > traced > registry max."""
    if halo is not None:
        return HaloSpec.zero(names).merge_max(halo)
    if tendency is not None:
        if state_spaces is None:
            raise ValueError(
                "tracing a tendency needs state_spaces= to build "
                "the tracer state")
        traced = trace_halo(tendency, state_spaces, registry)
        return HaloSpec.zero(names).merge_max(traced)
    return _registry_halo(names, registry, state_spaces)


def _registry_halo(
    names: tuple[str, ...],
    registry: object,
    state_spaces: tuple[SpaceLike, ...] | None = None,
) -> HaloSpec:
    """
    Derive the provisional halo from a dispatch registry.

    Description
    -----------
    The per-operator maximum of ``requirements(space).halo`` over
    the registry's space-keyed entries, per coordinate name (grid
    lifecycle step 2) — exact under the iteration-1
    sync-after-every-operator contract. With `state_spaces` given,
    entries are scoped to the meshes those spaces live on
    (iteration-1 reading of "operators that can actually fire").
    A duck-typed registry without an ``items`` surface contributes
    nothing (zero halo).

    Parameters
    ----------
    names : tuple[str, ...]
        The grid's coordinate names.
    registry : object
        The (duck-typed) operator registry.
    state_spaces : tuple[SpaceLike, ...] | None, optional
        The state-field spaces scoping the demands (default: None).

    Returns
    -------
    HaloSpec
        The per-name provisional ghost widths.
    """
    widths = dict.fromkeys(names, 0)
    items = getattr(registry, "items", None)
    if not callable(items):
        return HaloSpec(widths)
    scope_meshes = None
    if state_spaces is not None:
        scope_meshes = {
            id(factor.mesh)
            for space in state_spaces
            for factor in space.factors}
    for key, op in items():
        if not isinstance(key, tuple):
            continue  # kind-only entries carry no space to size on
        space = key[1]
        if scope_meshes is not None:
            mesh = getattr(space, "mesh", None)
            if mesh is None or id(mesh) not in scope_meshes:
                continue
        requirements = getattr(op, "requirements", None)
        if requirements is None:
            continue
        halo = requirements(space).halo
        for name in space.names:
            if name in widths:
                widths[name] = max(widths[name], halo)
    return HaloSpec(widths)


def _shardable_names(
    meshes: tuple[object, ...],
    halo: HaloSpec,
    devices: int,
) -> tuple[str, ...]:
    """
    Coordinate names admissible for ghost-sharding over `devices`.

    Description
    -----------
    A name qualifies when its mesh's ghost family declares the
    ``GHOST`` strategy and the per-shard extent satisfies the
    negotiation constraints: the cell count divides the device
    count, and ``cells_per_shard`` covers ``min_local_size`` and
    ``halo + 1`` (the short staggered last shard must still hold a
    full exchange edge).
    """
    shardable: list[str] = []
    for mesh in meshes:
        n_cells = getattr(mesh, "n_cells", None)
        if not n_cells or n_cells % devices:
            continue
        cells = n_cells // devices
        ghost = False
        min_local = 1
        for attr in _GHOST_FAMILY:
            try:
                space = getattr(mesh, attr)
            except (AttributeError, ValueError, NotImplementedError):
                continue  # factory absent on this mesh type/topology
            traits = mesh.decomposition_traits(space)
            if HaloStrategy.GHOST in traits.strategies:
                ghost = True
                min_local = max(min_local, traits.min_local_size)
        if not ghost:
            continue
        for name in mesh.names:
            try:
                width = halo[name]
            except KeyError:
                width = 0
            if cells >= max(min_local, width + 1):
                shardable.append(name)
    return tuple(shardable)
