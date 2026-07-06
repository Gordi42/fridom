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

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.framework2.grid.decomposition.halo import HaloSpec
    from fridom.framework2.grid.decomposition.layout import Layout


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
