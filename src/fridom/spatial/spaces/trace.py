"""
``TraceSpace``: the non-broadcasting boundary-trace factor.

Description
-----------
Owning plan: ``design/plans/active/boundary_trace_plan.md`` (§2). The
codomain of a boundary trace along one axis — the field's values on a
single boundary-adjacent row, carried as a size-1 factor on the
**parent** mesh. Distinct from ``ConstantSpace`` in exactly one way,
and that difference is the whole point: a trace is *not* constant, so
it never broadcasts into the interior. The strict space algebra then
auto-rejects ``Trace x full`` arithmetic (a wind-stress / surface-flux
mistake becomes a loud ``SpaceMismatchError`` at trace time) while
``Center(x) ⊗ Center(y) ⊗ Trace(z)`` still shards, stores, and
dispatches exactly like a ``Profile`` (sharded horizontals, replicated
size-1 z, no halo). Obtained as ``mesh.trace(node_set, side)`` on a
bounded structured 1D mesh.
"""
# Wave: TraceSpace (boundary-trace factor) + the Side vocabulary
from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING

from fridom.spatial.spaces.function_space import (
    _FACTORY_TOKEN,
    FunctionSpace,
    space_key,
)

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.bc import BCStructure
    from fridom.spatial.decomposition.layout import Layout
    from fridom.spatial.meshes.mesh import Mesh
    from fridom.spatial.scalars import Scalars, Variance
    from fridom.spatial.spaces.nodal import NodeSet


class Side(Enum):

    """
    The boundary side a trace is taken from.

    Description
    -----------
    The static ``low``/``high`` marker of a boundary along one 1D
    factor: ``LOW`` is the ``x_min`` wall (boundary component 0),
    ``HIGH`` the ``x_max`` wall (boundary component 1). Part of the
    interning key of every ``TraceSpace`` (and, later, of the
    boundary trace/embed operators), so a trace of the same node set
    at the two walls is two distinct interned spaces.
    """

    LOW = auto()
    HIGH = auto()


class TraceSpace(FunctionSpace):

    """
    Boundary-trace values along one factor: a size-1, non-constant row.

    Description
    -----------
    A single boundary-adjacent DOF, interned on the **parent** mesh
    (the mesh the trace was taken from), so ``TensorProductSpace``
    algebra and per-factor operator lifts keep working. Unlike a
    ``ConstantSpace`` it is *not* constant (``is_constant`` is
    ``False``): it collapses its axis to one replicated size-1 slot
    for storage/locality purposes (``collapses_axis`` is ``True``),
    but carries no broadcast sanction — the design win of
    ``boundary_trace_plan.md`` §2 is precisely that ``Trace ⊔ full``
    and ``Trace ⊔ Constant`` reject rather than silently broadcasting
    into the interior. Interned one per
    ``(mesh, parent_node_set, side, depth, scalars)``; ``bc`` is the
    free structure (a trace carries no homogeneous BC of its own).

    Parameters
    ----------
    mesh : Mesh
        The parent mesh the trace was taken from.
    scalars : Scalars
        The field of scalars the space is defined over.
    bc : BCStructure
        The (free) BC structure; a trace is always BC-free.
    parent_node_set : NodeSet
        The node set of the traced factor (the embed reads its target
        row and staggered offset off this).
    side : Side
        The boundary side (``LOW`` / ``HIGH``).
    depth : int, optional
        The signed true-node index from that side; only ``0`` (the
        boundary row) is implemented (default: 0).
    layout : Layout | None, optional
        The negotiated device layout; None for a bare space
        (default: None).
    variance : Variance | None, optional
        The component variance; None means scalar/no-variance
        (default: None).
    _token : object
        The owning mesh's private factory token (construction only
        through ``mesh.trace``).
    """

    _repr_label = "Trace"

    def __init__(self, mesh: Mesh, scalars: Scalars, bc: BCStructure,
                 *, parent_node_set: NodeSet, side: Side,
                 depth: int = 0,
                 layout: Layout | None = None,
                 variance: Variance | None = None,
                 _token: object = None) -> None:
        """Guarded constructor; see the class docstring."""
        super().__init__(mesh, scalars, bc, layout=layout,
                         variance=variance, _token=_token)
        if depth != 0:
            raise NotImplementedError(
                "interior fixed-depth traces (depth != 0) are not "
                "implemented: an interior plane sits on an arbitrary "
                "shard and would force the deferred partial-gather; "
                "only the boundary trace (depth=0) exists "
                "(boundary_trace_plan.md §2)")
        self._parent_node_set: NodeSet = parent_node_set
        self._side: Side = side
        self._depth: int = depth

    # ================================================================
    #  Constancy predicates
    # ================================================================
    @property
    def collapses_axis(self) -> bool:
        """Always ``True``: a trace collapses its axis to size 1.

        The storage/locality role of the constancy split
        (``boundary_trace_plan.md`` §2): replicated, size-1, no halo
        — like ``ConstantSpace`` — *without* the broadcast sanction
        (``is_constant`` stays ``False``).
        """
        return True

    # ================================================================
    #  Defining attributes
    # ================================================================
    @property
    def shape(self) -> tuple[int, ...]:
        """Always (1,): a single boundary-adjacent DOF."""
        return (1,)

    @property
    def parent_node_set(self) -> NodeSet:
        """The node set of the traced parent factor."""
        return self._parent_node_set

    @property
    def side(self) -> Side:
        """The boundary side the trace was taken from."""
        return self._side

    @property
    def depth(self) -> int:
        """The signed true-node index from the side (only 0 now)."""
        return self._depth

    # ================================================================
    #  Variant interning machinery (internal)
    # ================================================================
    def _variant_key(self, scalars: Scalars,
                     layout: Layout | None,
                     variance: Variance | None) -> tuple:
        """Return the (scalars, layout, variance) variant's key.

        One entry per (mesh, parent node set, side, depth, scalars),
        plus the layout and variance when set; bc is always the free
        structure.
        """
        return space_key(type(self), self._parent_node_set,
                         self._side, self._depth, scalars,
                         layout=layout, variance=variance)

    def _construct(self, scalars: Scalars,
                   layout: Layout | None,
                   variance: Variance | None) -> TraceSpace:
        """Build (not intern) the requested variant."""
        return type(self)(self._mesh, scalars, self._bc,
                          parent_node_set=self._parent_node_set,
                          side=self._side, depth=self._depth,
                          layout=layout, variance=variance,
                          _token=_FACTORY_TOKEN)

    # ================================================================
    #  Repr
    # ================================================================
    def _repr_details(self) -> tuple[str, ...]:
        """Add the traced node set and side to the repr."""
        return (f"{self._parent_node_set.name}",
                f"side={self._side.name}")
