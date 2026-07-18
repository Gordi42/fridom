r"""
Boundary trace / embed / scatter machinery (the ``TraceSpace`` verbs).

Description
-----------
Owning plan: ``design/plans/active/boundary_trace_plan.md`` (§3). The
sanctioned 3D<->2D boundary path of the operator framework — the
generalized replacement for raw ``.data`` boundary surgery
(``hydrostatic/core.py``'s ``_masked_w_faces``). Five operators, all
riding the single new primitive, the boundary trace:

- :class:`BoundaryTrace` (``"trace"``): a field's boundary-adjacent
  row along one axis as a size-1 ``TraceSpace`` factor (the surface
  w-face ``w(0)``, a cell's top-cell value, a traced ``grid.measure``
  / ``grid.metric`` / ``immersed.fraction`` slice).
- :class:`BoundaryEmbed` (``"embed"``): the mutual VJP — a 2D trace
  materialized sparsely back into its full-3D parent (boundary row
  set, zeros elsewhere).
- :class:`BoundaryScatterAdd` / :class:`BoundaryScatterSet`
  (``"scatter_add"`` / ``"scatter_set"``): row-scatter of a 2D trace
  into a full field's boundary row (add / overwrite); the fully-2D H7
  lowering and the absorbtion of ``_masked_w_faces``.
- :class:`AsProfile` (``"as_profile"``): the opt-in bridge Trace ->
  ``ConstantSpace`` (into the existing Constant-z machinery).
- :class:`Adopt` (``"adopt"``): the reverse retag ``ConstantSpace`` ->
  Trace (e.g. wrapping a wind-stress input file as boundary-located).

Slice and scatter are mutual native VJPs (no ``custom_vjp`` anywhere,
step path — §5). Every operator works in the **true frame** like
:class:`~fridom.spatial.operators.integrate.Integral`: index the
physical boundary row of ``f.data`` (``Side.LOW`` -> index 0,
``Side.HIGH`` -> index -1, robust to Dirichlet-eliminated shapes) and
``store`` on the codomain, so padded-storage index math never enters.
Future boundary consumers (wind stress, surface buoyancy flux, bottom
drag, SST diagnostics) route here instead of touching ``.data``.
"""
# Wave: boundary trace / embed / scatter / conversions (plan §3)
from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, final

import jax.numpy as jnp

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.fields.storage import store
from fridom.spatial.operators.base import (
    BinaryOperator,
    FieldLike,
    OperatorRequirements,
    SeparableOperator,
    UnaryOperator,
    resolve_codomain,
)
from fridom.spatial.operators.interned import interned
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.coefficient import CoefficientSpace
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
from fridom.spatial.spaces.trace import Side, TraceSpace

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike


# whether the (LOW, HIGH) boundary node is a member of the node set on
# a bounded mesh (a Dirichlet condition on a member node eliminates its
# value DOF; mirror of ``spaces/nodal.py`` / ``operators/staggering.py``)
_SIDE_MEMBER: dict[NodeSet, tuple[bool, bool]] = {
    NodeSet.CENTER: (False, False),
    NodeSet.LEFT: (True, False),
    NodeSet.RIGHT: (False, True),
    NodeSet.OUTER: (True, True),
    NodeSet.INNER: (False, False),
}

#: LOW -> boundary component 0 (x_min), HIGH -> component 1 (x_max)
_SIDE_INDEX: dict[Side, int] = {Side.LOW: 0, Side.HIGH: 1}


def _boundary_row_node_set(factor: FunctionSpace) -> NodeSet | None:
    """
    Return the node set a factor's boundary-adjacent row belongs to.

    Description
    -----------
    A boundary trace records a topological :class:`NodeSet` on the
    parent mesh. Nodal factors carry their own; the FV ``CellAvg``
    co-locates with cell centres, so its boundary cell is a
    ``CENTER`` row (``embed`` therefore lands the FV trace on the
    co-located nodal ``Center`` — the FV scatter-add path stays on
    ``CellAvg`` natively). Every other factor (``FaceAvg`` — whose
    bounded dual cells never touch the wall — coefficient, constant,
    trace) has no boundary row and returns ``None``.

    Parameters
    ----------
    factor : FunctionSpace
        A bare 1D factor space.

    Returns
    -------
    NodeSet | None
        The co-located node set, or None when there is no boundary row.
    """
    if isinstance(factor, NodalSpace):
        return factor.node_set
    if isinstance(factor, CellAvg):
        return NodeSet.CENTER
    return None


def _require_traceable_side(
    factor: FunctionSpace, side: Side, operation: str,
) -> None:
    """
    Reject a side whose boundary DOF is Dirichlet-eliminated.

    Description
    -----------
    Strictly finer than the both-sides ``require_dof_preserving_bc``
    (``restrict``): a field Dirichlet at one wall is still traceable
    at the other. Only a nodal factor can drop a DOF; a Dirichlet
    condition on a *member* boundary node of ``side`` eliminates that
    boundary value from the space (``spaces/nodal.py`` shape note), so
    the boundary row does not exist there and the trace/embed/scatter
    is ill-posed.

    Parameters
    ----------
    factor : FunctionSpace
        A bare 1D nodal factor space.
    side : Side
        The requested boundary side.
    operation : str
        The dispatch kind named in the error message.

    Raises
    ------
    SpaceMismatchError
        If ``side``'s boundary DOF is Dirichlet-eliminated.
    """
    if not isinstance(factor, NodalSpace):
        return
    membership = _SIDE_MEMBER.get(factor.node_set, (False, False))
    index = _SIDE_INDEX[side]
    components = factor.bc.components
    if (membership[index] and index < len(components)
            and components[index] is BC.DIRICHLET):
        wall = ("LOW", "HIGH")[index]
        raise SpaceMismatchError(
            f"{factor!r} has no boundary DOF at the {wall} wall: a "
            f"Dirichlet condition on the {wall} member node of "
            f"{factor.node_set.name} eliminates that boundary value "
            "from the space, so it cannot be traced there — trace the "
            "opposite wall, or keep the field BC-free",
            left=factor, operation=operation)


def _boundary_index(node_count: int, side: Side) -> int:
    """True-frame index of the boundary-adjacent row (§3, true frame)."""
    return 0 if side is Side.LOW else node_count - 1


def _sole_trace_axis(space: SpaceLike) -> str:
    """
    Return the name of the space's single ``TraceSpace`` factor.

    Parameters
    ----------
    space : SpaceLike
        A bare product space (the 2D operand of a scatter/embed).

    Returns
    -------
    str
        The traced coordinate name.

    Raises
    ------
    SpaceMismatchError
        Unless the space carries exactly one boundary-trace factor.
    """
    axes = tuple(
        name for factor in space.factors
        if isinstance(factor, TraceSpace) for name in factor.names)
    if len(axes) != 1:
        raise SpaceMismatchError(
            "a boundary-trace operand must carry exactly one "
            f"TraceSpace factor, got {len(axes)} in {space!r}",
            left=space, operation="scatter")
    return axes[0]


# ================================================================
#  BoundaryTrace: full row -> 2D Trace
# ================================================================
@final
@interned
class BoundaryTrace(SeparableOperator):

    """
    Extract a field's boundary-adjacent row as a 2D ``TraceSpace``.

    Description
    -----------
    A ``SeparableOperator`` that collapses one bounded axis to its
    boundary-adjacent row: ``Side.LOW`` the ``x_min`` row (true index
    0), ``Side.HIGH`` the ``x_max`` row (true index -1). Nodal and FV
    ``CellAvg`` factors only; a periodic axis (no boundary),
    coefficient / constant / trace factors, ``FaceAvg`` (no wall dual
    cell), ``depth != 0`` (deferred interior slices), and — side-aware
    — a wall whose boundary DOF is Dirichlet-eliminated all raise. The
    codomain is ``mesh.trace(node_set, side)`` (the FV trace lands on
    the co-located ``CENTER`` row). Works in the true frame like
    ``Integral`` (slice ``f.data`` with keepdims, then ``store``); the
    traced axis is declared ``layout="local"`` and a sharded axis is
    resharded onto a negotiated axis-local layout and back (the proven
    ``CumulativeIntegral`` pattern).

    Parameters
    ----------
    side : Side
        The boundary side (``Side.LOW`` / ``Side.HIGH``).
    depth : int, optional
        The signed true-node index from ``side``; only ``0`` (the
        boundary row) is implemented (default: 0).
    """

    dispatch_kind: ClassVar[str | None] = "trace"

    def __init__(self, side: Side, depth: int = 0) -> None:
        """Validate and store the boundary side and depth."""
        if not isinstance(side, Side):
            raise TypeError(
                f"side must be a Side member (Side.LOW / Side.HIGH), "
                f"got {side!r}")
        self._side: Side = side
        self._depth: int = depth

    def _intern_key(self) -> tuple:
        """Structural key: the side and depth (D6)."""
        return (self._side, self._depth)

    @property
    def side(self) -> Side:
        """The boundary side the trace is taken from."""
        return self._side

    @property
    def depth(self) -> int:
        """The signed true-node index from the side (only 0 now)."""
        return self._depth

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve ``full factor -> mesh.trace(node_set, side)``.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D nodal / average factor space.

        Returns
        -------
        FunctionSpace
            The ``TraceSpace`` codomain factor (scalars preserved).
        """
        if isinstance(domain, (ConstantSpace, TraceSpace)):
            raise SpaceMismatchError(
                f"a collapsed factor {domain!r} carries no boundary to "
                "trace (it is already a single size-1 row)",
                left=domain, operation="trace")
        if isinstance(domain, CoefficientSpace):
            raise SpaceMismatchError(
                f"no trace signature on the coefficient factor "
                f"{domain!r}: transform back to a nodal/average factor "
                "first", left=domain, operation="trace")
        node_set = _boundary_row_node_set(domain)
        if node_set is None:
            raise SpaceMismatchError(
                f"no boundary-adjacent row on {domain!r}: the boundary "
                "trace covers nodal factors and the FV CellAvg "
                "(FaceAvg dual cells never reach the wall)",
                left=domain, operation="trace")
        if domain.mesh.periodic:
            raise SpaceMismatchError(
                f"cannot trace a boundary of the periodic factor "
                f"{domain!r}: a periodic axis has no boundary",
                left=domain, operation="trace")
        if self._depth != 0:
            raise SpaceMismatchError(
                f"interior fixed-depth traces (depth={self._depth}) are "
                "deferred; only the boundary row (depth=0) is "
                "implemented (boundary_trace_plan.md §2)",
                left=domain, operation="trace")
        _require_traceable_side(domain, self._side, "trace")
        trace: FunctionSpace = domain.mesh.trace(node_set, self._side)
        if domain.scalars is Scalars.COMPLEX:
            trace = trace.as_complex()
        return trace

    def requirements(
        self,
        domain: FunctionSpace,  # noqa: ARG002 — axis-local, halo 0
    ) -> OperatorRequirements:
        """Declare halo = 0, layout = "local" (the traced axis)."""
        return OperatorRequirements(halo=0, layout="local")

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Slice the boundary row along ``axis``, resharding if sharded.

        Description
        -----------
        On an already axis-local operand the boundary row is sliced
        directly. When ``axis`` is sharded the operand is resharded
        onto ``decomposition.layout_for((axis,))`` through the
        ``Reshard`` movement operator, sliced, and the (collapsed)
        result resharded back to the operand's layout — the proven
        ``CumulativeIntegral`` axis-local pattern.

        Parameters
        ----------
        f : FieldLike
            The operand field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The 2D trace field (default metadata: new quantity).
        """
        layout = f.function_space.layout
        decomposition = getattr(f.grid, "decomposition", None)
        if (decomposition is not None and layout is not None
                and decomposition.device_count > 1
                and not layout.is_local(axis)):
            from fridom.spatial.operators.movement import (  # noqa: PLC0415 — keep movement off the operator import path
                Reshard,
            )
            local = decomposition.layout_for((axis,))
            f_local = Reshard(f.grid, local)(f)
            traced = self._slice_boundary(f_local, axis)
            return Reshard(f.grid, layout)(traced)
        return self._slice_boundary(f, axis)

    def _slice_boundary(self, f: FieldLike, axis: str) -> FieldLike:
        """Slice the boundary row (true frame) and store on the trace."""
        space = f.function_space
        bare = space.bare
        axis_index = bare.names.index(axis)
        node_count = bare.factor(axis).shape[0]
        index = _boundary_index(node_count, self._side)
        selector: list[object] = [slice(None)] * f.data.ndim
        selector[axis_index] = slice(index, index + 1)
        row = f.data[tuple(selector)]
        codomain = resolve_codomain(self, space).with_layout(
            space.layout)
        stored = store(f.grid.decomposition, codomain, row)
        return type(f)(f.grid, codomain, stored, None)


# ================================================================
#  BoundaryEmbed: 2D Trace -> sparse full row (mutual VJP of trace)
# ================================================================
@final
@interned
class BoundaryEmbed(SeparableOperator):

    """
    Materialize a 2D trace sparsely into its full-3D parent row.

    Description
    -----------
    The mutual VJP of :class:`BoundaryTrace`: a ``TraceSpace`` factor
    expands back to its parent node set's BC-free nodal space
    (``mesh.nodal(parent_node_set)``), the boundary row set to the
    trace value and every interior row zero. The side and target row
    come off the domain's ``TraceSpace`` (``Side.LOW`` -> index 0,
    ``Side.HIGH`` -> index -1). The codomain is always BC-free, so its
    boundary DOF always exists (no side-aware rejection is reachable).
    ``trace(embed(t)) == t`` exactly; ``embed(trace(f)) == f`` on the
    boundary row and zero elsewhere (an FV trace lands on the
    co-located nodal ``Center``, not ``CellAvg`` — use the scatter-add
    path to stay on ``CellAvg``).
    """

    dispatch_kind: ClassVar[str | None] = "embed"

    def _intern_key(self) -> tuple:
        """Structural key: side/depth ride the domain's TraceSpace (D6)."""
        return ()

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve ``TraceSpace -> mesh.nodal(parent_node_set)``.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D ``TraceSpace`` factor.

        Returns
        -------
        FunctionSpace
            The BC-free parent nodal codomain factor.
        """
        if not isinstance(domain, TraceSpace):
            raise SpaceMismatchError(
                f"no embed signature on {domain!r}: embed materializes "
                "a boundary trace (TraceSpace) back into its full "
                "parent row",
                left=domain, operation="embed")
        parent: FunctionSpace = domain.mesh.nodal(domain.parent_node_set)
        if domain.scalars is Scalars.COMPLEX:
            parent = parent.as_complex()
        return parent

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:
        """
        Sparse-materialize the trace value into the parent row.

        Parameters
        ----------
        f : FieldLike
            The operand trace field.
        axis : str
            The resolved coordinate axis.

        Returns
        -------
        FieldLike
            The sparse full field (default metadata: new quantity).
        """
        space = f.function_space
        bare = space.bare
        trace_factor = bare.factor(axis)
        codomain = resolve_codomain(self, space).with_layout(
            space.layout)
        axis_index = codomain.bare.names.index(axis)
        node_count = codomain.bare.factor(axis).shape[0]
        index = _boundary_index(node_count, trace_factor.side)
        value = f.data
        out_shape = list(value.shape)
        out_shape[axis_index] = node_count
        zeros = jnp.zeros(tuple(out_shape), dtype=value.dtype)
        selector: list[object] = [slice(None)] * value.ndim
        selector[axis_index] = slice(index, index + 1)
        full = zeros.at[tuple(selector)].set(value)
        stored = store(f.grid.decomposition, codomain, full)
        return type(f)(f.grid, codomain, stored, None)


# ================================================================
#  BoundaryScatterAdd / BoundaryScatterSet: 2D Trace -> full row
# ================================================================
class _BoundaryScatter(BinaryOperator):

    """
    Row-scatter of a 2D trace into a full field's boundary row.

    Description
    -----------
    Shared base of the add / set variants: ``(full, trace) -> full``,
    a bespoke codomain that does **not** route through the binary
    base's sanctioned lifts (a constant broadcast would silently
    accept the collapsed trace into every interior row). The full
    field's factor along the traced axis must be the trace's parent
    node set on the same mesh (the FV ``CellAvg`` matches a
    ``CENTER`` trace, co-located); every other factor must match, and
    the side-aware Dirichlet guard applies. The overwrite (``set``)
    variant is what absorbs ``_masked_w_faces`` (plan §6). Slice and
    scatter are mutual native VJPs (no ``custom_vjp``).
    """

    #: True overwrites the boundary row, False adds into it
    _overwrite: ClassVar[bool]

    def _intern_key(self) -> tuple:
        """Structural key: the mode is fixed per concrete class (D6)."""
        return ()

    def codomain(
        self,
        domain_full: SpaceLike,
        domain_trace: SpaceLike,
        *more: SpaceLike,
    ) -> SpaceLike:
        """
        Resolve ``(full, trace) -> full`` (bespoke, no lifts).

        Parameters
        ----------
        domain_full : SpaceLike
            The full field's bare product space.
        domain_trace : SpaceLike
            The 2D trace operand's bare product space.
        *more : SpaceLike
            Rejected: the scatter is strictly binary.

        Returns
        -------
        SpaceLike
            The full field's space (the target of the scatter).
        """
        if more:
            raise SpaceMismatchError(
                f"{type(self).__name__} is binary (full, trace); got "
                f"{2 + len(more)} operands", left=domain_full,
                operation=self.dispatch_kind)
        axis = _sole_trace_axis(domain_trace)
        trace_factor = domain_trace.factor(axis)
        try:
            full_factor = domain_full.factor(axis)
        except KeyError:
            raise SpaceMismatchError(
                f"the full field {domain_full!r} does not resolve the "
                f"traced axis {axis!r} of {domain_trace!r}",
                left=domain_full, right=domain_trace,
                operation=self.dispatch_kind) from None
        node_set = _boundary_row_node_set(full_factor)
        if (node_set is not trace_factor.parent_node_set
                or full_factor.mesh is not trace_factor.mesh):
            raise SpaceMismatchError(
                f"the full field's factor {full_factor!r} along "
                f"{axis!r} must be the trace's parent node set "
                f"({trace_factor.parent_node_set.name}) on the same "
                "mesh; scatter writes only into that boundary row",
                left=full_factor, right=trace_factor,
                operation=self.dispatch_kind)
        _require_traceable_side(
            full_factor, trace_factor.side, self.dispatch_kind)
        for name in domain_trace.names:
            if name == axis:
                continue
            if domain_trace.factor(name) is not domain_full.factor(name):
                raise SpaceMismatchError(
                    f"the trace and full field disagree off the traced "
                    f"axis at {name!r}: {domain_trace.factor(name)!r} "
                    f"vs {domain_full.factor(name)!r}",
                    left=domain_trace, right=domain_full,
                    operation=self.dispatch_kind)
        return domain_full

    def _apply(
        self,
        full: FieldLike,
        trace: FieldLike,
        *more: FieldLike,  # noqa: ARG002 — codomain rejected extras
    ) -> FieldLike:
        """
        Scatter the trace value into ``full``'s boundary row.

        Parameters
        ----------
        full : FieldLike
            The target field (its boundary row is written).
        trace : FieldLike
            The 2D trace operand.
        *more : FieldLike
            Never present (rejected by ``codomain``).

        Returns
        -------
        FieldLike
            The scattered field (target metadata kept).
        """
        space = full.function_space
        bare = space.bare
        axis = _sole_trace_axis(trace.function_space.bare)
        trace_factor = trace.function_space.bare.factor(axis)
        axis_index = bare.names.index(axis)
        node_count = bare.factor(axis).shape[0]
        index = _boundary_index(node_count, trace_factor.side)
        target = full.data
        value = trace.data
        selector: list[object] = [slice(None)] * target.ndim
        selector[axis_index] = slice(index, index + 1)
        if self._overwrite:
            out = target.at[tuple(selector)].set(value)
        else:
            out = target.at[tuple(selector)].add(value)
        stored = store(full.grid.decomposition, space, out)
        return type(full)(full.grid, space, stored, full.metadata)


@final
@interned
class BoundaryScatterAdd(_BoundaryScatter):

    """Row-scatter-ADD of a 2D trace into a full field's boundary row."""

    dispatch_kind: ClassVar[str | None] = "scatter_add"
    _overwrite: ClassVar[bool] = False


@final
@interned
class BoundaryScatterSet(_BoundaryScatter):

    """Row-scatter-SET (overwrite) — the ``_masked_w_faces`` absorber."""

    dispatch_kind: ClassVar[str | None] = "scatter_set"
    _overwrite: ClassVar[bool] = True


# ================================================================
#  Conversions: pure retags on identical size-1 storage
# ================================================================
@final
@interned
class AsProfile(SeparableOperator):

    """
    Opt-in bridge Trace -> ``ConstantSpace`` (broadcast sanction).

    Description
    -----------
    A pure retag: a ``TraceSpace`` factor becomes its mesh's
    ``ConstantSpace`` on identical size-1 storage (data untouched, own
    native VJP). The sanctioned way to feed a boundary value into every
    existing Constant-z consumer (the 2D spectral solve, ``measure``,
    export) — an *explicit* broadcast sanction, never the implicit
    relocation the strict algebra refuses.
    """

    dispatch_kind: ClassVar[str | None] = "as_profile"

    def _intern_key(self) -> tuple:
        """Structural key: no constructor state (D6)."""
        return ()

    def codomain(self, domain: FunctionSpace) -> FunctionSpace:
        """
        Resolve ``TraceSpace -> mesh.constant``.

        Parameters
        ----------
        domain : FunctionSpace
            The bare 1D ``TraceSpace`` factor.

        Returns
        -------
        FunctionSpace
            The mesh's ``ConstantSpace`` (scalars preserved).
        """
        if not isinstance(domain, TraceSpace):
            raise SpaceMismatchError(
                f"no as_profile signature on {domain!r}: it bridges a "
                "boundary trace (TraceSpace) into the Constant-z "
                "machinery", left=domain, operation="as_profile")
        constant: FunctionSpace = domain.mesh.constant
        if domain.scalars is Scalars.COMPLEX:
            constant = constant.as_complex()
        return constant

    def _apply_factor(self, f: FieldLike, axis: str) -> FieldLike:  # noqa: ARG002 — pure retag
        """Retag onto the ConstantSpace codomain (data untouched)."""
        space = f.function_space
        codomain = resolve_codomain(self, space).with_layout(
            space.layout)
        return type(f)(
            f.grid, codomain,
            f._data,  # noqa: SLF001 — retag reuses identical storage
            f.metadata, halo_valid=f.halo_valid)


@final
@interned
class Adopt(UnaryOperator):

    """
    Retag a ``ConstantSpace`` factor as a boundary ``TraceSpace``.

    Description
    -----------
    The reverse of :class:`AsProfile`: a whole-space retag that adopts
    a ``ConstantSpace`` factor along ``axis`` as a boundary-located
    ``mesh.trace(node_set, side)`` on identical size-1 storage (data
    untouched, own native VJP) — e.g. wrapping a wind-stress input file
    as boundary-located so the strict algebra keeps it off the
    interior. A ``UnaryOperator`` (not separable): the separable base
    short-circuits to identity on a ``ConstantSpace`` axis, so the
    conversion *from* Constant must run the whole-space path. The
    node set, side, and axis are the interning key.

    Parameters
    ----------
    node_set : NodeSet
        The parent node set to locate the trace on.
    side : Side
        The boundary side (``Side.LOW`` / ``Side.HIGH``).
    axis : str
        The coordinate name of the ``ConstantSpace`` factor to adopt.
    """

    dispatch_kind: ClassVar[str | None] = "adopt"

    def __init__(
        self, node_set: NodeSet, side: Side, axis: str,
    ) -> None:
        """Validate and store the node set, side, and axis."""
        if not isinstance(node_set, NodeSet):
            raise TypeError(
                f"node_set must be a NodeSet member, got {node_set!r}")
        if not isinstance(side, Side):
            raise TypeError(
                f"side must be a Side member (Side.LOW / Side.HIGH), "
                f"got {side!r}")
        if not isinstance(axis, str):
            raise TypeError(
                f"axis must be a coordinate name (str), got {axis!r}")
        self._node_set: NodeSet = node_set
        self._side: Side = side
        self._axis: str = axis

    def _intern_key(self) -> tuple:
        """Structural key: axis, node set, side (D6)."""
        return (self._axis, self._node_set, self._side)

    @property
    def axis(self) -> str:
        """The coordinate name of the adopted factor."""
        return self._axis

    def codomain(self, domain: SpaceLike) -> SpaceLike:
        """
        Retag the ``ConstantSpace`` factor on ``axis`` as a trace.

        Parameters
        ----------
        domain : SpaceLike
            The bare product (or single-factor) space.

        Returns
        -------
        SpaceLike
            The space with the ``axis`` factor replaced by the trace.
        """
        bare = domain.bare
        try:
            factor = bare.factor(self._axis)
        except KeyError:
            raise SpaceMismatchError(
                f"adopt is bound to axis {self._axis!r}, absent from "
                f"{bare!r}", left=bare, operation="adopt") from None
        if not isinstance(factor, ConstantSpace):
            raise SpaceMismatchError(
                f"adopt retags a ConstantSpace factor as a boundary "
                f"trace; the factor along {self._axis!r} is {factor!r}",
                left=factor, operation="adopt")
        trace: FunctionSpace = factor.mesh.trace(self._node_set,
                                                 self._side)
        if factor.scalars is Scalars.COMPLEX:
            trace = trace.as_complex()
        return bare.replace(**{self._axis: trace})

    def _apply(self, f: FieldLike) -> FieldLike:
        """Retag onto the trace codomain (data untouched)."""
        space = f.function_space
        codomain = resolve_codomain(self, space).with_layout(
            space.layout)
        return type(f)(
            f.grid, codomain,
            f._data,  # noqa: SLF001 — retag reuses identical storage
            f.metadata, halo_valid=f.halo_valid)
