"""
Shared window-alignment plumbing of the staggered stencil operators.

Description
-----------
Support module behind ``FiniteDifference`` and ``LinearInterp``
(owning class doc: ``design/specs/grid/classes/operators_stencils.md``).
It owns the one piece of logic both kernels share: aligning a
``size``-point slice-based kernel (``operators.stencil_kernels``)
over the **storage-shaped** (halo-extended) local array so that the
outputs land on the codomain's true DOFs — the staggering direction
is pure window alignment (rules section 3.5).

The alignment calculus, in units of the (uniform) cell width: the
first true node of a nodal factor sits at a fixed offset from
``x_min`` (``Center`` 0.5, ``Left``/``Outer`` 0.0, ``Right``/``Inner``
1.0), and a ``size``-point kernel output sits at the midpoint of its
input window. BC tags are accepted as long as they drop no DOFs
(the tag governs only the ghost fill, ``decomposition/tensor.py``);
a Dirichlet component on a *member* boundary node (``Left`` /
``Right`` / ``Outer``) eliminates that value DOF and is rejected.
Output slot ``m`` of the codomain storage frame is
fed by the input window starting at storage slot
``m + i0`` with ``i0 = delta - (size - 1) / 2``, where ``delta`` is
the codomain-minus-domain first-node offset (always a half-integer
apart, so ``i0`` is an integer for the even-size staggered family).

Kernels compute over the **full** storage extent (every window that
fits), so outputs land in the codomain's ghost slots too; edge slots
whose windows would leave the storage are zero-filled and repaired by
the post-application sync. This is what lets un-synced
``SeparableComposite`` chains consume ghost depth kernel by kernel
(halo/storage contract, decomposition cluster doc).
"""
# Wave 2C: staggered window alignment (FiniteDifference, LinearInterp)
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.errors import SpaceMismatchError
from fridom.framework2.grid.operators.base import (
    FieldLike,
    Operator,
    resolve_codomain,
)
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from jax import Array

    from fridom.framework2.grid.fields.metadata import FieldMetadata
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )

# first-node offset from x_min per node set, in cell-width units
# (DOF-complete nodal spaces of the structured 1D meshes)
_FIRST_NODE_OFFSET: dict[NodeSet, float] = {
    NodeSet.CENTER: 0.5,
    NodeSet.LEFT: 0.0,
    NodeSet.RIGHT: 1.0,
    NodeSet.OUTER: 0.0,
    NodeSet.INNER: 1.0,
}

# whether the (left, right) boundary node is a member of the node
# set on a bounded mesh (mirror of the halo-fill geometry table in
# ``decomposition/tensor.py``; ``spaces/nodal.py`` shape note)
_BOUNDARY_MEMBERSHIP: dict[NodeSet, tuple[bool, bool]] = {
    NodeSet.CENTER: (False, False),
    NodeSet.LEFT: (True, False),
    NodeSet.RIGHT: (False, True),
    NodeSet.OUTER: (True, True),
    NodeSet.INNER: (False, False),
}


def require_dof_preserving_bc(
    factor: FunctionSpace, operation: str,
) -> None:
    """
    Reject BC tags that drop boundary DOFs off a nodal factor.

    Description
    -----------
    A Dirichlet component whose boundary node is a *member* of the
    node set eliminates that value DOF (``spaces/nodal.py`` shape
    note), so the factor's lattice no longer matches the BC-free
    staggering calculus: the first true node moves one cell inward
    (the halo-fill geometry of ``decomposition/tensor.py`` shifts
    its nearest-DOF distance the same way) and the window alignment
    below would feed the kernels phantom slots. ``Center`` and
    ``Inner`` carry no boundary members, and a Neumann tag keeps
    the boundary value a true DOF, so those always pass — the BC
    tag then governs only the ghost fill.

    Parameters
    ----------
    factor : FunctionSpace
        A bare 1D nodal factor space.
    operation : str
        The dispatch kind named in the error message.

    Raises
    ------
    SpaceMismatchError
        If a Dirichlet component drops a member boundary DOF.
    """
    membership = _BOUNDARY_MEMBERSHIP.get(factor.node_set, ())
    dropped = tuple(
        ("left", "right")[side]
        for side, (kind, member) in enumerate(
            zip(factor.bc.components, membership, strict=False))
        if member and kind is BC.DIRICHLET)
    if dropped:
        raise SpaceMismatchError(
            f"{factor!r} drops its {'/'.join(dropped)} boundary "
            f"DOF: a Dirichlet condition on a member node of "
            f"{factor.node_set.name} eliminates the boundary value "
            "from the space, so the staggered stencils cannot align "
            "their windows on it (out of scope in this iteration); "
            "keep such fields on Center/Inner (no boundary members) "
            "or on the BC-free sibling",
            left=factor, operation=operation)


def first_node_offset(factor: FunctionSpace) -> float:
    """
    First-node offset of a nodal factor, in cell widths.

    Description
    -----------
    BC tags are accepted as long as they drop no DOFs: the node
    positions are then identical to the BC-free sibling's and the
    offset table applies unchanged (the tag governs only the ghost
    fill). Dirichlet on a member node set (``Left`` / ``Right`` /
    ``Outer``) raises through ``require_dof_preserving_bc``.

    Parameters
    ----------
    factor : FunctionSpace
        A bare 1D factor space.

    Returns
    -------
    float
        Distance of the first true DOF from ``x_min`` in units of
        the cell width.

    Raises
    ------
    SpaceMismatchError
        If the factor is not a nodal space of the staggered
        node-set family, or its BC tag drops a boundary DOF.
    """
    if (isinstance(factor, NodalSpace)
            and factor.node_set in _FIRST_NODE_OFFSET):
        require_dof_preserving_bc(factor, "stencil alignment")
        return _FIRST_NODE_OFFSET[factor.node_set]
    raise SpaceMismatchError(
        f"{factor!r} is not a staggered nodal factor; the "
        "iteration-1 stencil kernels cover the "
        "Center/Left/Right/Outer/Inner node sets only",
        left=factor, operation="stencil alignment")


def uniform_spacing(factor: FunctionSpace) -> float:
    """
    Uniform cell width of the factor's mesh.

    Description
    -----------
    Iteration-1 stand-in for the ``grid.measure(space, name=...)``
    accessor (not yet implemented): on a uniform ``IntervalMesh`` the
    dual and primal measures collapse to the constant ``mesh.dx``,
    read at trace time (never baked into operator state).

    Parameters
    ----------
    factor : FunctionSpace
        A bare 1D factor space.

    Returns
    -------
    float
        The uniform cell width.

    Raises
    ------
    NotImplementedError
        If the factor's mesh has no uniform ``dx`` descriptor.
    """
    dx = getattr(factor.mesh, "dx", None)
    if dx is None:
        raise NotImplementedError(
            f"{factor.mesh!r} has no uniform cell width; nonuniform "
            "measure fields arrive with grid.measure")
    return dx


def apply_staggered(
    op: Operator,
    f: FieldLike,
    axis: str,
    size: int,
    kernel: Callable[[Array, int], Array],
    metadata: FieldMetadata | None,
) -> FieldLike:
    """
    Run an aligned ``size``-point kernel along ``axis``.

    Description
    -----------
    The shared ``_apply_factor`` body: resolves the codomain, aligns
    the kernel over the halo-extended storage (module docstring), and
    builds the result field on the bare codomain via the plumbing
    constructor. Ghost/edge slots the kernel cannot compute are
    zero-filled; the operator base's post-application sync repairs
    them.

    Parameters
    ----------
    op : Operator
        The (bound) separable kernel being applied.
    f : FieldLike
        The operand field (storage-shaped ``_data``).
    axis : str
        The resolved coordinate axis.
    size : int
        The stencil size (number of input points per output).
    kernel : Callable[[Array, int], Array]
        Array kernel mapping (storage, axis index) to the full
        stencil output (length shrinks by ``size - 1``).
    metadata : FieldMetadata | None
        Metadata of the result (None resets to the default record).

    Returns
    -------
    FieldLike
        The result field on the bare codomain.
    """
    space = f.function_space
    bare = space.bare
    domain_factor = bare.factor(axis)
    codomain = resolve_codomain(op, space)
    codomain_factor = codomain.factor(axis)
    delta = (first_node_offset(codomain_factor)
             - first_node_offset(domain_factor))
    i0 = delta - (size - 1) / 2
    if i0 != int(i0):  # pragma: no cover — even-size family only
        raise ValueError(
            f"misaligned staggering: offset {delta} with stencil "
            f"size {size} does not land on the lattice")
    m0 = -int(i0)

    axis_index = bare.names.index(axis)
    storage = f._data  # noqa: SLF001 — documented storage seam
    decomposition = f.grid.decomposition
    # the codomain storage lives in the operand's layout (the base
    # re-attaches it); resolving the bare codomain would pick the
    # default layout instead of the operand's pencil
    out_shape = decomposition.storage_shape(codomain, space.layout)
    s_out = out_shape[axis_index]
    n_out = codomain_factor.shape[0]
    try:
        width = decomposition.halo[axis]
    except KeyError:
        width = 0

    # per-side stencil reach beyond the true region; the halo must
    # cover it (frame-independent: equivalent to the storage-bounds
    # check on one shard, and the per-block condition on many)
    reach_right = (n_out - domain_factor.shape[0]) + size - 1 - m0
    if m0 > width or reach_right > width:
        raise ValueError(
            f"the negotiated halo width {width} along {axis!r} is "
            f"too small for the {size}-point stencil of "
            f"{type(op).__name__}; renegotiate with a registry that "
            "declares the wider requirement")
    full = kernel(storage, axis_index)
    length = full.shape[axis_index]
    lo = max(0, m0)
    hi = min(s_out, m0 + length)
    index: list[slice] = [slice(None)] * full.ndim
    index[axis_index] = slice(lo - m0, hi - m0)
    piece = full[tuple(index)]
    pads = [(0, 0)] * full.ndim
    pads[axis_index] = (lo, s_out - hi)
    data = jnp.pad(piece, pads)
    # halo-validity claim (task 1.8, stage B): the kernel computed
    # every output ghost slot its window reaches, so on a *periodic*
    # axis the result keeps the operand's valid layers minus the
    # per-side maximum reach (stencils commute with the wrap fill).
    # On bounded axes the claim is zero: stenciling the input's
    # BC-structured/extrapolated fill is not the BC-consistent fill
    # of the *output* field, so those ghost slots must be refilled
    # at the next consumption.
    if getattr(domain_factor.mesh, "periodic", False):
        valid = f.halo_valid.consume(axis, max(m0, reach_right, 0))
    else:
        valid = f.halo_valid.consume(axis, f.halo_valid[axis])
    return type(f)(f.grid, codomain, data, metadata,
                   halo_valid=valid)
