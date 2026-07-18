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

from fractions import Fraction
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.spatial.bc import BC
from fridom.spatial.errors import SpaceMismatchError
from fridom.spatial.operators.base import (
    FieldLike,
    Operator,
    resolve_codomain,
)
from fridom.spatial.operators.stencil_kernels import (
    one_sided_weights,
)
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from jax import Array

    from fridom.spatial.fields.metadata import FieldMetadata
    from fridom.spatial.spaces.function_space import (
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


def window_reach(
    domain: FunctionSpace, codomain: FunctionSpace,
    size: int, m0: int,
) -> tuple[int, int]:
    """
    Per-side reach (in slots) of a kernel aligned at ``m0``.

    Description
    -----------
    The window-alignment calculus of ``apply_staggered`` /
    ``apply_fv_staggered`` read off directly: a ``size``-point kernel
    whose output slot ``t`` lands at storage index ``t + m0`` reads
    ``m0`` slots beyond the true region on the left and
    ``(n_out - n_in) + size - 1 - m0`` on the right. Both are clamped
    to ``>= 0``. This is the two-sided twin of the storage-bounds
    check; the caller supplies ``m0`` (the midpoint value for a
    centered kernel, the biased offset for a WENO kernel), so the
    reach is exact for both.

    Parameters
    ----------
    domain : FunctionSpace
        The bare 1D domain factor.
    codomain : FunctionSpace
        The bare 1D codomain factor.
    size : int
        The stencil size (number of input points per output).
    m0 : int
        The window alignment (output slot ``t`` fills index ``t + m0``).

    Returns
    -------
    tuple[int, int]
        The (below, above) reach in slots (>= 0).
    """
    below = max(0, m0)
    above = max(
        0, (codomain.shape[0] - domain.shape[0]) + size - 1 - m0)
    return below, above


def footprint_reach(size: int, m0: int) -> tuple[int, int]:
    r"""
    Per-shard stencil footprint of a ``size``-point kernel at ``m0``.

    Description
    -----------
    The ghost depth a ``size``-point kernel aligned at ``m0`` reads
    below/above **each output slot's own index** — the per-shard
    halo-exchange demand, as opposed to :func:`window_reach`, which
    additionally folds in the *global* codomain/domain length
    difference (the physical-boundary staggering deficit). That global
    term belongs to the boundary-legality reach
    (:func:`exterior_reach`, consumed by
    :func:`require_grounded_bounded_sides`) but is wrong as a per-shard
    halo demand: a generic interior shard carries equal true
    input/output counts, so the deficit never lightens its exchange
    (the whole global deficit lands on a single physical-boundary
    shard, whose extra reach is toward the locally-filled wall). This
    is why the requirements reach — consumed by ``_ensure_valid`` and
    the negotiation tracer, both of which must see the exchange demand
    — must use the footprint, not :func:`exterior_reach`: a staggered
    first difference ``Center -> Inner`` has exterior reach ``(0, 0)``
    (the shrinking codomain cancels the stencil overhang at the global
    wall) yet a genuine footprint of one slot, and reading the former
    silently elides the sync of a sharded walled operand.

    Periodic factors already satisfy ``n_out == n_in``, so the
    footprint coincides with :func:`window_reach` there (the two-sided
    storage optimization is preserved bit-for-bit); only bounded
    staggered kernels differ.

    Parameters
    ----------
    size : int
        The stencil size (number of input points per output).
    m0 : int
        The window alignment (output slot ``t`` reads input window
        ``[t + m0, t + m0 + size - 1]``).

    Returns
    -------
    tuple[int, int]
        The (below, above) footprint reach in slots (>= 0).
    """
    return max(0, m0), max(0, size - 1 - m0)


def _midpoint_m0(
    domain: FunctionSpace, codomain: FunctionSpace, size: int,
) -> int:
    """Window alignment of a midpoint-staggered ``size``-point kernel."""
    delta = first_node_offset(codomain) - first_node_offset(domain)
    return -int(delta - (size - 1) / 2)


def exterior_reach(
    domain: FunctionSpace, codomain: FunctionSpace, size: int,
) -> tuple[int, int]:
    """
    Per-side exterior reach (in slots) of a midpoint-aligned kernel.

    Description
    -----------
    :func:`window_reach` at the midpoint alignment — how many input
    slots beyond the true region the true-shape output of a
    ``size``-point staggered kernel reads on each side. A positive
    reach means the signature needs exterior values there.

    Parameters
    ----------
    domain : FunctionSpace
        The bare 1D nodal domain factor.
    codomain : FunctionSpace
        The bare 1D nodal codomain factor.
    size : int
        The stencil size (number of input points per output).

    Returns
    -------
    tuple[int, int]
        The (left, right) exterior reach in slots (>= 0).
    """
    return window_reach(
        domain, codomain, size, _midpoint_m0(domain, codomain, size))


def reach_or(
    op: object, domain: FunctionSpace, size: int, fallback: int,
) -> tuple[int, int]:
    """
    Midpoint :func:`footprint_reach`, or the symmetric fallback.

    Description
    -----------
    The reach helper for a nodal-staggered operator's two-sided
    ``requirements``: resolves the codomain factor and returns the
    per-shard :func:`footprint_reach` at the midpoint alignment when
    the factors are staggered-nodal, and the symmetric
    ``(fallback, fallback)`` otherwise — a ``Fourier`` retag row, or a
    product/whole-space query where the operator has no 1D signature —
    whose ghost reach is the declared symmetric ``fallback`` and whose
    physical stencil geometry is not defined. The footprint (not
    :func:`exterior_reach`) is the halo-exchange demand the
    requirements must publish: on a bounded axis a staggered stencil's
    exterior reach can cancel to zero while its per-shard footprint
    does not, and the requirements feed both ``_ensure_valid`` and the
    negotiation tracer, which must sync/size a sharded walled operand.

    Parameters
    ----------
    op : object
        The operator (its ``codomain`` resolves the codomain factor).
    domain : FunctionSpace
        The bare domain factor.
    size : int
        The stencil size.
    fallback : int
        The symmetric width to use when the geometry is undefined.

    Returns
    -------
    tuple[int, int]
        The two-sided reach.
    """
    try:
        codomain = op.codomain(domain)
    except SpaceMismatchError:
        return (fallback, fallback)
    return footprint_reach(size, _midpoint_m0(domain, codomain, size))


def require_grounded_bounded_sides(
    domain: FunctionSpace,
    codomain: FunctionSpace,
    size: int,
    operation: str,
    opt_in: str,
    *,
    one_sided: bool = False,
) -> None:
    """
    Enforce the R1 legality rule on a bounded signature.

    Description
    -----------
    An operator row exists on a bounded operand iff every side its
    true-shape output needs exterior values from carries declared BC
    structure (the tag grounds the mirror ghost fill) — a BC-free
    side defines no exterior values, and the storage layer never
    invents them (R1, boundary_plan.md). The explicit escape is the
    per-operator ``boundary="one_sided"`` opt-in (R2), which patches
    the boundary windows from true DOFs only and is legal on fully
    BC-free domains.

    Parameters
    ----------
    domain : FunctionSpace
        The bare 1D nodal domain factor (bounded mesh).
    codomain : FunctionSpace
        The resolved 1D codomain factor.
    size : int
        The stencil size.
    operation : str
        The dispatch kind named in the error message.
    opt_in : str
        The spelled-out one-sided opt-in named in the hint.
    one_sided : bool, optional
        Whether the operator instance opted into the one-sided
        boundary closure (default: False).

    Raises
    ------
    SpaceMismatchError
        If a BC-free bounded side is asked for exterior values
        without the one-sided opt-in.
    """
    reach = exterior_reach(domain, codomain, size)
    components = domain.bc.components
    needy = tuple(
        ("left", "right")[side]
        for side, kind in enumerate(components)
        if reach[side] > 0 and kind is BC.NONE)
    if not needy:
        return
    if one_sided and domain.bc.is_free:
        return
    raise SpaceMismatchError(
        f"no {operation} signature on {domain!r}: the true-shape "
        f"output needs exterior values at the {'/'.join(needy)} "
        "wall, which a BC-free bounded side does not define — "
        "declare BC structure (mesh.nodal(..., bc=...)) or opt "
        f"into {opt_in}",
        left=domain, operation=operation)


def patch_one_sided_edges(
    f: FieldLike,
    result: FieldLike,
    axis: str,
    *,
    size: int,
    points: int,
    derivative: int,
    spacing: float,
) -> FieldLike:
    """
    Overwrite boundary outputs with one-sided stencils (2d, R2).

    Description
    -----------
    The explicit opt-in closure of the ``boundary="one_sided"``
    operator variants (boundary_plan.md): every output of the
    standard window kernel whose window reaches beyond the true
    region of a bounded BC-free axis is recomputed from the
    ``points`` nearest **true** DOFs, with exact moment-solved
    weights at the output's actual offset. Interior outputs are
    untouched, so the interior order is preserved; the boundary
    patches are accurate to degree ``points - 1``.

    The patches write at static storage indices of the physical
    edges, so the applied axis must be **undistributed** — the
    variants declare ``layout="local"`` and the caller guards
    through :func:`require_local_axis`.

    Parameters
    ----------
    f : FieldLike
        The operand field (storage-shaped ``_data``).
    result : FieldLike
        The standard kernel's result field (bare codomain).
    axis : str
        The resolved coordinate axis.
    size : int
        The standard window size (determines the patched counts).
    points : int
        One-sided stencil size of the patches.
    derivative : int
        0 (value) or 1 (first derivative).
    spacing : float
        The uniform cell width (scales derivative weights).

    Returns
    -------
    FieldLike
        The result field with the boundary windows patched.
    """
    bare = f.function_space.bare
    domain_factor = bare.factor(axis)
    codomain_factor = result.function_space.bare.factor(axis)
    o_in = Fraction(
        first_node_offset(domain_factor)).limit_denominator(2)
    o_out = Fraction(
        first_node_offset(codomain_factor)).limit_denominator(2)
    i0 = (o_out - o_in) - Fraction(size - 1, 2)
    m0 = -int(i0)
    n_in = domain_factor.shape[0]
    n_out = codomain_factor.shape[0]
    left_count = max(0, m0)
    right_count = max(0, (n_out - n_in) + size - 1 - m0)
    if n_in < points:
        raise NotImplementedError(
            f"the one-sided boundary patch needs {points} true DOFs "
            f"along {axis!r}, got {n_in}")

    axis_index = bare.names.index(axis)
    try:
        width = f.grid.decomposition.halo[axis]
    except KeyError:
        width = 0
    storage = f._data  # noqa: SLF001 — documented storage seam
    out = result._data  # noqa: SLF001 — documented storage seam

    def take(arr: Array, index: int) -> Array:
        slices: list[object] = [slice(None)] * arr.ndim
        slices[axis_index] = index
        return arr[tuple(slices)]

    def put(arr: Array, index: int, value: Array) -> Array:
        slices: list[object] = [slice(None)] * arr.ndim
        slices[axis_index] = index
        return arr.at[tuple(slices)].set(value)

    scale = spacing ** (-derivative)
    for t in range(left_count):
        offsets = tuple(o_in + j - (o_out + t) for j in range(points))
        weights = one_sided_weights(offsets, derivative)
        value = sum(w * take(storage, width + j)
                    for w, j in zip(weights, range(points),
                                    strict=True)) * scale
        out = put(out, width + t, value)
    for t in range(right_count):
        target = n_out - 1 - t
        offsets = tuple(
            o_in + (n_in - points + j) - (o_out + target)
            for j in range(points))
        weights = one_sided_weights(offsets, derivative)
        value = sum(w * take(storage, width + n_in - points + j)
                    for w, j in zip(weights, range(points),
                                    strict=True)) * scale
        out = put(out, width + target, value)
    return type(result)(result.grid, result.function_space,
                        out, result.metadata,
                        halo_valid=result.halo_valid)


def require_local_axis(f: FieldLike, axis: str) -> None:
    """
    Raise unless ``axis`` is undistributed on ``f``'s layout.

    Parameters
    ----------
    f : FieldLike
        The operand field.
    axis : str
        The applied coordinate axis.

    Raises
    ------
    NotImplementedError
        If the axis is device-distributed.
    """
    layout = f.function_space.layout
    if layout is not None and dict(layout.device_axes).get(axis):
        raise NotImplementedError(
            "one-sided boundary variants patch the physical edges "
            f"at static indices, so {axis!r} must be undistributed "
            "(layout='local'); reshard first")


def uniform_spacing(factor: FunctionSpace) -> float:
    """
    Uniform cell width of the factor's mesh.

    Description
    -----------
    The constant special case of the ``grid.measure`` metric fields
    (concepts section 2.7): on a uniform mesh the dual and primal
    measures collapse to the constant ``mesh.dx``, read at trace
    time (never baked into operator state) and folded by XLA.
    Non-uniform meshes carry no ``dx`` descriptor; their spacing
    enters through :func:`divide_by_codomain_measure` instead.

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
            "spacing enters through the grid.measure fields")
    return dx


def mapped_mesh(mesh: object) -> bool:
    """
    Whether a 1D mesh carries a coordinate map (stretched axis).

    Description
    -----------
    The mesh-level spelling of :func:`mapped_factor`, for callers
    that hold the mesh rather than a factor space (the model-level
    ``bind`` guards of the biased schemes, which vet
    ``grid.factors``).

    Parameters
    ----------
    mesh : object
        A 1D mesh (``IntervalMesh``, ``MappedIntervalMesh``, ...).

    Returns
    -------
    bool
        True iff the mesh exposes a non-None ``coordinate_map``.
    """
    return getattr(mesh, "coordinate_map", None) is not None


def mapped_factor(factor: FunctionSpace) -> bool:
    """
    Whether the factor's mesh carries a coordinate map.

    Description
    -----------
    The routing predicate of the stencil spacing denominators
    (concepts section 2.7): ``False`` selects the uniform
    scalar-``dx`` fast path (the constant special case), ``True``
    the measure-field division of
    :func:`divide_by_codomain_measure`.

    It is also the **refusal** predicate of every uniform-offset
    stencil wider than two points (``FiniteDifference`` order > 2,
    the biased WENO/upwind reconstructions): those rows are the
    uniform-mesh weights, and the two-point measure field can only
    ground a 2nd-order division — a wide row divided by it is
    consistent but silently 2nd order, so the operators raise
    instead (see :func:`mapped_order_hint`).

    Parameters
    ----------
    factor : FunctionSpace
        A bare 1D factor space.

    Returns
    -------
    bool
        True iff the mesh exposes a non-None ``coordinate_map``.
    """
    return mapped_mesh(factor.mesh)


def mapped_order_hint(what: str) -> str:
    """
    Shared "why" clause of the mapped high-order refusals.

    Description
    -----------
    The one sentence every mapped guard of a uniform-offset stencil
    repeats (``FiniteDifference``'s order > 2 guard, the biased
    reconstructions, the biased advection modules): the stencil is a
    *computational-coordinate* row, so a mapped mesh needs the
    computational-space chain rule with an order-matched discrete
    Jacobian; the two-point measure field the grid materializes caps
    the achievable order at 2.

    Parameters
    ----------
    what : str
        The refusing stencil, named in the message
        (e.g. "the biased face reconstructions").

    Returns
    -------
    str
        The composed reason clause.
    """
    return (
        f"{what} are uniform-offset (computational-coordinate) rows, "
        "so on a stretched mesh they are not the design-order "
        "weights: a mapped high-order stencil needs the "
        "computational-space chain rule with an order-matched "
        "discrete Jacobian, and the two-point measure field caps the "
        "order at 2 — the scheme would silently drop to 2nd order "
        "(deferred; coordinate-systems plan)")


def divide_by_codomain_measure(
    result: FieldLike, operand: FieldLike, axis: str,
) -> FieldLike:
    """
    Divide a unit-spacing difference by its codomain measure field.

    Description
    -----------
    The mapped-mesh spacing route of the two-point difference
    kernels (rules sections 2.7, 3.9): the denominator of a
    staggered difference is the **codomain's own measure** — the
    primal cell width when landing on ``Center``/``CellAvg``, the
    dual center-to-center spacing when landing on the face family —
    materialized from the grid at trace time and divided in the
    storage frame. The measure field is synced first, so on a
    periodic axis the ghost slots the kernel computed stay valid
    (the measure's wrap fill is its exact periodic extension) and
    the result's halo-validity claim carries over unchanged; on
    bounded axes the claim is already zero. Uniform meshes never
    reach this route (scalar fast path, see
    :func:`uniform_spacing`).

    The divide is VJP-sealed (double-``jnp.where``): a bounded axis's
    measure has exactly-zero ghost slots, so the raw quotient is a
    masked singularity whose reverse pass is ``0/0 -> NaN`` (the
    forward value is discarded by the post-application sync). The seal
    is bitwise-transparent on every valid cell and on periodic axes,
    so it costs no forward accuracy (AGENTS.md differentiability
    policy).

    Parameters
    ----------
    result : FieldLike
        The unit-spacing kernel result (bare codomain, storage
        frame of the operand's layout).
    operand : FieldLike
        The operand field (supplies the grid and layout).
    axis : str
        The resolved coordinate axis.

    Returns
    -------
    FieldLike
        The measure-scaled result (halo-validity claim kept).
    """
    grid = operand.grid
    space = result.function_space
    # query in the operand's layout: grid.measure resolves a None
    # layout to the default, matching the kernel's storage frame
    query = space.with_layout(operand.function_space.layout)
    measure = grid.sync(grid.measure(query, name=axis))
    # storage-frame divide, VJP-sealed (AGENTS.md diff policy): a
    # bounded axis's measure carries exactly-zero ghost slots (the
    # never-valid storage padding), so the raw ``result / measure`` is
    # a masked singularity -- the forward quotient there is discarded
    # by the post-application sync, but its reverse VJP is ``0/0 ->
    # NaN`` and poisons ``jax.grad`` through the difference. The
    # double-``jnp.where`` seals the reverse pass while staying bitwise
    # identical on every valid cell (measure > 0) and on periodic axes
    # (the wrap fill is strictly positive, so ``bad`` is empty).
    m = measure._data  # noqa: SLF001 — storage seam
    bad = m == 0.0
    safe = jnp.where(bad, 1.0, m)
    quotient = result._data / safe  # noqa: SLF001 — storage seam
    data = jnp.where(bad, 0.0, quotient)
    return type(result)(grid, space, data, result.metadata,
                        halo_valid=result.halo_valid)


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
    # per-side reach ``(m0, reach_right)`` (stencils commute with the
    # wrap fill; the low side keeps its spare when the stencil only
    # reaches high, and vice versa). On bounded axes the claim is
    # zero: stenciling the input's BC-structured/extrapolated fill is
    # not the BC-consistent fill of the *output* field, so those ghost
    # slots must be refilled at the next consumption.
    if getattr(domain_factor.mesh, "periodic", False):
        valid = f.halo_valid.consume(
            axis, (max(m0, 0), max(reach_right, 0)))
    else:
        valid = f.halo_valid.reset(axis)
    return type(f)(f.grid, codomain, data, metadata,
                   halo_valid=valid)
