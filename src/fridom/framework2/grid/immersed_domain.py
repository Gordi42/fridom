"""
``ImmersedDomain``: the grid-owned masked-domain descriptor.

Description
-----------
Owning class doc: ``notes/framework2/classes/grid.md``; rules in
``notes/framework2/02_rules.md`` section 3.7. A static descriptor of
the wet region (successor of ``WaterMask``): it holds the declaring
callable and static parameters only — no arrays, no fields — and
materializes per-space masks/fractions on demand at trace time,
exactly like ``grid.evaluation_nodes`` (rules section 2.7).

Iteration-1 subset: the **boolean** masks (fraction in {0, 1},
``WaterMask`` parity — fidelity-ladder point 1). Cut-cell fractions,
transition sets, and the level-set generalization are designed-for.
Masked domains keep the full product shape (section 3.7): masked
DOFs are stored-but-dead, and masking correctness is owned by
operators and modules, not by the type system.

Halo note (below the iteration-1 contract): derived fields are
routed through the ordinary storage path (``pad`` + ``sync``), so
periodic wraps and shard-edge exchanges carry the correct
neighboring mask values; the BC-structured fill of *physical*
boundaries on bounded meshes is extrapolation-based and thresholded
here — mask-aware operators (designed-for) own their own ghost
discipline.
"""
# Wave 4: ImmersedDomain, Slip
from __future__ import annotations

import inspect
from enum import Enum, auto
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.framework2.grid.bc import BC
from fridom.framework2.grid.fields.metadata import FieldMetadata
from fridom.framework2.grid.fields.scalar_field import ScalarField
from fridom.framework2.grid.fields.storage import store
from fridom.framework2.grid.meshes.interval import IntervalMesh
from fridom.framework2.grid.spaces.average import CellAvg, FaceAvg
from fridom.framework2.grid.spaces.coefficient import CoefficientSpace
from fridom.framework2.grid.spaces.constant import ConstantSpace
from fridom.framework2.grid.spaces.nodal import NodalSpace, NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike

# the iteration-1 boolean subset: declared fractions are wet above
# this threshold (a boolean indicator and a {0, 1} fraction agree)
_WET_THRESHOLD = 0.5


class Slip(Enum):

    """
    Immersed-boundary structure: staggered-mask combination rule.

    Description
    -----------
    Structure (no-slip vs free-slip) is the *combination rule* used
    to derive staggered masks from the cell mask (rules section
    3.7), a parameter of the derivation, not stored per-space state:
    ``NO_SLIP`` keeps a face wet only when **both** adjacent cells
    are wet (today's ``face = AND(adjacent centers)``); ``FREE_SLIP``
    keeps it wet when **either** adjacent cell is wet.
    """

    NO_SLIP = auto()
    FREE_SLIP = auto()


class ImmersedDomain:

    """
    Static wet-region descriptor; per-space masks derived on demand.

    Description
    -----------
    Declares the wet fraction as a callable of the physical
    coordinates (keyword-matched to the grid's coordinate names,
    like ``create_field(init=...)``). Iteration 1 derives the
    boolean subset: declared values are thresholded at ``0.5``, so a
    boolean indicator and a {0, 1} fraction are equivalent. The
    descriptor is bound to its grid at grid construction
    (``immersed=`` kwarg) or by the pre-freeze ``grid.with_immersed``
    update; it is fully static (no arrays, not a pytree) and every
    ``fraction``/``mask`` call materializes to the local shard at
    trace time.

    Parameters
    ----------
    init : Callable[..., jax.Array]
        The wet fraction as a function of the physical coordinates
        (iteration 1: thresholded to the boolean subset).
    slip : Slip, optional
        The default staggered-mask derivation rule
        (default: Slip.NO_SLIP).
    """

    def __init__(
        self,
        init: Callable[..., jax.Array],
        *,
        slip: Slip = Slip.NO_SLIP,
    ) -> None:
        """Declare the wet fraction; see the class docstring."""
        if not callable(init):
            raise TypeError(
                f"init must be a callable of the physical "
                f"coordinates, got {init!r}")
        if not isinstance(slip, Slip):
            raise TypeError(
                f"slip must be a Slip member, got {slip!r}")
        self._init: Callable[..., jax.Array] = init
        self._slip: Slip = slip
        self._grid: Grid | None = None

    # ================================================================
    #  Identity (static aux discipline, matching Grid)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Identity comparison: ``self is other`` (static aux)."""
        return self is other

    def __hash__(self) -> int:
        """Identity hash, consistent with ``__eq__``."""
        return id(self)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def slip(self) -> Slip:
        """The default staggered-mask derivation rule."""
        return self._slip

    # ================================================================
    #  Grid binding (called by the grid at attachment)
    # ================================================================
    def _bind(self, grid: Grid) -> None:
        """Bind the descriptor to its grid (idempotent per grid)."""
        if self._grid is not None and self._grid is not grid:
            raise ValueError(
                "this ImmersedDomain is already attached to another "
                "grid; descriptors are grid-bound — build a new one "
                "per grid")
        self._grid = grid

    def _bound_grid(self) -> Grid:
        """Return the bound grid, raising when unattached."""
        if self._grid is None:
            raise RuntimeError(
                "this ImmersedDomain is not attached to a grid; "
                "pass it as Grid(..., immersed=...) or attach it "
                "via grid.with_immersed(...)")
        return self._grid

    # ================================================================
    #  Derived per-space fields (derive-on-demand, section 3.7)
    # ================================================================
    def fraction(
        self,
        space: SpaceLike,
        *,
        fraction: ScalarField | None = None,
    ) -> ScalarField:
        """
        Wet fraction on ``space`` (iteration 1: values in {0, 1}).

        Description
        -----------
        The fraction transfer is geometric and slip-independent
        (rules section 3.7); for the iteration-1 staircase geometry
        the transferred face fraction is exactly the AND of the
        adjacent cell fractions (a face touching land is a wall).

        Parameters
        ----------
        space : SpaceLike
            The target space (must resolve every grid coordinate).
        fraction : ScalarField | None, optional
            Explicit cell-fraction data (module-owned state) used
            instead of the static declaration (default: None).

        Returns
        -------
        ScalarField
            The {0, 1} wet fraction, tagged with ``space``.
        """
        grid = self._bound_grid()
        space = grid._laid_out(space)  # noqa: SLF001 — grid seam
        cells = self._cell_mask(space, fraction)
        wet = _derive(space, cells, jnp.logical_and)
        stored = store(grid.decomposition, space,
                       wet.astype(dtype_real()))
        stored = jnp.clip(stored, 0.0, 1.0)
        return ScalarField(grid, space, stored,
                           FieldMetadata.create(name="wet_fraction"))

    def mask(
        self,
        space: SpaceLike,
        *,
        slip: Slip | None = None,
        fraction: ScalarField | None = None,
    ) -> ScalarField:
        """
        Boolean wet mask on ``space``, derived by the slip rule.

        Parameters
        ----------
        space : SpaceLike
            The target space (must resolve every grid coordinate).
        slip : Slip | None, optional
            Override of the constructor's combination rule, for
            mixed-physics diagnostics (default: None).
        fraction : ScalarField | None, optional
            Explicit cell-fraction data (module-owned state) used
            instead of the static declaration (default: None).

        Returns
        -------
        ScalarField
            The boolean wet mask, tagged with ``space``.
        """
        grid = self._bound_grid()
        space = grid._laid_out(space)  # noqa: SLF001 — grid seam
        slip = self._slip if slip is None else slip
        if not isinstance(slip, Slip):
            raise TypeError(
                f"slip must be a Slip member, got {slip!r}")
        combine = (jnp.logical_and if slip is Slip.NO_SLIP
                   else jnp.logical_or)
        cells = self._cell_mask(space, fraction)
        wet = _derive(space, cells, combine)
        stored = store(grid.decomposition, space,
                       wet.astype(dtype_real()))
        return ScalarField(grid, space, stored > _WET_THRESHOLD,
                           FieldMetadata.create(name="wet_mask"))

    def transition(
        self,
        space: SpaceLike,
        *,
        fraction: ScalarField | None = None,
    ) -> ScalarField:
        """
        Wet/dry transition-set indicator on ``space`` (designed-for).

        Parameters
        ----------
        space : SpaceLike
            The target space.
        fraction : ScalarField | None, optional
            Explicit cell-fraction data (default: None).

        Returns
        -------
        ScalarField
            Never returns in iteration 1.
        """
        raise NotImplementedError(
            "transition-set indicators are designed-for; iteration 1 "
            "implements the boolean mask subset only")

    # ================================================================
    #  Base cell mask (the single declared datum, materialized)
    # ================================================================
    def _cell_mask(
        self,
        space: SpaceLike,
        fraction: ScalarField | None,
    ) -> jax.Array:
        """
        Materialize the boolean cell mask in ``space`` factor order.

        Description
        -----------
        Evaluates the declared indicator at the cell centers of the
        space's meshes (the collocation approximation of the volume
        fraction, rules section 3.7) — or broadcasts the explicit
        ``fraction`` data — and thresholds at 0.5 (the iteration-1
        boolean subset).

        Parameters
        ----------
        space : SpaceLike
            The (laid-out) target space.
        fraction : ScalarField | None
            Explicit cell-fraction field, or None for the static
            declaration.

        Returns
        -------
        jax.Array
            The boolean cell mask, one axis per space factor.
        """
        grid = self._bound_grid()
        factors = space.factors
        _validate_factors(space, grid)
        cell_shape = tuple(
            factor.mesh.n_cells for factor in factors)
        if fraction is not None:
            return self._explicit_cells(space, fraction, cell_shape)
        params = tuple(inspect.signature(self._init).parameters)
        names = tuple(
            name for factor in factors for name in factor.names)
        if set(params) != set(names):
            raise TypeError(
                "the immersed indicator must name exactly the grid "
                f"coordinate names {names}, got {params}")
        coords: dict[str, jax.Array] = {}
        for axis, factor in enumerate(factors):
            shape = [1] * len(factors)
            shape[axis] = cell_shape[axis]
            coords[factor.names[0]] = _cell_centers(
                factor.mesh).reshape(shape)
        values = jnp.asarray(self._init(**coords))
        return jnp.broadcast_to(values, cell_shape) > _WET_THRESHOLD

    def _explicit_cells(
        self,
        space: SpaceLike,
        fraction: ScalarField,
        cell_shape: tuple[int, ...],
    ) -> jax.Array:
        """
        Validate and broadcast explicit cell-fraction data.

        Parameters
        ----------
        space : SpaceLike
            The (laid-out) target space.
        fraction : ScalarField
            The module-owned cell-fraction field: every non-constant
            factor must be a ``Center``/``CellAvg`` factor of one of
            the target space's meshes.
        cell_shape : tuple[int, ...]
            The cell shape in target factor order.

        Returns
        -------
        jax.Array
            The boolean cell mask, one axis per space factor.
        """
        grid = self._bound_grid()
        if fraction.grid is not grid:
            raise ValueError(
                "the explicit fraction field lives on a different "
                "grid than this immersed domain")
        meshes = tuple(factor.mesh for factor in space.factors)
        source = fraction.function_space
        axes: dict[object, int] = {}
        for axis, factor in enumerate(source.factors):
            if isinstance(factor, ConstantSpace):
                continue
            is_center = (isinstance(factor, CellAvg)
                         or (isinstance(factor, NodalSpace)
                             and factor.node_set is NodeSet.CENTER))
            if not is_center or factor.mesh not in meshes:
                raise ValueError(
                    "explicit fraction data lives on the cell "
                    "family (Center/CellAvg factors of the target "
                    f"space's meshes); got {factor!r}")
            axes[factor.mesh] = axis
        data = fraction.data
        # move the source axes into target factor order, inserting
        # broadcast axes for meshes the source is constant along
        order = [axes[mesh] for mesh in meshes if mesh in axes]
        data = jnp.transpose(
            data, (*order, *(i for i in range(data.ndim)
                             if i not in order)))
        shape = tuple(
            cell_shape[i] if mesh in axes else 1
            for i, mesh in enumerate(meshes))
        data = data.reshape(shape)
        return jnp.broadcast_to(data, cell_shape) > _WET_THRESHOLD


# ================================================================
#  Per-factor staggering transfer (iteration-1 boolean rule)
# ================================================================
def _validate_factors(space: SpaceLike, grid: Grid) -> None:
    """Reject factor families outside the iteration-1 mask rules."""
    for factor in space.factors:
        if isinstance(factor, CoefficientSpace):
            # a value error (bad space choice), not a type error
            raise ValueError(  # noqa: TRY004
                f"masks have no coefficient-space representation "
                f"({factor!r}); masked domains on spectral bases "
                "are volume penalization (designed-for, rules "
                "section 3.7)")
        if isinstance(factor, ConstantSpace):
            # a value error (bad space choice), not a type error
            raise ValueError(  # noqa: TRY004
                f"the mask is not resolved along {factor!r}; "
                "constant factors cannot carry the wet region")
    names = tuple(
        name for factor in space.factors for name in factor.names)
    if set(names) != set(grid.names):
        raise ValueError(
            "immersed masks are derived on spaces resolving every "
            f"grid coordinate {grid.names}; this space resolves "
            f"{names}")


def _cell_centers(mesh: object) -> jax.Array:
    """
    Materialize the 1D cell-center coordinates of one mesh.

    Parameters
    ----------
    mesh : Mesh
        A grid mesh factor (iteration 1: ``IntervalMesh``).

    Returns
    -------
    jax.Array
        The ``n_cells`` cell-center coordinates.
    """
    if not isinstance(mesh, IntervalMesh):
        raise NotImplementedError(
            f"immersed masks on {type(mesh).__name__} arrive in a "
            "later wave; iteration 1 covers IntervalMesh")
    x_min = mesh.extent[0]
    return x_min + (jnp.arange(mesh.n_cells, dtype=dtype_real())
                    + 0.5) * mesh.dx


def _derive(
    space: SpaceLike,
    cells: jax.Array,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """
    Transfer the cell mask onto the space's staggered node sets.

    Parameters
    ----------
    space : SpaceLike
        The (laid-out) target space.
    cells : jax.Array
        The boolean cell mask, one axis per factor.
    combine : Callable[[jax.Array, jax.Array], jax.Array]
        The two-neighbor combination rule (AND / OR).

    Returns
    -------
    jax.Array
        The boolean mask at the space's true shape.
    """
    arr = cells
    for axis, factor in enumerate(space.factors):
        arr = _stagger_axis(arr, axis, factor, combine)
    return arr


def _take(arr: jax.Array, axis: int, sl: slice) -> jax.Array:
    """Slice ``arr`` with ``sl`` along ``axis``."""
    index: list[slice] = [slice(None)] * arr.ndim
    index[axis] = sl
    return arr[tuple(index)]


def _stagger_axis(
    arr: jax.Array,
    axis: int,
    factor: FunctionSpace,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """
    Transfer the cell mask to one factor's node set along ``axis``.

    Description
    -----------
    Cell-positioned factors (``Center``/``CellAvg``) keep the cell
    mask; face-family node sets combine the two adjacent cells with
    ``combine`` — wrapping on periodic meshes, with a dry exterior
    on bounded ones (``WaterMask`` parity). BC-constrained boundary
    DOFs are dropped exactly like the space shapes drop them.

    Parameters
    ----------
    arr : jax.Array
        The mask, cell-positioned along ``axis``.
    axis : int
        The array axis of this factor.
    factor : FunctionSpace
        The factor space owning the axis.
    combine : Callable[[jax.Array, jax.Array], jax.Array]
        The two-neighbor combination rule (AND / OR).

    Returns
    -------
    jax.Array
        The mask on the factor's node set along ``axis``.
    """
    node_set, membership, components = _axis_geometry(factor)
    if node_set is NodeSet.CENTER:
        staggered = arr
    elif factor.mesh.periodic:
        staggered = _stagger_periodic(arr, axis, node_set, combine)
    else:
        staggered = _stagger_bounded(arr, axis, node_set, combine)
    return _drop_constrained(staggered, axis, membership, components)


def _axis_geometry(
    factor: FunctionSpace,
) -> tuple[NodeSet, tuple[bool, bool], tuple[BC, ...]]:
    """
    Resolve one factor's node set, membership, and BC components.

    Parameters
    ----------
    factor : FunctionSpace
        A nodal or average factor space.

    Returns
    -------
    tuple[NodeSet, tuple[bool, bool], tuple[BC, ...]]
        The node set the mask is derived at, whether the (left,
        right) end DOF sits on the boundary, and the BC structure
        (empty for average factors: no DOF drops).
    """
    if isinstance(factor, CellAvg):
        return NodeSet.CENTER, (False, False), ()
    if isinstance(factor, FaceAvg):
        node_set = (NodeSet.RIGHT if factor.mesh.periodic
                    else NodeSet.INNER)
        return node_set, (False, False), ()
    if isinstance(factor, NodalSpace):
        node_set = factor.node_set
        return (node_set, _BOUNDARY_MEMBERSHIP[node_set],
                factor.bc.components)
    raise NotImplementedError(
        f"immersed masks on {factor!r} are not defined in "
        "iteration 1")


def _stagger_periodic(
    arr: jax.Array,
    axis: int,
    node_set: NodeSet,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """
    Combine adjacent cells onto a periodic face node set.

    Parameters
    ----------
    arr : jax.Array
        The mask, cell-positioned along ``axis``.
    axis : int
        The array axis of this factor.
    node_set : NodeSet
        The (non-center) target node set.
    combine : Callable[[jax.Array, jax.Array], jax.Array]
        The two-neighbor combination rule (AND / OR).

    Returns
    -------
    jax.Array
        The mask at the node set along ``axis``.
    """
    if node_set is NodeSet.RIGHT:
        return combine(arr, jnp.roll(arr, -1, axis))
    if node_set is NodeSet.LEFT:
        return combine(jnp.roll(arr, 1, axis), arr)
    # unreachable through the space factories (outer/inner raise on
    # periodic meshes)
    raise NotImplementedError(  # pragma: no cover
        f"node set {node_set} is undefined on periodic meshes")


def _stagger_bounded(
    arr: jax.Array,
    axis: int,
    node_set: NodeSet,
    combine: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """
    Combine adjacent cells onto a bounded face node set.

    Description
    -----------
    The exterior counts as dry (``WaterMask`` parity): a boundary
    face has a single true neighbor, combined against dry.

    Parameters
    ----------
    arr : jax.Array
        The mask, cell-positioned along ``axis``.
    axis : int
        The array axis of this factor.
    node_set : NodeSet
        The (non-center) target node set.
    combine : Callable[[jax.Array, jax.Array], jax.Array]
        The two-neighbor combination rule (AND / OR).

    Returns
    -------
    jax.Array
        The mask at the node set along ``axis``.
    """
    dry_shape = list(arr.shape)
    dry_shape[axis] = 1
    dry = jnp.zeros(tuple(dry_shape), dtype=bool)
    tail = slice(1, None)
    head = slice(None, -1)
    if node_set is NodeSet.RIGHT:
        right = jnp.concatenate(
            [_take(arr, axis, tail), dry], axis=axis)
        return combine(arr, right)
    if node_set is NodeSet.LEFT:
        left = jnp.concatenate(
            [dry, _take(arr, axis, head)], axis=axis)
        return combine(left, arr)
    if node_set is NodeSet.OUTER:
        return combine(jnp.concatenate([dry, arr], axis=axis),
                       jnp.concatenate([arr, dry], axis=axis))
    if node_set is NodeSet.INNER:
        return combine(_take(arr, axis, head),
                       _take(arr, axis, tail))
    raise NotImplementedError(
        f"immersed masks at {node_set} are not defined in "
        "iteration 1")


def _drop_constrained(
    arr: jax.Array,
    axis: int,
    membership: tuple[bool, bool],
    components: tuple[BC, ...],
) -> jax.Array:
    """
    Drop BC-constrained boundary DOFs, like the space shapes do.

    Parameters
    ----------
    arr : jax.Array
        The mask at the full node set along ``axis``.
    axis : int
        The array axis of this factor.
    membership : tuple[bool, bool]
        Whether the (left, right) end DOF sits on the boundary.
    components : tuple[BC, ...]
        The factor's BC structure (empty: no drops).

    Returns
    -------
    jax.Array
        The mask at the factor's true DOF count along ``axis``.
    """
    start: int | None = None
    stop: int | None = None
    if components:
        left_bc, right_bc = components
        if membership[0] and left_bc is not BC.NONE:
            start = 1
        if membership[1] and right_bc is not BC.NONE:
            stop = -1
    if start is None and stop is None:
        return arr
    return _take(arr, axis, slice(start, stop))


# whether the (left, right) boundary DOF is a member of the node set
# on a bounded mesh (mirrors the shape rule of the nodal spaces)
_BOUNDARY_MEMBERSHIP: dict[NodeSet, tuple[bool, bool]] = {
    NodeSet.CENTER: (False, False),
    NodeSet.LEFT: (True, False),
    NodeSet.RIGHT: (False, True),
    NodeSet.OUTER: (True, True),
    NodeSet.INNER: (False, False),
}
