r"""
``GridTransfer``: adjoint restriction / prolongation between two grids.

Description
-----------
Owning class doc: ``design/plans/active/multigrid_pathway_plan.md`` §A4
(MG-D1/D2). A free-standing **grid-pair** operator (not a ``.to`` kind
and not a registry row): built after both a fine grid and its coarse
sibling exist, it transfers real scalar fields on the collocated cell
spaces (``Center`` nodal, ``CellAvg``) between them. A ``ConstantSpace``
broadcast factor (shape ``(1,)``) is carried through untransferred — it
has no data axis to coarsen and drops out of the volume weighting — so a
barotropic ``Profile`` field (cell horizontals, constant vertical)
transfers on its horizontal factors alone (GM-D3).

The pair is **adjoint** under the two grids' measure-weighted L2
products,

.. math::
    \langle R f, g \rangle_H = \langle f, P g \rangle_h ,

and **conservative** (``P`` preserves constants, so ``R`` preserves
integrals). Prolongation ``P`` is implemented explicitly — piecewise
constant at ``order=1``, separable cell-centered linear at ``order=2``
(``3/4`` own + ``1/4`` neighbor for a factor-2 axis, with a one-sided
row on bounded axes) — and restriction ``R`` is its **literal
measure-weighted adjoint** ``R = M_H^{-1} P^T M_h`` (``jax.linear_transpose``
of ``P``, wrapped in the two cell-volume weightings), so the adjoint
identity holds by construction on every row, boundary rows included.

Kernels run in the **logical frame** (``field.data``, halos/padding
stripped) and re-enter storage through the target grid's field factory;
per-shard-aligned axes stay shard-local under GSPMD (the order-1
restriction is a collective-free block reduce, order-2 costs
halo-class permutes), and a non-nesting blocking falls back to the
global reblock path (correct, slower) that GSPMD inserts automatically.
"""
# Multigrid pathway plan, phase A (A4): GridTransfer
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.framework.utils import dtype_real, jaxify
from fridom.spatial.decomposition.decomposition import (
    check_level_shardability,
)
from fridom.spatial.fields.storage import factor_axes
from fridom.spatial.meshes.structured_1d import StructuredMesh1D
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodalSpace, NodeSet
from fridom.spatial.spaces.tensor_product import TensorProductSpace

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike

#: the transfer-pair orders iteration 1 realizes
_ORDERS = (1, 2)


@partial(jaxify, dynamic=())
class GridTransfer:

    """
    Restriction / prolongation between two coexisting grids.

    Description
    -----------
    Grid-pair-bound (CS-15 shape): validates same coordinate names,
    per-name integer cell-count ratios (1 = uncoarsened axis), and one
    shared device set, then re-checks both levels' shardability (the A5
    gate). See the module docstring for the adjoint construction.

    Parameters
    ----------
    fine : Grid
        The fine grid (transfer source for ``restrict``).
    coarse : Grid
        The coarse grid; every axis ratio fine/coarse must be a
        positive integer.
    order : int, optional
        Transfer-pair order, 1 (piecewise constant) or 2 (cell-centered
        linear) (default: 2).
    """

    def __init__(self, fine: Grid, coarse: Grid, *,
                 order: int = 2) -> None:
        """Bind the grid pair; validate ratios, devices, levels."""
        if order not in _ORDERS:
            raise ValueError(
                f"GridTransfer order must be one of {_ORDERS} "
                f"(1 = piecewise constant, 2 = cell-centered linear), "
                f"got {order!r}")
        if fine.names != coarse.names:
            raise ValueError(
                "GridTransfer needs the fine and coarse grids to carry "
                f"the same coordinate names, got {fine.names} and "
                f"{coarse.names}")
        if _device_set(fine) != _device_set(coarse):
            raise ValueError(
                "GridTransfer needs the fine and coarse grids on one "
                "shared device set (a hierarchy level lives on the same "
                "device mesh, MG-D5)")
        self._fine: Grid = fine
        self._coarse: Grid = coarse
        self._order: int = order
        fine_meshes = {name: mesh for mesh in fine.factors
                       for name in mesh.names}
        coarse_meshes = {name: mesh for mesh in coarse.factors
                         for name in mesh.names}
        ratios: dict[str, int] = {}
        periodic: dict[str, bool] = {}
        for name in fine.names:
            f_mesh = fine_meshes[name]
            c_mesh = coarse_meshes[name]
            ratios[name] = _axis_ratio(name, f_mesh, c_mesh)
            periodic[name] = bool(getattr(f_mesh, "periodic", False))
        self._ratios: dict[str, int] = ratios
        self._periodic: dict[str, bool] = periodic
        self._fine_meshes: dict[str, object] = fine_meshes
        self._coarse_meshes: dict[str, object] = coarse_meshes
        check_level_shardability(
            fine.factors, fine.decomposition.default_layout,
            fine.decomposition.halo, fine.decomposition.device_count,
            level="fine level")
        check_level_shardability(
            coarse.factors, coarse.decomposition.default_layout,
            coarse.decomposition.halo,
            coarse.decomposition.device_count, level="coarse level")

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def fine(self) -> Grid:
        """The fine grid of the pair."""
        return self._fine

    @property
    def coarse(self) -> Grid:
        """The coarse grid of the pair."""
        return self._coarse

    @property
    def order(self) -> int:
        """The transfer-pair order (1 or 2)."""
        return self._order

    @property
    def ratios(self) -> dict[str, int]:
        """Per-name fine/coarse cell-count ratios (a copy)."""
        return dict(self._ratios)

    # ================================================================
    #  Transfer operators
    # ================================================================
    def prolong(self, field: ScalarField) -> ScalarField:
        r"""
        Coarse-grid field -> fine-grid field (``P``).

        Description
        -----------
        The measure-free interpolation: piecewise constant at
        ``order=1``, separable cell-centered linear at ``order=2``.
        ``P`` preserves constants, so its adjoint ``restrict`` preserves
        integrals.

        Parameters
        ----------
        field : ScalarField
            A real scalar field on the coarse grid's collocated cell
            spaces (``Center`` / ``CellAvg``, spanning every
            coordinate).

        Returns
        -------
        ScalarField
            The prolonged field on the fine grid (same metadata).
        """
        self._check_input(field, self._coarse, "prolong")
        space = field.function_space
        specs = self._axis_specs(space)
        fine_data = self._prolong_array(field.data, specs)
        fine_space = self._sibling_space(space, self._fine_meshes)
        return self._fine.create_field(
            fine_space, data=fine_data, metadata=field.metadata)

    def restrict(self, field: ScalarField) -> ScalarField:
        r"""
        Fine-grid field -> coarse-grid field (``R = M_H^{-1} P^T M_h``).

        Description
        -----------
        The literal measure-weighted adjoint of :meth:`prolong`: weight
        by the fine cell volume, transpose ``P``, divide by the coarse
        cell volume. Adjoint to ``prolong`` under the two grids'
        measure-weighted L2 products by construction, hence conservative
        (``order=1`` is the volume-weighted block average, a left
        inverse of ``prolong``).

        Parameters
        ----------
        field : ScalarField
            A real scalar field on the fine grid's collocated cell
            spaces (``Center`` / ``CellAvg``, spanning every
            coordinate).

        Returns
        -------
        ScalarField
            The restricted field on the coarse grid (same metadata).
        """
        self._check_input(field, self._fine, "restrict")
        space = field.function_space
        specs = self._axis_specs(space)
        coarse_space = self._sibling_space(space, self._coarse_meshes)
        names = self._coarsened_names(space)
        weighted = field.data * _volume(self._fine, space, names)

        def prolong_fn(coarse: jax.Array) -> jax.Array:
            return self._prolong_array(coarse, specs)

        template = jax.ShapeDtypeStruct(
            tuple(coarse_space.shape), weighted.dtype)
        (cotangent,) = jax.linear_transpose(
            prolong_fn, template)(weighted)
        coarse_vol = _volume(self._coarse, coarse_space, names)
        coarse_data = cotangent / coarse_vol
        return self._coarse.create_field(
            coarse_space, data=coarse_data, metadata=field.metadata)

    # ================================================================
    #  Internal helpers
    # ================================================================
    def _prolong_array(
        self,
        data: jax.Array,
        specs: tuple[tuple[int, int, bool], ...],
    ) -> jax.Array:
        """Apply the per-axis prolongation over the coarsened axes."""
        for axis, ratio, periodic in specs:
            data = _prolong_axis(data, axis, ratio, self._order,
                                 periodic)
        return data

    def _axis_specs(
        self, space: SpaceLike,
    ) -> tuple[tuple[int, int, bool], ...]:
        """List ``(axis, ratio, periodic)`` for the coarsened axes."""
        specs: list[tuple[int, int, bool]] = []
        for factor, axis in factor_axes(space):
            name = factor.names[0]
            ratio = self._ratios[name]
            if ratio > 1:
                specs.append((axis, ratio, self._periodic[name]))
        return tuple(specs)

    def _coarsened_names(self, space: SpaceLike) -> tuple[str, ...]:
        """Return the coordinate names this transfer coarsens."""
        return tuple(
            factor.names[0] for factor in space.factors
            if self._ratios[factor.names[0]] > 1)

    def _sibling_space(
        self,
        space: SpaceLike,
        target_meshes: dict[str, object],
    ) -> SpaceLike:
        """Rebuild the space's cell families on the target meshes."""
        factors = tuple(
            _sibling_factor(factor, target_meshes[factor.names[0]])
            for factor in space.factors)
        if len(factors) == 1:
            return factors[0]
        return TensorProductSpace.of(*factors)

    def _check_input(
        self, field: ScalarField, grid: Grid, op: str,
    ) -> None:
        """Reject fields off-grid, complex, or off the cell family."""
        if field.grid is not grid:
            raise ValueError(
                f"GridTransfer.{op} expects a field on the "
                f"{'fine' if grid is self._fine else 'coarse'} grid of "
                "the pair; got a field on a different grid")
        space = field.function_space
        if space.scalars is not Scalars.REAL:
            raise ValueError(
                f"GridTransfer.{op} is real scalar fields only "
                "(iteration 1); complex and coefficient-space fields "
                "are designed-for")
        names = tuple(name for factor in space.factors
                      for name in factor.names)
        if set(names) != set(grid.names):
            raise ValueError(
                f"GridTransfer.{op} needs a cell field spanning every "
                f"coordinate {grid.names}, got a field over {names}")
        for factor in space.factors:
            if not _is_cell_factor(factor):
                raise ValueError(
                    f"GridTransfer.{op} transfers collocated cell "
                    "factors (Center nodal or CellAvg), plus a "
                    "pass-through ConstantSpace; the factor "
                    f"{factor!r} is a staggered face or coefficient "
                    "space (designed-for: coupling fluxes, CS-16 "
                    "traces)")


# ================================================================
#  Per-axis prolongation kernels (measure-free; logical frame)
# ================================================================
def _prolong_axis(
    arr: jax.Array,
    axis: int,
    ratio: int,
    order: int,
    periodic: bool,
) -> jax.Array:
    """
    Prolong one coarse axis to the fine cell count (``ratio`` sub-cells).

    Description
    -----------
    Splits into a ``(m, ratio)`` block along ``axis`` and fills each
    sub-cell: at ``order=1`` the parent value (piecewise constant), at
    ``order=2`` the cell-centered linear blend of the parent and one
    neighbor (``own + neighbor`` weights summing to 1, so constants are
    preserved). Neighbors wrap on periodic axes and edge-clamp on
    bounded ones — the clamp *is* the one-sided boundary row.

    Parameters
    ----------
    arr : jax.Array
        The coarse array (logical frame).
    axis : int
        The coarsened array axis.
    ratio : int
        The fine/coarse cell-count ratio (>= 2 here).
    order : int
        The transfer order (1 or 2).
    periodic : bool
        Whether the axis wraps.

    Returns
    -------
    jax.Array
        The array with ``axis`` grown by ``ratio``.
    """
    own = jnp.expand_dims(arr, axis + 1)
    if order == 1:
        target = list(own.shape)
        target[axis + 1] = ratio
        blocks = jnp.broadcast_to(own, tuple(target))
        return _merge_axis(blocks, axis)
    own_w, left_w, right_w = _order2_weights(ratio)
    if periodic:
        left = jnp.roll(arr, 1, axis=axis)
        right = jnp.roll(arr, -1, axis=axis)
    else:
        left = _clamp_edge(arr, axis, side=-1)
        right = _clamp_edge(arr, axis, side=1)
    w_shape = [1] * own.ndim
    w_shape[axis + 1] = ratio
    blocks = (own * own_w.reshape(w_shape)
              + jnp.expand_dims(left, axis + 1) * left_w.reshape(w_shape)
              + jnp.expand_dims(right, axis + 1)
              * right_w.reshape(w_shape))
    return _merge_axis(blocks, axis)


def _order2_weights(
    ratio: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""
    Cell-centered linear sub-cell weights (own, left, right).

    Description
    -----------
    Fine sub-cell ``j`` of a coarse cell sits at the computational
    offset ``t_j = (j + 0.5) / ratio - 0.5`` from the parent center (in
    coarse-cell units). Linear interpolation puts weight ``1 - |t_j|``
    on the parent and ``|t_j|`` on the left neighbor when ``t_j < 0``,
    else the right neighbor — so ``own + left + right == 1`` per
    sub-cell (constant preservation). For a factor-2 axis this is the
    ``3/4`` own + ``1/4`` neighbor stencil.

    Parameters
    ----------
    ratio : int
        The fine/coarse cell-count ratio.

    Returns
    -------
    tuple[jax.Array, jax.Array, jax.Array]
        The per-sub-cell own, left, and right weight vectors.
    """
    j = jnp.arange(ratio, dtype=dtype_real())
    t = (j + 0.5) / ratio - 0.5
    own_w = 1.0 - jnp.abs(t)
    left_w = jnp.where(t < 0.0, jnp.abs(t), 0.0)
    right_w = jnp.where(t >= 0.0, jnp.abs(t), 0.0)
    return own_w, left_w, right_w


def _clamp_edge(arr: jax.Array, axis: int, *, side: int) -> jax.Array:
    """
    Shift ``arr`` by one along ``axis`` with edge clamping.

    Description
    -----------
    ``side == -1`` returns the left neighbor ``arr[I - 1]`` with
    ``arr[0]`` clamped onto the first slot; ``side == +1`` the right
    neighbor ``arr[I + 1]`` with ``arr[-1]`` clamped onto the last —
    the two one-sided boundary rows of the cell-centered stencil.

    Parameters
    ----------
    arr : jax.Array
        The coarse array.
    axis : int
        The axis to shift along.
    side : int
        ``-1`` for the left neighbor, ``+1`` for the right.

    Returns
    -------
    jax.Array
        The clamped-shifted array (same shape).
    """
    n = arr.shape[axis]
    if side < 0:
        head = _slice_axis(arr, axis, 0, 1)
        body = _slice_axis(arr, axis, 0, n - 1)
        return jnp.concatenate([head, body], axis=axis)
    body = _slice_axis(arr, axis, 1, n)
    tail = _slice_axis(arr, axis, n - 1, n)
    return jnp.concatenate([body, tail], axis=axis)


def _slice_axis(
    arr: jax.Array, axis: int, start: int, stop: int,
) -> jax.Array:
    """Slice ``arr[start:stop]`` along ``axis``."""
    index: list[slice] = [slice(None)] * arr.ndim
    index[axis] = slice(start, stop)
    return arr[tuple(index)]


def _merge_axis(arr: jax.Array, axis: int) -> jax.Array:
    """Merge the block sub-axis ``axis + 1`` back into ``axis``."""
    shape = list(arr.shape)
    merged = shape[axis] * shape[axis + 1]
    new_shape = [*shape[:axis], merged, *shape[axis + 2:]]
    return arr.reshape(new_shape)


# ================================================================
#  Measures and validation
# ================================================================
def _volume(
    grid: Grid, space: SpaceLike, names: tuple[str, ...],
) -> jax.Array | float:
    """
    Product of the per-axis cell measures over the coarsened names.

    Description
    -----------
    The diagonal cell-volume weighting of the coarsened axes only
    (``grid.measure`` per name — the primal cell width, mapped-mesh
    aware). Passthrough axes cancel in ``M_H^{-1} P^T M_h`` and are
    omitted; an empty set yields the scalar ``1``.

    Parameters
    ----------
    grid : Grid
        The grid owning the field/space.
    space : SpaceLike
        The (cell) space to weight.
    names : tuple[str, ...]
        The coarsened coordinate names.

    Returns
    -------
    jax.Array | float
        The broadcastable cell-volume weight (``1`` when ``names`` is
        empty).
    """
    volume: jax.Array | float = 1.0
    for name in names:
        volume = volume * grid.measure(space, name).data
    return volume


def _axis_ratio(name: str, fine_mesh: object, coarse_mesh: object) -> int:
    """Validate and return the integer fine/coarse ratio of one axis."""
    fine_n = getattr(fine_mesh, "n_cells", None)
    coarse_n = getattr(coarse_mesh, "n_cells", None)
    if fine_n is None or coarse_n is None:
        raise ValueError(
            f"GridTransfer needs cell-counted meshes along {name!r}, "
            f"got {fine_mesh!r} and {coarse_mesh!r}")
    if coarse_n < 1 or fine_n % coarse_n != 0:
        raise ValueError(
            f"the fine/coarse cell-count ratio along {name!r} must be "
            f"a positive integer, got {fine_n} / {coarse_n}")
    ratio = fine_n // coarse_n
    if (ratio > 1 and (not isinstance(fine_mesh, StructuredMesh1D)
                       or bool(getattr(fine_mesh, "periodic", False))
                       != bool(getattr(coarse_mesh, "periodic", False)))):
        raise ValueError(
            f"a coarsened axis {name!r} needs matching-topology "
            "StructuredMesh1D factors on both grids")
    return ratio


def _is_cell_factor(factor: FunctionSpace) -> bool:
    """Whether ``factor`` transfers as a cell factor (or passes through).

    A ``ConstantSpace`` broadcast factor passes through: it carries no
    data axis to transfer (shape ``(1,)``, never coarsened, ratio 1) and
    drops out of the coarsened-name volume weighting, so a barotropic
    ``Profile`` cell field (cell x/y, constant z) transfers exactly on
    its horizontal factors (GM-D3). Otherwise the factor must be a
    collocated cell space (``Center`` nodal / ``CellAvg``).
    """
    if factor.is_constant:
        return True
    if isinstance(factor, CellAvg):
        return True
    return (isinstance(factor, NodalSpace)
            and factor.node_set is NodeSet.CENTER)


def _sibling_factor(
    factor: FunctionSpace, target_mesh: object,
) -> FunctionSpace:
    """Rebuild one cell factor's family on the target (sibling) mesh."""
    if factor.is_constant:
        return target_mesh.constant
    if isinstance(factor, CellAvg):
        return target_mesh.cell_avg
    if (isinstance(factor, NodalSpace)
            and factor.node_set is NodeSet.CENTER):
        return target_mesh.center
    raise ValueError(  # pragma: no cover — _check_input screens first
        f"GridTransfer transfers Center/CellAvg cell factors only, "
        f"got {factor!r}")


def _device_set(grid: Grid) -> frozenset[int]:
    """Return the device ids backing a grid's decomposition."""
    mesh = grid.decomposition.device_mesh
    return frozenset(int(dev.id) for dev in mesh.devices.flatten())
