"""
xarray export of spatial fields (``f.xr``).

Description
-----------
Owning class doc: ``design/specs/grid/classes/grid.md``, section 4
(Export). The field-side ``xr`` properties delegate here. Rules:

- **Axis-position labels are xgcm-style**, derived from the node
  set per factor: a center-positioned factor exports under the
  plain coordinate name (``x``), staggered node sets under
  ``<name>_<position>`` (``x_right``, ``y_outer``, ...), with the
  comodo ``c_grid_axis_shift`` attribute on shifted positions
  (-0.5 for ``left``, +0.5 for the face family; ``outer``/``inner``
  are disambiguated by their length, comodo-style).
- **Single-field exports use plain axis names**
  (``positions_in_names=False``, the ``ScalarField.xr`` default,
  owner decision 2026-07-11): a lone ``DataArray`` has no sibling
  variable to collide with, so its staggered dims export under the
  plain coordinate name (``x``) and the position survives in the
  ``c_grid_axis_shift`` attribute. Multi-variable exports
  (``VectorField.xr``, the ``fr.io.Writer`` store) keep the
  position-suffixed names: two components staggered differently
  along the same axis carry different coordinate values and cannot
  share a dim name in one ``Dataset``.
- **Average spaces export coordinate labels, not positions**:
  ``CellAvg`` under the cell-center label, ``FaceAvg`` under the
  face label, each carrying ``representation: "cell_mean"`` so
  round-trips do not silently reinterpret the DOFs as samples.
  These labels are export-layer metadata only — deliberately not
  space properties (averages have no mathematical position).
- **Coefficient factors export wavenumber coordinates** under
  ``k<name>`` (matching ``grid.wavenumbers`` naming), with
  ``representation: "wavenumber"`` (``"mode_index"`` for bases that
  are not wavenumber-indexed, e.g. Chebyshev).
- **Collapsed factors are squeezed**: a size-1 ``ConstantSpace``
  (broadcast placeholder, rules section 3.3) or ``TraceSpace``
  (boundary row) axis does not appear in the exported dims — a
  traced boundary field exports as the 2D slice it represents.
- **Data path**: values are gathered to the global true shape via
  ``decomposition.gather`` (halo and padding never leave the
  decomposition layer); coordinates come from
  ``grid.evaluation_nodes`` / ``grid.wavenumbers``.
- **Metadata**: ``FieldMetadata`` maps to ``DataArray`` attrs
  (``long_name``, ``units``, and the ``nc_attrs`` pairs); the
  metadata ``name`` becomes the ``DataArray`` name.
- **The nodes of a space** (``grid.nodes(space)`` / ``field.nodes()``,
  :func:`nodes_dataset`) are the plain-name coordinate skeleton of a
  single-field export with no data — the plotting view of the grid —
  plus, as data variables, every ``maps=`` physical coordinate the
  space resolves (the map value at the nodes, ``params=`` threaded)
  and the boolean ``wet`` mask of an immersed domain.

xarray is an optional (dev) dependency; it is imported lazily.
"""
# Wave 4: xarray export
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import numpy as np

from fridom.spatial.spaces.average import CellAvg, FaceAvg
from fridom.spatial.spaces.coefficient import (
    ChebyshevSpace,
    CoefficientSpace,
)
from fridom.spatial.spaces.nodal import NodalSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import xarray as xr

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField
    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike

# comodo/xgcm axis shift per staggered position (center: no shift)
_AXIS_SHIFT: dict[str, float] = {
    "left": -0.5,
    "right": 0.5,
    "outer": 0.5,
    "inner": 0.5,
}


def _import_xarray() -> object:
    """Import xarray lazily, with a friendly install pointer."""
    try:
        import xarray  # noqa: PLC0415 (deferred import of optional/heavy dependency)
    except ImportError as error:  # pragma: no cover — dev dep
        raise ImportError(
            "xarray is required for the .xr export; install the dev "
            "extra (uv sync --extra dev) or pip install xarray"
        ) from error
    return xarray


def _position(factor: FunctionSpace) -> str:
    """
    Resolve the xgcm axis-position label of one factor.

    Parameters
    ----------
    factor : FunctionSpace
        A nodal or average factor space.

    Returns
    -------
    str
        One of ``center``/``left``/``right``/``outer``/``inner``.
    """
    if isinstance(factor, CellAvg):
        return "center"
    if isinstance(factor, FaceAvg):
        return "right" if factor.mesh.periodic else "inner"
    if isinstance(factor, NodalSpace):
        return factor.node_set.name.lower()
    raise NotImplementedError(
        f"xarray export of {factor!r} is not defined in iteration 1")


@dataclass(frozen=True)
class SpaceLayout:

    """
    Values-free coordinate skeleton of a function space.

    Description
    -----------
    The field-independent half of :class:`ExportLayout`: the exported
    dims, their 1-D coordinate node vectors and attributes, and the
    storage-axis positions that survive the squeeze of the collapsed
    factors. Built by :func:`space_layout`; :func:`export_layout`
    adds the field's name, attributes, dtype and shape,
    :func:`nodes_dataset` uses it as the dataset of the nodes.

    Parameters
    ----------
    dims : tuple[str, ...]
        The exported dim names, in storage-axis order.
    coords : dict[str, np.ndarray]
        Per-dim 1-D coordinate node vector (real).
    coord_attrs : dict[str, dict[str, object]]
        Per-dim coordinate attributes (e.g. ``c_grid_axis_shift``,
        ``representation``, ``units``).
    kept_axes : tuple[int, ...]
        The storage-axis position of each exported dim (collapsed
        factors omitted).
    """

    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    coord_attrs: dict[str, dict[str, object]]
    kept_axes: tuple[int, ...]


@dataclass(frozen=True)
class ExportLayout:

    """
    Values-free export layout of a ``ScalarField``.

    Description
    -----------
    The label/coordinate/attribute skeleton of a field's ``xarray``
    export, built by :func:`export_layout` **without gathering the
    field data**. ``scalar_to_dataarray`` pairs it with the gathered
    values; the ``fr.io.Writer`` builds its store from the
    layout at bind, so bind needs neither ``decomposition.gather``
    nor ``xarray`` (see ``gather_free_output_plan.md``).

    Storage-axis mapping: ``dims``/``coords`` describe the *exported*
    axes only. Collapsed factors (``ConstantSpace`` broadcast
    placeholders and ``TraceSpace`` boundary rows) are squeezed out,
    so the exported axes are a subset of the field's storage axes
    (one per ``space.factors`` entry).
    ``kept_axes[i]`` is the storage-axis position of ``dims[i]``; the
    dropped positions are exactly the squeezed constant factors, and
    ``shape[i] == len(coords[dims[i]])`` is that axis' global true
    length.

    Parameters
    ----------
    name : str
        The exported variable name (the field metadata ``name``).
    dims : tuple[str, ...]
        The exported dim names, in storage-axis order.
    coords : dict[str, np.ndarray]
        Per-dim 1-D coordinate node vector (real).
    coord_attrs : dict[str, dict[str, object]]
        Per-dim coordinate attributes (e.g. ``c_grid_axis_shift``,
        ``representation``).
    attrs : dict[str, object]
        The variable attributes (``long_name``, ``units``, and the
        ``nc_attrs`` pairs).
    dtype : np.dtype
        The stored value dtype (the field storage dtype).
    shape : tuple[int, ...]
        The global true shape of the exported axes.
    kept_axes : tuple[int, ...]
        The storage-axis position of each exported dim (constant
        factors omitted).
    """

    name: str
    dims: tuple[str, ...]
    coords: dict[str, np.ndarray]
    coord_attrs: dict[str, dict[str, object]]
    attrs: dict[str, object]
    dtype: np.dtype
    shape: tuple[int, ...]
    kept_axes: tuple[int, ...]


def _host_labels(arr: object) -> np.ndarray:
    """
    Materialize a coordinate node vector on the host.

    Description
    -----------
    The coordinate labels in :func:`export_layout` are small 1-D
    vectors read off ``grid.evaluation_nodes`` / ``grid.wavenumbers``.
    Under a single-controller run (one process addressing every device)
    the vector is fully addressable and this is a plain ``np.asarray``,
    so the single-process labels are unchanged. Under a real
    multi-process run (``jax.distributed``) the vector is sharded across
    processes and non-addressable; it is then gathered collectively with
    ``multihost_utils.process_allgather(arr, tiled=True)`` — a
    collective that fires only in that genuine multi-process context,
    where :func:`export_layout` is already called symmetrically on every
    rank. ``tiled=True`` is required (a bare ``process_allgather`` errors
    on a sharded array).

    Parameters
    ----------
    arr : object
        The coordinate node vector (a ``jax`` array or array-like).

    Returns
    -------
    np.ndarray
        The host coordinate vector.
    """
    if isinstance(arr, jax.Array) and not arr.is_fully_addressable:
        from jax.experimental import multihost_utils  # noqa: PLC0415
        return np.asarray(
            multihost_utils.process_allgather(arr, tiled=True))
    return np.asarray(arr)


def space_layout(
    grid: Grid,
    space: SpaceLike,
    *,
    positions_in_names: bool = True,
) -> SpaceLayout:
    """
    Build the values-free coordinate skeleton of a function space.

    Description
    -----------
    Realizes the label/coordinate/attribute rules in the module
    docstring for the factors of ``space`` (no field, no
    ``decomposition.gather``, no ``xarray``). Collapsed factors are
    squeezed; the surviving storage-axis positions are recorded on
    ``SpaceLayout.kept_axes``. The coordinate node vectors are read
    through ``_host_labels``, a collective under a multi-process run.

    Parameters
    ----------
    grid : Grid
        The grid the space's factors live on.
    space : SpaceLike
        The function space to describe.
    positions_in_names : bool, optional
        Suffix staggered dims xgcm-style (``x_right``); False
        exports every position under the plain axis name, keeping
        the position in the ``c_grid_axis_shift`` attribute
        (default: True).

    Returns
    -------
    SpaceLayout
        The dims/coords/attrs skeleton and the storage-axis mapping.
    """
    dims: list[str] = []
    coords: dict[str, np.ndarray] = {}
    coord_attrs: dict[str, dict[str, object]] = {}
    kept_axes: list[int] = []
    for axis, factor in enumerate(space.factors):
        if len(factor.shape) != 1:
            raise NotImplementedError(
                f"xarray export of the multi-axis factor {factor!r} "
                "is not defined in iteration 1")
        if factor.collapses_axis:
            continue
        name = factor.names[0]
        attrs: dict[str, object] = {}
        if isinstance(factor, CoefficientSpace):
            dim = f"k{name}"
            vector = grid.wavenumbers(space, name=name)
            attrs["representation"] = (
                "mode_index" if isinstance(factor, ChebyshevSpace)
                else "wavenumber")
        else:
            position = _position(factor)
            dim = (name if position == "center" or not positions_in_names
                   else f"{name}_{position}")
            vector = grid.evaluation_nodes(space, name=name)
            if position in _AXIS_SHIFT:
                attrs["c_grid_axis_shift"] = _AXIS_SHIFT[position]
            if isinstance(factor, CellAvg | FaceAvg):
                attrs["representation"] = "cell_mean"
            # a chart that knows what its coordinate values MEAN
            # (angles on the sphere) states it here; everything else
            # takes the model's scaling-rendered unit at the writer
            declared = grid.coordinate_units.get(name)
            if declared is not None:
                attrs["units"] = declared
        labels = _host_labels(vector.data).reshape(-1)
        if np.iscomplexobj(labels):
            # wavenumbers are real by construction; the accessor
            # stores them at the coefficient space's complex dtype
            labels = labels.real
        dims.append(dim)
        coords[dim] = labels
        coord_attrs[dim] = attrs
        kept_axes.append(axis)
    return SpaceLayout(
        dims=tuple(dims),
        coords=coords,
        coord_attrs=coord_attrs,
        kept_axes=tuple(kept_axes),
    )


def export_layout(
    field: ScalarField,
    *,
    positions_in_names: bool = True,
) -> ExportLayout:
    """
    Build the values-free export layout of a ``ScalarField``.

    Description
    -----------
    Realizes the label/coordinate/attribute rules in the module
    docstring without touching the field data (no
    ``decomposition.gather``, no ``xarray``): the
    :func:`space_layout` of the field's space, plus the field's name,
    attributes, dtype and global shape. Constant factors are
    squeezed; the surviving storage-axis positions are recorded on
    ``ExportLayout.kept_axes`` so callers can drop the same positions
    from a gathered array (see :func:`gathered_values`).

    Parameters
    ----------
    field : ScalarField
        The field whose layout to describe.
    positions_in_names : bool, optional
        Suffix staggered dims xgcm-style (``x_right``); False
        exports every position under the plain axis name, keeping
        the position in the ``c_grid_axis_shift`` attribute. Plain
        names are only safe for a lone ``DataArray``; datasets
        combining differently staggered variables need the suffixed
        names (default: True).

    Returns
    -------
    ExportLayout
        The dims/coords/attrs skeleton and the storage-axis mapping.
    """
    skeleton = space_layout(
        field.grid, field.function_space,
        positions_in_names=positions_in_names)
    metadata = field.metadata
    return ExportLayout(
        name=metadata.name,
        dims=skeleton.dims,
        coords=skeleton.coords,
        coord_attrs=skeleton.coord_attrs,
        attrs={
            "long_name": metadata.long_name,
            "units": metadata.units,
            **dict(metadata.nc_attrs),
        },
        dtype=np.dtype(field._data.dtype),  # noqa: SLF001 — storage seam
        shape=tuple(len(skeleton.coords[dim]) for dim in skeleton.dims),
        kept_axes=skeleton.kept_axes,
    )


def gathered_values(
    field: ScalarField, layout: ExportLayout,
) -> np.ndarray:
    """
    Gather a field to the global true shape of its layout.

    Description
    -----------
    The values half of the export split: ``decomposition.gather`` to
    the global true array (halo and padding never leave the
    decomposition layer), then squeeze the constant-factor axes that
    ``layout`` dropped. Requires no ``xarray`` — the
    ``fr.io.Writer`` write path consumes it directly.

    Parameters
    ----------
    field : ScalarField
        The field to gather.
    layout : ExportLayout
        The field's layout (its ``kept_axes`` select the surviving
        storage axes).

    Returns
    -------
    np.ndarray
        The host global true-shape values.
    """
    space = field.function_space
    values = np.asarray(field.grid.decomposition.gather(
        field._data, space))  # noqa: SLF001 — storage seam
    dropped = tuple(axis for axis in range(values.ndim)
                    if axis not in layout.kept_axes)
    if dropped:
        values = np.squeeze(values, axis=dropped)
    return values


def scalar_to_dataarray(
    field: ScalarField,
    *,
    positions_in_names: bool = True,
) -> xr.DataArray:
    """
    Export a ``ScalarField`` to an ``xarray.DataArray``.

    Description
    -----------
    Realizes the export rules in the module docstring; the entry
    point is the ``ScalarField.xr`` property (which passes
    ``positions_in_names=False``). Composes the values-free
    :func:`export_layout` with :func:`gathered_values` and the
    ``xarray`` assembly.

    Parameters
    ----------
    field : ScalarField
        The field to export.
    positions_in_names : bool, optional
        Suffix staggered dims xgcm-style (``x_right``); False
        exports every position under the plain axis name, keeping
        the position in the ``c_grid_axis_shift`` attribute. Plain
        names are only safe for a lone ``DataArray``; datasets
        combining differently staggered variables need the suffixed
        names (default: True).

    Returns
    -------
    xr.DataArray
        The global true-shape data with labeled coordinates.
    """
    xarray = _import_xarray()
    layout = export_layout(field, positions_in_names=positions_in_names)
    values = gathered_values(field, layout)
    return xarray.DataArray(
        values,
        dims=layout.dims,
        coords={
            dim: (dim, layout.coords[dim], layout.coord_attrs[dim])
            for dim in layout.dims},
        name=layout.name,
        attrs=layout.attrs)


def vector_to_dataset(vector: VectorField) -> xr.Dataset:
    """
    Export a ``VectorField`` to an ``xarray.Dataset``.

    Description
    -----------
    One data variable per component, keyed by component name; the
    per-component coordinate labels merge (staggered components
    contribute distinct dims). The entry point is the
    ``VectorField.xr`` property.

    Parameters
    ----------
    vector : VectorField
        The collection to export.

    Returns
    -------
    xr.Dataset
        The dataset of the components' exports.
    """
    xarray = _import_xarray()
    return xarray.Dataset({
        name: scalar_to_dataarray(component)
        for name, component in vector.components.items()})


def _data_variable(
    field: ScalarField,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Gather one field as a plain-name xarray variable tuple."""
    layout = export_layout(field, positions_in_names=False)
    return (layout.dims, gathered_values(field, layout))


def nodes_dataset(
    grid: Grid,
    space: SpaceLike,
    *,
    params: Mapping[str, ScalarField] | VectorField | None = None,
) -> xr.Dataset:
    """
    Export the nodes of one function space as an ``xarray.Dataset``.

    Description
    -----------
    The plotting view of the grid, the entry points being
    ``grid.nodes(space)`` and ``field.nodes()``. The dataset is the
    plain-name coordinate skeleton of a single-field export with no
    data: the dims are the space's coordinate names, each dimension
    coordinate the 1-D node vector of its factor **at the factor's
    own position** (centres for ``center``/``cell_avg``, faces for
    the face family, the position kept in ``c_grid_axis_shift``),
    collapsed factors squeezed. Two kinds of data variable join it:
    every ``maps=`` physical coordinate the space resolves
    (``grid.mapping.mapped_coords`` lists what each one needs), the
    map value at the nodes under ``params=`` — the static defaults
    when None, the current geometry when the model state is passed —
    and, on a space resolving every grid coordinate of an immersed
    grid, the boolean ``wet`` mask. The tensor coordinates stay 1-D
    (xarray broadcasts them when a plot asks for two), so
    ``ds.plot.scatter(x="y", y="z")`` draws the nodes,
    ``ds.isel(x=0)`` takes a section and ``ds.stack(node=ds.dims)``
    is the list of points. Gathered once on the host; under a
    multi-process run the gather is collective (call it on every
    rank).

    Parameters
    ----------
    grid : Grid
        The grid the space's factors live on.
    space : SpaceLike
        The function space whose nodes to export.
    params : Mapping[str, ScalarField] | VectorField | None, optional
        Parameter fields of the grid's mapping by name, or a
        ``VectorField`` (the model state) they are picked out of;
        None evaluates the static defaults (default: None).

    Returns
    -------
    xr.Dataset
        The node coordinates, with the physical coordinates and the
        wet mask as data variables where the grid carries them.
    """
    xarray = _import_xarray()
    skeleton = space_layout(grid, space, positions_in_names=False)
    coords = {
        dim: (dim, skeleton.coords[dim], skeleton.coord_attrs[dim])
        for dim in skeleton.dims}
    # the coordinates the space resolves with a placed node set: a
    # mapped coordinate needs every coordinate it depends on, the
    # immersed mask every coordinate of the grid
    resolved = {
        name for factor in space.factors
        if isinstance(factor, NodalSpace | CellAvg | FaceAvg)
        for name in factor.names}
    variables: dict[str, tuple[tuple[str, ...], np.ndarray]] = {}
    mapping = grid.mapping
    if mapping is not None:
        for name, deps in mapping.mapped_coords.items():
            if set(deps) <= resolved:
                variables[name] = _data_variable(
                    mapping.positions(space, name, params=params))
    immersed = grid.immersed
    if immersed is not None and resolved == set(grid.names):
        variables["wet"] = _data_variable(immersed.mask(space))
    return xarray.Dataset(variables, coords=coords)
