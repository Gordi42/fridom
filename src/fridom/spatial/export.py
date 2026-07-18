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
  (``VectorField.xr``, the ``fr.model.io.Writer`` store) keep the
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
    import xarray as xr

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )

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
class ExportLayout:

    """
    Values-free export layout of a ``ScalarField``.

    Description
    -----------
    The label/coordinate/attribute skeleton of a field's ``xarray``
    export, built by :func:`export_layout` **without gathering the
    field data**. ``scalar_to_dataarray`` pairs it with the gathered
    values; the ``fr.model.io.Writer`` builds its store from the
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
    ``decomposition.gather``, no ``xarray``). Constant factors are
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
    grid = field.grid
    space = field.function_space
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
        labels = _host_labels(vector.data).reshape(-1)
        if np.iscomplexobj(labels):
            # wavenumbers are real by construction; the accessor
            # stores them at the coefficient space's complex dtype
            labels = labels.real
        dims.append(dim)
        coords[dim] = labels
        coord_attrs[dim] = attrs
        kept_axes.append(axis)
    metadata = field.metadata
    return ExportLayout(
        name=metadata.name,
        dims=tuple(dims),
        coords=coords,
        coord_attrs=coord_attrs,
        attrs={
            "long_name": metadata.long_name,
            "units": metadata.units,
            **dict(metadata.nc_attrs),
        },
        dtype=np.dtype(field._data.dtype),  # noqa: SLF001 — storage seam
        shape=tuple(len(coords[dim]) for dim in dims),
        kept_axes=tuple(kept_axes),
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
    ``fr.model.io.Writer`` write path consumes it directly.

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
