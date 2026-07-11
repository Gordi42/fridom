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
- **Constant factors are squeezed**: a ``ConstantSpace`` axis is a
  broadcast placeholder (rules section 3.3) and does not appear in
  the exported dims.
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

from typing import TYPE_CHECKING

import numpy as np

from fridom.spatial.spaces.average import CellAvg, FaceAvg
from fridom.spatial.spaces.coefficient import (
    ChebyshevSpace,
    CoefficientSpace,
)
from fridom.spatial.spaces.constant import ConstantSpace
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


def scalar_to_dataarray(field: ScalarField) -> xr.DataArray:
    """
    Export a ``ScalarField`` to an ``xarray.DataArray``.

    Description
    -----------
    Realizes the export rules in the module docstring; the entry
    point is the ``ScalarField.xr`` property.

    Parameters
    ----------
    field : ScalarField
        The field to export.

    Returns
    -------
    xr.DataArray
        The global true-shape data with labeled coordinates.
    """
    xarray = _import_xarray()
    grid = field.grid
    space = field.function_space
    values = np.asarray(grid.decomposition.gather(
        field._data, space))  # noqa: SLF001 — storage seam
    dims: list[str] = []
    coords: dict[str, tuple[str, np.ndarray, dict[str, object]]] = {}
    squeeze: list[int] = []
    for axis, factor in enumerate(space.factors):
        if len(factor.shape) != 1:
            raise NotImplementedError(
                f"xarray export of the multi-axis factor {factor!r} "
                "is not defined in iteration 1")
        if isinstance(factor, ConstantSpace):
            squeeze.append(axis)
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
            dim = (name if position == "center"
                   else f"{name}_{position}")
            vector = grid.evaluation_nodes(space, name=name)
            if position in _AXIS_SHIFT:
                attrs["c_grid_axis_shift"] = _AXIS_SHIFT[position]
            if isinstance(factor, CellAvg | FaceAvg):
                attrs["representation"] = "cell_mean"
        labels = np.asarray(vector.data).reshape(-1)
        if np.iscomplexobj(labels):
            # wavenumbers are real by construction; the accessor
            # stores them at the coefficient space's complex dtype
            labels = labels.real
        dims.append(dim)
        coords[dim] = (dim, labels, attrs)
    if squeeze:
        values = np.squeeze(values, axis=tuple(squeeze))
    metadata = field.metadata
    return xarray.DataArray(
        values,
        dims=tuple(dims),
        coords=coords,
        name=metadata.name,
        attrs={
            "long_name": metadata.long_name,
            "units": metadata.units,
            **dict(metadata.nc_attrs),
        })


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
