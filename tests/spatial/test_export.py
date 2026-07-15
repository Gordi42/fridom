"""Tests for fridom.spatial.export (the ``f.xr`` surface)."""
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from jax.experimental import multihost_utils

import fridom.spatial.export as export_module
from fridom.spatial.export import (
    ExportLayout,
    _host_labels,
    export_layout,
    gathered_values,
    scalar_to_dataarray,
)
from fridom.spatial.fields.metadata import FieldMetadata
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.grid import Grid
from fridom.spatial.meshes.chebyshev import ChebyshevMesh
from fridom.spatial.meshes.interval import IntervalMesh
from fridom.spatial.operators.fourier import Fourier


@pytest.fixture
def mx():
    return IntervalMesh(8, (0.0, 1.0), name="x")  # periodic


@pytest.fixture
def my():
    return IntervalMesh(6, (0.0, 2.0), periodic=False, name="y")


@pytest.fixture
def grid(mx, my):
    return Grid((mx, my))


def init(x, y):
    return jnp.sin(2.0 * jnp.pi * x) + jnp.cos(y) + x * y


def nodes(grid, space, name):
    """Return the expected 1D coordinate vector of one factor."""
    return np.asarray(
        grid.evaluation_nodes(space, name=name).data).reshape(-1)


# ================================================================
#  Center product (the default field)
# ================================================================
def test_center_dataarray(grid):
    field = grid.create_field(init=init, name="temp", units="K")
    da = field.xr
    assert isinstance(da, xr.DataArray)
    assert da.dims == ("x", "y")
    assert da.name == "temp"
    assert da.attrs["units"] == "K"
    assert da.attrs["long_name"] == "Unnamed"
    assert np.array_equal(da.values, np.asarray(field.data))
    space = field.function_space
    assert np.array_equal(da.coords["x"].values,
                          nodes(grid, space, "x"))
    assert np.array_equal(da.coords["y"].values,
                          nodes(grid, space, "y"))
    # center positions carry no axis shift
    assert "c_grid_axis_shift" not in da.coords["x"].attrs
    assert "representation" not in da.coords["x"].attrs


def test_default_metadata_attrs(grid):
    da = grid.create_field(init=init).xr
    assert da.name == "unnamed"
    assert da.attrs["units"] == "n/a"


def test_nc_attrs_are_mapped(grid):
    metadata = FieldMetadata.create(
        name="b", long_name="Buoyancy", units="m/s^2",
        nc_attrs={"standard_name": "buoyancy"})
    da = grid.create_field(init=init, metadata=metadata).xr
    assert da.name == "b"
    assert da.attrs["long_name"] == "Buoyancy"
    assert da.attrs["standard_name"] == "buoyancy"


# ================================================================
#  Staggered node sets (xgcm-style dim naming, the dataset form)
# ================================================================
def test_right_outer_dims_and_shifts(grid, mx, my):
    space = mx.right * my.outer
    field = grid.create_field(space, init=lambda x, y: x * y)
    da = scalar_to_dataarray(field)
    assert da.dims == ("x_right", "y_outer")
    assert da.shape == (8, 7)
    assert np.array_equal(da.values, np.asarray(field.data))
    assert np.array_equal(da.coords["x_right"].values,
                          nodes(grid, space, "x"))
    assert np.array_equal(da.coords["y_outer"].values,
                          nodes(grid, space, "y"))
    assert da.coords["x_right"].attrs["c_grid_axis_shift"] == 0.5
    assert da.coords["y_outer"].attrs["c_grid_axis_shift"] == 0.5


def test_left_and_inner_dims(grid, mx, my):
    da = scalar_to_dataarray(
        grid.create_field(mx.left * my.inner,
                          init=lambda x, y: x + y))
    assert da.dims == ("x_left", "y_inner")
    assert da.coords["x_left"].attrs["c_grid_axis_shift"] == -0.5
    assert da.coords["y_inner"].attrs["c_grid_axis_shift"] == 0.5


# ================================================================
#  Single-field .xr export uses plain axis names
# ================================================================
def test_scalar_xr_uses_plain_names(grid, mx, my):
    space = mx.right * my.outer
    field = grid.create_field(space, init=lambda x, y: x * y)
    da = field.xr
    # a lone DataArray has nothing to collide with: plain names,
    # position preserved in the comodo shift attribute
    assert da.dims == ("x", "y")
    assert np.array_equal(da.coords["x"].values,
                          nodes(grid, space, "x"))
    assert np.array_equal(da.coords["y"].values,
                          nodes(grid, space, "y"))
    assert da.coords["x"].attrs["c_grid_axis_shift"] == 0.5
    assert da.coords["y"].attrs["c_grid_axis_shift"] == 0.5


def test_scalar_xr_plain_names_left_inner(grid, mx, my):
    da = grid.create_field(mx.left * my.inner,
                           init=lambda x, y: x + y).xr
    assert da.dims == ("x", "y")
    assert da.coords["x"].attrs["c_grid_axis_shift"] == -0.5
    assert da.coords["y"].attrs["c_grid_axis_shift"] == 0.5


# ================================================================
#  Average spaces (labels, not positions)
# ================================================================
def test_cell_avg_labels(grid, mx, my):
    space = mx.cell_avg * my.cell_avg
    field = grid.create_field(space, init=lambda x, y: x + y)
    da = field.xr
    # cell averages are labeled with the cell-center coordinates
    assert da.dims == ("x", "y")
    assert np.array_equal(da.coords["x"].values,
                          nodes(grid, space, "x"))
    # ... and marked as means, so round-trips do not reinterpret
    # the DOFs as point samples
    assert da.coords["x"].attrs["representation"] == "cell_mean"
    assert da.coords["y"].attrs["representation"] == "cell_mean"
    assert np.array_equal(da.values, np.asarray(field.data))


def test_face_avg_labels(grid, mx, my):
    space = mx.face_avg * my.face_avg
    da = scalar_to_dataarray(
        grid.create_field(space, init=lambda x, y: x * y))
    # face labels: right faces on the periodic axis, inner faces on
    # the bounded one
    assert da.dims == ("x_right", "y_inner")
    assert da.shape == (8, 5)
    assert da.coords["x_right"].attrs["representation"] == "cell_mean"
    assert da.coords["y_inner"].attrs["representation"] == "cell_mean"
    assert np.array_equal(da.coords["y_inner"].values,
                          nodes(grid, space, "y"))


# ================================================================
#  Coefficient spaces (wavenumber coordinates)
# ================================================================
def test_fourier_coefficient_field():
    mesh = IntervalMesh(8, (0.0, 1.0), name="x")
    grid = Grid((mesh,))
    space = mesh.fourier(origin=mesh.center)
    field = grid.create_field(
        space, init_coeff=lambda kx: jnp.exp(-0.1 * kx**2), name="c")
    da = field.xr
    assert da.dims == ("kx",)
    assert da.shape == (5,)  # Hermitian half spectrum of n = 8
    assert np.array_equal(da.values, np.asarray(field.data))
    expected = np.asarray(
        grid.wavenumbers(space).data).reshape(-1).real
    assert np.array_equal(da.coords["kx"].values, expected)
    assert not np.iscomplexobj(da.coords["kx"].values)
    assert da.coords["kx"].attrs["representation"] == "wavenumber"


def test_mixed_fourier_product(grid):
    field = Fourier(grid, axes=("x",)).forward(
        grid.create_field(init=init, name="spec"))
    da = field.xr
    assert da.dims == ("kx", "y")
    assert np.array_equal(da.values, np.asarray(field.data))
    assert da.coords["kx"].attrs["representation"] == "wavenumber"
    assert "representation" not in da.coords["y"].attrs


def test_chebyshev_mode_indices():
    mesh = ChebyshevMesh(8, (0.0, 1.0), name="z")
    grid = Grid((mesh,))
    space = mesh.chebyshev(mesh.outer)
    da = grid.create_field(space).xr
    assert da.dims == ("kz",)
    assert np.array_equal(da.coords["kz"].values, np.arange(9))
    assert da.coords["kz"].attrs["representation"] == "mode_index"


# ================================================================
#  Constant factors are squeezed
# ================================================================
def test_constant_axis_squeezed(grid):
    field = grid.create_field(init=init).integrate("x")
    da = field.xr
    assert da.dims == ("y",)
    assert np.array_equal(da.values,
                          np.asarray(field.data).reshape(-1))


# ================================================================
#  The values-free layout helper (export_layout / gathered_values)
# ================================================================
def test_export_layout_shape_no_gather(grid, mx, my):
    # values-free: dims/coords/attrs without touching the field data
    space = mx.right * my.outer
    field = grid.create_field(space, init=lambda x, y: x * y)
    layout = export_layout(field)
    assert isinstance(layout, ExportLayout)
    assert layout.dims == ("x_right", "y_outer")
    assert layout.shape == (8, 7)
    assert layout.kept_axes == (0, 1)  # both storage axes exported
    assert np.array_equal(layout.coords["x_right"],
                          nodes(grid, space, "x"))
    assert layout.coord_attrs["x_right"]["c_grid_axis_shift"] == 0.5
    assert layout.dtype == np.asarray(field.data).dtype
    # gathered_values reproduces the DataArray values exactly
    values = gathered_values(field, layout)
    assert np.array_equal(values, np.asarray(field.data))
    assert np.array_equal(values, scalar_to_dataarray(field).values)


def test_export_layout_squeezes_constant_axis(grid):
    # a constant factor is dropped from dims but kept_axes records
    # which storage axis survived (axis 1, the y factor)
    field = grid.create_field(init=init).integrate("x")
    layout = export_layout(field)
    assert layout.dims == ("y",)
    assert layout.kept_axes == (1,)
    assert layout.shape == (6,)
    values = gathered_values(field, layout)
    assert values.shape == (6,)
    assert np.array_equal(values,
                          np.asarray(field.data).reshape(-1))


def test_export_layout_plain_names(grid, mx, my):
    # positions_in_names=False mirrors the ScalarField.xr default
    space = mx.left * my.inner
    field = grid.create_field(space, init=lambda x, y: x + y)
    layout = export_layout(field, positions_in_names=False)
    assert layout.dims == ("x", "y")
    assert layout.coord_attrs["x"]["c_grid_axis_shift"] == -0.5


# ================================================================
#  VectorField export
# ================================================================
def test_vector_dataset(grid, mx, my):
    u = grid.create_field(mx.right * my.center,
                          init=lambda x, y: x + y, name="u")
    v = grid.create_field(mx.center * my.outer,
                          init=lambda x, y: x * y, name="v")
    ds = VectorField({"u": u, "v": v}).xr
    assert isinstance(ds, xr.Dataset)
    assert sorted(ds.data_vars) == ["u", "v"]
    assert ds["u"].dims == ("x_right", "y")
    assert ds["v"].dims == ("x", "y_outer")
    assert np.array_equal(ds["u"].values, np.asarray(u.data))
    assert np.array_equal(ds["v"].values, np.asarray(v.data))
    # staggered and centered labels of one mesh coexist
    assert "x" in ds.coords
    assert "x_right" in ds.coords


# ================================================================
#  Device-count invariance (genuine under the forced-4 suite)
# ================================================================
def build_grid(device_ids):
    mesh_x = IntervalMesh(16, (0.0, 1.0), name="x")
    mesh_y = IntervalMesh(16, (0.0, 2.0), periodic=False, name="y")
    return Grid((mesh_x, mesh_y), device_ids=device_ids)


def test_export_is_device_count_invariant(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many, one = build_grid(None), build_grid((0,))
    da_many = many.create_field(init=init, name="q").xr
    da_one = one.create_field(init=init, name="q").xr
    assert da_many.dims == da_one.dims
    assert np.array_equal(da_many.values, da_one.values)
    for dim in da_many.dims:
        assert np.array_equal(da_many.coords[dim].values,
                              da_one.coords[dim].values)


def test_spectral_export_is_device_count_invariant(forced_devices):
    if forced_devices is not None:
        assert jax.device_count() == forced_devices
    many, one = build_grid(None), build_grid((0,))
    da_many = Fourier(many, axes=("x",)).forward(
        many.create_field(init=init)).xr
    da_one = Fourier(one, axes=("x",)).forward(
        one.create_field(init=init)).xr
    assert da_many.dims == da_one.dims == ("kx", "y")
    assert np.array_equal(da_many.values, da_one.values)
    assert np.array_equal(da_many.coords["kx"].values,
                          da_one.coords["kx"].values)


# ================================================================
#  The multi-host coordinate gather (_host_labels)
# ================================================================
def test_host_labels_plain_for_addressable(grid):
    # single-process / fully-addressable: a plain np.asarray, so the
    # single-process labels are unchanged (no collective is fired).
    field = grid.create_field(init=init, name="q")
    vector = grid.evaluation_nodes(field.function_space, name="x")
    out = _host_labels(vector.data)
    assert isinstance(out, np.ndarray)
    assert np.array_equal(out.reshape(-1),
                          np.asarray(vector.data).reshape(-1))


def test_host_labels_plain_for_ndarray():
    # a plain numpy array is not a jax.Array: the plain branch again
    out = _host_labels(np.array([4.0, 5.0]))
    assert np.array_equal(out, np.array([4.0, 5.0]))


def test_host_labels_gathers_non_addressable(monkeypatch):
    # the genuine multi-process branch: a non-addressable jax.Array is
    # gathered collectively (tiled). Simulate a non-addressable array
    # and patch process_allgather so no real collective is needed.
    class FakeArray:
        is_fully_addressable = False

    seen = {}

    def fake_allgather(arr, *, tiled):
        seen["tiled"] = tiled
        seen["arr"] = arr
        return np.array([1.0, 2.0, 3.0])

    monkeypatch.setattr(
        multihost_utils, "process_allgather", fake_allgather)
    monkeypatch.setattr(
        export_module, "jax", types.SimpleNamespace(Array=FakeArray))
    fake = FakeArray()
    out = _host_labels(fake)
    assert seen["tiled"] is True  # tiled=True is mandatory for a shard
    assert seen["arr"] is fake
    assert np.array_equal(out, np.array([1.0, 2.0, 3.0]))
