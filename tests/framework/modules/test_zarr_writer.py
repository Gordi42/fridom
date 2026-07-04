"""Tests for the ZarrWriter module."""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import zarr

import fridom.framework as fr


# ================================================================
#  Fixtures
# ================================================================
@pytest.fixture
def parent_directory():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def directory_name(parent_directory):
    yield Path(parent_directory) / "snapshots"
    # clean up
    if (Path(parent_directory) / "snapshots").exists():
        shutil.rmtree(Path(parent_directory) / "snapshots")


@pytest.fixture
def zarr_module(directory_name):
    return fr.modules.ZarrWriter(
        filename="test",
        write_trigger=fr.ClockTrigger(time_interval=np.timedelta64(1, "m")),
        directory=directory_name,
    )


@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(shape=(16, 8), domain_size=(1, 1))
    mset = fr.ModelSettingsBase(grid=grid)
    mset.time_stepper.dt = np.timedelta64(30, "s")
    def _state_constructor() -> fr.VectorField:
        var1 = fr.ScalarField(
            mset, name="var1", long_name="Variable 1", units="unit1")
        var2 = fr.ScalarField(
            mset, name="var2", long_name="Variable 2", units="unit2")
        return fr.VectorField(mset, field_list=[var1, var2])
    mset.state_constructor = _state_constructor
    return mset


# ================================================================
#  Tests
# ================================================================
def test_initialization(zarr_module, directory_name):
    assert zarr_module.filename == Path(directory_name) / "test"
    assert zarr_module.directory == directory_name
    assert zarr_module.write_trigger is not None
    assert zarr_module.restart_trigger is None
    assert zarr_module.add_timestamp is False
    assert zarr_module.get_variables is not None


def test_setup_creates_directory(zarr_module, directory_name, mset):
    assert not directory_name.exists()
    zarr_module.setup(mset=mset.setup())
    assert directory_name.exists()


def test_model_run(mset, zarr_module, directory_name):
    mset.diagnostics.add_module(zarr_module)
    mset.setup()
    model = fr.Model(mset)
    model.run(runlen=np.timedelta64(1, "h"))

    file_path = directory_name / "test.zarr"
    assert file_path.exists()

    store = zarr.open(file_path)
    assert store.attrs["Conventions"] == "CF-1.10"
    assert store["var1"].attrs["units"] == "unit1"
    assert store["var2"].attrs["units"] == "unit2"
    assert store["var1"].shape == (61, 8, 16)
    assert store["x"].shape == (16,)
    assert store["y"].shape == (8,)
    assert np.allclose(store["time"][:3], [0.0, 60.0, 120.0])


def test_start_closes_open_file(zarr_module, mset):
    zarr_module.setup(mset=mset.setup())
    zarr_module._file_is_open = True
    zarr_module.start()
    assert not zarr_module._file_is_open


def test_stop_closes_open_file(zarr_module, mset):
    zarr_module.setup(mset=mset.setup())
    zarr_module._file_is_open = True
    zarr_module.stop()
    assert not zarr_module._file_is_open
    assert zarr_module._zarr_store is None


def test_repeated_close(zarr_module, mset):
    # closing an already closed file must not raise
    zarr_module.setup(mset=mset.setup())
    zarr_module._close_file()
    zarr_module._close_file()
    assert not zarr_module._file_is_open


def test_reset_without_restart_trigger(zarr_module, mset):
    zarr_module.setup(mset=mset.setup())
    zarr_module.reset()
    assert not zarr_module._file_is_open


def test_custom_get_variables(directory_name, mset):
    # only the selected variables are written
    def get_variables(mz):
        return [mz.z["var1"]]

    zarr_module = fr.modules.ZarrWriter(
        filename="test",
        write_trigger=fr.ClockTrigger(time_interval=np.timedelta64(1, "m")),
        directory=directory_name,
        get_variables=get_variables,
    )
    mset.diagnostics.add_module(zarr_module)
    mset.setup()
    model = fr.Model(mset)
    model.run(runlen=np.timedelta64(2, "m"))

    store = zarr.open(directory_name / "test.zarr")
    assert "var1" in store
    assert "var2" not in store


def test_nc_attrs_and_high_dimensions(directory_name):
    # a four dimensional grid uses generic coordinate names, and custom
    # netcdf attributes of the fields are stored
    grid = fr.grid.cartesian.Grid(shape=(4, 4, 4, 4),
                                  domain_size=(1, 1, 1, 1))
    mset = fr.ModelSettingsBase(grid=grid)
    mset.time_stepper.dt = np.timedelta64(30, "s")

    def _state_constructor() -> fr.VectorField:
        var = fr.ScalarField(mset, name="var", units="unit",
                             nc_attrs={"custom_attr": "custom_value"})
        return fr.VectorField(mset, field_list=[var])

    mset.state_constructor = _state_constructor
    zarr_module = fr.modules.ZarrWriter(
        filename="test", directory=directory_name)
    zarr_module.setup(mset=mset.setup())

    mz = fr.ModelState(mset)
    zarr_module.update(mz=mz)
    zarr_module.stop()

    store = zarr.open(directory_name / "test.zarr")
    for name in ("x0", "x1", "x2", "x3"):
        assert name in store
    assert store["var"].attrs["custom_attr"] == "custom_value"
    assert store["var"].shape == (1, 4, 4, 4, 4)


def test_reset(directory_name, mset):
    zarr_module = fr.modules.ZarrWriter(
        filename="test",
        write_trigger=fr.ClockTrigger(time_interval=np.timedelta64(1, "m")),
        restart_trigger=fr.ClockTrigger(
            time_interval=np.timedelta64(30, "m")),
        directory=directory_name,
    )
    zarr_module.setup(mset=mset.setup())
    zarr_module.reset()
    assert not zarr_module._file_is_open


@pytest.mark.parametrize(*(
    "restart_trigger, expected_files",
    [
        (None,
         ["test.zarr"]),
        (fr.ClockTrigger(time_interval=np.timedelta64(30, "m")),
         ["test_0s.zarr", "test_30:00s.zarr", "test_01:00:00s.zarr"]),
    ],
))
def test_restart_trigger(restart_trigger, expected_files, directory_name,
                         mset):
    zarr_module = fr.modules.ZarrWriter(
        filename="test",
        write_trigger=fr.ClockTrigger(time_interval=np.timedelta64(1, "m")),
        restart_trigger=restart_trigger,
        directory=directory_name,
    )
    mset.diagnostics.add_module(zarr_module)
    mset.setup()
    model = fr.Model(mset)
    model.run(runlen=np.timedelta64(1, "h"))

    assert len(list(directory_name.glob("*.zarr"))) == len(expected_files)
    for file in expected_files:
        assert (directory_name / file).exists()


@pytest.mark.parametrize(*(
    "filename, clock, expected",
    [
        ("snap", fr.Clock(), "snap_0s.zarr"),
        ("snap.zarr", fr.Clock(), "snap_0s.zarr"),
        ("snap", fr.Clock(start_date=np.datetime64("2020-01-01")),
         "snap_2020-01-01T00:00:00.zarr"),
    ],
))
def test_format_filename(filename, clock, expected, directory_name):
    zarr_module = fr.modules.ZarrWriter(
        filename=filename, directory=directory_name)
    zarr_module.add_timestamp = True
    assert zarr_module._format_filename(clock).name == expected
