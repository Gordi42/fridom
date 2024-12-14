"""Tests for the NetCDFWriter module."""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from netCDF4 import Dataset

import fridom.framework as fr


@pytest.fixture
def parent_directory():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname

@pytest.fixture
def directory_name(parent_directory):
    yield Path(parent_directory) / "snapshots"
    # clean up
    if (Path(parent_directory) / "snapshots").exists():
        # remove the directory with all its contents
        shutil.rmtree(Path(parent_directory) / "snapshots")

@pytest.fixture
def netcdf_module(directory_name):
    return fr.modules.NetCDFWriter(
        filename="test.cdf",
        write_trigger = fr.ClockTrigger(time_interval=np.timedelta64(1, "m")),
        directory=directory_name,
    )

@pytest.fixture
def mset():
    grid = fr.grid.cartesian.Grid(N=(128, 64), L=(1, 1))
    mset = fr.ModelSettingsBase(grid=grid)
    mset.setup()
    def _state_constructor() -> fr.StateBase:
        var1 = fr.FieldVariable(
            mset, name="var1", long_name="Variable 1", units="unit1")
        var2 = fr.FieldVariable(
            mset, name="var2", long_name="Variable 2", units="unit2")
        return fr.StateBase(mset, field_list=[var1, var2])
    mset.state_constructor = _state_constructor
    return mset

def test_setup(netcdf_module, directory_name, mset):
    # check that the directory is created
    assert not directory_name.exists()
    netcdf_module.setup(mset=mset)
    # now the directory should exist
    assert directory_name.exists()

def test_model_run(mset, netcdf_module, directory_name):
    mset.diagnostics.add_module(netcdf_module)
    mset.setup()
    # check that the model runs without error
    model = fr.Model(mset)
    model.run(runlen=np.timedelta64(1, "h"))

    time = fr.utils.humanize_number(0, "seconds")

    # check that the file is created
    file_path = directory_name / f"test_{time}.cdf"
    assert file_path.exists()

    # open the file and check the contents
    parallel = False

    with Dataset(file_path, "r", parallel=parallel) as ncfile:
        assert "var1" in ncfile.variables
        assert "var2" in ncfile.variables
        assert ncfile.variables["var1"].units == "unit1"
        assert ncfile.variables["var2"].units == "unit2"
        assert ncfile.variables["var1"].shape == (61, 64, 128)


def test_initialization(netcdf_module, directory_name):
    assert netcdf_module.filename == Path(directory_name) / "test.cdf"
    assert netcdf_module.directory == directory_name
    assert netcdf_module.write_trigger is not None
    assert netcdf_module.restart_trigger is None
    assert netcdf_module.get_variables is not None

def test_setup_creates_directory(netcdf_module, directory_name, mset):
    assert not directory_name.exists()
    netcdf_module.setup(mset=mset)
    assert directory_name.exists()

def test_start_closes_open_file(netcdf_module, mset):
    netcdf_module.setup(mset=mset)
    netcdf_module._file_is_open = True
    netcdf_module.start()
    assert not netcdf_module._file_is_open

def test_stop_closes_open_file(netcdf_module, mset):
    netcdf_module.setup(mset=mset)
    netcdf_module._file_is_open = True
    netcdf_module.stop()
    assert not netcdf_module._file_is_open
    assert netcdf_module._ncfile is None

def test_update_writes_data(netcdf_module, mset):
    netcdf_module.setup(mset=mset)
    netcdf_module.start()
    model_state = fr.ModelState(mset)
    model_state.clock.tick(np.timedelta64(1, "m"))
    netcdf_module.update(model_state)
    assert netcdf_module._file_is_open

@pytest.mark.parametrize(*(
    "restart_trigger, expected_files",
    [
        (None,
         ["test_0s.cdf"]),
        (fr.ClockTrigger(time_interval=np.timedelta64(30, "m")),
         ["test_0s.cdf", "test_30:00s.cdf", "test_01:00:00s.cdf"]),
        (fr.ClockTrigger(start_date=10*60, stop_date=25*60, time_interval=5*60),
         ["test_0s.cdf", "test_10:00s.cdf", "test_15:00s.cdf", "test_20:00s.cdf"]),
    ],
))
def test_restart_trigger(restart_trigger, expected_files, directory_name, mset):
    netcdf_module = fr.modules.NetCDFWriter(
        filename="test.cdf",
        write_trigger = fr.ClockTrigger(time_interval=np.timedelta64(1, "m")),
        restart_trigger=restart_trigger,
        directory=directory_name,
    )
    mset.diagnostics.add_module(netcdf_module)
    mset.setup()
    model = fr.Model(mset)
    model.run(runlen=np.timedelta64(1, "h"))

    # check if the number of files is correct
    assert len(list(directory_name.glob("*.cdf"))) == len(expected_files)

    for file in expected_files:
        assert (directory_name / file).exists()

@pytest.mark.parametrize(*(
    "time_stamp, expected_filename",
    [
        (True, "test_0s"),
        (False, "test"),
    ],
))
def test_no_timestamp(
    time_stamp, expected_filename, netcdf_module, directory_name, mset):
    netcdf_module.add_timestamp = time_stamp

    # check if the format filename is correct
    filename = netcdf_module._format_filename(fr.Clock())
    assert filename.stem == expected_filename

    mset.diagnostics.add_module(netcdf_module)
    mset.setup()
    model = fr.Model(mset)
    model.run(runlen=np.timedelta64(1, "h"))

    # check if the number of files is correct
    assert len(list(directory_name.glob("*.cdf"))) == 1
    assert (directory_name / f"{expected_filename}.cdf").exists()
