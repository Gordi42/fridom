import pytest
import fridom.framework as fr
import tempfile
import os
import numpy as np
from mpi4py import MPI
from netCDF4 import Dataset
from copy import deepcopy

@pytest.fixture()
def parent_directory():
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmpdirname = MPI.COMM_WORLD.bcast(tmpdirname, root=0)
        yield tmpdirname

def test_temporary_directory(parent_directory):
    # check if the tmpdirname is the same in every process
    all_paths = MPI.COMM_WORLD.gather(parent_directory, root=0)
    if MPI.COMM_WORLD.Get_rank() == 0:
        assert all([path == parent_directory for path in all_paths])

    assert os.path.exists(parent_directory)

@pytest.fixture()
def directory_name(parent_directory):
    # return f"{parent_directory}/snapshots"
    return os.path.join(parent_directory, "snapshots")

@pytest.fixture()
def netcdf_module(directory_name):
    return fr.modules.NetCDFWriter(
        filename="test.cdf",
        write_interval=np.timedelta64(1, 'm'),
        directory=directory_name
    )

@pytest.fixture()
def mset(backend):
    grid = fr.grid.cartesian.Grid(N=(128, 64), L=(1, 1))
    mset = fr.ModelSettingsBase(grid=grid)
    mset.setup()
    position = fr.grid.cartesian.Position(
        tuple([fr.grid.cartesian.AxisOffset.CENTER] * 2))
    def state_constructor():
        var1 = fr.FieldVariable(
            mset, name="var1", long_name="Variable 1", units="unit1", 
            position=position)
        var2 = fr.FieldVariable(
            mset, name="var2", long_name="Variable 2", units="unit2", 
            position=position)
        return fr.StateBase(mset, field_list=[var1, var2])
    mset.state_constructor = state_constructor
    return mset

def test_netCDFWriterNew(netcdf_module, directory_name, mset):
    MPI.COMM_WORLD.barrier()
    # check that the directory is created
    assert not os.path.exists(directory_name)
    MPI.COMM_WORLD.barrier()
    netcdf_module.setup(mset=mset)
    # now the directory should exist
    assert os.path.exists(directory_name)

def test_model_run(mset, netcdf_module, directory_name):
    
    mset.diagnostics.add_module(netcdf_module)
    mset.setup()
    # check that the model runs without error
    model = fr.Model(mset)
    model.run(runlen=np.timedelta64(1, 'h'))

    time = np.datetime64(1, 's')

    # check that the file is created
    file_path = os.path.join(directory_name, f"test_{time}.cdf")
    assert os.path.exists(file_path)

    # open the file and check the contents
    if MPI.COMM_WORLD.Get_size() > 1:
        parallel = True
    else:
        parallel = False

    with Dataset(file_path, "r", parallel=parallel) as ncfile:
        assert "var1" in ncfile.variables
        assert "var2" in ncfile.variables
        assert ncfile.variables["var1"].units == "unit1"
        assert ncfile.variables["var2"].units == "unit2"
        assert ncfile.variables["var1"].shape == (60, 64, 128)
