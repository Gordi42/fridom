import os
import time as system_time
import multiprocessing as mp
from netCDF4 import Dataset
import numpy

from fridom.Framework.ModelSettingsBase import ModelSettingsBase
from fridom.Framework.GridBase import GridBase


class NetCDFWriter:
    """
    Writes the data to a NetCDF file on a separate process.

    Attributes:
        filename (str)              : Name of the NetCDF file.
        binary_files (list)         : Names of the binary files.
        parallel_writer (mp.Process): Parallel writer process.
        input_queue (mp.Queue)      : Queue to communicate with the writer.
        snap_slice (tuple)          : Slice for snapshot.

    Methods:
        write_cdf: Write data to binary files and add them to the NetCDF file.
        close    : Close the NetCDF file.
    """

    def __init__(self, mset: ModelSettingsBase, grid:GridBase) -> None:
        """
        Constructor.
        """
        self.grid = grid
        self.mset = mset
        self.var_names      = None
        self.var_long_names = None
        self.var_unit_names = None
        self.filename = os.path.join("snapshots", mset.snap_filename)
        self.is_active = False
        return

    def set_var_names(self, 
                      var_names:list, 
                      var_long_names:list, 
                      var_unit_names:list):
        """
        Prepare the NetCDF file for writing by giving information about the
        variable names and units. This function should be called
        before the model is started.
        """
        self.var_names      = var_names
        self.var_long_names = var_long_names
        self.var_unit_names = var_unit_names
        name = self.filename.split(".")[0]
        self.binary_files = [name + "_" + var_name + "_bin.npy" for var_name in var_names]
        return

    def start(self):
        """
        Start the parallel writer process.
        """
        # only start if snapshots are enabled
        if self.mset.enable_snap:
            # create snapshot folder if it doesn't exist
            if not os.path.exists("snapshots"):
                os.makedirs("snapshots")

            # delete old binary files
            for binary_file in self.binary_files:
                if os.path.exists(binary_file):
                    os.remove(binary_file)

            # snap slice
            sel = self.mset.snap_slice

            # launch parallel writer
            get = lambda x: x.get() if self.mset.gpu else x
            x = [get(xi[sel[i]]) for i, xi in enumerate(self.grid.x)]
            self.input_queue = mp.Queue()
            self.parallel_writer = mp.Process(
                target=parallel_writer, args=(
                    self.mset, x, self.filename, self.input_queue, 
                    self.var_names, self.var_long_names, self.var_unit_names, 
                    self.binary_files))
            self.parallel_writer.start()

            self.is_active = True
        return


    def write_cdf(self, variables, time):
        """
        Write data to binary files and add them to the NetCDF file.

        Args:
            variables (list): List of variables to write.
            time (float):   Current model time.
        """
        # wait until all binary files are deleted
        for binary_file in self.binary_files:
            while os.path.exists(binary_file):
                pass

        # write data to binary file
        cp = self.grid.cp
        binary_files = self.binary_files
        for name, var in zip(binary_files, variables):
            cp.save(name, var[self.mset.snap_slice])

        # add binary file to cdf file
        self.input_queue.put(time)
        return


    def close(self):
        """
        Close the NetCDF file.
        """
        if self.is_active:
            self.input_queue.put("STOP")
            self.parallel_writer.join()
        self.is_active = False
        return



def parallel_writer(mset:ModelSettingsBase, x_in, filename, input_queue, 
                    var_names, var_long_names, var_unit_names,
                    binary_files):
    """
    Parallel writer process.

    Args:
        mset (ModelSettings) : ModelSettings object.
        x_in (np.ndarray)    : coordinates.
        filename (str)       : Name of the NetCDF file.
        input_queue (Queue)  : Queue to communicate with main process.
        var_names (list)     : Names of the variables.
        var_long_names (list): Long names of the variables.
        var_unit_names (list): Unit names of the variables.
        binary_files (list)  : Names of the binary files.
    """
    # check if file already exists
    if os.path.exists(filename):
        os.remove(filename)

    # create cdf file
    ncfile = Dataset(filename, mode='w', format='NETCDF4')

    # ================================================================
    #  CREATE DIMENSIONS
    # ================================================================
    if len(x_in) <= 3:
        x_names = ['x', 'y', 'z'][:len(x_in)]
    else:
        x_names = [f"x{i}" for i in range(len(x_in))]
    for x, name in zip(x_in, x_names):
        ncfile.createDimension(name, x.size)
    time_dim = ncfile.createDimension('time', None)


    # ================================================================
    #  CREATE ATTRIBUTES
    # ================================================================
        
    # General Info
    ncfile.description = f'fridom: {mset.model_name}'
    ncfile.created     = system_time.ctime(system_time.time())

    # Model Settings
    for key in dir(mset):
        val = getattr(mset, key)
        if isinstance(val, (int, float, str)):
            if not key.startswith("_"):
                if isinstance(val, bool):
                    val = int(val)
                setattr(ncfile, key, val)

    # ================================================================
    #  CREATE VARIABLES
    # ================================================================
    dtype = mset.dtype

    # Coordinates
    x = [ncfile.createVariable(name, dtype, (name,)) for name in x_names]
    time = ncfile.createVariable('time', dtype, ('time',))

    for xi, name in zip(x, x_names):
        xi.units    = 'm'
        xi.long_name = f'{name}-coordinate'

    time.units = 's'
    time.long_name = 'time'

    for xi, xii in zip(x, x_in):
        xi[:] = xii

    dims = tuple(["time"] + x_names[::-1])

    # State Variables
    vars = []
    for var_name, var_long_name, var_unit_name in zip(
            var_names, var_long_names, var_unit_names):
        var = ncfile.createVariable(var_name, dtype, dims)
        var.units = var_unit_name
        var.long_name = var_long_name
        vars.append(var)

    # ================================================================
    #  START MAIN LOOP
    # ================================================================
    while True:
        model_time = input_queue.get()
        if model_time == "STOP":
            break

        ti = time.size
        time[ti] = model_time

        # wait until all files are written
        for binary_file in binary_files:
            while not os.path.exists(binary_file):
                pass

        for var, binary_file in zip(vars, binary_files):
            var[ti,:] = numpy.load(binary_file).T
            os.remove(binary_file)

    ncfile.close()
    return




