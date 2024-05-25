from fridom.framework.modelsettings_base import ModelSettingsBase
from fridom.framework.model_state import ModelStateBase
from fridom.framework.state_base import StateBase
from fridom.framework.modules.module import \
    Module, start_module, update_module, stop_module

class NetCDFWriter(Module):
    """
    # Base class for netCDF writers.
    Writes the data to a NetCDF file on a separate process.

    Child classes must implement the following methods:
    ## Must implement:
    - init: Initialization of the module, specifying variable names
    - get_variables: Get the variables from the model state to write to the 
    NetCDF file.
    """
    def __init__(self,
                 name = "NetCDFWriter",
                 filename = "snap",
                 snap_interval = 100,
                 snap_slice = None,
                 var_names = None,
                 var_long_names = None,
                 var_unit_names = None,
                 ) -> None:
        """
        # Args:
        - name (str)            : Name of the module.
        - filename (str)        : Name of the NetCDF file. (without extension)
        - snap_interval (int)   : Interval between snapshots.
        - snap_slice (tuple)    : Which part of the grid to save.
        - var_names (list)      : Names of the variables. (should be set by child)
        - var_long_names (list) : Long names of the variables. (should be set by child)
        - var_unit_names (list) : Unit names of the variables. (should be set by child)
        """
        import os
        filename = os.path.join("snapshots", filename)
        name = filename.split(".")[0]
        binary_files = [name + "_" + var_name + "_bin.npy" for var_name in var_names]
        super().__init__(
            name = name,
            filename = filename,
            is_active = False,
            snap_interval = snap_interval,
            snap_slice = snap_slice,
            var_names = var_names,
            var_long_names = var_long_names,
            var_unit_names = var_unit_names,
            binary_files = binary_files)
        return
    
    @start_module
    def start(self):
        """
        Start the parallel writer process.
        """
        import os
        import multiprocessing as mp
        # create snapshot folder if it doesn't exist
        if not os.path.exists("snapshots"):
            os.makedirs("snapshots")

        # delete old binary files
        for binary_file in self.binary_files:
            if os.path.exists(binary_file):
                os.remove(binary_file)

        # snap slice
        sel = self.snap_slice
        if sel is None:
            sel = tuple([slice(None)]*len(self.grid.x))

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

    @update_module
    def update(self, mz: ModelStateBase, _dz: StateBase):
        """
        Write data to binary files and add them to the NetCDF file.
        """
        # check if it is time to write
        if mz.it % self.snap_interval != 0:
            return  # not time to write

        # wait until all binary files are deleted
        import os
        for binary_file in self.binary_files:
            while os.path.exists(binary_file):
                pass

        # write data to binary file
        cp = self.grid.cp
        binary_files = self.binary_files
        for name, var in zip(binary_files, self.get_variables(mz)):
            cp.save(name, var[self.mset.snap_slice])

        # add binary file to cdf file
        self.input_queue.put(mz.time)
        return

    @stop_module
    def stop(self):
        """
        Stop the parallel writer process.
        """
        if self.is_active:
            self.input_queue.put("STOP")
            self.parallel_writer.join()
        self.is_active = False
        return

    def get_variables(self, mz: ModelStateBase):
        """
        This method should be overwritten by the user to return the variables
        that should be written to the NetCDF file.

        ## Returns:
        - List of Field Variables
        """
        return []


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
    import os, time as system_time, numpy
    from netCDF4 import Dataset
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