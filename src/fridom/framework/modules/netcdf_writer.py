# Import external modules
from typing import TYPE_CHECKING, Union
import numpy as np
import os
from netCDF4 import Dataset
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.modules.module import Module, setup_module, module_method
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_state import ModelState

class NetCDFWriter(Module):
    """
    Writing model output to NetCDF files.
    
    Parameters
    ----------
    `write_interval` : `np.timedelta64`
        The interval at which the data should be written to the file.
    `filename` : `str`, optional
        The name of the file to write to. Default is "snap" (no directory).
    `directory` : `str`, optional
        The directory where the files should be stored. Default is "snapshots".
    `start_time` : `np.datetime64`, optional
        The time at which the first file should be written. Default is 
    `end_time` : `np.datetime64`, optional
        The time at which the last file should be written. Default is None.
    `restart_interval` : `np.timedelta64`, optional
        The interval at which a new file should be created. Default is None.
    `snap_slice` : `tuple`, optional
        The slice of the grid that should be written to the file. Default is None.
    `name` : `str`, optional
        The name of the module. Default is "NetCDFWriter".
    `get_variables` : `callable`, (default: None)
        A function that returns a list of field variables that should be written 
        to the file. If None, all fields of the State object will be written.
        The function signature of get_variables is:
        `get_variables(mz: 'ModelState') -> list[FieldVariable]`
    
    Examples
    --------
    The following example shows how to create a netCDF output from a 
    nonhydrostatic model using the SingleWave initial condition.

    .. code-block:: python

        import fridom.nonhydro as nh
        import numpy as np
        import matplotlib.pyplot as plt

        # create a netCDF writer that outputs u, v, w, b, and p
        nc_writer = nh.modules.NetCDFWriter(
            get_variables = lambda mz: [mz.u, mz.v, mz.w, mz.b, mz.z_diag.p],
            write_interval = np.timedelta64(1, 's'))

        # create the model
        grid = nh.grid.cartesian.Grid(
            N=[128]*3, L=[1]*3, periodic_bounds=(True, True, True))
        mset = nh.ModelSettings(grid=grid, dsqr=0.02, Ro=0.0)
        mset.time_stepper.dt = np.timedelta64(10, 'ms')
        # add the netCDF writer to the diagnostics
        mset.diagnostics.add_module(nc_writer)
        mset.setup()
        z = nh.initial_conditions.SingleWave(mset, kx=2, ky=0, kz=1)
        model = nh.Model(mset)
        model.z = z
        model.run(runlen=np.timedelta64(10, 's'))

    """
    def __init__(self,
                 write_interval: np.timedelta64,
                 filename: str = "snap",
                 start_time: Union[np.datetime64, None] = None,
                 end_time: Union[np.datetime64, None] = None,
                 restart_interval: Union[np.timedelta64, None] = None,
                 snap_slice: tuple | None = None,
                 directory: str | None = None,
                 name: str = "NetCDFWriter",
                 get_variables: 'callable | None' = None,
                 ):
        directory = directory or "snapshots"
        filename = os.path.join(directory, filename)
        super().__init__(name = name)

        if get_variables is None:
            def get_variables(mz: 'ModelState'):
                return mz.z.field_list

        if snap_slice is not None:
            raise NotImplementedError("snap_slice is not implemented yet.")

        # ----------------------------------------------------------------
        #  Set Attributes
        # ----------------------------------------------------------------
        self.directory = directory
        self.filename = filename
        self.start_time = start_time
        self.end_time = end_time
        self.write_interval = write_interval
        self.restart_interval = restart_interval
        self.snap_slice = snap_slice
        self.get_variables = get_variables

        # private attributes
        self._current_start_time = None
        self._last_checkpoint_time = None
        self._last_write_time = None
        self._file_is_open = False
        self._ncfile = None
        return

    @setup_module
    def setup(self) -> None:
        # create snapshot folder if it doesn't exist
        config.logger.verbose(f"Touching snapshot directory: {self.directory}")
        os.makedirs(self.directory, exist_ok=True)

        # snap slice:
        if self.snap_slice is None:
            self.snap_slice = tuple([slice(None)]*self.grid.n_dims)
        return


    @module_method
    def start(self):
        if self._file_is_open:
            config.logger.warning(
                "NetCDFWriter: start() called while a file is already open.",
                "Continue with closing the file")
            self._close_file()
        return

    @module_method
    def stop(self):
        if self._file_is_open:
            self._close_file()
        self._current_start_time = None
        self._last_checkpoint_time = None
        self._last_write_time = None
        return

    @module_method
    def update(self, mz: 'ModelState') -> 'ModelState':
        time = mz.time
        # ----------------------------------------------------------------
        #  Check if the model time is in the writing range
        # ----------------------------------------------------------------
        # check if the model time is smaller than the start time
        if self.start_time is not None and time < self.start_time:
            return mz
        # check if the model time is larger than the end time
        if self.end_time is not None:
            if time > self.end_time and not self._file_is_open:
                return mz
            if time > self.end_time and self._file_is_open:
                self._close_file()
                return mz

        # ----------------------------------------------------------------
        #  Check if it is time to write
        # ----------------------------------------------------------------
        if self._last_write_time is None or self._last_checkpoint_time is None:
            time_to_write = True
        else:
            next_write_time = self._last_write_time + self.write_interval
            if (self._last_checkpoint_time < next_write_time and
                time >= next_write_time):
                time_to_write = True
            else:
                time_to_write = False
        self._last_checkpoint_time = time
        if not time_to_write:
            return mz

        # ----------------------------------------------------------------
        #  Cehck if the current start time is set
        # ----------------------------------------------------------------
        if self._current_start_time is None:
            self._current_start_time = time

        # ----------------------------------------------------------------
        #  Check if the file should be closed
        # ----------------------------------------------------------------
        if self.restart_interval is not None:
            next_restart_time = self._current_start_time + self.restart_interval
            if time >= next_restart_time and self._file_is_open:
                self._close_file()

        # ----------------------------------------------------------------
        #  Create a new file if the current file is not open
        # ----------------------------------------------------------------
        if not self._file_is_open:
            start_time = self._current_start_time
            if self.restart_interval is not None:
                while time - start_time >= self.restart_interval:
                    start_time += self.restart_interval
            self._current_start_time = start_time
            self._create_file(mz)

        # ----------------------------------------------------------------
        #  Write data
        # ----------------------------------------------------------------
        self._write_data(mz)
        self._last_write_time = time
        return mz

    def _create_file(self, mz: 'ModelState'):
        # ----------------------------------------------------------------
        #  Create the filename
        # ----------------------------------------------------------------
        base, ext = os.path.splitext(self.filename)
        ext = ext.lower()
        base = base if ext in [".nc", ".cdf"] else self.filename
        ext = ext if ext in [".nc", ".cdf"] else ".cdf"
        filename = f"{base}_{self._current_start_time}{ext}"

        # ----------------------------------------------------------------
        #  Create the NetCDF file
        # ----------------------------------------------------------------
        config.logger.info(f"Creating NetCDF file: {filename}")
        try:
            ncfile = Dataset(filename, "w", format="NETCDF4", parallel=True)
        except: 
            config.logger.warning(
                "Failed to create NetCDF file in parallel mode. Trying parallel=False")
            ncfile = Dataset(filename, "w", format="NETCDF4", parallel=False)

        dtype = config.dtype_real
        n_dims = self.grid.n_dims
        if n_dims <= 3:
            x_names = ['x', 'y', 'z'][:n_dims]
        else:
            x_names = [f"x{i}" for i in range(n_dims)]
        # ----------------------------------------------------------------
        #  General attributes
        # ----------------------------------------------------------------
        ncfile.description = f"fridom: {self.mset.model_name}"
        import time as system_time
        ncfile.created = system_time.ctime(system_time.time())

        # ----------------------------------------------------------------
        #  Create the dimensions
        # ----------------------------------------------------------------
        for i, name in enumerate(x_names):
            nx = len(self.grid.x_global[i][self.snap_slice[i]])
            ncfile.createDimension(name, nx)
        time_dim = ncfile.createDimension('time', None)

        # ----------------------------------------------------------------
        #  Create the variables
        # ----------------------------------------------------------------

        # Coordinate variables
        x = [ncfile.createVariable(name, dtype, (name,)) for name in x_names]
        time = ncfile.createVariable("time", dtype, ("time",))

        for xi, name in zip(x, x_names):
            xi.units = "m"
            xi.long_name = f"{name} coordinate"

        time.units = f"seconds since {self.start_time}"
        time.long_name = "UTC time"
        time.calendar = "standard"
        time.standard_name = "time"
        time.set_collective(True)

        # store the coordinates
        for i in range(n_dims):
            x[i][:] = utils.to_numpy(self.grid.x_global[i][self.snap_slice[i]])

        # create the output variables
        for var in self.get_variables(mz):
            nc_var = ncfile.createVariable(
                var.name, dtype, ("time", *x_names[::-1]))
            nc_var.units = var.units
            nc_var.long_name = var.long_name
            for key, value in var.nc_attrs.items():
                setattr(nc_var, key, value)
            nc_var.set_collective(True)

        # ----------------------------------------------------------------
        #  Store the attributes
        # ----------------------------------------------------------------
        self._file_is_open = True
        self._ncfile = ncfile
        return

    def _write_data(self, mz: 'ModelState'):
        time = self._ncfile.variables["time"]
        time_ind = time.size
        global_slice = self.grid.get_subdomain().global_slice
        inner_slice = self.grid.get_subdomain().inner_slice
        ind = time_ind, *global_slice[::-1]

        time[time_ind] = mz.time
        for var in self.get_variables(mz):
            nc_var = self._ncfile.variables[var.name]
            arr = var[inner_slice]
            nc_var[ind] = utils.to_numpy(arr.T)
        return

    def _close_file(self):
        self._ncfile.close()
        self._ncfile = None
        self._file_is_open = False
        return