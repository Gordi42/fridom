"""netcdf_writer.py - Writing model output to NetCDF files."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from netCDF4 import Dataset

import fridom.framework as fr

if TYPE_CHECKING:
    from collections.abc import Callable


class NetCDFWriter(fr.modules.Module):

    """
    Writing model output to NetCDF files.

    Parameters
    ----------
    write_trigger : fr.ClockTrigger, optional
        The trigger that determines when the data should be written to the file.
        Default is None which means that the data will be written at every time step.
    restart_trigger : fr.ClockTrigger, optional
        The trigger that determines when a new file should be created.
        Default is None which means that only one file will be created.
    filename : str, optional
        The name of the file to write to. Default is "snap" (no directory).
    directory : str, optional
        The directory where the files should be stored. Default is "snapshots".
    get_variables : callable, (default: None)
        A function that returns a list of scalar fields that should be written
        to the file. If None, all fields of the State object will be written.
        The function signature of get_variables is:
        `get_variables(mz: 'ModelState') -> list[ScalarField]`

    """

    name = "NetCDFWriter"
    def __init__(self,
                 write_trigger: fr.ClockTrigger | None = None,
                 restart_trigger: fr.ClockTrigger | None = None,
                 filename: str = "snap",
                 directory: str | None = None,
                 get_variables: Callable | None = None,
                 ) -> None:
        super().__init__()

        directory = directory or "snapshots"
        filename = Path(directory) / filename
        self.execute_at_start = True

        if get_variables is None:
            def get_variables(mz: fr.ModelState) -> list[fr.ScalarField]:
                return mz.z.field_list

        # ----------------------------------------------------------------
        #  Set Attributes
        # ----------------------------------------------------------------
        self.directory = directory
        self.filename = filename
        self.write_trigger = write_trigger or fr.ClockTrigger()
        self.restart_trigger = restart_trigger
        self._snap_slice = None
        self._add_timestamp = True
        self.get_variables = get_variables

        # private attributes
        self._file_is_open = False
        self._ncfile = None

    def _on_setup(self) -> None:
        # create snapshot folder if it doesn't exist
        fr.log.verbose(f"Touching snapshot directory: {self.directory}")
        Path(self.directory).mkdir(parents=True, exist_ok=True)

        # snap slice:
        if self._snap_slice is None:
            self._snap_slice = tuple([slice(None)]*self.grid.n_dims)

    @fr.modules.module_method
    def start(self) -> None:  # noqa: D102
        if self._file_is_open:
            msg = "NetCDFWriter: start() called while a file is already open."
            fr.log.warning(msg)
            self._close_file()

    @fr.modules.module_method
    def stop(self) -> None:  # noqa: D102
        if self._file_is_open:
            self._close_file()

    def _on_reset(self) -> None:
        self.write_trigger.reset()
        if self.restart_trigger is not None:
            self.restart_trigger.reset()

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        # ----------------------------------------------------------------
        #  Check if it is time to write
        # ----------------------------------------------------------------
        if not self.write_trigger.check(mz.clock):
            return mz

        # ----------------------------------------------------------------
        #  Check if the file should be restarted
        # ----------------------------------------------------------------
        if self.restart_trigger is not None and self.restart_trigger.check(mz.clock):
            self._close_file()

        # ----------------------------------------------------------------
        #  Create a new file if the current file is not open
        # ----------------------------------------------------------------
        if not self._file_is_open:
            self._create_file(mz)

        # ----------------------------------------------------------------
        #  Write data
        # ----------------------------------------------------------------
        self._write_data(mz)
        return mz

    def _format_filename(self, clock: fr.Clock) -> Path:
        """
        Add a timestamp to the filename.

        Parameters
        ----------
        clock : fr.Clock
            The clock of the model with the current time.

        Returns
        -------
        Path
            The formatted filename.

        """
        # we first remove the suffix from the filename, if the suffix is .nc or .cdf
        suffix = self.filename.suffix.lower()
        if suffix in [".nc", ".cdf"]:
            base_name = self.filename.parent / self.filename.stem
        else:
            base_name = self.filename
            suffix = ".cdf"
        # add the timestamp to the filename
        if not self.add_timestamp:
            return base_name.with_name(f"{base_name.stem}{suffix}")
        tot_time = clock.get_total_time()
        if isinstance(tot_time, np.datetime64):
            time_stamp = tot_time
        else:
            time_stamp = fr.utils.humanize_number(tot_time, unit="seconds")
            time_stamp = time_stamp.replace(" ", "_")
        return base_name.with_name(f"{base_name.stem}_{time_stamp}{suffix}")

    def _create_file(self, mz: fr.ModelState) -> None:
        # ----------------------------------------------------------------
        #  Make sure that there is no file open
        # ----------------------------------------------------------------
        self._close_file()
        # ----------------------------------------------------------------
        #  Create the filename
        # ----------------------------------------------------------------
        filename = self._format_filename(mz.clock)

        # ----------------------------------------------------------------
        #  Create the NetCDF file
        # ----------------------------------------------------------------
        fr.log.info(f"Creating NetCDF file: {filename}")
        # check if the model is running in parallel
        parallel = (self.grid.domain_decomp.parallel)
        ncfile = Dataset(filename, "w", format="NETCDF4", parallel=parallel)

        dtype = fr.config.dtype_real
        n_dims = self.grid.n_dims
        if n_dims <= 3:  # noqa: PLR2004
            x_names = ["x", "y", "z"][:n_dims]
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
        _time_dim = ncfile.createDimension("time", None)

        # ----------------------------------------------------------------
        #  Create the variables
        # ----------------------------------------------------------------

        # Coordinate variables
        x = [ncfile.createVariable(name, dtype, (name,)) for name in x_names]
        time = ncfile.createVariable("time", dtype, ("time",))

        for xi, name in zip(x, x_names, strict=False):
            xi.units = "m"
            xi.long_name = f"{name} coordinate"

        time.units = "seconds"
        time.long_name = "UTC time"
        time.calendar = "standard"
        time.standard_name = "time"
        if parallel:
            time.set_collective(True)

        # store the coordinates
        for i in range(n_dims):
            x[i][:] = fr.utils.to_numpy(self.grid.x_global[i][self.snap_slice[i]])

        # create the output variables
        for var in self.get_variables(mz):
            nc_var = ncfile.createVariable(
                var.name, dtype, ("time", *x_names[::-1]))
            nc_var.units = var.units
            nc_var.long_name = var.long_name
            for key, value in var.nc_attrs.items():
                setattr(nc_var, key, value)
            if parallel:
                nc_var.set_collective(True)

        # ----------------------------------------------------------------
        #  Store the attributes
        # ----------------------------------------------------------------
        self._file_is_open = True
        self._ncfile = ncfile

    def _write_data(self, mz: fr.ModelState) -> None:
        time = self._ncfile.variables["time"]
        time_ind = time.size
        ind = time_ind, *tuple(slice(None) for _ in range(self.grid.n_dims))

        time[time_ind] = mz.clock.passed_time
        for var in self.get_variables(mz):
            nc_var = self._ncfile.variables[var.name]
            arr = var.unpad()[self.snap_slice]
            nc_var[ind] = fr.utils.to_numpy(arr.T)

    def _close_file(self) -> None:
        if self._ncfile is not None:
            fr.log.debug(f"Closing NetCDF file: {self._ncfile.filepath()}")
            self._ncfile.close()
        self._ncfile = None
        self._file_is_open = False

    # ----------------------------------------------------------------
    #  Properties
    # ----------------------------------------------------------------

    @property
    def snap_slice(self) -> tuple[slice, ...]:
        """The slice of the grid that should be written to the file."""
        return self._snap_slice

    @snap_slice.setter
    def snap_slice(self, value: tuple[slice, ...]) -> None:
        # we close the file if the slice changes
        self._close_file()
        # set the new slice
        self._snap_slice = value

    @property
    def add_timestamp(self) -> bool:
        """Whether a timestamp should be added to the filename."""
        return self._add_timestamp

    @add_timestamp.setter
    def add_timestamp(self, value: bool) -> None:
        self._add_timestamp = value
