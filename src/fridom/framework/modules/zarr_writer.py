"""Writing model output to zarr stores."""
from __future__ import annotations

import time as system_time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import zarr
from zarr.errors import ZarrUserWarning

import fridom.framework as fr

if TYPE_CHECKING:
    from collections.abc import Callable

warnings.filterwarnings("ignore", category=ZarrUserWarning)


class ZarrWriter(fr.modules.Module):

    """
    Writing model output to zarr stores.

    Parameters
    ----------
    write_trigger : fr.ClockTrigger, optional
        The trigger that determines when the data should be written to the
        file. Default is None which means that the data will be written at
        every time step.
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

    name = "ZarrWriter"
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
        self._add_timestamp = (restart_trigger is not None)
        self.get_variables = get_variables

        # private attributes
        self._file_is_open = False
        self._zarr_store = None
        self._var_arrs = None
        self._time = None

    def _on_setup(self) -> None:
        # create snapshot folder if it doesn't exist
        fr.log.verbose(f"Touching snapshot directory: {self.directory}")
        Path(self.directory).mkdir(parents=True, exist_ok=True)

    @fr.modules.module_method
    def start(self) -> None:  # noqa: D102
        if self._file_is_open:
            msg = "ZarrWriter: start() called while a file is already open."
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
        if (self.restart_trigger is not None
                and self.restart_trigger.check(mz.clock)):
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
        # we first remove the suffix from the filename, if the suffix is
        # .nc or .cdf
        suffix = self.filename.suffix.lower()
        if suffix == ".zarr":
            base_name = self.filename.parent / self.filename.stem
        else:
            base_name = self.filename
            suffix = ".zarr"
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

    def _generate_file(self, mz: fr.ModelState, filename: Path) -> None:
        # ----------------------------------------------------------------
        #  Create the zarr store
        # ----------------------------------------------------------------
        fr.log.info(f"Creating zarr store: {filename}")

        store = zarr.group(filename, overwrite=True)

        dtype = fr.config.dtype_real
        n_dims = self.grid.n_dims
        if n_dims <= 3:  # noqa: PLR2004
            x_names = ["x", "y", "z"][:n_dims]
        else:
            x_names = [f"x{i}" for i in range(n_dims)]

        # ----------------------------------------------------------------
        #  General attributes
        # ----------------------------------------------------------------
        store.attrs["Conventions"] = "CF-1.10"
        store.attrs["description"] = f"fridom: {self.mset.model_name}"
        store.attrs["history"] = (
            f"Created on {system_time.ctime(system_time.time())}")

        # ----------------------------------------------------------------
        #  Create the dimensions
        # ----------------------------------------------------------------
        time = store.create_array(
            "time",
            shape=(0,),
            chunks=(1,),
            dtype=dtype,
            dimension_names=("time",),
        )
        time.attrs.update(
            {
                "units": "seconds",
                "long_name": "UTC time",
                "calendar": "standard",
                "standard_name": "time",
            },
        )


        for i, d in enumerate(x_names):
            coord_arr = store.create_array(
                d,
                data=self.grid.x_global[i],
                dimension_names=(d,),
            )
            coord_arr.attrs.update(
                {
                    "units": "m",
                    "long_name": f"{d} coordinate",
                    "axis": d.upper(),
                },
            )

        # ----------------------------------------------------------------
        #  Create the variables
        # ----------------------------------------------------------------
        for var in self.get_variables(mz):
            zarr_arr = store.create_array(
                var.name,
                shape=(0, *var.unpad().shape[::-1]),
                chunks=self._get_chunk_shape(var),
                dtype=dtype,
                dimension_names=("time", *x_names[::-1]),
            )
            zarr_arr.attrs["units"] = var.units
            zarr_arr.attrs["long_name"] = var.long_name
            for key, value in var.nc_attrs.items():
                zarr_arr.attrs[key] = value

        zarr.consolidate_metadata(store.store_path)

    def _create_file(self, mz: fr.ModelState) -> None:
        # ----------------------------------------------------------------
        #  Make sure that there is no file open
        # ----------------------------------------------------------------
        self._close_file()
        # ----------------------------------------------------------------
        #  Create the filename
        # ----------------------------------------------------------------
        filename = self._format_filename(mz.clock)

        if fr.utils.I_AM_MAIN_RANK:
            self._generate_file(mz, filename)

        fr.utils.mpi_barrier()

        # ----------------------------------------------------------------
        #  Store the attributes
        # ----------------------------------------------------------------
        self._file_is_open = True
        self._zarr_store = zarr.open_consolidated(filename)
        self._time = self._zarr_store["time"]

        var_arrs = {}
        for var in self.get_variables(mz):
            var_arrs[var.name] = self._zarr_store[var.name]
        self._var_arrs = var_arrs

    def _get_chunk_shape(self, var: fr.ScalarField) -> tuple[int, ...]:
        if fr.config.backend_is_jax:
            return (1, *var.unpad().addressable_shards[0].data.shape[::-1])
        return (1, *var.unpad().T.shape)


    def _write_data(self, mz: fr.ModelState) -> None:
        # add a new time step
        time = self._time
        time.append([mz.clock.passed_time])

        # add data for each variable
        for var in self.get_variables(mz):
            self._write_data_var(var)

    def _write_data_var(self, var: fr.ScalarField) -> None:
        nt = self._time.shape[0]
        var_arr = self._var_arrs[var.name]
        new_shape = (nt, *var_arr.shape[1:])
        var_arr.resize(new_shape)
        if not fr.config.backend_is_jax:
            var_arr[nt - 1, ...] = var.unpad().T
            return

        for shard in var.unpad().T.addressable_shards:
            indexer = (nt - 1, *shard.index)
            var_arr[indexer] = shard.data

    def _close_file(self) -> None:
        if self._zarr_store is not None:
            fr.log.debug(f"Closing Zarr store: {self._zarr_store.store_path}")
            zarr.consolidate_metadata(self._zarr_store.store_path)

        del self._zarr_store
        self._var_arrs = None
        self._time = None
        self._ncfile = None
        self._file_is_open = False

    # ----------------------------------------------------------------
    #  Properties
    # ----------------------------------------------------------------

    @property
    def add_timestamp(self) -> bool:
        """Whether a timestamp should be added to the filename."""
        return self._add_timestamp

    @add_timestamp.setter
    def add_timestamp(self, value: bool) -> None:
        self._add_timestamp = value
