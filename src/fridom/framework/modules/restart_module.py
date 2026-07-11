"""Restart module for cluster computing."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

import fridom.framework as fr

# reference time for the realtime restart interval (job start time)
_LOAD_TIME = time.time()


class RestartModule(fr.modules.Module):

    """
    Automatically restart the job at a given interval.

    Description
    -----------

    Parameters
    ----------
    realtime_interval : np.timedelta64, optional
        The interval in real time at which the model should restart.
        If None, the model will not restart based on real time.
    clock_trigger : fr.ClockTrigger, optional
        Triggers the restart based on the model time.
        If None, the model will not restart based on model time.
    restart_command : str | Callable, optional
        The command to start the job.
        If the command is a string:
            The model will restart by running the command in a subprocess.
        If the command is a callable:
            The model will restart by calling the function. This function
            should simply restart the job. It should have no arguments,
            and should return nothing.
        If the command is None:
            The model will try to find the command from the environment.
            If the command is not found, the model will not be able to restart.
    file_path : Path, optional
        The path to the restart file.

    """

    name = "Restart Module"
    def __init__(self,
                 realtime_interval: np.timedelta64 | None = None,
                 clock_trigger: fr.ClockTrigger | None = None,
                 restart_command: str | Callable | None = None,
                 file_path: Path | str = Path("restart/model.dill")) -> None:
        super().__init__()

        # ----------------------------------------------------------------
        #  Check the restart interval input
        # ----------------------------------------------------------------

        fr.exceptions.TooManyArgumentsError.check(
            max_args=1,
            realtime_interval=realtime_interval,
            clock_trigger=clock_trigger)

        if realtime_interval is None and clock_trigger is None:
            # No interval is set. Disable the module.
            fr.log.verbose(
                "No interval is set in RestartModule. Disabling the module.")
            self.disable()

        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self.realtime_interval = realtime_interval
        self.clock_trigger = clock_trigger
        self.file_path = file_path
        self.restart_command = restart_command
        self.file = None

    def _on_setup(self) -> None:
        self._touch_restart_directory()

    @fr.modules.module_method
    def should_restart(self, mz: fr.ModelState) -> bool:
        """Check if one of the restart conditions is met."""
        # ----------------------------------------------------------------
        #  Realtime interval
        # ----------------------------------------------------------------
        elapsed_time = np.timedelta64(
            int(time.time() - _LOAD_TIME), "s")
        interval = self.realtime_interval
        if interval is not None and elapsed_time >= interval:
            fr.log.info(
                "Realtime restart interval reached. Model will restart.")
            self.set_full_filename(mz.clock.it)
            return True
        # ----------------------------------------------------------------
        #  Modelclock trigger
        # ----------------------------------------------------------------
        if (self.clock_trigger is not None
                and self.clock_trigger.check(mz.clock)):
            fr.log.info("Modeltime trigger reached. Model will restart.")
            self.set_full_filename(mz.clock.it)
            return True
        return False

    @fr.modules.module_method
    def should_reload(self) -> bool:
        """
        Check if the model should reload from a restart file.

        Description
        -----------
        If a restart file is found in the directory, the model will reload.
        If multiple restart files are found, the model will reload from the
        restart file with the highest iteration number.
        """
        fr.log.verbose("Checking if restart files exist.")
        files = [f.name for f in self.file_path.parent.iterdir()]
        files = [f for f in files if f.startswith(self.file_path.stem)]
        if len(files) > 0:
            fr.log.info("Found restart files. Model will reload.")
            its = [int(f.split("_")[1]) for f in files]
            self.set_full_filename(max(its))
            return True
        fr.log.info("No restart files found. Model will not reload.")
        return False

    def _on_reset(self) -> None:
        if self.clock_trigger is not None:
            self.clock_trigger.reset()

    def set_full_filename(self, it: int) -> None:
        """Set the full filename with the iteration number and rank."""
        rank = fr.utils.get_my_rank()
        filename = f"{self.file_path.stem}_{it}_{rank}{self.file_path.suffix}"
        self.file = self.file_path.parent / filename

    def restart(self, model: fr.Model) -> None:
        """
        Restart the model.

        Description
        -----------
        This function stops the model, saves the model state to a file,
        and executes the restart command.

        Parameters
        ----------
        model : fr.Model
            The model to restart.

        """
        fr.log.info(
            "Stopping model at it: %d, time: %f",
            model.model_state.clock.it,
            model.model_state.clock.time)
        model.stop()
        model.save(self.file)
        fr.log.info(model.mset.timer)
        if isinstance(self.restart_command, str):
            self._restart_from_command()
            return  # pragma: no cover (sys.exit never returns)
        if isinstance(self.restart_command, Callable):
            self.restart_command()
            return
        msg = "No restart command is set. The model can't restart."
        raise ValueError(msg)

    def _restart_from_command(self) -> None:
        """Restart the model from the restart command."""
        if isinstance(self.restart_command, str):
            fr.log.info(f"Running restart command: {self.restart_command}")
        fr.utils.mpi_barrier()
        if fr.utils.MPI_AVAILABLE:
            result = subprocess.run(  # noqa: S603
                self.restart_command.split(),
                capture_output=True,
                text=True,
                check=False)
            fr.log.notice(result.stdout)
            if result.stderr:
                fr.log.error(result.stderr)
        fr.utils.mpi_barrier()
        sys.exit()

    def _touch_restart_directory(self) -> None:
        """Touch the restart directory."""
        fr.log.verbose("Touching the restart directory.")
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def info(self) -> dict:  # noqa: D102
        res = super().info
        if not self.is_enabled():
            return res
        if self.realtime_interval is not None:
            res["Realtime Restart Interval"] = self.realtime_interval
        if self.clock_trigger is not None:
            res["Clock Trigger"] = self.clock_trigger
        if self.restart_command is not None:
            res["Restart Command"] = self.restart_command
        res["File Path"] = self.file_path
        return res

    @property
    def realtime_interval(self) -> np.timedelta64 | None:
        """The interval in real time at which the model should restart."""
        return self._realtime_interval

    @realtime_interval.setter
    def realtime_interval(
            self, realtime_interval: np.timedelta64 | None) -> None:
        if realtime_interval is None:
            self._realtime_interval = None
            return
        # check that the interval is a timedelta
        if not isinstance(realtime_interval, np.timedelta64):
            msg = "The interval should be a numpy timedelta64."
            raise TypeError(msg)
        self._realtime_interval = realtime_interval

    @property
    def clock_trigger(self) -> fr.ClockTrigger | None:
        """
        Triggers the restart based on the model time.

        By default, the clock trigger will not trigger on the first step.
        Any other option will be overridden. To trigger on the first step,
        set the trigger_on_first_step to True after setting the clock trigger,
        e.g.:

        .. code-block:: python

            import fridom.framework as fr
            restart_module = fr.modules.RestartModule(
                clock_trigger=fr.ClockTrigger())
            restart_module.clock_trigger.trigger_on_first_step = True

        """
        return self._clock_trigger

    @clock_trigger.setter
    def clock_trigger(self, clock_trigger: fr.ClockTrigger | None) -> None:
        self._clock_trigger = clock_trigger
        if isinstance(clock_trigger, fr.ClockTrigger):
            clock_trigger.trigger_on_first_step = False

    @property
    def file_path(self) -> Path:
        """The path to the restart file."""
        return self._file_path

    @file_path.setter
    def file_path(self, file_path: Path | str) -> None:
        # cast to Path
        if isinstance(file_path, str):
            file_path = Path(file_path)
        # sanitize the filename
        filename = file_path.stem
        # replace "_" with "-"
        if "_" in filename:
            msg = "The filename should not contain the character '_'."
            msg += " Replacing '_' with '-' in the filename."
            fr.log.warning(msg)
            filename = filename.replace("_", "-")
        self._file_path = file_path.with_name(filename + file_path.suffix)

    @property
    def restart_command(self) -> str:
        """The command to restart the job."""
        return self._restart_command

    @restart_command.setter
    def restart_command(self, restart_command: str | None) -> None:
        self._restart_command = None
        if restart_command is not None:
            self._restart_command = restart_command
            return
        if not self.is_enabled():
            # do not query the environment for disabled modules
            return

        # Get the job id from the environment
        job_id = os.getenv("SLURM_JOB_ID")
        if job_id is None:
            msg = "No job id is found in the environment."
            msg += " The model will not be able to restart."
            fr.log.warning(msg)
            return

        if not job_id.isdigit():
            fr.log.warning(
                "Invalid job id. The model will not be able to restart.")
            return

        job_info = subprocess.run(  # noqa: S603
            ["/usr/bin/scontrol", "show", "job", job_id],
            capture_output=True,
            text=True,
            check=False)

        # extract the restart command from the job info
        command = None
        for line in job_info.stdout.split("\n"):
            if line.strip().startswith("Command="):
                command = line.split("=", 1)[1].strip()
        if command is None:
            fr.log.warning(
                "No restart command is set."
                " The model will not be able to restart.")
            return
        self._restart_command = f"sbatch {command}"
