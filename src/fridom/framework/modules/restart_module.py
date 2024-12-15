"""Restart module for cluster computing."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import fridom.framework as fr


class RestartModule(fr.modules.Module):
    name = "Restart Module"
    def __init__(self,
                 realtime_interval: np.timedelta64 | None = None,
                 modeltime_interval: np.timedelta64 | None = None,
                 iteration_interval: int | None = None,
                 restart_command: str | None = None,
                 filename: str = "model",
                 directory: str = "restart") -> None:
        super().__init__()

        # ----------------------------------------------------------------
        #  Check the restart interval input
        # ----------------------------------------------------------------

        num_interval_args = sum([realtime_interval is not None,
                                 modeltime_interval is not None,
                                 iteration_interval is not None])

        if num_interval_args == 0:
            # No interval is set. Disable the module.
            fr.log.verbose(
                "No interval is set in RestartModule. Disabling the module.")
            self.disable()

        # ----------------------------------------------------------------
        #  Check the filename
        # ----------------------------------------------------------------

        if "-" in filename:
            fr.log.warning(
                "The filename should not contain the character '-' Replacing '-' with '_' in the filename.")
            filename = filename.replace("-", "_")
        # remove the extension from the filename
        if "." in filename:
            base, ext = os.path.splitext(filename)
            ext = ext.lower()
            filename = base if ext in [".dill", ".pkl", ".pickle"] else filename

        # ----------------------------------------------------------------
        #  Set the restart command
        # ----------------------------------------------------------------
        if restart_command is None and self.is_enabled == True:
            job_id = os.getenv("SLURM_JOB_ID")
            if job_id is None:
                fr.log.warning(
                    "No restart command is set. The model will not be able to restart.")
                restart_command = None
            else:
                import subprocess
                job_info = subprocess.run(
                        ["scontrol", "show", "job", job_id],
                        capture_output=True,
                        text=True, check=False)
                command = None
                for line in job_info.stdout.split("\n"):
                    if line.strip().startswith("Command="):
                        command = line.split("=", 1)[1].strip()
                if command is None:
                    fr.log.warning(
                        "No restart command is set. The model will not be able to restart.")
                restart_command = f"sbatch {command}"

        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self.realtime_interval = realtime_interval
        self.modeltime_interval = modeltime_interval
        self.iteration_interval = iteration_interval
        self.restart_command = restart_command
        self.filename = filename
        self.directory = directory
        self.file = None

        # private attributes
        self._last_restart_modeltime = None
        self._last_restart_iteration = None

    @fr.modules.module_method
    def setup(self, mset: fr.ModelSettingsBase) -> None:
        super().setup(mset)
        fr.log.verbose("Touching the restart directory.")
        os.makedirs(self.directory, exist_ok=True)

    @fr.modules.module_method
    def should_restart(self, mz: fr.ModelState) -> bool:
        # ----------------------------------------------------------------
        #  Realtime interval
        # ----------------------------------------------------------------
        if self.realtime_interval is not None:
            # get elapsed time
            elapsed_time = np.timedelta64(int(time.time() - fr.config.load_time), "s")
            if elapsed_time >= self.realtime_interval:
                fr.log.info(
                    "Realtime restart interval reached. Model will restart.")
                self.set_full_filename(mz.clock.it)
                return True
        # ----------------------------------------------------------------
        #  Modeltime interval
        # ----------------------------------------------------------------
        elif self.modeltime_interval is not None:
            if self._last_restart_modeltime is None:
                self._last_restart_modeltime = mz.clock.time
            elapsed_time = mz.clock.time - self._last_restart_modeltime
            if elapsed_time >= self.modeltime_interval:
                fr.log.info(
                    "Modeltime restart interval reached. Model will restart.")
                self.set_full_filename(mz.clock.it)
                self._last_restart_modeltime = mz.clock.time
                self._last_restart_iteration = mz.clock.it
                return True
        # ----------------------------------------------------------------
        #  Iteration interval
        # ----------------------------------------------------------------
        elif self.iteration_interval is not None:
            if self._last_restart_iteration is None:
                self._last_restart_iteration = mz.clock.it
            elapsed_time = mz.clock.it - self._last_restart_iteration
            if elapsed_time >= self.iteration_interval:
                fr.log.info(
                    "Iteration restart interval reached. Model will restart.")
                self.set_full_filename(mz.clock.it)
                self._last_restart_modeltime = mz.clock.time
                self._last_restart_iteration = mz.clock.it
                return True
        return False

    @fr.modules.module_method
    def should_reload(self) -> bool:
        fr.log.verbose("Checking if restart files exist.")
        files = os.listdir(self.directory)
        files = [f for f in files if f.startswith(self.filename)]
        if len(files) > 0:
            fr.log.info("Found restart files. Model will reload.")
            its = [int(f.split("_")[1]) for f in files]
            self.set_full_filename(max(its))
            return True
        fr.log.info("No restart files found. Model will not reload.")
        return False

    @fr.modules.module_method
    def reset(self) -> None:
        self._last_restart_modeltime = None
        self._last_restart_iteration = None

    def set_full_filename(self, it: int) -> None:
        if fr.utils.MPI_AVAILABLE:
            rank = fr.utils.MPI.COMM_WORLD.Get_rank()
        else:
            rank = 0
        filename = f"{self.filename}_{it}_{rank}.dill"
        self.file = os.path.join(self.directory, filename)

    def restart(self, model: fr.Model) -> None:
        fr.log.info(
            "Stopping model at it: %d, time: %f",
            model.model_state.clock.it,
            model.model_state.clock.time)
        model.stop()
        model.save(self.file)
        fr.log.info(model.mset.timer)
        fr.log.info("Spawning new sbatch job:")
        fr.log.info(self.restart_command)
        fr.utils.mpi_barrier()
        if fr.utils.MPI_AVAILABLE:
            import subprocess
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

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def info(self) -> dict:
        res = super().info
        if not self.is_enabled:
            return res
        if self.realtime_interval is not None:
            res["Realtime Restart Interval"] = self.realtime_interval
        if self.modeltime_interval is not None:
            res["Modeltime Restart Interval"] = self.modeltime_interval
        if self.iteration_interval is not None:
            res["Iteration Restart Interval"] = self.iteration_interval
        if self.directory is not None:
            res["Directory"] = self.directory
        if self.filename is not None:
            res["Filename"] = self.filename
        return res
