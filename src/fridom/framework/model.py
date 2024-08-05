# Import external modules
from typing import TYPE_CHECKING, Union
from mpi4py import MPI
import numpy as np
import fridom.framework as fr
# Import internal modules
from fridom.framework import config
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase


class Model:
    """
    Base class for the model.

    Attributes:
        mset (ModelSettings)    : Model settings.
        grid (Grid)             : Grid.
        z (State)               : State variable.
        dz_list (list)          : List of tendency terms (for time stepping).
        pointer (np.ndarray)    : Pointer for time stepping.
        coeff_AB (np.ndarray)   : Adam-Bashforth coefficients.
        timer (TimingModule)    : Timer.
        it (int)                : Iteration counter.
        time (float)            : Model time.
        dz (State)              : Current tendency term.

    Methods:
        run()                   : Run the model for a given number of steps.
        step()                  : Perform one time step.
        reset()                 : Reset the model (pointers, tendencies)
    """

    def __init__(self, mset: 'ModelSettingsBase') -> None:
        """
        Constructor.
        """
        self.mset = mset

        # state variable
        from fridom.framework.model_state import ModelState
        self.model_state = ModelState(mset)

        # Timer
        self.timer = mset.timer

        # Modules
        self.restart_module = mset.restart_module
        self.tendencies  = mset.tendencies
        self.diagnostics = mset.diagnostics
        self.bc = mset.bc

        # Time stepper
        self.time_stepper = mset.time_stepper
        return

    def start(self):
        """
        Prepare the model for running.
        """
        # start all modules
        self.timer.total.start()
        self.restart_module.start()
        self.tendencies.start()
        self.diagnostics.start()
        self.time_stepper.start()
        self.bc.start()
        self.model_state.panicked = False

        # compile the modules
        from time import time
        fr.config.logger.notice("Compiling tendency modules")
        start_time = time()
        mz = fr.ModelState(self.mset)
        mz.dz = self.mset.state_constructor()
        self.tendencies.update(mz)
        fr.config.logger.notice(
            f"Compilation finished in {time()-start_time:.2f} seconds")
        return

    def stop(self):
        """
        Finish the model run.
        """
        self.restart_module.stop()
        self.tendencies.stop()
        self.diagnostics.stop()
        self.time_stepper.stop()
        self.bc.stop()
        self.timer.total.stop()
        return
        
    def reset(self) -> None:
        """
        Reset the model (pointers, tendencies).
        """
        self.restart_module.reset()
        self.tendencies.reset()
        self.diagnostics.reset()
        self.time_stepper.reset()
        self.bc.reset()
        self.model_state.reset()
        self.timer.reset()
        # to implement in child class
        return

    # ============================================================
    #   RUN MODEL
    # ============================================================

    def run(self, 
            steps: int | None = None, 
            runlen: Union[np.timedelta64, float, int, None] = None,
            start_time: Union[np.datetime64, float, int] = 0,
            end_time: Union[np.datetime64, float, int, None] = None,
            progress_bar=True) -> None:
        """
        Run the model
        
        Parameters
        ----------
        `steps` : `int` (default: None)
            Number of steps to run.
        `runlen` : `np.timedelta64 | float | int` (default: None)
            Length of the run.
        `start_time` : `np.datetime64 | float | int` (default: 0)
            Start time of the run.
        `end_time` : `np.datetime64 | float | int` (default: None)
            End time of the run.
        `progress_bar` : `bool` (default: True)
            Show progress bar.
        
        Raises
        ------
        `ValueError`
            Only one of `steps`, `runlen` or `end_time` can be given.
        """
        # ----------------------------------------------------------------
        #  Check input
        # ----------------------------------------------------------------
        # only one of steps, runlen or end_time can be given
        if sum([steps is not None, 
                runlen is not None, 
                end_time is not None]) > 1:
            raise ValueError("Only one of steps, runlen or end_time can be given.")

        # ----------------------------------------------------------------
        #  Convert time parameters to seconds
        # ----------------------------------------------------------------
        self.model_state.start_time = start_time
        datetime_formatting = False
        if isinstance(start_time, np.datetime64):
            datetime_formatting = True
            start_time = fr.utils.to_seconds(start_time)
        if isinstance(end_time, np.datetime64):
            datetime_formatting = True
            end_time = fr.utils.to_seconds(end_time)
        if isinstance(runlen, np.timedelta64):
            runlen = fr.utils.to_seconds(runlen)

        # set the start time
        self.model_state.time = start_time

        # ----------------------------------------------------------------
        #  Create the postfix formatter
        # ----------------------------------------------------------------
        if datetime_formatting:
            def postfix_formatter(mz):
                return f"It: {mz.it} - Time: {np.datetime64(int(mz.time), 's')}"
        else:
            def postfix_formatter(mz):
                human_time = fr.utils.humanize_number(mz.time, unit="seconds")
                return f"It: {mz.it} - Time: {human_time}"

        # ----------------------------------------------------------------
        #  Calculate number of steps / end time
        # ----------------------------------------------------------------
        # calculate end time if runlen is given
        if runlen is not None:
            end_time = start_time + runlen

        # calculate the final iteration step if steps is given
        if steps is not None:
            first_it = self.model_state.it
            final_it = first_it + steps
            
        # ----------------------------------------------------------------
        #  Load the model
        # ----------------------------------------------------------------
        # check if the model needs to be reloaded
        if self.restart_module.should_reload():
            self.load(self.restart_module.file)

        # start the model
        self.start()
        
        # ----------------------------------------------------------------
        #  Set up the progress bar
        # ----------------------------------------------------------------
        from fridom.framework.utils import ProgressBar
        pbar = ProgressBar(disable=not progress_bar)

        # ----------------------------------------------------------------
        #  Initial diagnostics
        # ----------------------------------------------------------------
        for module in self.diagnostics.module_list:
            if module.execute_at_start:
                self.model_state = module.update(self.model_state)

        # ----------------------------------------------------------------
        #  Main loop: Given number of setps
        # ----------------------------------------------------------------
        if steps is not None:
            start_it = self.model_state.it
            config.logger.info(
                f"Running model from iteration {start_it} to {final_it}")
            
            # loop over the given number of steps
            for i in range(start_it, final_it):
                self.step()

                if self.model_state.panicked:
                    config.logger.warning(
                        "Something went wrong. Stopping model.")
                    break

                # update the progress bar
                with self.timer["Progress Bar"]:
                    mz = self.model_state
                    pbar.update(
                        value = 100 * (i-first_it+1) / steps,
                        postfix = postfix_formatter(mz))

        # ----------------------------------------------------------------
        #  Main loop: Given run length
        # ----------------------------------------------------------------
        elif end_time is not None:
            config.logger.info(
                f"Running model from {self.model_state.time} to {end_time}")

            # loop until the end time is reached
            while self.model_state.time < end_time:
                self.step()

                if self.model_state.panicked:
                    config.logger.warning(
                        "Something went wrong. Stopping model.")
                    break

                # update the progress bar
                with self.timer["Progress Bar"]:
                    mz = self.model_state
                    pbar.update(
                        value = 100*(mz.time-start_time) / (end_time-start_time),
                        postfix = postfix_formatter(mz))
        
        # close the progress bar
        pbar.close()

        # stop the model
        self.stop()

        config.logger.info(
            f"Model run finished at it: {self.model_state.it}, time: {self.model_state.time}")
        config.logger.info(self.mset.timer)

        return

    # ============================================================
    #   SINGLE TIME STEP
    # ============================================================

    def step(self) -> None:
        """
        Update the model state by one time step.
        """
        # synchronize the state vector (ghost points)
        with self.timer["sync"]:
            self.z.sync()

        # perform the time step
        self.model_state = self.time_stepper.update(mz=self.model_state)

        # check if there are any nans in the state variable
        with self.timer["check_nan"]:
            if self.model_state.it % self.mset.nan_check_interval == 0:
                if self.model_state.z.has_nan():
                    config.logger.critical(
                        "State variable contains NaNs. Stopping model.")
                    self.model_state.panicked = True

        # make diagnostics
        self.model_state = self.diagnostics.update(self.model_state)

        # check if the model should restart
        if self.restart_module.should_restart(self.model_state):
            self.restart()
        return

    def restart(self) -> None:
        config.logger.info(
            f"Stopping model at it: {self.model_state.it}, time: {self.model_state.time}")
        self.stop()
        self.save(self.restart_module.file)
        config.logger.info(self.mset.timer)
        config.logger.info("Spawning new sbatch job:")
        config.logger.info(self.restart_module.restart_command)
        MPI.COMM_WORLD.Barrier()
        if MPI.COMM_WORLD.Get_rank() == 0:
            import subprocess
            result = subprocess.run(
                self.restart_module.restart_command.split(), 
                capture_output=True, text=True)
            config.logger.notice(result.stdout)
            if result.stderr:
                config.logger.error(result.stderr)
        MPI.COMM_WORLD.Barrier()
        exit()

    
    # ============================================================
    #   Getters and setters
    # ============================================================

    @property
    def z(self):
        """
        Returns the current state variable.
        """
        return self.model_state.z
    
    @z.setter
    def z(self, value):
        """
        Set the current state variable.
        """
        self.model_state.z = value
        return

    # ============================================================
    #   OTHER METHODS
    # ============================================================

    def load(self, file: str) -> None:
        # underscores are not allowed in the filename
        import dill
        # get a list of all files in the directory that start with the filename
        with open(file, "rb") as f:
            model = dill.load(f)
            model.mset.grid = self.mset.grid

        for key, attr in vars(model).items():
            setattr(self, key, attr)
        return

    def save(self, file: str) -> None:
        import dill
        with open(file, "wb") as f:
            config.logger.verbose(f"Saving model to {file}")
            grid = self.mset.grid
            # remove the grid from the model before pickling
            self.mset.grid = None
            dill.dump(self, f)
            # restore the grid
            self.mset.grid = grid
        return
