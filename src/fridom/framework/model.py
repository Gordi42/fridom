# Import external modules
from typing import TYPE_CHECKING
from mpi4py import MPI
import numpy as np
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
            steps=None, 
            runlen=None, 
            start_time=np.datetime64(0, 's'),
            end_time=None,
            progress_bar=True) -> None:
        """
        Run the model
        
        Parameters
        ----------
        `steps` : `int`
            Number of steps to run.
        `runlen` : `np.timedelta64`
            Length of the run.
        `start_time` : `np.datetime64` (default np.datetime64(0, 's'))
            Start time of the run.
        `end_time` : `np.datetime64`
            End time of the run.
        `progress_bar` : `bool`
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

        # set the start time
        self.model_state.time = start_time

        # ----------------------------------------------------------------
        #  Calculate number of steps / end time
        # ----------------------------------------------------------------
        # calculate end time if runlen is given
        if runlen is not None:
            end_time = self.model_state.time + runlen

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
        
        # ----------------------------------------------------------------
        #  Set up the progress bar
        # ----------------------------------------------------------------
        from fridom.framework.utils import ProgressBar
        pbar = ProgressBar(disable=not progress_bar)

        # start the model
        self.start()

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

                # update the progress bar
                mz = self.model_state
                pbar.update(
                    value = 100 * (i-first_it+1) / steps,
                    postfix = f"It: {mz.it} - Time: {mz.time}")

        # ----------------------------------------------------------------
        #  Main loop: Given run length
        # ----------------------------------------------------------------
        elif end_time is not None:
            config.logger.info(
                f"Running model from {self.model_state.time} to {end_time}")

            # loop until the end time is reached
            while self.model_state.time < end_time:
                self.step()
                # update the progress bar
                mz = self.model_state
                pbar.update(
                    value = 100*(mz.time-start_time) / (end_time-start_time),
                    postfix = f"It: {mz.it} - Time: {mz.time}")
        
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
        self.time_stepper.update_tendency()
        self.model_state.dz = self.time_stepper.dz

        # synchronize the state vector (ghost points)
        self.timer.get("sync").start()
        self.z.sync()
        self.timer.get("sync").stop()

        # apply boundary conditions to the state variable
        # self.model_state = self.bc.update(mz=self.model_state)

        # calculate tendency
        self.model_state = self.tendencies.update(mz=self.model_state)

        self.time_stepper.dz = self.model_state.dz
        self.model_state = self.time_stepper.update(mz=self.model_state)

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
