# Import external modules
from typing import TYPE_CHECKING
from mpi4py import MPI
from tqdm import tqdm
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
        self.model_state.time = mset.start_time

        # Timer
        self.timer = mset.timer

        # Modules
        from copy import deepcopy
        self.tendencies  = deepcopy(mset.tendencies)
        self.diagnostics = deepcopy(mset.diagnostics)
        self.bc = deepcopy(mset.bc)

        # Time stepper
        self.time_stepper = deepcopy(mset.time_stepper)
        return

    def start(self):
        """
        Prepare the model for running.
        """
        # start all modules
        self.tendencies.start(mset=self.mset)
        self.diagnostics.start(mset=self.mset)
        self.time_stepper.start(mset=self.mset)
        self.bc.start(mset=self.mset)
        return

    def stop(self):
        """
        Finish the model run.
        """
        self.tendencies.stop()
        self.diagnostics.stop()
        self.time_stepper.stop()
        self.bc.stop()
        return
        


    # ============================================================
    #   RUN MODEL
    # ============================================================

    def run(self, 
            steps=None, 
            runlen=None, 
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
        `end_time` : `np.datetime64`
            End time of the run.
        `progress_bar` : `bool`
            Show progress bar.
        
        Raises
        ------
        `ValueError`
            Only one of `steps`, `runlen` or `end_time` can be given.
        """
        # only one of steps, runlen or end_time can be given
        if sum([steps is not None, 
                runlen is not None, 
                end_time is not None]) > 1:
            raise ValueError("Only one of steps, runlen or end_time can be given.")

        # calculate end time if runlen is given
        if runlen is not None:
            end_time = self.model_state.time + runlen
        
        # progress bar
        if MPI.COMM_WORLD.Get_rank() != 0:
            progress_bar = False
        bar_format = "{percentage:3.2f}%|{bar}| [{elapsed}<{remaining}, {rate_fmt}]{postfix}"
        pbar = tqdm(
            total=100, disable=not progress_bar, bar_format=bar_format, unit="%")

        # start the model
        self.start()

        # main loop
        self.timer.total.start()
        if steps is not None:
            for i in range(steps):
                self.step()
                pbar.n = 100 * (i+1) / steps
                pbar.set_postfix_str(f"It: {self.model_state.it} - Time: {self.model_state.time}")
                pbar.refresh()
        elif end_time is not None:
            while self.model_state.time < end_time:
                self.step()
                pbar.n = 100 * ( (self.model_state.time - self.mset.start_time) 
                                   / (end_time - self.mset.start_time) )
                pbar.set_postfix_str(f"It: {self.model_state.it} - Time: {self.model_state.time}")
                pbar.refresh()
        pbar.close()
        self.timer.total.stop()

        # stop the model
        self.stop()

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
        self.bc.update(mz=self.model_state)

        # calculate tendency
        self.tendencies.update(mz=self.model_state)

        self.time_stepper.update(mz=self.model_state)

        # make diagnostics
        self.diagnostics.update(self.model_state)
        return

    
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

    def reset(self) -> None:
        """
        Reset the model (pointers, tendencies).
        """
        self.tendencies.reset()
        self.diagnostics.reset()
        self.time_stepper.reset()
        self.bc.reset()
        self.model_state.reset()
        self.timer.reset()
        # to implement in child class
        return

    def load(self, 
             filename: str = "model", 
             directory: str | None = None) -> 'Model':
        # underscores are not allowed in the filename
        import dill
        import os
        import numpy as np
        # get a list of all files in the directory that start with the filename
        filename = filename.replace("_", "-")
        directory = directory or "restart"

        files = os.listdir(directory)
        files = [f for f in files if f.startswith(filename)]
        
        # get the dates of the files
        dates = [f.split("_")[1] for f in files]
        dates = [np.datetime64(d) for d in dates]

        # get the latest date
        date = max(dates)

        rank = MPI.COMM_WORLD.Get_rank()
        filename = f"{filename}_{date}_{rank}.dill"

        file = os.path.join(directory, filename)
        with open(file, "rb") as f:
            model = dill.load(f)
            model.mset.grid = self.mset.grid

        for key, attr in vars(model).items():
            setattr(self, key, attr)
        return

    def save(self, 
             filename: str = "model", 
             directory: str | None = None) -> None:
        filename = filename.replace("_", "-")
        import dill
        import os
        directory = directory or "restart"
        os.makedirs(directory, exist_ok=True)
        time = self.model_state.time
        rank = MPI.COMM_WORLD.Get_rank()
        filename = f"{filename}_{time}_{rank}.dill"
        file = os.path.join(directory, filename)
        with open(file, "wb") as f:
            grid = self.mset.grid
            # remove the grid from the model before pickling
            self.mset.grid = None
            dill.dump(self, f)
            # restore the grid
            self.mset.grid = grid
