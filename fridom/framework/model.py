from abc import abstractmethod
import numpy as np

from fridom.framework.grid_base import GridBase
from fridom.framework.state_base import StateBase
from fridom.framework.model_state import ModelStateBase


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

    def __init__(self, grid: GridBase) -> None:
        """
        Constructor.

        Args:
            grid (Grid)             : Grid.
        """
        self.mset = grid.mset
        self.grid = grid

        # state variable
        self.model_state = grid.mset.model_state_constructor(grid)

        # Timer
        from fridom.framework.timing_module import TimingModule
        self.timer = TimingModule()

        # Modules
        from copy import deepcopy
        self.tendencies  = deepcopy(grid.mset.tendencies)
        self.diagnostics = deepcopy(grid.mset.diagnostics)

        # Time stepper
        self.time_stepper = deepcopy(grid.mset.time_stepper)
        return

    def start(self):
        """
        Prepare the model for running.
        """
        # start all modules
        self.tendencies.start(grid=self.grid, timer=self.timer)
        self.diagnostics.start(grid=self.grid, timer=self.timer)
        self.time_stepper.start(grid=self.grid, timer=self.timer)
        return

    def stop(self):
        """
        Finish the model run.
        """
        self.tendencies.stop()
        self.diagnostics.stop()
        self.time_stepper.stop()
        return
        


    # ============================================================
    #   RUN MODEL
    # ============================================================

    def run(self, steps=None, runlen=None) -> None:
        """
        Run the model for a given number of steps or a given time.

        Args:
            steps (int)     : Number of steps to run.
            runlen (float)  : Time to run. (preferred over steps)
        """
        # check if steps or runlen is given
        if runlen is not None:
            steps = runlen / self.time_stepper.dt
        
        # progress bar
        from tqdm import tqdm
        tq = tqdm if self.mset.enable_tqdm else lambda x: x

        # start the model
        self.start()

        # main loop
        self.timer.total.start()
        for _ in tq(range(int(steps))):
            self.step()
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
        dz = self.time_stepper.dz

        # calculate tendency
        self.tendencies.update(mz=self.model_state, dz=dz)

        self.time_stepper.update(mz=self.model_state)

        # make diagnostics
        self.diagnostics.update(self.model_state, dz)
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
        self.model_state.it = 0
        self.timer.reset()
        self.z *= 0
        # to implement in child class
        return


# remove symbols from namespace
del abstractmethod, GridBase, StateBase