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
        time_stepping()         : Perform Adam-Bashforth time stepping.
        update_pointer()        : Update pointer for the time stepping.
        update_coeff_AB()       : Upward ramping of Adam-Bashforth coefficients  
                                  after restart.
        reset()                 : Reset the model (pointers, tendencies)
    """

    def __init__(self, grid: GridBase) -> None:
        """
        Constructor.

        Args:
            grid (Grid)             : Grid.
        """
        mset = grid.mset
        self.mset = mset
        self.grid = grid
        cp = grid.cp

        # state variable
        self.model_state = mset.model_state_constructor(grid)

        # time stepping variables
        self.dz_list = [mset.state_constructor(grid) for _ in range(mset.time_levels)]
        self.pointer = np.arange(mset.time_levels, dtype=cp.int32)
        self.coeffs = [
            cp.asarray(mset.AB1), cp.asarray(mset.AB2),
            cp.asarray(mset.AB3), cp.asarray(mset.AB4)
        ]
        self.coeff_AB = cp.zeros(mset.time_levels, dtype=mset.dtype)

        # Timer
        from fridom.framework.timing_module import TimingModule
        self.timer = TimingModule()
        self.timer.add_component("Time Stepping")

        # Modules
        from copy import deepcopy
        self.tendencies  = deepcopy(mset.tendencies)
        self.diagnostics = deepcopy(mset.diagnostics)
        return

    def start(self):
        """
        Prepare the model for running.
        """
        # start all modules
        self.tendencies.start(grid=self.grid, timer=self.timer)
        self.diagnostics.start(grid=self.grid, timer=self.timer)
        return

    def stop(self):
        """
        Finish the model run.
        """
        self.tendencies.stop()
        self.diagnostics.stop()
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
            steps = runlen / self.mset.dt
        
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
        self.update_pointer()
        self.update_coeff_AB()
        
        start_timer = lambda x: self.timer.get(x).start()
        end_timer   = lambda x: self.timer.get(x).stop()

        # calculate tendency
        self.tendencies.update(mz=self.model_state, dz=self.dz)

        # Adam Bashforth time stepping
        start_timer("Time Stepping")
        self.time_stepping()
        end_timer("Time Stepping")

        self.model_state.it += 1
        self.model_state.time += self.mset.dt

        # make diagnostics
        self.diagnostics.update(self.model_state, self.dz)
        return


    # ============================================================
    #   TIME STEPPING
    # ============================================================
    def time_stepping(self) -> None:
        """
        Perform Adam-Bashforth time stepping.
        """
        dt = self.mset.dt
        for i in range(self.mset.time_levels):
            self.z += self.dz_list[self.pointer[i]] * dt * self.coeff_AB[i]
        return


    def update_pointer(self) -> None:
        """
        Update pointer for Adam-Bashforth time stepping.
        """
        self.pointer = np.roll(self.pointer, 1)
        return


    def update_coeff_AB(self) -> None:
        """
        Upward ramping of Adam-Bashforth coefficients after restart.
        """
        # current time level (ctl)
        # maximum ctl is the number of time levels - 1
        cp = self.grid.cp
        ctl = min(self.it, self.mset.time_levels-1)

        # list of Adam-Bashforth coefficients
        coeffs = self.coeffs

        # choose Adam-Bashforth coefficients of current time level
        self.coeff_AB[:]      = 0
        self.coeff_AB[:ctl+1] = cp.asarray(coeffs[ctl])
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

    @property
    def dz(self):
        """
        Returns a pointer on the current tendency term.
        """
        return self.dz_list[self.pointer[0]]

    @dz.setter
    def dz(self, value):
        """
        Set the current tendency term.
        """
        self.dz_list[self.pointer[0]] = value
        return

    @property
    def it(self):
        """
        Returns the current iteration counter.
        """
        return self.model_state.it
    
    @it.setter
    def it(self, value):
        """
        Set the current iteration counter.
        """
        self.model_state.it = value
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
        self.it = 0
        self.timer.reset()
        self.z *= 0
        for dz in self.dz_list:
            dz *= 0
        # to implement in child class
        return

    @property
    def time(self) -> float:
        """
        Model time.
        """
        return self.it * self.mset.dt


# remove symbols from namespace
del abstractmethod, GridBase, StateBase