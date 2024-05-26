from abc import abstractmethod
import numpy as np

from fridom.framework.grid_base import GridBase
from fridom.framework.state_base import StateBase
from fridom.framework.model_state import ModelStateBase


class ModelBase:
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
        total_tendency()        : Calculate the tendency.
        nonlinear_tendency()    : Calculate nonlinear tendency.
        time_stepping()         : Perform Adam-Bashforth time stepping.
        update_pointer()        : Update pointer for the time stepping.
        update_coeff_AB()       : Upward ramping of Adam-Bashforth coefficients  
                                  after restart.
        diagnostics()           : Print diagnostics of the model.
        reset()                 : Reset the model (pointers, tendencies)
    """

    def __init__(self, 
                 grid: GridBase, 
                 State: StateBase,
                 ModelState: ModelStateBase, 
                 is_spectral=False) -> None:
        """
        Constructor.

        Args:
            grid (Grid)             : Grid.
            State (State)           : State class.
        """
        mset = grid.mset
        self.mset = mset
        self.grid = grid
        cp = grid.cp

        # state variable
        self.model_state = ModelState(grid=grid, is_spectral=is_spectral)

        # time stepping variables
        self.dz_list = [State(grid, is_spectral=is_spectral) for _ in range(mset.time_levels)]
        self.pointer = np.arange(mset.time_levels, dtype=cp.int32)
        self.coeffs = [
            cp.asarray(mset.AB1), cp.asarray(mset.AB2),
            cp.asarray(mset.AB3), cp.asarray(mset.AB4)
        ]
        self.coeff_AB = cp.zeros(mset.time_levels, dtype=mset.dtype)

        # Timer
        from fridom.framework.timing_module import TimingModule
        self.timer = TimingModule()
        self.timer.add_component("Total Tendency")
        self.timer.add_component("Time Stepping")

        # Modules
        from copy import deepcopy
        self.tendency_modules = deepcopy(mset.tendency_modules)
        self.diagnostics_modules = deepcopy(mset.diagnostic_modules)
        return

    def start(self):
        """
        Prepare the model for running.
        """
        # start all modules
        for module in self.tendency_modules:
            module.start(grid=self.grid, timer=self.timer)
        for module in self.diagnostics_modules:
            module.start(grid=self.grid, timer=self.timer)
        return

    def stop(self):
        """
        Finish the model run.
        """
        for module in self.tendency_modules:
            module.stop()
        for module in self.diagnostics_modules:
            module.stop()
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
        start_timer("Total Tendency")
        self.total_tendency()
        end_timer("Total Tendency")

        # loop over tendency modules
        for module in self.tendency_modules:
            module.update(mz=self.model_state, dz=self.dz)

        # Adam Bashforth time stepping
        start_timer("Time Stepping")
        self.time_stepping()
        end_timer("Time Stepping")

        self.model_state.it += 1
        self.model_state.time += self.mset.dt

        # loop over diagnostics modules
        for module in self.diagnostics_modules:
            module.update(self.model_state, self.dz)
        return


    # ============================================================
    #   TENDENCIES
    # ============================================================

    @abstractmethod
    def total_tendency(self) -> None:
        """
        This is the main method of the model. It calculates the right hand side of the model equations. All models must implement this method.
        The result of the calculation must be stored in self.dz[:]
        $\partial_t z = f(z, t)$ (f is the rhs)
        """
        # to implement in child class
        return

    def nonlinear_tendency(self) -> StateBase:
        """
        Calculate nonlinear tendency. Models do not have to implement this method. It is called in some balancing routines.

        Returns:
            dz (State)  : Nonlinear tendency.
        """
        # to implement in child class
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