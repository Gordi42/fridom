from fridom.nonhydro.grid import Grid
from fridom.nonhydro.state import State
from fridom.framework.timing_module import TimingModule

class SourceTendency:
    """
    Parent class for source terms in the model.

    ## Methods:
        update:     Function that is called every time step
                    to update the source term.
    """
    def __init__(self, grid: Grid, timer: TimingModule):
        mset = grid.mset
        self.mset = mset
        self.grid = grid
        self.timer = timer

        self.z = None # to be set
        return

    def __call__(self, dz: State, time: float):
        """
        Adds the source term to the state tendency.

        ## Arguments:
            dz (State):     state tendency
            time (float):   current time in simulation
        """
        # start the timer
        self.timer.get("Source").start()

        self.z.update(time)

        dz.u[:] += self.z.u
        dz.v[:] += self.z.v
        dz.w[:] += self.z.w
        dz.b[:] += self.z.b

        # stop the timer
        self.timer.get("Source").stop()
        return

# remove symbols from namespace
del Grid, State, TimingModule