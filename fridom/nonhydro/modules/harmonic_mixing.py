from fridom.nonhydro.grid import Grid
from fridom.nonhydro.state import State
from fridom.framework.timing_module import TimingModule


class HarmonicMixing:
    """
    This class computes the harmonic mixing tendency of the model.
    """

    def __init__(self, grid: Grid, timer: TimingModule):
        """
        Constructor of the Harmonic mixing class.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        mset = grid.mset
        self.mset = mset
        self.grid = grid
        self.timer = timer

    def __call__(self, z: State, dz:State):
        """
        Compute the harmonic mixing tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : Tendency of the state.
        """
        # start the timer
        self.timer.get("Harmonic Mixing").start()

        # compute the harmonic friction tendency
        b = z.b
        kh = self.mset.kh; kv = self.mset.kv; 

        # [TODO] boundary conditions
        dz.b += (b.diff_2(0) + b.diff_2(1))*kh + b.diff_2(2)*kv

        # stop the timer
        self.timer.get("Harmonic Mixing").stop()

        return 

# remove symbols from the namespace
del Grid, State, TimingModule