from fridom.nonhydro.grid import Grid
from fridom.nonhydro.state import State
from fridom.framework.timing_module import TimingModule


class HarmonicFriction:
    """
    This class computes the harmonic friction tendency of the model.
    """

    def __init__(self, grid: Grid, timer: TimingModule):
        """
        Constructor of the Harmonic Friction class.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        mset = grid.mset
        self.mset = mset
        self.grid = grid
        self.timer = timer

    def __call__(self, z: State, dz:State):
        """
        Compute the harmonic friction tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : Tendency of the state.
        """
        # start the timer
        self.timer.get("Harmonic Friction").start()

        # compute the harmonic friction tendency
        u = z.u; v = z.v; w = z.w
        ah = self.mset.ah; av = self.mset.av; 

        # [TODO] boundary conditions
        dz.u += (u.diff_2(0) + u.diff_2(1))*ah + u.diff_2(2)*av
        dz.v += (v.diff_2(0) + v.diff_2(1))*ah + v.diff_2(2)*av
        dz.w += (w.diff_2(0) + w.diff_2(1))*ah + w.diff_2(2)*av

        # stop the timer
        self.timer.get("Harmonic Friction").stop()

        return 

# remove symbols from the namespace
del Grid, State, TimingModule