from fridom.nonhydro.grid import Grid
from fridom.nonhydro.state import State
from fridom.framework.field_variable import FieldVariable
from fridom.framework.timing_module import TimingModule


class PressureGradientTendency:
    """
    This class computes the pressure gradient tendency of the model.
    """

    def __init__(self, grid: Grid, timer: TimingModule):
        """
        Constructor.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        mset = grid.mset
        self.mset = mset
        self.grid = grid
        self.timer = timer

        self.dx1 = mset.dtype(1.0) / mset.dx
        self.dy1 = mset.dtype(1.0) / mset.dy
        self.dz1 = mset.dtype(1.0) / mset.dz

    def __call__(self, p: FieldVariable, dz: State) -> None:
        """
        Compute the pressure gradient tendency of the model.

        Args:
            p (FieldVariable) : Pressure
            dz (State)        : State tendency
        """
        # start the timer
        self.timer.get("Pressure Gradient").start()

        cp = dz.cp
        p_pad = cp.pad(p, ((0,1), (0,1), (0,1)), 'wrap')

        # Slices
        c = slice(None,-1); f = slice(1,None)
        xf = (f,c,c); xc = (c,c,c)
        yf = (c,f,c); yc = (c,c,c)
        zf = (c,c,f); zc = (c,c,c)

        # remove pressure gradient
        dz.u -= (p_pad[xf] - p_pad[xc])*self.dx1 
        dz.v -= (p_pad[yf] - p_pad[yc])*self.dy1 
        dz.w -= (p_pad[zf] - p_pad[zc])*self.dz1 / self.mset.dsqr

        # apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            dz.u[-1,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            dz.v[:,-1,:] = 0
        if not self.mset.periodic_bounds[2]:
            dz.w[:,:,-1] = 0

        # stop the timer
        self.timer.get("Pressure Gradient").stop()
        return

# remove symbols from the namespace
del Grid, State, TimingModule, FieldVariable