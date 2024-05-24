from fridom.NonHydrostatic.ModelSettings import ModelSettings
from fridom.NonHydrostatic.Grid import Grid
from fridom.NonHydrostatic.State import State
from fridom.Framework.TimingModule import TimingModule


class LinearTendency:
    """
    This class computes the linear tendency of the model.
    """

    def __init__(self, mset: ModelSettings, grid: Grid, timer: TimingModule):
        """
        Constructor of the LinearTendency class.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        self.mset = mset
        self.grid = grid
        self.timer = timer

        self.quarter = mset.dtype(0.25)
        self.half = mset.dtype(0.5)

    def __call__(self, z: State, dz: State) -> None:
        """
        Compute the linear tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : Linear tendency of the state.
        """
        # start the timer
        self.timer.get("Linear Tendency").start()

        # compute the linear tendency
        u = z.u; v = z.v; w = z.w; b = z.b
        dsqr = self.mset.dsqr
        f_cor = self.grid.f_array
        N2 = self.grid.N2_array
        cp = z.cp

        # Padding for averaging
        up = cp.pad(u, ((1,0), (0,1), (0,0)), 'wrap')
        vp = cp.pad(v, ((0,1), (1,0), (0,0)), 'wrap')
        wp = cp.pad(w, ((0,0), (0,0), (1,0)), 'wrap')
        bp = cp.pad(b, ((0,0), (0,0), (0,1)), 'wrap')

        # No boundary conditions required here

        # Slices
        f = slice(1,None); b = slice(None,-1)
        q = self.quarter  # 0.25
        h = self.half     # 0.5

        # calculate u-tendency
        dz.u[:] = (vp[f,f] + vp[f,b] + vp[b,f] + vp[b,b]) * q * f_cor

        # calculate v-tendency
        dz.v[:] = (up[f,f] + up[f,b] + up[b,f] + up[b,b]) * q * (-f_cor)

        # calculate w-tendency
        dz.w[:] = (bp[:,:,f] + bp[:,:,b]) * h / dsqr
        
        # apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            dz.u[-1,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            dz.v[:,-1,:] = 0
        if not self.mset.periodic_bounds[2]:
            dz.w[:,:,-1] = 0

        # calculate b-tendency
        dz.b[:] = - (wp[:,:,f] + wp[:,:,b]) * h * N2

        # stop the timer
        self.timer.get("Linear Tendency").stop()

        return

# remove symbols from the namespace
del ModelSettings, Grid, State, TimingModule