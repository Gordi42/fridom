from fridom.NonHydrostatic.ModelSettings import ModelSettings
from fridom.NonHydrostatic.Grid import Grid
from fridom.NonHydrostatic.State import State
from fridom.Framework.TimingModule import TimingModule


class BiharmonicMixing:
    """
    This class computes the biharmonic mixing tendency of the model.
    """

    def __init__(self, mset: ModelSettings, grid: Grid, timer: TimingModule):
        """
        Constructor of the Biharmonic Mixing class.

        mset (ModelSettings) : ModelSettings object.
        grid (Grid)          : Grid object.
        """
        self.mset = mset
        self.grid = grid
        self.timer = timer

        self.dx2 = mset.dtype(1.0) / mset.dx**2
        self.dy2 = mset.dtype(1.0) / mset.dy**2
        self.dz2 = mset.dtype(1.0) / mset.dz**2

    def __call__(self, z: State, dz:State):
        """
        Compute the biharmonic mixing tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : Tendency of the state.
        """
        # start the timer
        self.timer.get("Biharmonic Mixing").start()

        # compute the biharmonic mixing tendency
        # shorthand notation
        dx2 = self.dx2; dy2 = self.dy2; dz2 = self.dz2
        khbi = self.mset.khbi; kvbi = self.mset.kvbi; 

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
        xf = (f,c,c); xb = (b,c,c)
        yf = (c,f,c); yb = (c,b,c)
        zf = (c,c,f); zb = (c,c,b)
        cc = (c,c,c)

        def biharmonic_function(p, h_coeff, v_coeff):
            """
            Calculate biharmonic friction / mixing.

            Args:
                p (FieldVariable)   : Field variable.

            Returns:
                res (FieldVariable) : Biharmonic friction / mixing.
            """
            # Padding with periodic boundary conditions
            p = z.cp.pad(p, ((2,2), (2,2), (2,2)), 'wrap')

            # Apply boundary conditions
            if not self.mset.periodic_bounds[0]:
                p[:2,:,:]  = 0; p[-2:,:,:] = 0
            if not self.mset.periodic_bounds[1]:
                p[:,:2,:]  = 0; p[:,-2:,:] = 0
            if not self.mset.periodic_bounds[2]:
                p[:,:,:2]  = 0; p[:,:,-2:] = 0

            # first two derivatives
            tmp_h = (p[xf] - 2*p[cc] + p[xb])*dx2*h_coeff + \
                    (p[yf] - 2*p[cc] + p[yb])*dy2*h_coeff
            tmp_v = (p[zf] - 2*p[cc] + p[zb])*dz2*v_coeff

            # last two derivatives
            res = (tmp_h[xf] - 2*tmp_h[cc] + tmp_h[xb])*dx2 + \
                  (tmp_h[yf] - 2*tmp_h[cc] + tmp_h[yb])*dy2 + \
                  (tmp_v[zf] - 2*tmp_v[cc] + tmp_v[zb])*dz2
            return res


        # biharmonic friction / mixing
        dz.b[:] -= biharmonic_function(z.b, khbi, kvbi)

        # stop the timer
        self.timer.get("Biharmonic Mixing").stop()

        return 

# remove symbols from the namespace
del ModelSettings, Grid, State, TimingModule