from fridom.NonHydrostatic.ModelSettings import ModelSettings
from fridom.NonHydrostatic.Grid import Grid
from fridom.NonHydrostatic.State import State
from fridom.Framework.TimingModule import TimingModule


class NonlinearTendency:
    """
    This class computes the nonlinear tendency of the model.
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
        self.dx1 = mset.dtype(1.0) / mset.dx
        self.dy1 = mset.dtype(1.0) / mset.dy
        self.dz1 = mset.dtype(1.0) / mset.dz

        # add a timer
        self.timer.add_component('Nonlinear Tendency')

    def __call__(self, z: State, dz: State) -> None:
        """
        Compute the nonlinear tendency of the model.

        Args:
            z (State)  : State object.
            dz (State) : Nonlinear tendency of the state.
        """
        # start the timer
        self.timer.get("Nonlinear Tendency").start()

        # compute the nonlinear tendency

        # shorthand notation
        dx1 = self.dx1; dy1 = self.dy1; dz1 = self.dz1
        quarter = self.quarter

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
        xf = (f,c,c); xb = (b,c,c)
        yf = (c,f,c); yb = (c,b,c)
        zf = (c,c,f); zb = (c,c,b)
        cc = (c,c,c)
        
        # Padding with periodic boundary conditions
        periodic = lambda x: z.cp.pad(x, ((2,2), (2,2), (2,2)), 'wrap')
        u  = periodic(z.u)
        v  = periodic(z.v)
        w  = periodic(z.w)
        bu = periodic(z.b)

        # boundary conditions
        if not self.mset.periodic_bounds[0]:
            u[:2,:,:]  = 0; v[:2,:,:]  = 0; w[:2,:,:]  = 0; bu[:2,:,:]  = 0
            u[-2:,:,:] = 0; v[-2:,:,:] = 0; w[-2:,:,:] = 0; bu[-2:,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            u[:,:2,:]  = 0; v[:,:2,:]  = 0; w[:,:2,:]  = 0; bu[:,:2,:]  = 0
            u[:,-2:,:] = 0; v[:,-2:,:] = 0; w[:,-2:,:] = 0; bu[:,-2:,:] = 0
        if not self.mset.periodic_bounds[2]:
            u[:,:,:2]  = 0; v[:,:,:2]  = 0; w[:,:,:2]  = 0; bu[:,:,:2]  = 0
            u[:,:,-2:] = 0; v[:,:,-2:] = 0; w[:,:,-2:] = 0; bu[:,:,-2:] = 0


        # function to calculate nonlinear tendency
        def flux_divergence(p, slice):
            """
            Calculate the flux divergence of a field p.

            Args:
                p (FieldVariable)   : Field variable.
                slice (tuple)       : Slicing tuple for averaging.

            Returns:
                res (FieldVariable) : Flux divergence.
            """
            # flux in x-,y-,z-direction
            fe = (u[cc] + u[slice]) * (p[cc] + p[xf]) * quarter
            fn = (v[cc] + v[slice]) * (p[cc] + p[yf]) * quarter
            ft = (w[cc] + w[slice]) * (p[cc] + p[zf]) * quarter

            res = (fe[cc] - fe[xb])*dx1 + \
                  (fn[cc] - fn[yb])*dy1 + \
                  (ft[cc] - ft[zb])*dz1
            return res

        # calculate nonlinear tendency
        dz.u[:] -= flux_divergence(u, xf) * self.mset.Ro
        dz.v[:] -= flux_divergence(v, yf) * self.mset.Ro
        dz.w[:] -= flux_divergence(w, zf) * self.mset.Ro
        dz.b[:] -= flux_divergence(bu, cc) * self.mset.Ro

        # stop the timer
        self.timer.get("Nonlinear Tendency").stop()

        return 