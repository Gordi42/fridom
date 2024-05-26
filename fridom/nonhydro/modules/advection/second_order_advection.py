from fridom.nonhydro.state import State
from fridom.nonhydro.model_state import ModelState
from fridom.framework.modules.module import Module, start_module, update_module

class SecondOrderAdvection(Module):
    """
    This advection scheme is a second-order centered advection scheme.
    It is numerically the same as centered advection scheme + linear interpolation
    but may be more efficient.
    """
    def __init__(self):
        """
        This advection scheme is a second-order centered advection scheme.
        It is numerically the same as centered advection scheme + linear interpolation
        but may be more efficient.

        ## Arguments:
        - None
        """
        super().__init__(name="Second Order Advection")

    @start_module
    def start(self):
        mset = self.mset
        self.quarter = mset.dtype(0.25)
        self.dx1 = mset.dtype(1.0) / mset.dx
        self.dy1 = mset.dtype(1.0) / mset.dy
        self.dz1 = mset.dtype(1.0) / mset.dz

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the advection term of the state vector z.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Tendency of the state.
        """
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
        periodic = lambda x: mz.z.cp.pad(x, ((2,2), (2,2), (2,2)), 'wrap')
        u  = periodic(mz.z.u)
        v  = periodic(mz.z.v)
        w  = periodic(mz.z.w)
        bu = periodic(mz.z.b)

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
        return 

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    scheme = Second Order Advection\n"
        return res

# remove symbols from the namespace
del State, ModelState, Module, update_module, start_module