from fridom.nonhydro.state import State
from fridom.framework.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module


class BiharmonicFriction(Module):
    """
    This class computes the biharmonic friction tendency of the model.

    Computes:
    $ dz.u -= ah \\nabla^4 u + kh \\partial_z^4 u $
    $ dz.v -= ah \\nabla^4 v + kh \\partial_z^4 v $
    $ dz.w -= ah \\nabla^4 w + kh \\partial_z^4 w $
    where:
    - `ah`: Horizontal biharmonic friction coefficient.
    - `av`: Vertical biharmonic friction coefficient.
    """
    def __init__(self, ah: float = 0, av: float = 0):
        """
        ## Arguments:
        - `ah`: Horizontal harmonic friction coefficient.
        - `av`: Vertical harmonic friction coefficient.
        """
        super().__init__(name="Biharmonic Friction", ah=ah, av=av)

    @start_module
    def start(self):
        # cast the parameters to the correct data type
        self.ah = self.mset.dtype(self.ah)
        self.av = self.mset.dtype(self.av)

        # compute the grid spacing
        mset = self.mset
        self.dx2 = mset.dtype(1.0) / mset.dx**2
        self.dy2 = mset.dtype(1.0) / mset.dy**2
        self.dz2 = mset.dtype(1.0) / mset.dz**2
        return

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the biharmonic friction tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Tendency of the state.
        """
        # shorthand notation
        dx2 = self.dx2; dy2 = self.dy2; dz2 = self.dz2
        ah = self.ah; av = self.av

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
            p = self.grid.cp.pad(p, ((2,2), (2,2), (2,2)), 'wrap')

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


        # biharmonic friction 
        dz.u[:] -= biharmonic_function(mz.z.u, ah, av)
        dz.v[:] -= biharmonic_function(mz.z.v, ah, av)
        dz.w[:] -= biharmonic_function(mz.z.w, ah, av)
        return 

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    ah: {self.ah}\n    av: {self.av}"
        return res

# remove symbols from the namespace
del State, ModelState, Module, update_module, start_module