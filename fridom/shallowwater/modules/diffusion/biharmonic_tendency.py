from fridom.shallowwater.state import State
from fridom.shallowwater.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module


class BiharmonicTendency(Module):
    """
    This class computes the biharmonic friction and mixing tendency of the model.
    """
    def __init__(self, ah: float = 0.0, kh: float = 0.0):
        """
        ## Arguments:
        - `kh`: Biharmonic friction coefficient.
        - `kv`: Biharmonic mixing coefficient.
        """
        super().__init__(name="Biharmonic Tendency", ah=ah, kh=kh)

    @start_module
    def start(self):
        # cast the parameters to the correct data type
        self.ah = self.mset.dtype(self.ah)
        self.kh = self.mset.dtype(self.kh)

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the biharmonic tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Tendency of the state.
        """
        # shorthand notation
        dx2 = self.grid.dx2; dy2 = self.grid.dy2
        ah = self.ah; kh = self.kh

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
        xf = (f,c); xb = (b,c)
        yf = (c,f); yb = (c,b)
        cc = (c,c)

        def biharmonic_function(p, coeff):
            """
            Calculate biharmonic friction / mixing.

            Args:
                p (FieldVariable)   : Field variable.

            Returns:
                res (FieldVariable) : Biharmonic friction / mixing.
            """
            # Padding with periodic boundary conditions
            p = self.grid.cp.pad(p, ((2,2), (2,2)), 'wrap')

            # Apply boundary conditions
            if not self.mset.periodic_bounds[0]:
                p[:2,:]  = 0; p[-2:,:] = 0
            if not self.mset.periodic_bounds[1]:
                p[:,:2]  = 0; p[:,-2:] = 0

            # first two derivatives
            tmp_h = (p[xf] - 2*p[cc] + p[xb])*dx2*coeff + \
                    (p[yf] - 2*p[cc] + p[yb])*dy2*coeff

            # last two derivatives
            res = (tmp_h[xf] - 2*tmp_h[cc] + tmp_h[xb])*dx2 + \
                  (tmp_h[yf] - 2*tmp_h[cc] + tmp_h[yb])*dy2
            return res

        # biharmonic friction / mixing
        dz.u[:] -= biharmonic_function(mz.z.u, ah)
        dz.v[:] -= biharmonic_function(mz.z.v, ah)
        dz.h[:] -= biharmonic_function(mz.z.h, kh)
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    ah: {self.ah}\n    kh: {self.kh}"
        return res

# remove symbols from the namespace
del State, ModelState, Module, update_module, start_module