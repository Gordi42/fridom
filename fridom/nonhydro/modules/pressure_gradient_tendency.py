from fridom.nonhydro.state import State
from fridom.nonhydro.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module


class PressureGradientTendency(Module):
    """
    This class computes the pressure gradient tendency of the model.
    """
    def __init__(self):
        super().__init__(name="Pressure Gradient")

    @start_module
    def start(self):
        # compute the grid spacing
        mset = self.mset
        self.dx1 = mset.dtype(1.0) / mset.dx
        self.dy1 = mset.dtype(1.0) / mset.dy
        self.dz1 = mset.dtype(1.0) / mset.dz
        return

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the pressure gradient tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Tendency of the state.
        """
        cp = dz.cp
        p_pad = cp.pad(mz.p, ((0,1), (0,1), (0,1)), 'wrap')

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
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    discretization: Finite difference\n"
        return res

# remove symbols from the namespace
del State, ModelState, Module, update_module, start_module