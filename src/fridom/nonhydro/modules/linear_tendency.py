from fridom.nonhydro.state import State
from fridom.framework.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module


class LinearTendency(Module):
    """
    This class computes the linear tendency of the model.
    """
    def __init__(self):
        super().__init__(name="Linear Tendency")

    @start_module
    def start(self):
        self.quarter = self.mset.dtype(0.25)
        self.half = self.mset.dtype(0.5)

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the linear tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Linear tendency of the state.
        """
        # compute the linear tendency
        u = mz.z.u; v = mz.z.v; w = mz.z.w; b = mz.z.b
        dsqr = self.mset.dsqr
        f_cor = self.grid.f_array
        N2 = self.grid.N2_array
        cp = self.grid.cp

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

        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    discretization: Finite Difference\n"
        return res

# remove symbols from the namespace
del State, Module, update_module, start_module, ModelState