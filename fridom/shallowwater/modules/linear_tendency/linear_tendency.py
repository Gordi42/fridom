from fridom.shallowwater.state import State
from fridom.framework.model_state import ModelState
from fridom.framework.modules.module import Module, update_module


class LinearTendency(Module):
    """
    This class computes the linear tendency of the model.
    """
    def __init__(self):
        super().__init__(name="Linear Tendency")

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the linear tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Linear tendency of the state.
        """
        u = mz.z.u; v = mz.z.v; h = mz.z.h
        f_cor = self.grid.f_array
        dx1 = self.grid.dx1; dy1 = self.grid.dy1
        csqr = self.mset.csqr

        # Padding for averaging
        up = u.pad_raw(((1,1), (1,1)))
        vp = v.pad_raw(((1,1), (1,1)))
        hp = h.pad_raw(((1,1), (1,1)))

        if not self.mset.periodic_bounds[0]:
            up[0,:] = 0; up[-2:,:] = 0
            vp[0,:] = 0; vp[-1,:] = 0
        if not self.mset.periodic_bounds[1]:
            vp[:,0] = 0; vp[:,-2:] = 0
            up[:,0] = 0; up[:,-1] = 0

        # Slices
        f = slice(2,None); b = slice(None,-2); c = slice(1,-1)
        q = self.grid.quarter  # 0.25

        # Coriolis tendency
        dz.u[:] = (vp[c,c] + vp[f,c] + vp[c,b] + vp[f,b]) * q * f_cor
        dz.v[:] = (up[c,c] + up[b,c] + up[c,f] + up[b,f]) * q * (-f_cor)

        # Pressure gradient tendency
        dz.u[:] -= (hp[f,c] - hp[c,c]) * dx1
        dz.v[:] -= (hp[c,f] - hp[c,c]) * dy1

        # Horizontal divergence tendency
        dz.h[:] = -((up[c,c] - up[b,c])*dx1 + (vp[c,c] - vp[c,b])*dy1)*csqr

        # apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            dz.u[-1,:] = 0
        if not self.mset.periodic_bounds[1]:
            dz.v[:,-1] = 0
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    discretization: Finite Difference\n"
        return res

# remove symbols from the namespace
del State, Module, update_module, ModelState