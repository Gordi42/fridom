from fridom.shallowwater.state import State
from fridom.shallowwater.model_state import ModelState
from fridom.framework.modules.module import Module, update_module

class SadournyAdvection(Module):
    """
    # Advection scheme by Sadourny (1975).
    This advection scheme conserves the total energy of the system.
    """
    def __init__(self):
        super().__init__(name="Advection")

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the linear tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Linear tendency of the state.
        """
        # shorthand notation
        dx1 = self.grid.dx1; dy1 = self.grid.dy1
        Ro  = self.mset.Ro
        half = self.grid.half; quar = self.grid.quarter

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
        xf = (f,c); xb = (b,c); xc = (c,c)
        yf = (c,f); yb = (c,b); yc = (c,c)

        # Padding
        up = mz.z.u.pad_raw(((2,2), (2,2)))
        vp = mz.z.v.pad_raw(((2,2), (2,2)))
        hp = mz.z.h.pad_raw(((2,2), (2,2)))

        # Apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            up[:2,:] = 0; up[-3:,:] = 0
            vp[:2,:] = 0; vp[-2:,:] = 0
            hp[:2,:] = 0; hp[-2:,:] = 0
        if not self.mset.periodic_bounds[1]:
            up[:,:2] = 0; up[:,-2:] = 0
            vp[:,:2] = 0; vp[:,-3:] = 0
            hp[:,:2] = 0; hp[:,-2:] = 0

        # advection of layer thickness with flux divergence
        fe = up[xc] * (hp[xf] + hp[xc]) * half
        fn = vp[yc] * (hp[yf] + hp[yc]) * half

        dz.h[:] -= (fe[xc] - fe[xb])*dx1*Ro + (fn[yc] - fn[yb])*dy1*Ro

        # advection of momentum with potential vorticity and kinetic energy
        hf = Ro * hp + self.mset.csqr
        q = (vp[xf] - vp[xc])*dx1 - (up[yf] - up[yc])*dy1
        # apply boundary conditions to potential vorticity
        if not self.mset.periodic_bounds[0]:
            q[0,:] = -(up[2,c] - up[1,c])*dx1;
            q[-1,:] = -(up[-1,c] - up[-2,c])*dx1
        if not self.mset.periodic_bounds[1]:
            q[:,0] = (vp[c,2] - vp[c,1])*dy1;
            q[:,-1] = (vp[c,-1] - vp[c,-2])*dy1
        q /= (hf[c,c] + hf[f,c] + hf[c,f] + hf[f,f])*quar
        k = (up[xc]**2 + up[xb]**2 + vp[yc]**2 + vp[yb]**2)*quar

        fe = up[xc] * (hf[xf] + hf[xc]) * half
        fn = vp[yc] * (hf[yf] + hf[yc]) * half

        dz.u[:] += (q[yc] * (fn[xc] + fn[xf]) + q[yb] * (fn[c,b] + fn[f,b]))*quar*Ro
        dz.u[:] -= (k[xf] - k[xc])*dx1 * Ro

        dz.v[:] -= (q[xc] * (fe[yc] + fe[yf]) + q[xb] * (fe[b,c] + fe[b,f]))*quar*Ro
        dz.v[:] -= (k[yf] - k[yc])*dy1 * Ro

        # apply boundary conditions
        if not self.mset.periodic_bounds[0]:
            dz.u[-1,:] = 0
        if not self.mset.periodic_bounds[1]:
            dz.v[:,-1] = 0
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    discretization: based on Sadourny (1975)\n"
        return res

# remove symbols from the namespace
del State, Module, update_module, ModelState