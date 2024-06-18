from fridom.nonhydro.state import State
from fridom.framework.model_state import ModelState
from fridom.framework.modules.module import Module, update_module, start_module


class TendencyDivergence(Module):
    """
    This class computes the divergence of the tendency of the model.
    """

    def __init__(self):
        """
        # Tendency Divergence
        This class computes the divergence of the tendency of the model.
        The divergence of the tendency is required for the pressure solver.
        ## Arguments:
        - None
        """
        super().__init__(name="Tendency Divergence")

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
        Compute the divergence of the tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Tendency of the state.
        """
        cp = self.grid.cp

        u_pad = cp.pad(dz.u, ((1,0), (0,0), (0,0)), 'wrap')
        v_pad = cp.pad(dz.v, ((0,0), (1,0), (0,0)), 'wrap')
        w_pad = cp.pad(dz.w, ((0,0), (0,0), (1,0)), 'wrap')

        # boundary conditions
        if not self.mset.periodic_bounds[0]:
            u_pad[0,:,:] = 0
        if not self.mset.periodic_bounds[1]:
            v_pad[:,0,:] = 0
        if not self.mset.periodic_bounds[2]:
            w_pad[:,:,0] = 0

        # Slices
        c = slice(1,None); b = slice(None,-1); n = slice(None)
        xb = (b,n,n); xc = (c,n,n)
        yb = (n,b,n); yc = (n,c,n)
        zb = (n,n,b); zc = (n,n,c)

        # calculate divergence
        mz.z_diag.div[:] = (u_pad[xc] - u_pad[xb])*self.dx1 + \
                           (v_pad[yc] - v_pad[yb])*self.dy1 + \
                           (w_pad[zc] - w_pad[zb])*self.dz1
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    discretization = Finite Difference\n"
        return res

# remove symbols from the namespace
del State, ModelState, Module, update_module, start_module