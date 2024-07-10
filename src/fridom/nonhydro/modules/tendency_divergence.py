# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.framework.modules.module import Module, update_module, start_module
# Import type information
if TYPE_CHECKING:
    from fridom.nonhydro.state import State
    from fridom.framework.model_state import ModelState


class TendencyDivergence(Module):
    """
    This class computes the divergence of the tendency of the model.
    The divergence of the tendency is required for the pressure solver.
    """

    def __init__(self):
        super().__init__(name="Tendency Divergence")
        self.required_halo = 1

    @start_module
    def start(self):
        # compute the grid spacing
        dx, dy, dz = self.mset.grid.dx
        self.dx1 = config.dtype_real(1.0) / dx
        self.dy1 = config.dtype_real(1.0) / dy
        self.dz1 = config.dtype_real(1.0) / dz
        self.bc = self.mset.bc
        self.bc.start(mset=self.mset)
        return

    @update_module
    def update(self, mz: 'ModelState') -> None:
        dz = mz.dz
        dz.sync()
        u = dz.u; v = dz.v; w = dz.w
        # Slices
        c = slice(1,-1); b = slice(None,-2)
        # apply boundary conditions
        self.bc.apply_boundary_conditions(dz)

        mz.z_diag.div[c,c,c] = (u[c,c,c] - u[b,c,c])*self.dx1 + \
                               (v[c,c,c] - v[c,b,c])*self.dy1 + \
                               (w[c,c,c] - w[c,c,b])*self.dz1
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    discretization = Finite Difference\n"
        return res

    def apply_boundary_conditions(self, z: 'State') -> None:
        subdomain = self.grid.get_subdomain(spectral=False)
        # slices
        left  = slice(0, subdomain.halo)
        right = slice(-subdomain.halo - 1, None)
        # apply boundary conditions
        if not self.grid.periodic_bounds[0]:
            if subdomain.is_left_edge[0]:
                z.u[left,:,:] = 0
            if subdomain.is_right_edge[0]:
                z.u[right,:,:] = 0
        if not self.grid.periodic_bounds[1]:
            if subdomain.is_left_edge[1]:
                z.v[:,left,:] = 0
            if subdomain.is_right_edge[1]:
                z.v[:,right,:] = 0
        if not self.grid.periodic_bounds[2]:
            if subdomain.is_left_edge[2]:
                z.w[:,:,left] = 0
            if subdomain.is_right_edge[2]:
                z.w[:,:,right] = 0