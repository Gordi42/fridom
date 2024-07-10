# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.modules import Module, update_module
# Import type information
if TYPE_CHECKING:
    from fridom.nonhydro.state import State
    from fridom.framework.model_state import ModelState

class BoundaryConditions(Module):
    """
    Set boundary conditions to the model state.
    
    Methods
    -------
    `apply_boundary_conditions(z: 'State') -> None`
        Apply the boundary conditions to the model state.
    """
    def __init__(self, name: str = "BoundaryConditions"):
        super().__init__(name=name)

    def update_boundary_conditions(self, mz: 'ModelState') -> None:
        self.apply_boundary_conditions(mz.z)
        return
    
    def apply_boundary_conditions(self, z: 'State') -> None:
        if all(self.grid.periodic_bounds):
            return
        
        subdomain = self.grid.get_subdomain(spectral=False)
        # slices
        left  = slice(0, subdomain.halo)
        right = slice(-subdomain.halo, None)
        righti = slice(-subdomain.halo - 1, None)  # include the right edge
        # apply boundary conditions
        if not self.grid.periodic_bounds[0]:
            if subdomain.is_left_edge[0]:
                z.u[left,:,:] = 0
                z.v[left,:,:] = 0
                z.w[left,:,:] = 0
                z.b[left,:,:] = 0
            if subdomain.is_right_edge[0]:
                z.u[righti,:,:] = 0
                z.v[right,:,:] = 0
                z.w[right,:,:] = 0
                z.b[right,:,:] = 0
        if not self.grid.periodic_bounds[1]:
            if subdomain.is_left_edge[1]:
                z.u[:,left,:] = 0
                z.v[:,left,:] = 0
                z.w[:,left,:] = 0
                z.b[:,left,:] = 0
            if subdomain.is_right_edge[1]:
                z.u[:,right,:] = 0
                z.v[:,righti,:] = 0
                z.w[:,right,:] = 0
                z.b[:,right,:] = 0
        if not self.grid.periodic_bounds[2]:
            if subdomain.is_left_edge[2]:
                z.u[:,:,left] = 0
                z.v[:,:,left] = 0
                z.w[:,:,left] = 0
                z.b[:,:,left] = 0
            if subdomain.is_right_edge[2]:
                z.u[:,:,right] = 0
                z.v[:,:,right] = 0
                z.w[:,:,righti] = 0
                z.b[:,:,right] = 0