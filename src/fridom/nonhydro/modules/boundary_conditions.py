# Import external modules
from typing import TYPE_CHECKING
from functools import partial
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.modules.module import Module
from fridom.framework.grid.cartesian import Grid
# Import type information
if TYPE_CHECKING:
    import numpy as np
    from fridom.nonhydro.state import State
    from fridom.framework.model_state import ModelState

@partial(utils.jaxjit, static_argnames=("grid",))
def apply_cartesian_bc(z: 'dict[str, np.ndarray]', 
                       grid: 'Grid') -> 'dict[str, np.ndarray]':
    subdomain = grid.get_subdomain(spectral=False)
    # slices
    left  = slice(0, subdomain.halo)
    right = slice(-subdomain.halo, None)
    righti = slice(-subdomain.halo - 1, None)  # include the right edge
    # apply boundary conditions
    if not grid.periodic_bounds[0]:
        if subdomain.is_left_edge[0]:
            z["u"] = utils.modify_array(z["u"], left, 0)
            z["v"] = utils.modify_array(z["v"], left, 0)
            z["w"] = utils.modify_array(z["w"], left, 0)
            z["b"] = utils.modify_array(z["b"], left, 0)
        if subdomain.is_right_edge[0]:
            z["u"] = utils.modify_array(z["u"], righti, 0)
            z["v"] = utils.modify_array(z["v"], right, 0)
            z["w"] = utils.modify_array(z["w"], right, 0)
            z["b"] = utils.modify_array(z["b"], right, 0)
    if not grid.periodic_bounds[1]:
        if subdomain.is_left_edge[1]:
            z["u"] = utils.modify_array(z["u"], left, 0)
            z["v"] = utils.modify_array(z["v"], left, 0)
            z["w"] = utils.modify_array(z["w"], left, 0)
            z["b"] = utils.modify_array(z["b"], left, 0)
        if subdomain.is_right_edge[1]:
            z["u"] = utils.modify_array(z["u"], right, 0)
            z["v"] = utils.modify_array(z["v"], righti, 0)
            z["w"] = utils.modify_array(z["w"], right, 0)
            z["b"] = utils.modify_array(z["b"], right, 0)
    if not grid.periodic_bounds[2]:
        if subdomain.is_left_edge[2]:
            z["u"] = utils.modify_array(z["u"], left, 0)
            z["v"] = utils.modify_array(z["v"], left, 0)
            z["w"] = utils.modify_array(z["w"], left, 0)
            z["b"] = utils.modify_array(z["b"], left, 0)
        if subdomain.is_right_edge[2]:
            z["u"] = utils.modify_array(z["u"], right, 0)
            z["v"] = utils.modify_array(z["v"], right, 0)
            z["w"] = utils.modify_array(z["w"], righti, 0)
            z["b"] = utils.modify_array(z["b"], right, 0)
    return z
    

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

        if isinstance(self.grid, Grid):
            z.arr_dict = apply_cartesian_bc(z.arr_dict, self.grid)
        else:
            raise NotImplementedError("Only Cartesian grid is supported.")
        