# Import external modules
from typing import TYPE_CHECKING
from functools import partial
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.modules.module import Module, module_method
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
        lleft = (left, slice(None), slice(None))
        rright = (right, slice(None), slice(None))
        rrighti = (righti, slice(None), slice(None))
        if subdomain.is_left_edge[0]:
            z["u"] = utils.modify_array(z["u"], lleft, 0)
            z["v"] = utils.modify_array(z["v"], lleft, 0)
            z["w"] = utils.modify_array(z["w"], lleft, 0)
            z["b"] = utils.modify_array(z["b"], lleft, 0)
        if subdomain.is_right_edge[0]:
            z["u"] = utils.modify_array(z["u"], rrighti, 0)
            z["v"] = utils.modify_array(z["v"], rright, 0)
            z["w"] = utils.modify_array(z["w"], rright, 0)
            z["b"] = utils.modify_array(z["b"], rright, 0)
    if not grid.periodic_bounds[1]:
        lleft = (slice(None), left, slice(None))
        rright = (slice(None), right, slice(None))
        rrighti = (slice(None), righti, slice(None))
        if subdomain.is_left_edge[1]:
            z["u"] = utils.modify_array(z["u"], lleft, 0)
            z["v"] = utils.modify_array(z["v"], lleft, 0)
            z["w"] = utils.modify_array(z["w"], lleft, 0)
            z["b"] = utils.modify_array(z["b"], lleft, 0)
        if subdomain.is_right_edge[1]:
            z["u"] = utils.modify_array(z["u"], rright, 0)
            z["v"] = utils.modify_array(z["v"], rrighti, 0)
            z["w"] = utils.modify_array(z["w"], rright, 0)
            z["b"] = utils.modify_array(z["b"], rright, 0)
    if not grid.periodic_bounds[2]:
        lleft = (slice(None), slice(None), left)
        rright = (slice(None), slice(None), right)
        rrighti = (slice(None), slice(None), righti)
        if subdomain.is_left_edge[2]:
            z["u"] = utils.modify_array(z["u"], lleft, 0)
            z["v"] = utils.modify_array(z["v"], lleft, 0)
            z["w"] = utils.modify_array(z["w"], lleft, 0)
            z["b"] = utils.modify_array(z["b"], lleft, 0)
        if subdomain.is_right_edge[2]:
            z["u"] = utils.modify_array(z["u"], rright, 0)
            z["v"] = utils.modify_array(z["v"], rright, 0)
            z["w"] = utils.modify_array(z["w"], rrighti, 0)
            z["b"] = utils.modify_array(z["b"], rright, 0)
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

    @module_method
    def update(self, mz: 'ModelState') -> 'ModelState':
        mz.z = self.apply_boundary_conditions(mz.z)
        return mz
    
    def apply_boundary_conditions(self, z: 'State') -> 'State':
        if all(self.grid.periodic_bounds):
            return z

        if isinstance(self.grid, Grid):
            z.arr_dict = apply_cartesian_bc(z.arr_dict, self.grid)
            return z
        else:
            raise NotImplementedError("Only Cartesian grid is supported.")
        