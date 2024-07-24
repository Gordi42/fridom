# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.modules.module import Module, setup_module, module_method
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_state import ModelState
    from fridom.nonhydro.state import State

@utils.jaxjit
def pressure_gradient_tendency(mset, p, dz):
    """
    Compute the pressure gradient tendency of the model.
    """
    p_grad = mset.grid.grad(p)

    dz["u"] -= p_grad[0]
    dz["v"] -= p_grad[1]
    dz["w"] -= p_grad[2] * mset.dsqr
    return dz

class PressureGradientTendency(Module):
    """
    This class computes the pressure gradient tendency of the model.
    """
    def __init__(self):
        super().__init__(name="Pressure Gradient")
        self.required_halo = 1

    @setup_module
    def setup(self):
        self.bc = self.mset.bc
        self.bc.setup(mset=self.mset)
        return

    @module_method
    def update(self, mz: 'ModelState') -> 'ModelState':
        mz.dz.arr_dict = pressure_gradient_tendency(
            self.mset, mz.z_diag.p.arr, mz.dz.arr_dict)

        self.bc.apply_boundary_conditions(mz.dz)
        return mz

    @property
    def info(self) -> dict:
        res = super().info
        res["Discretization"] = "Finite Difference"
        return res
