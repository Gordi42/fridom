# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.modules.module import Module, setup_module, module_method
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_state import ModelState


@utils.jaxjit
def linear_tendency(mset, z, dz):
    """
    Compute the linear tendency of the model.
    """
    # compute the linear tendency
    u = z["u"]; v = z["v"]; w = z["w"]; bu = z["b"]
    dsqr = mset.dsqr
    f_cor = mset.f_coriolis
    N2 = mset.N2

    # Slices
    c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
    q = 0.25

    # calculate u-tendency
    du = (v[c,c] + v[f,c] + v[c,b] + v[f,b]) * q * f_cor
    dz["u"] = utils.modify_array(dz["u"], (c, c), du)

    # calculate v-tendency
    dv = (u[c,c] + u[b,c] + u[c,f] + u[b,f]) * q * (-f_cor)
    dz["v"] = utils.modify_array(dz["v"], (c, c), dv)

    # calculate w-tendency
    dw = (bu[:,:,c] + bu[:,:,f]) * 0.5 / dsqr
    dz["w"] = utils.modify_array(dz["w"], (slice(None), slice(None), c), dw)

    # calculate b-tendency
    db = - (w[:,:,c] + w[:,:,b]) * 0.5 * N2
    dz["b"] = utils.modify_array(dz["b"], (slice(None), slice(None), c), db)

    return dz

class LinearTendency(Module):
    """
    This class computes the linear tendency of the model.
    """
    def __init__(self):
        super().__init__(name="Linear Tendency")
        self.required_halo = 1

    @setup_module
    def setup(self):
        self.quarter = config.dtype_real(0.25)
        self.half = config.dtype_real(0.5)
        return

    @module_method
    def update(self, mz: 'ModelState') -> None:
        mz.dz.arr_dict = linear_tendency(
            self.mset, mz.z.arr_dict, mz.dz.arr_dict)
        return

    @property
    def info(self) -> dict:
        res = super().info
        res["Discretization"] = "Finite Difference"
        return res