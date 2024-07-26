# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.modules.module import Module, setup_module, module_method
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_state import ModelState
    from fridom.framework.grid import InterpolationBase


class LinearTendency(Module):
    """
    This class computes the linear tendency of the model.
    """
    _dynamic_attributes = set(["mset"])
    def __init__(self, interpolation: 'InterpolationBase | None' = None):
        super().__init__(name="Linear Tendency")
        self.required_halo = 1
        self.interpolation = interpolation

    @setup_module
    def setup(self):
        if self.interpolation is None:
            self.interpolation = self.mset.grid._interp_mod
        else:
            self.interpolation.setup(mset=self.mset)
        return

    @module_method
    def update(self, mz: 'ModelState') -> 'ModelState':
        mz.dz = self.linear_tendency(mz.z, mz.dz)
        return mz

    @utils.jaxjit
    def linear_tendency(self, z, dz):
        """
        Compute the linear tendency of the model.
        """
        # compute the linear tendency
        u = z.u; v = z.v; w = z.w; bu = z.b
        dsqr = self.mset.dsqr
        f_cor = self.mset.f_coriolis
        N2 = self.mset.N2
        interp = self.interpolation.interpolate

        # calculate u-tendency
        dz.u.arr = interp(v.arr, v.position, u.position) * f_cor

        # calculate v-tendency
        dz.v.arr = interp(u.arr, u.position, v.position) * (-f_cor)

        # calculate w-tendency
        dz.w.arr = interp(bu.arr, bu.position, w.position) / dsqr

        # calculate b-tendency
        dz.b.arr = - interp(w.arr, w.position, bu.position) * N2

        return dz

    @property
    def info(self) -> dict:
        res = super().info
        res["Discretization"] = "Finite Difference"
        return res

utils.jaxify_class(LinearTendency)