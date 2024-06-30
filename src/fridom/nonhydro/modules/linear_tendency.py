# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.framework.modules.module import Module, update_module, start_module
# Import type information
if TYPE_CHECKING:
    from fridom.nonhydro.state import State
    from fridom.framework.model_state import ModelState


class LinearTendency(Module):
    """
    This class computes the linear tendency of the model.
    """
    def __init__(self):
        super().__init__(name="Linear Tendency")
        self.required_halo = 1

    @start_module
    def start(self):
        self.quarter = config.dtype_real(0.25)
        self.half = config.dtype_real(0.5)

    @update_module
    def update(self, mz: 'ModelState', dz: 'State') -> None:
        # compute the linear tendency
        u = mz.z.u; v = mz.z.v; w = mz.z.w; b = mz.z.b
        dsqr = self.mset.dsqr
        f_cor = self.mset.f_coriolis
        N2 = self.mset.N2

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)
        q = self.quarter  # 0.25
        h = self.half     # 0.5

        # calculate u-tendency
        dz.u[c,c] = (v[c,c] + v[f,c] + v[c,b] + v[f,b]) * q * f_cor

        # calculate v-tendency
        dz.v[c,c] = (u[c,c] + u[b,c] + u[c,f] + u[b,f]) * q * (-f_cor)

        # calculate w-tendency
        dz.w[:,:,c] = (b[:,:,f] + b[:,:,b]) * h / dsqr

        # calculate b-tendency
        dz.b[:,:,c] = - (w[:,:,f] + w[:,:,b]) * h * N2

        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    discretization: Finite Difference\n"
        return res