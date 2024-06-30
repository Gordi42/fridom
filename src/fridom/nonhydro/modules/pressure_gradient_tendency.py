# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.framework.modules.module import Module, update_module, start_module
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_state import ModelState
    from fridom.nonhydro.state import State


class PressureGradientTendency(Module):
    """
    This class computes the pressure gradient tendency of the model.
    """
    def __init__(self):
        super().__init__(name="Pressure Gradient")
        self.required_halo = 1

    @start_module
    def start(self):
        # compute the grid spacing
        dx, dy, dz = self.mset.grid.dx
        self.dx1 = config.dtype_real(1.0) / dx
        self.dy1 = config.dtype_real(1.0) / dy
        self.dz1 = config.dtype_real(1.0) / dz
        return

    @update_module
    def update(self, mz: 'ModelState', dz: 'State') -> None:
        p = mz.z_diag.p

        # Slices
        c = slice(1,-1); f = slice(2,None); b = slice(None,-2)

        # remove pressure gradient
        dz.u[c,:,:] -= (p[f,:,:] - p[b,:,:]) * self.dx1 
        dz.v[:,c,:] -= (p[:,f,:] - p[:,b,:]) * self.dy1 
        dz.w[:,:,c] -= (p[:,:,f] - p[:,:,b]) * self.dz1 / self.mset.dsqr
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += f"    discretization: Finite difference\n"
        return res