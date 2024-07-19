# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.framework.modules.module import Module, setup_module, module_method
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

    @setup_module
    def setup(self):
        # compute the grid spacing
        dx, dy, dz = self.mset.grid.dx
        self.dx1 = config.dtype_real(1.0) / dx
        self.dy1 = config.dtype_real(1.0) / dy
        self.dz1 = config.dtype_real(1.0) / dz
        self.bc = self.mset.bc
        self.bc.setup(mset=self.mset)
        return

    @module_method
    def update(self, mz: 'ModelState') -> None:
        p = mz.z_diag.p
        dz = mz.dz

        # Slices
        c = slice(1,-1); f = slice(2,None)

        # remove pressure gradient
        p_grad = self.grid.grad(p)
        dz.u -= p_grad[0]
        dz.v -= p_grad[1]
        dz.w -= p_grad[2] * self.mset.dsqr




        # dz.u[c,:,:] -= (p[f,:,:] - p[c,:,:]) * self.dx1 
        # dz.v[:,c,:] -= (p[:,f,:] - p[:,c,:]) * self.dy1 
        # dz.w[:,:,c] -= (p[:,:,f] - p[:,:,c]) * self.dz1 / self.mset.dsqr
        self.bc.apply_boundary_conditions(dz)
        return

    @property
    def info(self) -> dict:
        res = super().info
        res["Discretization"] = "Finite Difference"
        return res