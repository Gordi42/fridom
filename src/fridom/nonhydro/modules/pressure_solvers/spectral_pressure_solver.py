# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.modules.module import Module, module_method
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_state import ModelState

class SpectralPressureSolver(Module):
    """
    This class solves the pressure field with a spectral solver.
    """
    def __init__(self):
        super().__init__(name="Pressure Solver")
        self.required_halo = 0

    @module_method
    def update(self, mz: 'ModelState') -> None:
        ps = mz.z_diag.div.fft() / (-self.grid.k2_hat)
        ps.arr = set_constant_pressure(ps.arr, self.grid.k2_hat_zero)
        mz.z_diag.p = ps.fft()
        return

    @property
    def info(self) -> dict:
        res = super().info
        res["Solver"] = "Spectral"
        return res

@utils.jaxjit
def set_constant_pressure(pressure, where):
    return utils.modify_array(pressure, where, 0)