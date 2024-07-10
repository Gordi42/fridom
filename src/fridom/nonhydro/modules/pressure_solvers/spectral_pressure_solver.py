# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.framework.modules.module import Module, update_module
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

    @update_module
    def update(self, mz: 'ModelState') -> None:
        ps = mz.z_diag.div.fft() / (-self.grid.k2_hat)
        ps[self.grid.k2_hat_zero] = 0
        mz.z_diag.p[:] = ps.fft()
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    solver = Spectral\n"
        return res