# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.framework.modules.module import Module, update_module, start_module
# Import type information
if TYPE_CHECKING:
    from fridom.nonhydro.state import State
    from fridom.framework.model_state import ModelState


class TendencyDivergence(Module):
    """
    This class computes the divergence of the tendency of the model.
    The divergence of the tendency is required for the pressure solver.
    """

    def __init__(self):
        super().__init__(name="Tendency Divergence")

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
        u = mz.z.u; v = mz.z.v; w = mz.z.w
        # Slices
        c = slice(1,-1); b = slice(None,-2)

        mz.z_diag.div[c,c,c] = (u[c,c,c] - u[b,c,c])*self.dx1 + \
                               (v[c,c,c] - v[c,b,c])*self.dy1 + \
                               (w[c,c,c] - w[c,c,b])*self.dz1
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    discretization = Finite Difference\n"
        return res