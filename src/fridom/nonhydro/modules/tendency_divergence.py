# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.framework.modules.module import Module, setup_module, module_method
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_state import ModelState


class TendencyDivergence(Module):
    """
    This class computes the divergence of the tendency of the model.
    The divergence of the tendency is required for the pressure solver.
    """

    def __init__(self):
        super().__init__(name="Tendency Divergence")
        self.required_halo = 1

    @setup_module
    def setup(self):
        # compute the grid spacing
        self.bc = self.mset.bc
        self.bc.setup(mset=self.mset)
        return

    @module_method
    def update(self, mz: 'ModelState') -> None:
        dz = mz.dz
        dz.sync()
        # apply boundary conditions
        self.bc.apply_boundary_conditions(dz)

        mz.z_diag.div.arr = self.grid.div((dz.u, dz.v, dz.w))
        return

    @property
    def info(self) -> dict:
        res = super().info
        res["Discretization"] = "Finite Difference"
        return res