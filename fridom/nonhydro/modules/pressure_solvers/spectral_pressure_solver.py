from fridom.nonhydro.state import State
from fridom.nonhydro.model_state import ModelState
from fridom.framework.modules.module import Module, update_module

class SpectralPressureSolver(Module):
    """
    This class solves the pressure field with a spectral solver.
    """

    def __init__(self):
        """
        # Spectral pressure solver
        ## Arguments:
        - None
        """
        super().__init__(name="Pressure Solver")

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Solve for the pressure field.
        """
        ps = mz.div.fft() / (-self.grid.k2_hat)
        ps[self.grid.k2_hat_zero] = 0
        mz.p[:] = ps.fft()
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    solver = Spectral\n"
        return res


# remove symbols from namespace
del State, ModelState, Module, update_module