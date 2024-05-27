from fridom.shallowwater.state import State
from fridom.framework.model_state import ModelState
from fridom.framework.modules.module import Module, update_module


class LinearTendencySpectral(Module):
    """
    This class computes the linear tendency of the model using spectral methods.
    """
    def __init__(self):
        # TODO: Add option to use discrete wave numbers
        super().__init__(name="Linear Tendency")

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the linear tendency of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Linear tendency of the state.
        """
        Kx, Ky = self.grid.K

        # TODO: Implement possible fourier transforms here

        # Coriolis tendency
        dz.u[:] = self.mset.f0 * mz.z.v
        dz.v[:] = -self.mset.f0 * mz.z.u

        # Pressure gradient tendency
        dz.u[:] -= 1j * Kx * mz.z.h
        dz.v[:] -= 1j * Ky * mz.z.h

        # Horizontal divergence tendency
        dz.h[:] = -1j * (Kx * mz.z.u + Ky * mz.z.v) * self.mset.csqr
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    discretization: Spectral\n"
        return res

# remove symbols from the namespace
del State, Module, update_module, ModelState