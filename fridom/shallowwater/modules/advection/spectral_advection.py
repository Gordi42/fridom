from fridom.shallowwater.state import State
from fridom.shallowwater.model_state import ModelState
from fridom.framework.modules.module import Module, update_module

class SpectralAdvection(Module):
    """
    This class computes the advection of the model using spectral methods.
    """
    def __init__(self):
        super().__init__(name="Spectral Advection")

    @update_module
    def update(self, mz: ModelState, dz: State) -> None:
        """
        Compute the advection of the model.

        Args:
            mz (ModelState) : Model state.
            dz (State)      : Advection of the state.
        """
        Kx, Ky = self.grid.K

        u_hat = mz.z.u * self.grid.dealias_mask
        v_hat = mz.z.v * self.grid.dealias_mask
        h_hat = mz.z.h * self.grid.dealias_mask

        u = u_hat.fft()
        v = v_hat.fft()
        h = h_hat.fft()

        zeta = (1j * Kx * v_hat - 1j * Ky * u_hat).fft()
        ekin_hat = ((u**2 + v**2)*0.5).fft()

        duhdx = 1j * Kx * (u * h).fft()
        dvhdy = 1j * Ky * (v * h).fft()

        dz.h[:] -= (duhdx + dvhdy) * self.mset.Ro
        dz.u[:] += ((zeta * v).fft() - 1j * Kx * ekin_hat ) * self.mset.Ro
        dz.v[:] -= ((zeta * u).fft() + 1j * Ky * ekin_hat ) * self.mset.Ro
        return

    def __repr__(self) -> str:
        res = super().__repr__()
        res += "    discretization: spectral\n"
        return res
    
# remove symbols from the namespace
del State, Module, update_module, ModelState