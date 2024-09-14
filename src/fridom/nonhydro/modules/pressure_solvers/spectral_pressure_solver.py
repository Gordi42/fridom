import fridom.framework as fr
import fridom.nonhydro as nh
import numpy as np


@fr.utils.jaxify
class SpectralPressureSolver(fr.modules.Module):
    """
    This class solves the pressure field with a spectral solver.
    """
    name = "Spectral Pressure Solver"

    @fr.modules.module_method
    def setup(self, mset: 'nh.ModelSettings') -> None:
        super().setup(mset)
        match type(mset.grid):
            case nh.grid.cartesian.Grid:
                fft_required = True
                from fridom.framework.grid.cartesian import discrete_spectral_operators as dso
                k2 = [dso.k_hat_squared(kx, dx, use_discrete=True)
                      for (kx,dx) in zip(self.grid.K, self.grid.dx)]
            case nh.grid.spectral.Grid:
                fft_required = False
                k2 = [kx**2 for kx in self.grid.K]
            case _:
                raise ValueError("The spectral solver does not support this grid type.")

        # scaled discretized wave number squared
        k_squared = k2[0] + k2[1] + k2[2] / mset.dsqr

        # Compute the inverse of the wave number squared
        with np.errstate(divide='ignore', invalid='ignore'):
            k_squared_inv = 1 / k_squared

        # Set k2_hat_inv to zero where k2_hat is zero
        self.k_squared_inv = fr.config.ncp.where(k_squared == 0, 0, k_squared_inv)
        self.fft_required = fft_required
        return

    @fr.utils.jaxjit
    def solve_for_pressure(self, div: fr.FieldVariable) -> fr.FieldVariable:
        if self.fft_required:
            return ( - div.fft() * self.k_squared_inv).fft()
        else:
            return - div * self.k_squared_inv


    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.z_diag.p.arr = self.solve_for_pressure(mz.z_diag.div).arr
        return mz

    @property
    def info(self) -> dict:
        res = super().info
        res["Solver"] = "Spectral"
        return res
