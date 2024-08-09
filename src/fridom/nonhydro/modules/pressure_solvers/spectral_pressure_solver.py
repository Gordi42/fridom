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
        # check if the grid is cartesian
        if not isinstance(mset.grid, fr.grid.cartesian.Grid):
            raise ValueError("Spectral solver only works with Cartesian grids.")
        
        # compute the discretized wave number squared
        from fridom.framework.grid.cartesian import discrete_spectral_operators as dso
        k_dis = [dso.k_hat_squared(kx, dx, use_discrete=True)
                 for (kx,dx) in zip(self.grid.K, self.grid.dx)]

        # scaled discretized wave number squared
        k2_hat = k_dis[0] + k_dis[1] + k_dis[2] / mset.dsqr

        # Compute the inverse of the wave number squared
        with np.errstate(divide='ignore', invalid='ignore'):
            k2_hat_inv = 1 / k2_hat

        # Set k2_hat_inv to zero where k2_hat is zero
        self.k2_hat_inv = fr.config.ncp.where(k2_hat == 0, 0, k2_hat_inv)
        return

    @fr.utils.jaxjit
    def solve_for_pressure(self, div: fr.FieldVariable) -> fr.FieldVariable:
        return ( - div.fft() * self.k2_hat_inv).fft()


    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:
        mz.z_diag.p.arr = self.solve_for_pressure(mz.z_diag.div).arr
        return mz

    @property
    def info(self) -> dict:
        res = super().info
        res["Solver"] = "Spectral"
        return res
