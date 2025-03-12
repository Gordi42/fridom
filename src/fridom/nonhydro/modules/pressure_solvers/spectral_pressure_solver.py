"""Solver for the pressure field using a spectral method."""
from __future__ import annotations

from functools import partial

import numpy as np

import fridom.framework as fr
import fridom.nonhydro as nh


@partial(fr.utils.jaxify, dynamic=("k_squared_inv",))
class SpectralPressureSolver(fr.modules.Module):

    """Solve for the pressure field with a spectral solver."""

    name = "Spectral Pressure Solver"

    def _on_setup(self) -> None:
        match type(self.mset.grid):
            case nh.grid.cartesian.Grid:
                fft_required = True
                dso = fr.grid.cartesian.discrete_spectral_operators
                k2 = [dso.k_hat_squared(kx, dx, use_discrete=True)
                      for (kx,dx) in zip(self.grid.K, self.grid.dx)]
            case nh.grid.spectral.Grid:
                fft_required = False
                k2 = [kx**2 for kx in self.grid.K]
            case _:
                msg = "The spectral solver does not support this grid type."
                raise ValueError(msg)

        # scaled discretized wave number squared
        k_squared = k2[0] + k2[1] + k2[2] / self.mset.dsqr

        # Compute the inverse of the wave number squared
        with np.errstate(divide="ignore", invalid="ignore"):
            k_squared_inv = 1 / k_squared

        # Set k2_hat_inv to zero where k2_hat is zero
        self.k_squared_inv = fr.config.ncp.where(k_squared == 0, 0, k_squared_inv)
        self.fft_required = fft_required

    @fr.utils.jaxjit
    def _solve_for_pressure(self, div: fr.ScalarField) -> fr.ScalarField:
        if self.fft_required:
            return ( - div.fft() * self.k_squared_inv).ifft()
        return - div * self.k_squared_inv


    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.z_diag.p.arr = self._solve_for_pressure(mz.z_diag.div).arr
        return mz

    @property
    def info(self) -> dict:  # noqa: D102
        res = super().info
        res["Solver"] = "Spectral"
        return res
