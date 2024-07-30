# Import external modules
from typing import TYPE_CHECKING
import numpy as np
from numpy import ndarray
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.grid.cartesian import Grid as CartesianGridBase
# Import type information
if TYPE_CHECKING:
    from fridom.nonhydro.model_settings import ModelSettings

class Grid(CartesianGridBase):
    """
    Cartesian grid for the 3D non-hydrostatic model.
    
    Parameters
    ----------
    `N` : `list[int]`
        Number of grid points in each direction.
    `L` : `list[int]`
        Domain size in each direction (in meters).
    `periodic_bounds` : `list[bool]`
        Whether the domain is periodic in each direction.
    `decomposition` : `str`
        The decomposition of the domain ('slab' or 'pencil').
    
    Attributes
    ----------
    `omega_analytical` : `np.ndarray`
        Analytical dispersion relation (omega(kx, ky, kz)).
    `omega_space_discrete` : `np.ndarray`
        Dispersion relation with space-discretization effects
    `omega_time_discrete` : `np.ndarray`
        Dispersion relation with space-time-discretization effects
    """
    # update the list of dynamic attributes
    _dynamic_attributes = CartesianGridBase._dynamic_attributes + [
        'k2_hat', 'k2_hat_zero',
        '_omega_analytical', '_omega_space_discrete', '_omega_time_discrete']
    
    def __init__(self, N: list[int], L: list[int],
                 periodic_bounds: list[bool] = [True, True, True],
                 decomposition: str = 'slab'):
        if decomposition == 'slab':
            shared_axes = [0, 1]
        elif decomposition == 'pencil':
            shared_axes = [0]
        else:
            raise ValueError(f"Unknown decomposition {decomposition}")
        super().__init__(N, L, periodic_bounds, shared_axes)

    def setup(self, mset: 'ModelSettings'):
        super().setup(mset)

        ncp = config.ncp

        # discretized wave number squared
        k_dis = [2 * (1 - ncp.cos(kx * dx)) / dx**2 
                  for (kx,dx) in zip(self.K, self.dx)]

        # scaled discretized wave number squared
        k2_hat = k_dis[0] + k_dis[1] + k_dis[2] / mset.dsqr

        k2_hat_zero = config.ncp.where(k2_hat == 0)
        utils.modify_array(k2_hat, k2_hat_zero, 1)

        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self.k2_hat = k2_hat
        self.k2_hat_zero = k2_hat_zero
        return

    def omega(self, 
              k: tuple[float] | tuple[ndarray],
              use_discrete: bool = False
              ) -> ndarray:
        # shorthand notation
        ncp = config.ncp
        dsqr = self.mset.dsqr
        f2 = self.mset.f0**2
        N2 = self.mset.N2
        kx, ky, kz = k
        dx, dy, dz = self.dx

        if not ncp.allclose(f2, self.mset.f_coriolis.arr**2):
            config.logger.warning(
                "Dispersion relation may be wrong when f is varying.")
        if not ncp.allclose(N2, self.mset.N2):
            config.logger.warning(
                "Dispersion relation may be wrong when N is varying.")

        if use_discrete:
            # averaging operator (one hat squared)
            ohpm = lambda kx, dx: (1 + ncp.cos(kx*dx)) / 2
            # difference operator (k hat squared)
            khpm = lambda kx, dx: 2 * (1 - ncp.cos(kx*dx)) / dx**2
            # horizontal wave number squared
            kh2_hat = khpm(kx, dx) + khpm(ky, dy)

            with np.errstate(divide='ignore'):
                coriolis_part = ohpm(kx,dx) * ohpm(ky,dy) * f2 * khpm(kz,dz)
                buoyancy_part = ohpm(kz,dz) * N2 * kh2_hat
                denominator = dsqr * kh2_hat + khpm(kz,dz)
        else:
            with np.errstate(divide='ignore'):
                coriolis_part = f2 * kz**2
                buoyancy_part = N2 * (kx**2 + ky**2)
                denominator = dsqr * (kx**2 + ky**2) + kz**2

        om = ncp.sqrt((coriolis_part + buoyancy_part) / denominator)

        # set the result to zero where the denominator is zero
        s = (kx**2 + ky**2 + kz**2 > 0)
        om = ncp.where(s, om, 0)
        return om

utils.jaxify_class(Grid)