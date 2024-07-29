# Import external modules
from typing import TYPE_CHECKING
import numpy as np
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

        # dispersion relation are computed on demand
        self._omega_analytical     = None
        self._omega_time_discrete  = None
        self._omega_space_discrete = None

        return

    @property
    def omega_analytical(self):
        """
        Analytical dispersion relation.
        """
        if self._omega_analytical is None:
            # shorthand notation
            ncp = config.ncp
            dsqr = self.mset.dsqr

            f2 = self.mset.f0**2
            N2 = self.mset.N2

            if not ncp.allclose(f2, self.mset.f_coriolis.arr**2):
                print("Warning: Dispersion relation may be wrong when f is varying.")
            if not ncp.allclose(N2, self.mset.N2):
                print("Warning: Dispersion relation may be wrong when N is varying.")

            # squared wave number (horizontal and vertical)
            kh2 = self.K[0]**2 + self.K[1]**2
            kz2 = self.K[2]**2

            # avoid division by zero
            s = (kh2 + kz2 > 0)               

            # compute dispersion relation
            om = ncp.zeros_like(kh2)
            om = utils.modify_array(om, s, 
                ncp.sqrt((f2 * kz2 + N2 * kh2)[s] / (dsqr * kh2 + kz2)[s]))

            # store the result
            self._omega_analytical = om
        
        return self._omega_analytical

    @property
    def omega_space_discrete(self):
        """
        Dispersion relation with space-discretization effects.
        """
        if self._omega_space_discrete is None:
            # shorthand notation
            ncp = config.ncp
            dsqr = self.mset.dsqr
            kx, ky, kz = self.K
            dx, dy, dz = self.dx

            f2 = self.mset.f0**2
            N2 = self.mset.N2

            if not ncp.allclose(f2, self.mset.f_coriolis.arr**2):
                config.logger.warning(
                    "Dispersion relation may be wrong when f is varying.")
            if not ncp.allclose(N2, self.mset.N2):
                config.logger.warning(
                    "Dispersion relation may be wrong when N is varying.")

            # averaging operator (one hat squared)
            ohpm = lambda kx, dx: (1 + ncp.cos(kx*dx)) / 2
            # difference operator (k hat squared)
            khpm = lambda kx, dx: 2 * (1 - ncp.cos(kx*dx)) / dx**2

            # avoid division by zero
            s = (kx**2 + ky**2 + kz**2 > 0)

            # horizontal wave number squared
            kh2_hat = khpm(kx, dx) + khpm(ky, dy)

            # compute dispersion relation
            om_hat = ncp.zeros_like(kh2_hat)
            om_hat = utils.modify_array(om_hat, s,
                ncp.sqrt((ohpm(kx,dx) * ohpm(ky,dy) * f2 * khpm(kz,dz) + 
                          ohpm(kz,dz) * N2 * kh2_hat)[s] /
                         (dsqr * kh2_hat + khpm(kz,dz))[s]))

            # store the result
            self._omega_space_discrete = om_hat
        
        return self._omega_space_discrete
    
    @property
    def omega_time_discrete(self):
        """
        Dispersion relation with space-time-discretization effects.
        Warning: The computation is very slow.
        """
        if self._omega_time_discrete is None:
            from fridom.framework.time_steppers import AdamBashforth
            if not isinstance(self.mset.time_stepper, AdamBashforth):
                raise ValueError("Only AdamBashforth time stepper is supported.")
            
            # shorthand notation
            ncp = config.ncp
            Nx, Ny, Nz = self.N

            # get adam-bashforth coefficients
            ab = self.mset.time_stepper
            ab_coefficients = [ab.AB1, ab.AB2, ab.AB3, ab.AB4]
        
            # get the coefficients for the current time level
            coeff = ncp.array(ab_coefficients[ab.order-1])

            # construct polynomial coefficients for each grid point
            # tile the array such that coeff and omega_discrete have the same
            coeff = ncp.tile(coeff, (Nx, Ny, Nz, 1))
            omi   = self.omega_space_discrete[..., ncp.newaxis]

            # calculate the polynomial coefficients
            coeff = ncp.multiply(omi, coeff) * 1j * ab._dt_float

            # subtract 1 from the last coefficient
            last_col = (..., 0)
            coeff = utils.modify_array(coeff, last_col, coeff[last_col] - 1)

            # leading coefficient is 1
            coeff = ncp.pad(
                coeff, ((0,0), (0,0), (0,0), (1,0)), 
                'constant', constant_values=(1,0))

            # reverse the order of the coefficients
            coeff = coeff[..., ::-1]

            def find_roots(c):
                """
                Find the last root of the polynomial.

                Parameters:
                    c (1D array): Polynomial coefficients.

                Returns:
                    root (complex): Last root of the polynomial.
                """
                return np.roots(c)[-1]

            # find the roots of the polynomial
            from fridom.framework.to_numpy import to_numpy
            # root finding only works on the CPU
            coeff = to_numpy(coeff)
            roots = ncp.array(np.apply_along_axis(find_roots, -1, coeff))
        
            # calculate the dispersion relation with time discretization
            res = -1j * ncp.log(roots) / ab._dt_float

            # store the result
            self._omega_time_discrete = res
        return self._omega_time_discrete

utils.jaxify_class(Grid)