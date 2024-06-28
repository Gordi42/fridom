# Import external modules
from typing import TYPE_CHECKING
import numpy as np
# Import internal modules
from fridom.framework import config
from fridom.nonhydro.state import State
# Import type information
if TYPE_CHECKING:
    from fridom.nonhydro.model_settings import ModelSettings

class SingleWave(State):
    """
    An initial condition that consist of a single wave with a
    given wavenumber and a given mode.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `kx` : `float`
        The wavenumber in the x-direction.
    `ky` : `float`
        The wavenumber in the y-direction.
    `kz` : `float`
        The wavenumber in the z-direction.
    `s` : `int`
        The mode (0, 1, -1)
        0 => geostrophic mode
        1 => positive inertia-gravity mode
        -1 => negative inertia-gravity mode
    `phase` : `complex`
        The phase of the wave. (default: 0)
    `use_discrete` : `bool` (default: True)
        Whether to use the discrete eigenvectors or the analytical ones.
    
    Attributes
    ----------
    `omega` : `complex`
        The frequency of the wave (includes effects of time discretization)
        (only for inertia-gravity modes).
    `period` : `float`
        The period of the wave (includes effects of time discretization)
        (only for inertia-gravity modes).
    """
    def __init__(self, mset: 'ModelSettings', 
                 kx=6, ky=0, kz=4, s=1, phase=0, use_discrete=True) -> None:
        super().__init__(mset)

        # Shortcuts
        ncp = config.ncp
        grid = mset.grid

        # Find index of the wavenumber in the grid (nearest neighbor)
        ki = ncp.argmin(ncp.abs(grid.k_global[0] - kx))
        kj = ncp.argmin(ncp.abs(grid.k_global[1] - ky))
        kk = ncp.argmin(ncp.abs(grid.k_global[2] - kz))

        # save the index of the wave number and the wave number itself
        self.kx = grid.k_global[0][ki]
        self.ky = grid.k_global[1][kj]
        self.kz = grid.k_global[2][kk]
        self.k_loc = (ki, kj, kk)

        # Construct the spectral field of the corresponding mode
        # all zeros except for the mode
        from fridom.framework.field_variable import FieldVariable
        g = FieldVariable(grid, is_spectral=True)
        g[self.k_loc] = cp.exp(1j*phase)

        # Construct the eigenvector of the corresponding mode
        if use_discrete:
            from fridom.nonhydro.eigenvectors import VecQ
            q = VecQ(s, grid)
        else:
            from fridom.nonhydro.eigenvectors import VecQAnalytical
            q = VecQAnalytical(s, grid)

        # Construct the state
        z = (q * g).fft()

        # Normalize the state
        z /= z.norm_l2()
        
        # Set the state to itself
        self.u[:] = z.u; self.v[:] = z.v
        self.w[:] = z.w; self.b[:] = z.b

        # Freqency and period are calculated on demand
        self.__omega = None
        self.__period = None
        return

    @property
    def omega(self):
        """
        Calculate the complex freqency of the wave including the effects
        of the time discretization. The real part is the frequency and
        the imaginary part is the growth rate.
        """
        # Calculate the frequency only once
        if self.__omega is None:
            # Frequency including space discretization
            om = self.grid.omega_space_discrete[self.k_loc]

            # Start the calculation of the frequency 
            # including time discretization

            # Get the time stepping coefficients
            coeff = [self.mset.AB1, self.mset.AB2, self.mset.AB3, self.mset.AB4]
            coeff = np.array(coeff[self.mset.time_levels-1])

            # Construct the polynomial coefficients
            coeff = 1j * coeff * om.item() * self.mset.dt
            coeff[0] -= 1
            coeff = np.pad(coeff, (1,0), 'constant', constant_values=(1,0))
            
            # Calculate the roots of the polynomial
            roots = np.roots(coeff[::-1])[-1]

            # Calculate the complex frequency
            self.__omega = -1j * np.log(roots)/self.mset.dt
        return self.__omega

    @property
    def period(self):
        """
        Calculate the period of the wave including the effects
        of the time discretization.
        """
        # Calculate the period only once
        if self.__period is None:
            self.__period = 2*self.cp.pi/self.omega.real
        return self.__period

# remove symbols from namespace
del Grid, State