import numpy as np

from fridom.shallowwater.grid import Grid
from fridom.shallowwater.state import State


class SingleWave(State):
    """
    An initial condition that consist of a single wave with a 
    given wavenumber and a given mode.

    Attributes:
        kx (float)     : The wavenumber in the x-direction.
        ky (float)     : The wavenumber in the y-direction.
        mode (int)     : The mode (0, 1, -1)
        omega (complex): The frequency of the wave 
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
        period (float) : The period of the wave.
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
    """
    def __init__(self, grid:Grid, 
                 kx=6, ky=4, s=1, phase=0, use_discrete=True) -> None:
        """
        Constructor of the SingleWave initial condition.

        Arguments:
            grid (Grid)           : The grid.
            kx (float)            : The wavenumber in the x-direction.
            ky (float)            : The wavenumber in the y-direction.
            s (int)               : The mode (0, 1, -1)
                                    0 => geostrophic mode
                                    1 => positive inertia-gravity mode
                                   -1 => negative inertia-gravity mode
            phase (complex)       : The phase of the wave. (Default: 0)
            use_discrete (bool)   : Whether to use the discrete eigenvectors
                                    or the analytical ones. Default: True.
        """
        super().__init__(grid)

        # Shortcuts
        cp = self.cp

        # Find index of the wavenumber in the grid (nearest neighbor)
        ki = cp.argmin(cp.abs(grid.k[0] - kx))
        kj = cp.argmin(cp.abs(grid.k[1] - ky))

        # save the index of the wave number and the wave number itself
        self.k_loc = (ki, kj)
        self.kx = grid.k[0][ki]; self.ky = grid.k[1][kj]

        # Construct the spectral field of the corresponding mode
        # all zeros except for the mode
        from fridom.framework.field_variable import FieldVariable
        g = FieldVariable(grid, is_spectral=True)
        g[self.k_loc] = cp.exp(1j*phase)

        # Construct the eigenvector of the corresponding mode
        if use_discrete:
            from fridom.shallowwater.eigenvectors import VecQ
            q = VecQ(s, grid)
        else:
            from fridom.shallowwater.eigenvectors import VecQAnalytical
            q = VecQAnalytical(s, grid)

        # Construct the state
        z = (q * g).fft()

        # Normalize the state
        z /= cp.sqrt(z.norm_l2())
        
        # Set the state to itself
        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h

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
