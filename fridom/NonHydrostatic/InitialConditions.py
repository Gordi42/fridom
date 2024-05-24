import numpy as np

from fridom.NonHydrostatic.ModelSettings import ModelSettings
from fridom.NonHydrostatic.Grid import Grid
from fridom.NonHydrostatic.State import State


class Jet(State):
    """
    Superposition of a zonal jet and a geostrophic perturbation.
    Following the setup of Chouksey et al. 2022.
    For very large jet_strengths, convective instabilities can occur.
    """
    def __init__(self, mset: ModelSettings, grid: Grid, 
                 jet_strength=1, jet_width=0.16,
                 pert_strength=0.05, pert_wavenum=5,
                 geo_proj=True):
        """
        Constructor of the Instable Jet initial condition with 2 zonal jets.

        Arguments:
            mset              : The model settings.
            grid              : The grid.
            jet_strength      : The strength of the zonal jets.
            jet_width         : The width of the zonal jets.
            pert_strength     : The strength of the perturbation.
            pert_wavenum      : The wavenumber of the perturbation.
            geo_proj          : Whether to project the initial condition
                                to the geostrophic subspace. Default: True.
        """
        super().__init__(grid)
        cp = self.cp

        X  = grid.X[0]; Y  = grid.X[1]; Z  = grid.X[2]
        Lx = mset.L[0]; Ly = mset.L[1]; Lz = mset.L[2]

        # two opposite jets
        self.u[:] = -cp.exp(-(Y-Ly/4)**2/(jet_width)**2)
        self.u[:] += cp.exp(-(Y-3*Ly/4)**2/(jet_width)**2)
        self.u[:] *= jet_strength * cp.cos(2*cp.pi*Z/Lz)

        # add a small perturbation
        z_per = SingleWave(mset, grid, kx=pert_wavenum, ky=0, kz=0, s=0)
        z_per /= cp.max(cp.sqrt(z_per.u**2 + z_per.v**2 + z_per.w**2))

        self.u[:] += pert_strength * z_per.u
        self.v[:] += pert_strength * z_per.v
        self.w[:] += pert_strength * z_per.w
        self.b[:] += pert_strength * z_per.b

        if geo_proj:
            from fridom.NonHydrostatic.Projection import GeostrophicSpectral
            proj_geo = GeostrophicSpectral(mset, grid)
            z_geo = proj_geo(self)
            self.u[:] = z_geo.u; self.v[:] = z_geo.v; 
            self.w[:] = z_geo.w; self.b[:] = z_geo.b
        return


class BarotropicJet(State):
    """
    A Barotropic instable jet setup with 2 zonal jets and a perturbation
    on top of it. The jet positions in the y-direction are 
    at (1/4, 3/4)*Ly (opposing sign).
    """
    def __init__(self, mset:ModelSettings, grid:Grid, 
                 wavenum=5, waveamp=0.1, geo_proj=True):
        """
        Constructor of the Barotropic Jet initial condition with 2 zonal jets.

        Arguments:
            mset              : The model settings.
            grid              : The grid.
            wavenum           : The wavenumber of the perturbation.
            waveamp           : The amplitude of the perturbation.
            geo_proj          : Whether to project the initial condition
                                to the geostrophic subspace. Default: True.
        """
        super().__init__(grid)
        # Shortcuts
        cp = self.cp
        PI = cp.pi
        x  = grid.X[0]; y  = grid.X[1]; z  = grid.X[2]
        Lx = mset.L[0]; Ly = mset.L[1]; Lz = mset.L[2]

        # Construct the zonal jets
        self.u[:]  = 2.5*( cp.exp(-((y - 0.75*Ly)/(0.04*PI))**2) - 
                           cp.exp(-((y - 0.25*Ly)/(0.04*PI))**2) )

        # Construct the perturbation
        kx_p = 2*PI/Lx * wavenum
        self.v[:]  = waveamp * cp.sin(kx_p*x)

        if geo_proj:
            from fridom.NonHydrostatic.Projection import GeostrophicSpectral
            proj_geo = GeostrophicSpectral(mset, grid)
            z_geo = proj_geo(self)
            self.u[:] = z_geo.u; self.v[:] = z_geo.v; 
            self.w[:] = z_geo.w; self.b[:] = z_geo.b
        return


class SingleWave(State):
    """
    An initial condition that consist of a single wave with a 
    given wavenumber and a given mode.

    Attributes:
        kx (float)     : The wavenumber in the x-direction.
        ky (float)     : The wavenumber in the y-direction.
        kz (float)     : The wavenumber in the z-direction.
        mode (int)     : The mode (0, 1, -1)
        omega (complex): The frequency of the wave 
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
        period (float) : The period of the wave.
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
    """
    def __init__(self, mset: ModelSettings, grid:Grid, 
                 kx=6, ky=0, kz=4, s=1, phase=0, use_discrete=True) -> None:
        """
        Constructor of the SingleWave initial condition.

        Arguments:
            mset (ModelSettings)  : The model settings.
            grid (Grid)           : The grid.
            kx (float)            : The wavenumber in the x-direction.
            ky (float)            : The wavenumber in the y-direction.
            kz (float)            : The wavenumber in the z-direction.
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
        kk = cp.argmin(cp.abs(grid.k[2] - kz))

        # save the index of the wave number and the wave number itself
        self.k_loc = (ki, kj, kk)
        self.kx = grid.k[0][ki]; self.ky = grid.k[1][kj]; self.kz = grid.k[2][kk]

        # Construct the spectral field of the corresponding mode
        # all zeros except for the mode
        from fridom.Framework.FieldVariable import FieldVariable
        g = FieldVariable(grid, is_spectral=True)
        g[self.k_loc] = cp.exp(1j*phase)

        # Construct the eigenvector of the corresponding mode
        if use_discrete:
            from fridom.NonHydrostatic.Eigenvectors import VecQ
            q = VecQ(s, mset, grid)
        else:
            from fridom.NonHydrostatic.Eigenvectors import VecQAnalytical
            q = VecQAnalytical(s, mset, grid)

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


class WavePackage(State):
    """
    A single wave package.
    Attributes:
        kx (float)     : The wavenumber in the x-direction.
        ky (float)     : The wavenumber in the y-direction.
        kz (float)     : The wavenumber in the z-direction.
        mode (int)     : The mode (0, 1, -1)
        omega (complex): The frequency of the wave 
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
        period (float) : The period of the wave.
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
    """
    def __init__(self, mset: ModelSettings, grid:Grid, 
                 kx=6, ky=0, kz=4, s=1, phase=0, 
                 mask_pos=(0.5, None, 0.5), mask_width=(0.2, None, 0.2)) -> None:
        """
        Constructor of the initial condition.

        Arguments:
            mset (ModelSettings)  : The model settings.
            grid (Grid)           : The grid.
            kx (float)            : The wavenumber in the x-direction.
            ky (float)            : The wavenumber in the y-direction.
            kz (float)            : The wavenumber in the z-direction.
            s (int)               : The mode (0, 1, -1)
                                    0 => geostrophic mode
                                    1 => positive inertia-gravity mode
                                   -1 => negative inertia-gravity mode
            phase (real)          : The phase of the wave. (Default: 0)
        """
        super().__init__(grid)

        # Shortcuts
        cp = self.cp

        # Construct single wave
        z = SingleWave(mset, grid, kx, ky, kz, s, phase)

        if not s == 0:
            self.omega = z.omega
            self.period = z.period

        # Construct mask
        mask = cp.ones_like(grid.X[0])
        for x, pos, width in zip(grid.X, mask_pos, mask_width):
            if pos is not None and width is not None:
                mask *= cp.exp(-(x - pos)**2 / width**2)

        z.u *= mask
        z.v *= mask
        z.w *= mask
        z.b *= mask

        # Project onto the mode again
        from fridom.NonHydrostatic.Eigenvectors import VecQ, VecP
        q = VecQ(s, mset, grid)
        p = VecP(s, mset, grid)

        z = z.project(p, q)

        if not s == 0:
            z *= 2

        # save the state
        self.u[:] = z.u; self.v[:] = z.v
        self.w[:] = z.w; self.b[:] = z.b

        return


class VerticalMode(State):
    """
    An initial condition that consist of a single verticle mode with a 
    given wavenumber and a given mode.

    Attributes:
        kx (float)     : The wavenumber in the x-direction.
        ky (float)     : The wavenumber in the y-direction.
        kz (float)     : The wavenumber in the z-direction.
        mode (int)     : The mode (0, 1, -1)
        omega (complex): The frequency of the wave 
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
        period (float) : The period of the wave.
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
    """
    def __init__(self, mset: ModelSettings, grid:Grid, 
                 kx=6, ky=0, kz=4, s=1, phase=0, use_discrete=True) -> None:
        """
        Constructor of the initial condition.

        Arguments:
            mset (ModelSettings)  : The model settings.
            grid (Grid)           : The grid.
            kx (float)            : The wavenumber in the x-direction.
            ky (float)            : The wavenumber in the y-direction.
            kz (float)            : The wavenumber in the z-direction.
            s (int)               : The mode (0, 1, -1)
                                    0 => geostrophic mode
                                    1 => positive inertia-gravity mode
                                   -1 => negative inertia-gravity mode
            phase (real)          : The phase of the wave. (Default: 0)
        """
        super().__init__(grid)

        # Shortcuts
        cp = self.cp

        # Note that the fourier transformation assumes that the first value
        # is at z=0 and the last value is at z=(H-\Delta z). However, due to
        # the grid structure, the first value is at z=\Delta z and the last
        # value is at z=H. To satisfy the boundary condition w(z=H)=0, we
        # need to shift the values by one grid point. This can be done by 
        # adding a phase shift corresponding to a shift of one grid point.

        # Super position of two waves with opposite phase and opposite kz
        z1 = SingleWave(mset, grid, kx, ky, kz, s, phase)
        
        # calculate the phase shift of the first wave that corresponds to one
        # grid point.
        delta_phase_z1 = mset.dz * z1.kz
        # the phase shift of the second wave is the same but with opposite sign.
        # instead of shifting both waves by the same amount, we shift the second
        # wave by the difference of the two phase shifts.
        phase_shift_z2 = cp.pi - 2 * delta_phase_z1
        z2 = SingleWave(mset, grid, kx, ky, -kz, s, phase + phase_shift_z2)
        z_sum = z1 + z2

        # store frequency, period, and wave number
        self.kx = z1.kx; self.ky = z1.ky; self.kz = z1.kz
        if s == 0:
            pass
        else:
            self.omega = z1.omega
            self.period = z1.period

        self.u[:] = z_sum.u; self.v[:] = z_sum.v
        self.w[:] = z_sum.w; self.b[:] = z_sum.b

        return

# remove symbols from namespace
del ModelSettings, Grid, State