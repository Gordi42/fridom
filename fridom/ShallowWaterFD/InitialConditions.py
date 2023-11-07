import numpy 

from fridom.ShallowWaterFD.ModelSettings import ModelSettings
from fridom.ShallowWaterFD.Grid import Grid
from fridom.ShallowWaterFD.State import State
from fridom.Framework.FieldVariable import FieldVariable
from fridom.ShallowWaterFD.Eigenvectors import VecQ, VecP, VecQAnalytical
from fridom.ShallowWaterFD.Projection import GeostrophicSpectral, DivergenceSpectral


class Jet(State):
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
        super().__init__(mset, grid)
        # Shortcuts
        cp = self.cp
        PI = cp.pi
        x, y = tuple(grid.X)
        Lx, Ly = tuple(mset.L)

        # Construct the zonal jets
        self.u[:]  = 2.5*( cp.exp(-((y - 0.75*Ly)/(0.04*PI))**2) - 
                           cp.exp(-((y - 0.25*Ly)/(0.04*PI))**2) )

        # Construct the perturbation
        kx_p = 2*PI/Lx * wavenum
        self.v[:]  = waveamp * cp.sin(kx_p*x)

        if geo_proj:
            proj_geo = GeostrophicSpectral(mset, grid)
            z_geo = proj_geo(self)
            self.u[:] = z_geo.u; self.v[:] = z_geo.v; self.h[:] = z_geo.h
        return


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
    def __init__(self, mset: ModelSettings, grid:Grid, 
                 kx=6, ky=4, s=1, phase=0, use_discrete=True) -> None:
        """
        Constructor of the SingleWave initial condition.

        Arguments:
            mset (ModelSettings)  : The model settings.
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
        super().__init__(mset, grid)

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
        g = FieldVariable(mset, grid, is_spectral=True)
        g[self.k_loc] = cp.exp(1j*phase)

        # Construct the eigenvector of the corresponding mode
        if use_discrete:
            q = VecQ(s, mset, grid)
        else:
            q = VecQAnalytical(s, mset, grid)

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
            coeff = numpy.array(coeff[self.mset.time_levels-1])

            # Construct the polynomial coefficients
            coeff = 1j * coeff * om.item() * self.mset.dt
            coeff[0] -= 1
            coeff = numpy.pad(coeff, (1,0), 'constant', constant_values=(1,0))
            
            # Calculate the roots of the polynomial
            roots = numpy.roots(coeff[::-1])[-1]

            # Calculate the complex frequency
            self.__omega = -1j * numpy.log(roots)/self.mset.dt
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
        mode (int)     : The mode (0, 1, -1)
        omega (complex): The frequency of the wave 
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
        period (float) : The period of the wave.
                         (includes effects of time discretization)
                         (only for inertia-gravity modes).
    """
    def __init__(self, mset: ModelSettings, grid:Grid, 
                 kx=6, ky=0, s=1, phase=0, 
                 mask_pos=(0.5, 0.5), mask_width=(0.2, 0.2)) -> None:
        """
        Constructor of the initial condition.

        Arguments:
            mset (ModelSettings)  : The model settings.
            grid (Grid)           : The grid.
            kx (float)            : The wavenumber in the x-direction.
            ky (float)            : The wavenumber in the y-direction.
            s (int)               : The mode (0, 1, -1)
                                    0 => geostrophic mode
                                    1 => positive inertia-gravity mode
                                   -1 => negative inertia-gravity mode
            phase (real)          : The phase of the wave. (Default: 0)
        """
        super().__init__(mset, grid)

        # Shortcuts
        cp = self.cp

        # Construct single wave
        z = SingleWave(mset, grid, kx, ky, s, phase)

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
        z.h *= mask

        # Project onto the mode again
        q = VecQ(s, mset, grid)
        p = VecP(s, mset, grid)

        z = z.project(p, q)

        if not s == 0:
            z *= 2

        # save the state
        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h

        return


class Random(State):
    """
    Oceanic spectra with random phases.
    """
    def __init__(self, mset: ModelSettings, grid:Grid, 
                 d=7, k0=6, seed=0) -> None:
        super().__init__(mset, grid)
        # set coefficients for power law
        cp = self.cp
        Kx, Ky = tuple(grid.K)
        K = cp.sqrt(Kx**2 + Ky**2)
        
        b = (7.+d)/4.
        a = (4./7.)*b-1

        # Define Function for random phase
        kx_flat = Kx.flatten(); ky_flat = Ky.flatten()
        k_order = cp.max(cp.abs(cp.array([kx_flat, ky_flat])), axis=0)
        angle = cp.angle(kx_flat + 1j*ky_flat)
        if mset.gpu:
            sort = numpy.lexsort((angle.get(), k_order.get()))
            sort = cp.array(sort)
        else:
            sort = cp.lexsort((angle, k_order))

        default_rng = cp.random.default_rng

        def random_phase(seed):
            r = kx_flat*0 + 0j
            r[sort] = default_rng(seed).standard_normal(kx_flat.shape) + 1j*default_rng(2*seed).standard_normal(kx_flat.shape)
            return r.reshape(K.shape)

        k_hor = cp.sqrt(Kx**2 + Ky**2)
        kx_max = 2./3.*cp.amax(cp.abs(k_hor))
        large_k = (k_hor >= kx_max)
        # create random geostrophic state
        r =  random_phase(seed)
        H0 = cp.where(large_k, 0, cp.sqrt( K**6/(K**2 + a*k0**2)**(2*b) )*r)
        r =  random_phase(3*seed+2)
        Hp = cp.where(large_k, 0, cp.sqrt( K**6/(K**2 + a*k0**2)**(2*b) )*r)
        r =  random_phase(4*seed+3)
        Hm = cp.where(large_k, 0, cp.sqrt( K**6/(K**2 + a*k0**2)**(2*b) )*r)

        # project to eigenvectors

        z0 = (VecQ(0, mset, grid) * H0).fft()
        zp = (VecQ(1, mset, grid) * Hp).fft()
        zm = (VecQ(-1, mset, grid) * Hm).fft()

        # normalize
        for z in [z0, zp, zm]:
            z.h[:] -= cp.mean(z.h)
    
        scal0 = 0.2/cp.amax(z0.h)
        scalp = 0.02/cp.amax(zp.h)
        scalm = 0.02/cp.amax(zm.h)

        z = z0*scal0 + zp*scalp + zm*scalm

        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h