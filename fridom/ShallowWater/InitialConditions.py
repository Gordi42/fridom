import numpy as np

from fridom.ShallowWater.Grid import Grid
from fridom.ShallowWater.State import State


class Jet(State):
    """
    A Barotropic instable jet setup with 2 zonal jets and a perturbation
    on top of it. The jet positions in the y-direction are 
    at (1/4, 3/4)*Ly (opposing sign).
    """
    def __init__(self, grid:Grid, 
                 wavenum=5, waveamp=0.1, jet_pos=(0.25, 0.75), jet_width=0.04,
                 geo_proj=True):
        """
        Constructor of the Barotropic Jet initial condition with 2 zonal jets.

        Arguments:
            grid              : The grid.
            wavenum           : The wavenumber of the perturbation.
            waveamp           : The amplitude of the perturbation.
            geo_proj          : Whether to project the initial condition
                                to the geostrophic subspace. Default: True.
        """
        super().__init__(grid)
        # Shortcuts
        cp = self.cp
        mset = grid.mset
        PI = cp.pi
        x, y = tuple(grid.X)
        x = x + 0.5*mset.dg[0]
        y = y + 0.5*mset.dg[1]
        Lx, Ly = tuple(mset.L)

        # Construct the zonal jets
        self.u[:]  = 2.5*( cp.exp(-((y - jet_pos[1]*Ly)/(jet_width*PI))**2) - 
                           cp.exp(-((y - jet_pos[0]*Ly)/(jet_width*PI))**2) )

        # Construct the perturbation
        kx_p = 2*PI/Lx * wavenum
        self.h[:]  = waveamp * cp.sin(kx_p*x)

        if geo_proj:
            from fridom.ShallowWater.Projection import GeostrophicSpectral
            proj_geo = GeostrophicSpectral(grid)
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
        from fridom.Framework.FieldVariable import FieldVariable
        g = FieldVariable(grid, is_spectral=True)
        g[self.k_loc] = cp.exp(1j*phase)

        # Construct the eigenvector of the corresponding mode
        if use_discrete:
            from fridom.ShallowWater.Eigenvectors import VecQ
            q = VecQ(s, grid)
        else:
            from fridom.ShallowWater.Eigenvectors import VecQAnalytical
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
    def __init__(self, grid:Grid, 
                 kx=6, ky=0, s=1, phase=0, 
                 mask_pos=(0.5, 0.5), mask_width=(0.2, 0.2)) -> None:
        """
        Constructor of the initial condition.

        Arguments:
            grid (Grid)           : The grid.
            kx (float)            : The wavenumber in the x-direction.
            ky (float)            : The wavenumber in the y-direction.
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
        z = SingleWave(grid, kx, ky, s, phase)

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
        from fridom.ShallowWater.Eigenvectors import VecQ, VecP
        q = VecQ(s, grid)
        p = VecP(s, grid)

        z = z.project(p, q)

        if not s == 0:
            z *= 2

        # save the state
        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h

        return


class Random(State):
    """
    Oceanic spectra with random phases.
    Used in the OBTA paper.
    """
    def __init__(self, grid:Grid, 
                 d=7, k0=6, seed=12345, amplitude_geostrophy=0.2, 
                 amplitude_wave=0.1, wave_power_law=-2) -> None:
        super().__init__(grid)
        z_geo = GeostrophicSpectra(grid, d, k0, seed=seed)
        z_wav = WaveSpectra(grid, wave_power_law, seed=seed)
        z = z_geo * amplitude_geostrophy + z_wav * amplitude_wave
        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h
        return

class RandomPhase(State):
    """
    Calculates a random phase field with a presribed spectral scaling for
    the layer thickness h.
    """
    def __init__(self, grid:Grid, 
                 spectral_function, random_type="normal",
                 amplitude=1.0, seed=12345) -> None:
        """
        Arguments:
            spectral_function (callable) : with interface spectral_function(K)
            random_type (str)            : "uniform" or "normal"
                => uniform: The phase is randomized by multiplying the complex 
                            value e^ip to the state where p is uniformly
                            distributed between 0 and 2pi.
                => normal:  The phase is randomized by multiplying 
                            (a + ib) to the state where a and b are
                            normally distributed.
            amplitude (float)             : The resulting height field is
                                            normalized to this value.
            seed (int)                    : The seed for the random phase
        """
        super().__init__(grid)
        # get the wavenumber
        cp = self.cp
        mset = grid.mset
        Kx, Ky = tuple(grid.K)
        K = cp.sqrt(Kx**2 + Ky**2)
        k_hor = cp.sqrt(Kx**2 + Ky**2)

        # Define Function for random phase
        kx_flat = Kx.flatten(); ky_flat = Ky.flatten()
        k_order = cp.max(cp.abs(cp.array([kx_flat, ky_flat])), axis=0)
        angle = cp.angle(kx_flat + 1j*ky_flat)
        if mset.gpu:
            sort = np.lexsort((angle.get(), k_order.get()))
            sort = cp.array(sort)
        else:
            sort = cp.lexsort((angle, k_order))

        default_rng = cp.random.default_rng

        if random_type == "uniform":
            def random_phase(seed):
                # random phase between 0 and 2pi
                phase = default_rng(seed).uniform(0, 2*cp.pi, kx_flat.shape)
                return cp.exp(1j*phase).reshape(K.shape)

        elif random_type == "normal":
            def random_phase(seed):
                r = kx_flat*0 + 0j
                r[sort] = default_rng(seed).standard_normal(kx_flat.shape) + 1j*default_rng(2*seed).standard_normal(kx_flat.shape)
                return r.reshape(K.shape)
        else:
            raise ValueError("Unknown random phase type")

        kx_max = 2./3.*cp.amax(cp.abs(k_hor))
        large_k = (k_hor >= kx_max*1.0)
        spectra = cp.where(large_k, 0, spectral_function(K))
        # divide by K 
        spectra[K!=0] /= K[K!=0]

        from fridom.ShallowWater.State import State
        z = State(grid, is_spectral=True)
        z.h[:] = cp.sqrt(spectra) * random_phase(seed)
        z = z.fft()

        # normalize
        z.h[:] -= cp.mean(z.h)
        scal = amplitude/cp.amax(z.h)
        z *= scal

        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h
        return

class GeostrophicSpectra(State):
    """
    Oceanic spectra with random phases.
    """
    def __init__(self, grid:Grid, 
                 d=7, k0=6, seed=12345, random_type="normal") -> None:
        super().__init__(grid)
        # set coefficients for power law
        cp = self.cp
        mset = grid.mset
        
        b = (7.+d)/4.
        a = (4./7.)*b-1
        def spectral_function(K):
            return K**7/(K**2 + a*k0**2)**(2*b)

        z = RandomPhase(grid, spectral_function, random_type, 1.0, seed)
        from fridom.ShallowWater.Projection import GeostrophicSpectral
        geo_proj = GeostrophicSpectral(grid)
        z = geo_proj(z)
        max_amp = cp.amax(cp.abs(z.h))
        z /= max_amp

        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h
        return

    

class WaveSpectra(State):
    """
    Wave spectra with power law scaling of frequency.
    """
    def __init__(self, grid:Grid, 
                 power_law=-2, seed=12345,
                 random_type="normal") -> None:
        super().__init__(grid)
        # get the wavenumber
        cp = self.cp
        mset = grid.mset

        def spectral_function(K):
            spectra = cp.sqrt(mset.f0 ** 2 + mset.csqr * K ** 2)
            spectra[spectra!=0] **= power_law
            return spectra

        z = RandomPhase(grid, spectral_function, random_type, 1.0, seed)
        from fridom.ShallowWater.Projection import GeostrophicSpectral
        geo_proj = GeostrophicSpectral(grid)
        z = z - geo_proj(z)
        max_amp = cp.amax(cp.abs(z.h))
        z /= max_amp
        
        self.u[:] = z.u; self.v[:] = z.v; self.h[:] = z.h
        return

# remove symbols from namespace
del Grid, State