import numpy as np

from fridom.framework.grid_base_old import GridBaseOld
from fridom.nonhydro.model_settings import ModelSettings

class Grid(GridBaseOld):
    """
    Grid container with the meshgrid of the physical and spectral domain.

    Attributes:
        mset (ModelSettings)           : Model settings.
        x (list of 1D arrays)          : Physical domain.
        X (list of 3D arrays)          : Physical domain (meshgrid).
        k (list of 1D arrays)          : Spectral domain.
        K (list of 3D arrays)          : Spectral domain (meshgrid).
        N2_array (1D array)            : Buoyancy frequency squared.
        k2_hat (3D array)              : Scaled wave number squared.
        k2_hat_zero (3D array)         : Boolean array of where k2_hat is zero.
        omega_analytical (3D array)    : Analytical dispersion relation.
        omega_space_discrete (3D array): Space-discrete dispersion relation.
        omega_time_discrete (3D array) : Space-time-discrete dispersion relation.
    """

    def __init__(self, mset:ModelSettings) -> None:
        """
        Constructor.

        Parameters:
            mset (ModelSettings): Model settings.
        """
        super().__init__(mset)

        cp = self.cp; dtype = mset.dtype
        N = mset.N
        y = self.x[1]

        # buoyancy frequency squared
        self.N2_array = cp.ones((1,1,N[2]), dtype=dtype) * mset.N0**2
        self.f_array  = cp.zeros((1,N[1],1), dtype=dtype)
        self.f_array[:] = mset.f0 + mset.beta * y[None,:,None]

        self.N2_array = mset.N0**2
        self.f_array = mset.f0

        # discretized wave number squared
        k_dis = [2 * (1 - cp.cos(kx * dx)) / dx**2 
                  for (kx,dx) in zip(self.K, mset.dg)]

        # scaled discretized wave number squared
        self.k2_hat = k2_hat = k_dis[0] + k_dis[1] + k_dis[2] / mset.dsqr

        self.k2_hat_zero = (k2_hat == 0)
        k2_hat[self.k2_hat_zero] = 1

        # dispersion relation are computed on demand
        self._omega_analytical     = None
        self._omega_time_discrete  = None
        self._omega_space_discrete = None
        return

    @property
    def cpu(self) -> "Grid":
        """
        Create a copy of the grid on the CPU.
        """
        if self._cpu is None:
            if not self.mset.gpu:
                self._cpu = self
            else:
                mset_cpu = self.mset.copy()
                mset_cpu.gpu = False
                self._cpu = Grid(mset_cpu)
                # self._cpu.N2_array = self.N2_array.get()
                # self._cpu.f_array  = self.f_array.get()
        return self._cpu

    
    @property
    def omega_analytical(self):
        """
        Analytical dispersion relation.

        Returns:
            omega_analytical (3D array): Analytical dispersion relation. 
        """
        # check if the dispersion relation can be computed
        if self.mset.enable_varying_N:
            print("Warning: Dispersion relation may be wrong when N is varying.")
        if self.mset.enable_varying_f:
            print("Warning: Dispersion relation may be wrong when f is varying.")
        # compute only once
        if self._omega_analytical is None:
            # shorthand notation
            f0    = self.mset.f0; 
            N0    = self.mset.N0; 
            dsqr  = self.mset.dsqr
            sqrt  = self.cp.sqrt

            # squared wave number (horizontal and vertical)
            kh2   = self.K[0]**2 + self.K[1]**2
            kz2   = self.K[2]**2

            # avoid division by zero
            s     = (kh2 + kz2 > 0)               

            # compute dispersion relation
            om    = self.cp.zeros_like(kh2)
            om[s] = sqrt((f0**2 * kz2 + N0**2 * kh2)[s] / (dsqr * kh2 + kz2)[s])

            # store the result
            self._omega_analytical = om
        
        return self._omega_analytical


    @property
    def omega_space_discrete(self):
        """
        Discrete dispersion relation (only in space).
        """
        # check if the dispersion relation can be computed
        if self.mset.enable_varying_N:
            print("Warning: Dispersion relation may be wrong when N is varying.")
        if self.mset.enable_varying_f:
            print("Warning: Dispersion relation may be wrong when f is varying.")
        # compute only once
        if self._omega_space_discrete is None:
            # shorthand notation
            cp   = self.cp
            mset = self.mset
            dx = mset.dx; dy = mset.dy; dz = mset.dz
            f0 = mset.f0; N0 = mset.N0; dsqr = mset.dsqr
            kx = self.K[0]; ky = self.K[1]; kz = self.K[2]

            # averaging operator (one hat squared)
            ohpm = lambda kx, dx: (1 + cp.cos(kx*dx)) / 2
            # difference operator (k hat squared)
            khpm = lambda kx, dx: 2 * (1 - cp.cos(kx*dx)) / dx**2

            # avoid division by zero
            s = (kx**2 + ky**2 + kz**2 > 0)

            # horizontal wave number squared
            kh2_hat = khpm(kx, dx) + khpm(ky, dy)

            # compute dispersion relation
            om_hat    = cp.zeros_like(kh2_hat)
            om_hat[s] = cp.sqrt(
                (ohpm(kx,dx) * ohpm(ky,dy) * f0**2 * khpm(kz,dz) + 
                 ohpm(kz,dz) * N0**2 * kh2_hat)[s] / 
                 (dsqr * kh2_hat + khpm(kz,dz))[s] )

            # store the result
            self._omega_space_discrete = om_hat

        return self._omega_space_discrete


    @property
    def omega_time_discrete(self):
        """
        Discrete dispersion relation (in space and time). (Very slow)

        Returns:
            omega_time_discrete (3D array): Discrete dispersion relation.
        """
        # compute only once
        if self._omega_time_discrete is None:
            # shorthand notation
            cp = self.cp; mset = self.mset
            Nx = mset.N[0]; Ny = mset.N[1]; Nz = mset.N[2]

            # get adam bashforth coefficients
            ab_coefficients = [mset.AB1, mset.AB2,
                               mset.AB3, mset.AB4]

            # get the coefficients for the current time level
            coeff = ab_coefficients[mset.time_levels-1]

            # construct polynomial coefficients for each grid point
            # tile the array such that coeff and omega_discrete have the same
            coeff = cp.tile(coeff, (Nx, Ny, Nz, 1))
            omi   = self.omega_space_discrete[..., cp.newaxis]

            # calculate the polynomial coefficients
            coeff = cp.multiply(omi, coeff) * 1j * self.mset.dt
            coeff[..., 0] -= 1

            # leading coefficient is 1
            coeff = cp.pad(
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
                # root finding only works on the CPU
                if self.mset.gpu:
                    # convert to numpy array
                    return np.roots(c.get())[-1]
                else:
                    return cp.roots(c)[-1]

            # find the roots of the polynomial
            roots = cp.apply_along_axis(find_roots, -1, coeff)
        
            # calculate the dispersion relation with time discretization
            res = -1j * cp.log(roots) / self.mset.dt

            # store the result
            self._omega_time_discrete = res

        return self._omega_time_discrete

# remove symbols from namespace
del ModelSettings, GridBaseOld