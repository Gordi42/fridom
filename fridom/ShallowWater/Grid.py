import numpy

from fridom.Framework.GridBase import GridBase
from fridom.ShallowWater.ModelSettings import ModelSettings

class Grid(GridBase):
    """
    Grid container with the meshgrid of the physical and spectral domain.

    Attributes:
        mset (ModelSettings)           : Model settings.
        x (list of 1D arrays)          : Physical domain.
        X (list of 2D arrays)          : Physical domain (meshgrid).
        k (list of 1D arrays)          : Spectral domain.
        K (list of 2D arrays)          : Spectral domain (meshgrid).
        N2_array (1D array)            : Buoyancy frequency squared.
        k2_hat (2D array)              : Scaled wave number squared.
        k2_hat_zero (2D array)         : Boolean array of where k2_hat is zero.
        omega_analytical (2D array)    : Analytical dispersion relation.
        omega_space_discrete (2D array): Space-discrete dispersion relation.
        omega_time_discrete (2D array) : Space-time-discrete dispersion relation.
    """

    def __init__(self, mset:ModelSettings) -> None:
        """
        Constructor.

        Parameters:
            mset (ModelSettings): Model settings.
        """
        super().__init__(mset)
        self.mset = mset

        cp = self.cp; dtype = mset.dtype
        N = mset.N
        y = self.x[1]

        # constants for finite difference operators
        self.dx1 = mset.dtype(1) / mset.dx
        self.dy1 = mset.dtype(1) / mset.dy
        self.dx2 = self.dx1**2
        self.dy2 = self.dy1**2
        self.half = mset.dtype(0.5)
        self.quarter = mset.dtype(0.25)

        # buoyancy frequency squared
        self.f_array  = cp.zeros((1,N[1]), dtype=dtype)
        self.f_array[:] = mset.f0 + mset.beta * y[None,:]

        # discretized wave number squared
        k_dis = [2 * (1 - cp.cos(kx * dx)) / dx**2 
                  for (kx,dx) in zip(self.K, mset.dg)]

        # scaled discretized wave number squared
        self.k2_hat = k2_hat = k_dis[0] + k_dis[1]

        self.k2_hat_zero = (k2_hat == 0)
        k2_hat[self.k2_hat_zero] = 1

        # dealiasing mask (2/3 rule)
        kx_max = 2/3 * cp.max(self.K[0])
        ky_max = 2/3 * cp.max(self.K[1])
        kmax = (kx_max**2 + ky_max**2)**0.5
        k = (self.K[0]**2 + self.K[1]**2) ** 0.5
        self.dealias_mask = cp.ones_like(self.K[0])
        self.dealias_mask[k > kmax] = 0

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
                self._cpu.f_array  = self.f_array.get()
        return self._cpu

    
    @property
    def omega_analytical(self):
        """
        Analytical dispersion relation.

        Returns:
            omega_analytical (3D array): Analytical dispersion relation. 
        """
        # check if the dispersion relation can be computed
        if self.mset.enable_varying_f:
            print("Warning: Dispersion relation may be wrong when f is varying.")
        # compute only once
        if self._omega_analytical is None:
            # shorthand notation
            f0    = self.mset.f0; 
            csqr  = self.mset.csqr
            sqrt  = self.cp.sqrt
            kh2   = self.k2_hat

            # compute dispersion relation
            om    = sqrt(f0**2 + csqr * kh2)

            # store the result
            self._omega_analytical = om
        
        return self._omega_analytical


    @property
    def omega_space_discrete(self):
        """
        Discrete dispersion relation (only in space).
        """
        # check if the dispersion relation can be computed
        if self.mset.enable_varying_f:
            print("Warning: Dispersion relation may be wrong when f is varying.")
        # compute only once
        if self._omega_space_discrete is None:
            # shorthand notation
            cp   = self.cp
            mset = self.mset
            dx = mset.dx; dy = mset.dy
            f0 = mset.f0; csqr = mset.csqr
            kx, ky = tuple(self.K)

            # averaging operator (one hat squared)
            ohpm = lambda kx, dx: (1 + cp.cos(kx*dx)) / 2
            # difference operator (k hat squared)
            khpm = lambda kx, dx: 2 * (1 - cp.cos(kx*dx)) / dx**2

            # horizontal wave number squared
            kh2_hat = khpm(kx, dx) + khpm(ky, dy)

            # compute dispersion relation
            om_hat    = cp.sqrt(f0**2 * ohpm(kx,dx)*ohpm(ky,dy) + csqr * kh2_hat)

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
            Nx, Ny = tuple(mset.N)

            # get adam bashforth coefficients
            ab_coefficients = [mset.AB1, mset.AB2,
                               mset.AB3, mset.AB4]

            # get the coefficients for the current time level
            coeff = ab_coefficients[mset.time_levels-1]

            # construct polynomial coefficients for each grid point
            # tile the array such that coeff and omega_discrete have the same
            coeff = cp.tile(coeff, (Nx, Ny, 1))
            omi   = self.omega_space_discrete[..., cp.newaxis]

            # calculate the polynomial coefficients
            coeff = cp.multiply(omi, coeff) * 1j * self.mset.dt
            coeff[..., 0] -= 1

            # leading coefficient is 1
            coeff = cp.pad(
                coeff, ((0,0), (0,0), (1,0)), 
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
                    return numpy.roots(c.get())[-1]
                else:
                    return cp.roots(c)[-1]

            # find the roots of the polynomial
            roots = cp.apply_along_axis(find_roots, -1, coeff)
        
            # calculate the dispersion relation with time discretization
            res = -1j * cp.log(roots) / self.mset.dt

            # store the result
            self._omega_time_discrete = res

        return self._omega_time_discrete
