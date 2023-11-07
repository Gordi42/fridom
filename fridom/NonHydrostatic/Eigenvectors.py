from fridom.NonHydrostatic.ModelSettings import ModelSettings
from fridom.NonHydrostatic.Grid import Grid
from fridom.NonHydrostatic.State import State


class VecQ(State):
    """
    Discrete eigenvectors of the System matrix.
    See the documentation for more details.
    """

    def __init__(self, s, mset:ModelSettings, grid:Grid) -> None:
        """
        Constructor of the discrete eigenvectors of the System matrix.
        
        Arguments:
            s   : The mode of the eigenvector. 
                    (0  => geostrophic, 
                    "d" => divergent,
                     1  => positive inertial-gravity,
                    -1  => negative inertial-gravity)
            mset: The ModelSettings object.
            grid: The Grid object.
        """
        super().__init__(mset, grid, is_spectral=True)

        # Shortcuts
        cp = self.cp
        kx = grid.K[0]; ky = grid.K[1]; kz = grid.K[2]
        dx = mset.dx; dy = mset.dy; dz = mset.dz
        f0 = mset.f0; N0 = mset.N0; dsqr = mset.dsqr

        # Discrete spectral operators (one-hat-plus etc.)
        ohp  = lambda kx, dx: (1 + cp.exp(1j*kx*dx)) / 2
        ohm  = lambda kx, dx: (1 + cp.exp(-1j*kx*dx)) / 2
        khp  = lambda kx, dx: 1j * (1 - cp.exp(1j*kx*dx)) / dx
        khm  = lambda kx, dx: 1j * (cp.exp(-1j*kx*dx) - 1) / dx
        khpm = lambda kx, dx: 2 * (1 - cp.cos(kx*dx)) / dx**2
        ohpm = lambda kx, dx: (1 + cp.cos(kx*dx)) / 2
        
        # Discrete spectral horizontal wavenumber squared
        kh2_hat = khpm(kx, dx) + khpm(ky, dy)

        # Check the mode of the eigenvector
        is_geostrophic = (s == 0)
        is_divergent   = (s == "d")

        # Mask to separate inertial modes from inertia-gravity modes
        ng = (kx**2 + ky**2 == 0)       # inertial modes
        g = (kx**2 + ky**2 != 0)        # inertia-gravity modes

        if is_geostrophic:
            # Horizontal constant geostrophic mode
            self.u[ng] = 0
            self.v[ng] = 0
            self.w[ng] = 0
            self.b[ng] = 1

            # Geostrophic mode
            self.u[g] = (-ohp(kx,dx)*ohm(ky,dy)*ohp(kz,dz)*khp(ky,dy))[g]
            self.v[g] = ( ohm(kx,dx)*ohp(ky,dy)*ohp(kz,dz)*khp(kx,dx))[g]
            self.w[g] = 0
            self.b[g] = (ohpm(kx,dx)*ohpm(ky,dy)*f0*khp(kz,dz))[g]
        elif is_divergent:
            self.u[:] = khp(kx,dx)
            self.v[:] = khp(ky,dy)
            self.w[:] = khp(kz,dz)
            self.b[:] = 0
        else:
            # Inertial modes
            self.u[ng] = -1j*s
            self.v[ng] = 1
            self.w[ng] = 0
            self.b[ng] = 0

            # Inertia-gravity wave modes
            # Eigenvalues (frequency)
            om_hat = -s * self.grid.omega_space_discrete

            # Eigenvectors of inertia-gravity wave modes
            self.u[g] = (khm(kz,dz) * (1j*om_hat*khp(kx,dx) + 
                          ohp(kx,dx)*ohm(ky,dy)*f0*khp(ky,dy)))[g]
            self.v[g] = (khm(kz,dz) * (1j*om_hat*khp(ky,dy) -
                          ohm(kx,dx)*ohp(ky,dy)*f0*khp(kx,dx)))[g]
            self.w[g] = (-1j*om_hat*kh2_hat)[g]
            self.b[g] = (ohm(kz,dz) * N0**2 * kh2_hat)[g]


class VecP(State):
    """
    Projection vector on the discrete eigenvectors of the System matrix.
    See the documentation for more details.
    """

    def __init__(self, s, mset:ModelSettings, grid=Grid) -> None:
        """
        Constructor of the projector on the discrete eigenvectors.
        
        Arguments:
            s   : The mode of the eigenvector. 
                    (0  => geostrophic, 
                    "d" => divergent,
                     1  => positive inertial-gravity,
                    -1  => negative inertial-gravity)
            mset: The ModelSettings object.
            grid: The Grid object.
        """
        super().__init__(mset, grid, is_spectral=True)

        # Shortcuts
        cp = self.cp
        kx = grid.K[0]; ky = grid.K[1]; kz = grid.K[2]
        dx = mset.dx; dy = mset.dy; dz = mset.dz
        f0 = mset.f0; N0 = mset.N0; dsqr = mset.dsqr

        # Discrete spectral operators (one-hat-plus etc.)
        ohp = lambda kx, dx: (1 + cp.exp(1j*kx*dx)) / 2
        ohm = lambda kx, dx: (1 + cp.exp(-1j*kx*dx)) / 2
        khp = lambda kx, dx: 1j * (1 - cp.exp(1j*kx*dx)) / dx
        khm = lambda kx, dx: 1j * (cp.exp(-1j*kx*dx) - 1) / dx
        khpm = lambda kx, dx: 2 * (1 - cp.cos(kx*dx)) / dx**2
        ohpm = lambda kx, dx: (1 + cp.cos(kx*dx)) / 2
        
        # Discrete spectral horizontal wavenumber squared
        kh2_hat = khpm(kx, dx) + khpm(ky, dy)

        # Check the mode of the eigenvector
        is_geostrophic = (s == 0)
        is_divergent = (s == "d")

        # Mask to separate inertial modes from inertia-gravity modes
        g = (kx**2 + ky**2 != 0)

        # Construct the eigenvector
        q = VecQ(s, mset, grid)
        self.u = q.u.copy(); self.v = q.v.copy() 
        self.w = q.w.copy(); self.b = q.b.copy()

        if is_geostrophic:
            # Horizontal constant geostrophic mode is same as the eigenvector
            # Calculate the horizontal varying geostrophic modes:
            self.u[g] = (-N0**2*ohp(kx,dx)*ohm(ky,dy)*ohp(kz,dz)*khp(ky,dy))[g]
            self.v[g] = ( N0**2*ohm(kx,dx)*ohp(ky,dy)*ohp(kz,dz)*khp(kx,dx))[g]
            self.w[g] = 0
            self.b[g] = (ohpm(kx,dx)*ohpm(ky,dy)*f0*khp(kz,dz))[g]

        elif is_divergent:
            # Divergent modes are the same as the eigenvector
            pass
        else:
            # Inertial modes are the same as the eigenvector
            # Only calculate the inertia-gravity modes:
            # Scaling factor
            gamma = cp.zeros_like(kx)
            gamma[g] = (kh2_hat[g] + khpm(kz,dz)[g]) / \
                       (dsqr * kh2_hat[g] + khpm(kz,dz)[g])

            # Eigenvalues (frequency)
            om_hat = -s * self.grid.omega_space_discrete

            self.u[g] = (khm(kz,dz) * (1j*om_hat*khp(kx,dx) + 
                            ohp(kx,dx)*ohm(ky,dy)*f0*gamma*khp(ky,dy)))[g]
            self.v[g] = (khm(kz,dz) * (1j*om_hat*khp(ky,dy) -
                            ohm(kx,dx)*ohp(ky,dy)*f0*gamma*khp(kx,dx)))[g]
            self.w[g] = (-1j*om_hat*kh2_hat)[g]
            self.b[g] = (ohm(kz,dz) * gamma * kh2_hat)[g]

        # normalize the vector
        norm = cp.abs(q.dot(self))
        # avoid division by zero
        mask = (norm > 1e-10)
        self.u[mask] /= norm[mask]
        self.v[mask] /= norm[mask]
        self.w[mask] /= norm[mask]
        self.b[mask] /= norm[mask]
        # set to zero where the norm is zero
        zero_mask = (norm <= 1e-10)
        self.u[zero_mask] *= 0
        self.v[zero_mask] *= 0
        self.w[zero_mask] *= 0
        self.b[zero_mask] *= 0


class VecQAnalytical(State):
    """
    Analytical eigenvectors of the System matrix.
    See the documentation for more details.
    """
    def __init__(self, s, mset:ModelSettings, grid:Grid) -> None:
        """
        Constructor of the analytical eigenvectors of the System matrix.
        
        Arguments:
            s   : The mode of the eigenvector. 
                    (0  => geostrophic, 
                    "d" => divergent,
                     1  => positive inertial-gravity,
                    -1  => negative inertial-gravity)
            mset: The ModelSettings object.
            grid: The Grid object.
        """
        super().__init__(mset, grid, is_spectral=True)

        # Shortcuts
        kx = grid.K[0]; ky = grid.K[1]; kz = grid.K[2]
        kh2 = kx**2 + ky**2
        f0 = mset.f0; N0 = mset.N0; dsqr = mset.dsqr

        # Check the mode of the eigenvector
        is_geostrophic = (s == 0)
        is_divergent = (s == "d")

        # Mask to separate inertial modes from inertia-gravity modes
        ng = (kh2 == 0)
        g = (kh2 != 0)

        if is_geostrophic:
            # Horizontal constant geostrophic mode
            self.u[ng] = 0
            self.v[ng] = 0
            self.w[ng] = 0
            self.b[ng] = 1

            # Geostrophic mode
            self.u[g] = -ky[g]
            self.v[g] =  kx[g]
            self.w[g] = 0
            self.b[g] = f0 * kz[g]
        elif is_divergent:
            # Divergent mode
            self.u[:] = kx
            self.v[:] = ky
            self.w[:] = kz
            self.b[:] = 0
        else:
            # Inertial modes
            self.u[ng] = -1j*s
            self.v[ng] = 1
            self.w[ng] = 0
            self.b[ng] = 0

            # Inertia-gravity wave modes
            # Eigenvalues (frequency)
            om = -s * self.grid.omega_analytical

            # Eigenvectors of inertia-gravity wave modes
            self.u[g] = kz[g]*(1j*om[g]*kx[g] + f0*ky[g])
            self.v[g] = kz[g]*(1j*om[g]*ky[g] - f0*kx[g])
            self.w[g] = -1j*om[g]*kh2[g]
            self.b[g] = N0**2*kh2[g]


class VecPAnalytical(State):
    """
    Projection vector on the analytical eigenvectors of the System matrix.
    See the documentation for more details.
    """
    def __init__(self, s, mset:ModelSettings, grid=Grid) -> None:
        """
        Constructor of the projector on the analytical eigenvectors.
        
        Arguments:
            s   : The mode of the eigenvector. 
                    (0  => geostrophic, 
                    "d" => divergent,
                     1  => positive inertial-gravity,
                    -1  => negative inertial-gravity)
            mset: The ModelSettings object.
            grid: The Grid object.
        """
        super().__init__(mset, grid, is_spectral=True)

        # Shortcuts
        kx = grid.K[0]; ky = grid.K[1]; kz = grid.K[2]
        kh2 = kx**2 + ky**2
        f0 = mset.f0; N0 = mset.N0; dsqr = mset.dsqr

        # Check the mode of the eigenvector
        is_geostrophic = (s == 0)
        is_divergent = (s == "d")

        # Mask to separate inertial modes from inertia-gravity modes
        g = (kh2 != 0)

        # Construct the eigenvector
        q = VecQAnalytical(s, mset, grid)
        self.u = q.u.copy(); self.v = q.v.copy() 
        self.w = q.w.copy(); self.b = q.b.copy()

        if is_geostrophic:
            # Horizontal constant geostrophic mode is same as the eigenvector
            # Calculate the horizontal varying geostrophic modes:
            self.u[g] = -N0**2 * ky[g]
            self.v[g] =  N0**2 * kx[g]
            self.w[g] = 0
            self.b[g] = f0 * kz[g]
        elif is_divergent:
            # Divergent modes are the same as the eigenvector
            pass
        else:
            # Inertial modes are the same as the eigenvector
            # Only calculate the inertia-gravity modes:

            # Eigenvalues (frequency)
            om = -s * self.grid.omega_analytical
            # Scaling factor
            gamma = (kh2[g] + kz[g]**2) / (dsqr * kh2[g] + kz[g]**2)

            self.u[g] = kz[g]*(1j*om[g]*kx[g] + f0*gamma*ky[g])
            self.v[g] = kz[g]*(1j*om[g]*ky[g] - f0*gamma*kx[g])
            self.w[g] = -1j*om[g]*kh2[g]
            self.b[g] = gamma*kh2[g]

        # normalize the vector
        norm = self.cp.abs(q.dot(self))
        mask = (norm != 0)
        self.u[mask] /= norm[mask]
        self.v[mask] /= norm[mask]
        self.w[mask] /= norm[mask]
        self.b[mask] /= norm[mask]