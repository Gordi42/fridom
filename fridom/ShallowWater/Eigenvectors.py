from fridom.ShallowWater.ModelSettings import ModelSettings
from fridom.ShallowWater.Grid import Grid
from fridom.ShallowWater.State import State


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
                     1  => positive inertial-gravity,
                    -1  => negative inertial-gravity)
            mset: The ModelSettings object.
            grid: The Grid object.
        """
        super().__init__(mset, grid, is_spectral=True)

        # Shortcuts
        cp = self.cp
        kx, ky = tuple(grid.K)
        dx = mset.dx; dy = mset.dy;
        f0 = mset.f0; csqr = mset.csqr

        # Discrete spectral operators (one-hat-plus etc.)
        ohp  = lambda kx, dx: (1 + cp.exp(1j*kx*dx)) / 2
        ohm  = lambda kx, dx: (1 + cp.exp(-1j*kx*dx)) / 2
        khp  = lambda kx, dx: 1j * (1 - cp.exp(1j*kx*dx)) / dx
        ohpm = lambda kx, dx: (1 + cp.cos(kx*dx)) / 2
        
        f2_hat = f0**2 * ohpm(kx,dx) * ohpm(ky,dy)

        # Check the mode of the eigenvector
        is_geostrophic = (s == 0)

        om = -s * self.grid.omega_space_discrete

        self.u[:] = (-1j*f0*ohp(kx,dx)*ohm(ky,dy)*khp(ky,dy) + om*khp(kx,dx))
        self.v[:] = (+1j*f0*ohm(kx,dx)*ohp(ky,dy)*khp(kx,dx) + om*khp(ky,dy))
        self.h[:] = f2_hat - s**2 * om**2

        # Inertial mode
        self.u[kx**2+ky**2==0] = -1j*s
        self.v[kx**2+ky**2==0] = s**2
        self.h[kx**2+ky**2==0] = 1-s**2

        if is_geostrophic:
            self.u[f2_hat==0] = 0
            self.v[f2_hat==0] = 0



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
                     1  => positive inertial-gravity,
                    -1  => negative inertial-gravity)
            mset: The ModelSettings object.
            grid: The Grid object.
        """
        super().__init__(mset, grid, is_spectral=True)

        # Shortcuts
        cp = self.cp
        kx, ky = tuple(grid.K)
        csqr = mset.csqr

        # Mask to separate inertial modes from inertia-gravity modes
        g = (kx**2 + ky**2 != 0)

        # Construct the eigenvector
        q = VecQ(s, mset, grid)
        self.u = q.u.copy(); self.v = q.v.copy(); self.h = q.h.copy()
        self.h[g] /= csqr

        # normalize the vector
        norm = cp.abs(q.dot(self))
        # avoid division by zero
        mask = (norm > 1e-10)
        self.u[mask] /= norm[mask]
        self.v[mask] /= norm[mask]
        self.h[mask] /= norm[mask]
        # set to zero where the norm is zero
        zero_mask = (norm <= 1e-10)
        self.u[zero_mask] *= 0
        self.v[zero_mask] *= 0
        self.h[zero_mask] *= 0


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
        kx, ky = tuple(grid.K)
        kh2 = kx**2 + ky**2
        f0 = mset.f0

        # Mask to separate inertial modes from inertia-gravity modes
        ng = (kh2 == 0)

        om = -s * self.grid.omega_analytical

        self.u[:] = -1j * f0 * ky + om * kx
        self.v[:] = +1j * f0 * kx + om * ky
        self.h[:] = f0**2 - s**2 * om**2

        # Inertial mode
        self.u[ng] = -1j*s
        self.v[ng] = s**2
        self.h[ng] = 1-s**2


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
        cp = self.cp
        kx, ky = tuple(grid.K)
        csqr = mset.csqr

        # Mask to separate inertial modes from inertia-gravity modes
        g = (kx**2 + ky**2 != 0)

        # Construct the eigenvector
        q = VecQ(s, mset, grid)
        self.u = q.u.copy(); self.v = q.v.copy(); self.h = q.h.copy()
        self.h[g] /= csqr

        # normalize the vector
        norm = self.cp.abs(q.dot(self))
        mask = (norm != 0)
        self.u[mask] /= norm[mask]
        self.v[mask] /= norm[mask]
        self.h[mask] /= norm[mask]