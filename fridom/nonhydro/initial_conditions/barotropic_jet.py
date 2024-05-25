from fridom.nonhydro.grid import Grid
from fridom.nonhydro.state import State


class BarotropicJet(State):
    """
    A Barotropic instable jet setup with 2 zonal jets and a perturbation
    on top of it. The jet positions in the y-direction are 
    at (1/4, 3/4)*Ly (opposing sign).
    """
    def __init__(self, grid:Grid, 
                 wavenum=5, waveamp=0.1, geo_proj=True):
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
        x  = grid.X[0]; y  = grid.X[1]; z  = grid.X[2]
        Lx = mset.L[0]; Ly = mset.L[1]; Lz = mset.L[2]

        # Construct the zonal jets
        self.u[:]  = 2.5*( cp.exp(-((y - 0.75*Ly)/(0.04*PI))**2) - 
                           cp.exp(-((y - 0.25*Ly)/(0.04*PI))**2) )

        # Construct the perturbation
        kx_p = 2*PI/Lx * wavenum
        self.v[:]  = waveamp * cp.sin(kx_p*x)

        if geo_proj:
            from fridom.nonhydro.projection import GeostrophicSpectral
            proj_geo = GeostrophicSpectral(grid)
            z_geo = proj_geo(self)
            self.u[:] = z_geo.u; self.v[:] = z_geo.v; 
            self.w[:] = z_geo.w; self.b[:] = z_geo.b
        return

# remove symbols from namespace
del Grid, State