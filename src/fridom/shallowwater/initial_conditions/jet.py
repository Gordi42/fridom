from fridom.shallowwater.grid import Grid
from fridom.shallowwater.state import State


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
            from fridom.shallowwater.projection import GeostrophicSpectral
            proj_geo = GeostrophicSpectral(grid)
            z_geo = proj_geo(self)
            self.u[:] = z_geo.u; self.v[:] = z_geo.v; self.h[:] = z_geo.h
        return


# remove symbols from namespace
del Grid, State