from fridom.nonhydro.grid import Grid
from fridom.nonhydro.state import State


class Jet(State):
    """
    Superposition of a zonal jet and a geostrophic perturbation.
    Following the setup of Chouksey et al. 2022.
    For very large jet_strengths, convective instabilities can occur.
    """
    def __init__(self, grid: Grid, 
                 jet_strength=1, jet_width=0.16,
                 pert_strength=0.05, pert_wavenum=5,
                 geo_proj=True):
        """
        Constructor of the Instable Jet initial condition with 2 zonal jets.

        Arguments:
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
        mset = grid.mset

        X  = grid.X[0]; Y  = grid.X[1]; Z  = grid.X[2]
        Lx = mset.L[0]; Ly = mset.L[1]; Lz = mset.L[2]

        # two opposite jets
        self.u[:] = -cp.exp(-(Y-Ly/4)**2/(jet_width)**2)
        self.u[:] += cp.exp(-(Y-3*Ly/4)**2/(jet_width)**2)
        self.u[:] *= jet_strength * cp.cos(2*cp.pi*Z/Lz)

        # add a small perturbation
        from fridom.nonhydro.initial_conditions import SingleWave
        z_per = SingleWave(grid, kx=pert_wavenum, ky=0, kz=0, s=0)
        z_per /= cp.max(cp.sqrt(z_per.u**2 + z_per.v**2 + z_per.w**2))

        self.u[:] += pert_strength * z_per.u
        self.v[:] += pert_strength * z_per.v
        self.w[:] += pert_strength * z_per.w
        self.b[:] += pert_strength * z_per.b

        if geo_proj:
            from fridom.nonhydro.projection import GeostrophicSpectral
            proj_geo = GeostrophicSpectral(grid)
            z_geo = proj_geo(self)
            self.u[:] = z_geo.u; self.v[:] = z_geo.v; 
            self.w[:] = z_geo.w; self.b[:] = z_geo.b
        return

# remove symbols from namespace
del Grid, State