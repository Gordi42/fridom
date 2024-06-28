# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.nonhydro.state import State
# Import type information
if TYPE_CHECKING:
    from fridom.nonhydro.model_settings import ModelSettings


class BarotropicJet(State):
    """
    A Barotropic instable jet setup with 2 zonal jets and a perturbation
    on top of it. The jet positions in the y-direction are 
    at (1/4, 3/4)*Ly (opposing sign).

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `wavenum` : `int`
        The wavenumber of the perturbation.
    `waveamp` : `float`
        The amplitude of the perturbation.
    `geo_proj` : `bool`
        Whether to project the initial condition to the geostrophic subspace.
    
    """
    def __init__(self, mset: 'ModelSettings', 
                 wavenum=5, waveamp=0.1, geo_proj=True):
        super().__init__(mset)
        # Shortcuts
        ncp = config.ncp
        PI = ncp.pi
        X, Y, Z = mset.grid.X
        Lx, Ly, Lz = mset.grid.L

        # Construct the zonal jets
        self.u[:]  = 2.5*( ncp.exp(-((Y - 0.75*Ly)/(0.04*PI))**2) - 
                           ncp.exp(-((Y - 0.25*Ly)/(0.04*PI))**2) )

        # Construct the perturbation
        kx_p = 2*PI/Lx * wavenum
        self.v[:]  = waveamp * ncp.sin(kx_p*X)

        if geo_proj:
            from fridom.nonhydro.projection import GeostrophicSpectral
            proj_geo = GeostrophicSpectral(mset)
            z_geo = proj_geo(self)
            self.u[:] = z_geo.u; self.v[:] = z_geo.v; 
            self.w[:] = z_geo.w; self.b[:] = z_geo.b
        return