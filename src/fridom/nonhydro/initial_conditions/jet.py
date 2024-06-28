# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework import config
from fridom.nonhydro.state import State
# Import type information
if TYPE_CHECKING:
    from fridom.nonhydro.model_settings import ModelSettings


class Jet(State):
    """
    A 3D jet with horizontal and vertical shear.
    
    Description
    -----------
    Superposition of a zonal jet and a geostrophic perturbation.
    Following the setup of Chouksey et al. 2022.
    For very large jet_strengths, convective instabilities can occur.
    
    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `jet_strength` : `float`
        The strength of the zonal jets.
    `jet_width` : `float`
        The width of the zonal jets.
    `pert_strength` : `float`
        The strength of the perturbation.
    `pert_wavenum` : `int`
        The wavenumber of the perturbation.
    `geo_proj` : `bool`
        Whether to project the initial condition to the geostrophic subspace.
    """
    def __init__(self, mset: 'ModelSettings', 
                 jet_strength=1, jet_width=0.16,
                 pert_strength=0.05, pert_wavenum=5,
                 geo_proj=True):
        super().__init__(mset)
        ncp = config.ncp

        X, Y, Z = mset.grid.X
        Lx, Ly, Lz = mset.grid.L

        # two opposite jets
        self.u[:] = -ncp.exp(-(Y-Ly/4)**2/(jet_width)**2)
        self.u[:] += ncp.exp(-(Y-3*Ly/4)**2/(jet_width)**2)
        self.u[:] *= jet_strength * ncp.cos(2*ncp.pi*Z/Lz)

        # add a small perturbation
        from fridom.nonhydro.initial_conditions import SingleWave
        z_per = SingleWave(mset, kx=pert_wavenum, ky=0, kz=0, s=0)
        z_per /= ncp.max(ncp.sqrt(z_per.u**2 + z_per.v**2 + z_per.w**2))

        self.u[:] += pert_strength * z_per.u
        self.v[:] += pert_strength * z_per.v
        self.w[:] += pert_strength * z_per.w
        self.b[:] += pert_strength * z_per.b

        if geo_proj:
            from fridom.nonhydro.projection import GeostrophicSpectral
            proj_geo = GeostrophicSpectral(mset)
            z_geo = proj_geo(self)
            self.u[:] = z_geo.u; self.v[:] = z_geo.v; 
            self.w[:] = z_geo.w; self.b[:] = z_geo.b
        return