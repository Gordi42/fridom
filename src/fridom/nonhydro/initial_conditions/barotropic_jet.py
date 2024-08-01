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
    Barotropic instable jet setup with 2 zonal jets

    Description
    -----------
    A Barotropic instable jet setup with 2 zonal jets and a perturbation
    on top of it. The jet is given by:

    .. math::
        u = 2.5 \\left( \\exp\\left(-\\left(\\frac{y - 0.75 L_y}{\\sigma}\\right)^2\\right) -
                        \\exp\\left(-\\left(\\frac{y - 0.25 L_y}{\\sigma}\\right)^2\\right) 
                \\right)

    where :math:`L_y` is the domain length in the y-direction, 
    and :math:`\\sigma = 0.04 \\pi` is the width of the jet. The perturbation
    is given by:

    .. math::
        v = A \\sin \\left( \\frac{2 \\pi}{L_x} k_p x \\right)

    where :math:`A` is the amplitude of the perturbation and :math:`k_p` is the
    wavenumber of the perturbation. When `geo_proj` is set to True, the initial
    condition is projected to the geostrophic subspace using the geostrophic
    eigenvectors.

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
    
    Examples
    --------

    .. code-block:: python

        import fridom.nonhydro as nh
        import numpy as np
        grid = nh.grid.cartesian.Grid(
            N=(128, 128, 16), L=(4, 4, 1), periodic_bounds=(True, True, True))
        mset = nh.ModelSettings(
            grid=grid, f0=1, N2=1, dsqr=0.2**2, Ro=0.5)
        mset.time_stepper.dt = np.timedelta64(2, 'ms')
        mset.setup()
        z = nh.initial_conditions.BarotropicJet(
            mset, wavenum=5, waveamp=0.1, geo_proj=True)
        model = nh.Model(mset)
        model.z = z
        model.run(runlen=np.timedelta64(2, 's'))
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
        self.u.arr  = 2.5*( ncp.exp(-((Y - 0.75*Ly)/(0.04*PI))**2) - 
                            ncp.exp(-((Y - 0.25*Ly)/(0.04*PI))**2) )

        # Construct the perturbation
        kx_p = 2*PI/Lx * wavenum
        self.v.arr  = waveamp * ncp.sin(kx_p*X)

        if geo_proj:
            from fridom.nonhydro.projection import GeostrophicSpectral
            proj_geo = GeostrophicSpectral(mset)
            z_geo = proj_geo(self)
            self.fields = z_geo.fields
        return