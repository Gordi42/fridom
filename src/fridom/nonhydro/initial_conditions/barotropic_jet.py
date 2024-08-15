import fridom.nonhydro as nh


class BarotropicJet(nh.State):
    r"""
    Barotropic instable jet setup with 2 zonal jets

    Description
    -----------
    A Barotropic instable jet setup with 2 zonal jets and a perturbation
    on top of it. The jet is given by:

    .. math::
        u = 2.5 \left( \exp\left(-\left(\frac{y - 0.75 L_y}{\sigma L_y \pi}\right)^2\right) -
                        \exp\left(-\left(\frac{y - 0.25 L_y}{\sigma L_y \pi}\right)^2\right) 
                \right)

    where :math:`L_y` is the domain length in the y-direction, 
    and :math:`\sigma` is the width of the jet. The perturbation
    is given by:

    .. math::
        v = A \sin \left( \frac{2 \pi}{L_x} k_p x \right)

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
    `jet_width` : `float`
        The width of the jet.
    `geo_proj` : `bool`
        Whether to project the initial condition to the geostrophic subspace.
    """
    def __init__(self, 
                 mset: nh.ModelSettings, 
                 wavenum=5, 
                 waveamp=0.1, 
                 jet_width=0.04,
                 geo_proj=True):
        super().__init__(mset)
        # Shortcuts
        ncp = nh.config.ncp
        PI = ncp.pi
        X, Y, Z = mset.grid.X
        Lx, Ly, Lz = mset.grid.L
        width = jet_width * Ly * PI

        # Construct the zonal jets
        self.u.arr  = 2.5*( ncp.exp(-((Y - 0.75*Ly)/(width))**2) - 
                            ncp.exp(-((Y - 0.25*Ly)/(width))**2) )

        # Construct the perturbation
        kx_p = 2*PI/Lx * wavenum
        self.v.arr  = waveamp * ncp.sin(kx_p*X)

        if geo_proj:
            proj_geo = nh.projection.GeostrophicSpectral(mset)
            z_geo = proj_geo(self)
            self.fields = z_geo.fields
        return