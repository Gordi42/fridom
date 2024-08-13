import fridom.shallowwater as sw


class Jet(sw.State):
    """
    Two opposing instable jets.

    Description
    -----------
    An instable jet setup with 2 zonal jets and a small pressure perturbation
    on top of it. The jet is given by:

    .. math::
        u = 2.5 \\left( \\exp\\left(-\\left(\\frac{y - 0.75 L_y}{\\sigma}\\right)^2\\right) -
                        \\exp\\left(-\\left(\\frac{y - 0.25 L_y}{\\sigma}\\right)^2\\right) 
                \\right)

    where :math:`L_y` is the domain length in the y-direction, 
    and :math:`\\sigma = 0.04 \\pi` is the width of the jet. The perturbation
    is given by:

    .. math::
        p = A \\sin \\left( \\frac{2 \\pi}{L_x} k_p x \\right)

    where :math:`A` is the amplitude of the perturbation and :math:`k_p` is the
    wavenumber of the perturbation. When `geo_proj` is set to True, the initial
    condition is projected to the geostrophic subspace using the geostrophic
    eigenvectors.

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `wavenum` : `int`
        The relative wavenumber of the perturbation.
    `waveamp` : `float`
        The amplitude of the perturbation.
    `jet_pos` : `tuple`
        The relative position of the jets in the y-direction
    `jet_width` : `float`
        The width of the jets.
    `geo_proj` : `bool`
        Whether to project the initial condition to the geostrophic subspace.
    """
    def __init__(self, 
                 mset: sw.ModelSettings, 
                 wavenum: int = 5, 
                 waveamp: float = 0.1, 
                 jet_pos: tuple[float] = (0.25, 0.75), 
                 jet_width: float = 0.04,
                 geo_proj: bool = True):
        super().__init__(mset)
        # Shortcuts
        ncp = sw.config.ncp
        PI = ncp.pi
        X, Y = self.grid.X
        Lx, Ly = self.grid.L

        # Construct the zonal jets
        self.u.arr = 2.5*( ncp.exp(-((Y - jet_pos[1]*Ly)/(jet_width*PI))**2) - 
                           ncp.exp(-((Y - jet_pos[0]*Ly)/(jet_width*PI))**2) )

        # Construct the perturbation
        kx_p = 2 * PI / Lx * wavenum
        self.p.arr = waveamp * ncp.sin(kx_p*X)

        if geo_proj:
            proj_geo = sw.projection.GeostrophicSpectral(mset)
            z_geo = proj_geo(self)
            self.fields = z_geo.fields
        return
