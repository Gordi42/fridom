import fridom.shallowwater as sw


class Jet(sw.State):
    """
    Two opposing instable jets.

    Description
    -----------
    An instable jet setup with a small pressure perturbation
    on top of it. The jet is given by:

    .. math::
        u = \\exp\\left(-\\left(\\frac{y - p L_y}{\\sigma L_y}\\right)^2\\right)

    where :math:`L_y` is the domain length in the y-direction, 
    :math:`p` is the relative position of the jet
    and :math:`\\sigma` is the relative width of the jet. The perturbation
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
    `pos` : `float`
        The relative position of the jet in the y-direction
    `width` : `float`
        The relative width of the jet.
    `geo_proj` : `bool`
        Whether to project the initial condition to the geostrophic subspace.
    """
    def __init__(self, 
                 mset: sw.ModelSettings,
                 wavenum: int = 2,
                 waveamp: float = 0.1,
                 pos: float = 0.5,
                 width: float = 0.1,
                 geo_proj: bool = True):
        super().__init__(mset)
        # Shortcuts
        ncp = sw.config.ncp
        X, Y = self.grid.X
        Lx, Ly = self.grid.L

        # Construct the zonal jets
        z_jet = sw.State(mset)
        z_jet.u.arr = ncp.exp(- ((Y - pos * Ly)/(width * Ly))**2)

        # Project to geostrophic subspace
        if geo_proj:
            proj_geo = sw.projection.GeostrophicSpectral(mset)
            z_jet = proj_geo(z_jet)

        # Normalize the jet
        u_amp = (z_jet.u**2 + z_jet.v**2).max()
        z_jet /= u_amp**(1/2)

        # Construct the perturbation
        z_wave = sw.initial_conditions.SingleWave(mset, (wavenum, 0), s=0)
        u_amp = (z_wave.u**2 + z_wave.v**2).max()
        z_wave /= u_amp**(1/2)

        # return the sum
        z = z_jet + waveamp * z_wave
        self.fields = z.fields
