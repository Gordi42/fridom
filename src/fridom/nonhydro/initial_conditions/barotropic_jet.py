"""Barotropic instable jet initial condition."""
from __future__ import annotations

import fridom.nonhydro as nh


class BarotropicJet(nh.State):

    r"""
    Barotropic instable jet setup with 2 zonal jets.

    Description
    -----------
    A Barotropic instable jet setup with 2 zonal jets and a perturbation
    on top of it. The jet is given by:

    .. math::
        u = 2.5 \left(
            \exp\left(
                -\left(\frac{y - 0.75 L_y}{\sigma L_y \pi}\right)^2\right) -
            \exp\left(
                -\left(\frac{y - 0.25 L_y}{\sigma L_y \pi}\right)^2\right)
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
                 wavenum: int = 5,
                 waveamp: float = 0.1,
                 jet_width: float = 0.04,
                 geo_proj: bool = True) -> None:
        super().__init__(mset)
        # Shortcuts
        ncp = nh.config.ncp
        pi = ncp.pi
        x, y, _z = mset.grid.x_mesh
        lx, ly, _lz = mset.grid.domain_size
        width = jet_width * ly * pi

        # Construct the zonal jets
        self.u.arr  = 2.5*( ncp.exp(-((y - 0.75*ly)/(width))**2) -
                            ncp.exp(-((y - 0.25*ly)/(width))**2) )

        # Construct the perturbation
        kx_p = 2*pi/lx * wavenum
        self.v.arr  = waveamp * ncp.sin(kx_p*x)

        if geo_proj:
            proj_geo = nh.projection.GeostrophicSpectral(mset)
            z_geo = proj_geo(self)
            self.fields = z_geo.fields
