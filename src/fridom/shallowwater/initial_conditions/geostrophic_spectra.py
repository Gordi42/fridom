"""Initial conditions for the shallow water model with geostrophic spectra."""
from __future__ import annotations

from typing import Callable, TypeVar

import numpy as np

import fridom.shallowwater as sw

T = TypeVar("T", float, np.ndarray)

def geostrophic_energy_spectrum(kx: T, ky: T, d: float = 7, k0: float = 6) -> T:
    r"""
    Geostrophic energy spectrum.

    Description
    -----------
    Following the work of Masur & Oliver [2020], the energy spectrum :math:`S`
    is given by:

    .. math::
        S = \frac{k^7}{\left(k^2 + a k_0^2\right)^{2b}}

    where :math:`k = \sqrt{k_x^2 + k_y^2}` is the horizontal wavenumber, :math:`a`
    and :math:`b` are constants:

    .. math::
        a = \frac{4}{7}b - 1, \quad b = \frac{7+d}{4}

    where :math:`d` is the power law exponent for large horizontal wavenumbers
    (:math:`S(k) \sim k^{-d}` for :math:`k \to \infty`). The parameter
    :math:`k_0` is the wavenumber with the maximum energy.

    Parameters
    ----------
    kx : float | np.ndarray
        The horizontal wavenumber in the x-direction.
    ky : float | np.ndarray
        The horizontal wavenumber in the y-direction.
    d : float, optional (default=7)
        The power law exponent for large horizontal wavenumbers
        (:math:`S_h(k) \sim k^{-d}` for :math:`k \to \infty`).
    k0 : float, optional (default=6)
        The wavenumber with the maximum energy.

    Returns
    -------
    float | np.ndarray
        The spectral energy density.

    """
    ncp = sw.config.ncp

    # horizontal spectra
    kh = ncp.sqrt(kx**2 + ky**2)
    b = (7.+d)/4.
    a = (4./7.)*b-1
    return kh**7/(kh**2 + a*k0**2)**(2*b)


class RandomGeostrophicSpectra(sw.State):

    """
    Random geostrophic state with a given spectral energy density.

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings (need to be set up).
    `seed` : `int`
        Seed for the random number generator (for the phase)
    `spectral_energy_density` : `Callable(kx, ky)`
        Callable that returns the spectral energy density as a function of the
        wavenumbers `kx` and `ky`.

    """

    def __init__(self,
                 mset: sw.ModelSettings,
                 seed: int = 12345,
                 spectral_energy_density: Callable = geostrophic_energy_spectrum,
                 ) -> None:
        super().__init__(mset, is_spectral=False)

        ncp = sw.config.ncp
        grid = mset.grid
        kx, ky = grid.get_mesh(spectral=True)

        # construct the geostrophic eigenvectors
        q = grid.vec_q(s=0, use_discrete=True)

        # scale the geostrophic eigenvector such that they have energy 1
        energy_arr = q.spectral_ekin.arr
        energy_arr = ncp.where(energy_arr == 0, 1, energy_arr)
        q /= ncp.sqrt(energy_arr)

        # construct a random phase
        r = grid.create_random_array(seed=seed, spectral=True)

        # construct the spectral energy density
        spectra = spectral_energy_density(kx, ky)

        # construct the geostrophic state
        z = q * r * ncp.sqrt(spectra)

        # transform to physical space and normalize the state such that the
        # maximum velocity is 1
        z = z.ifft()
        u_max = z.u.max()
        z /= u_max

        # set the state
        self.fields = z.fields
