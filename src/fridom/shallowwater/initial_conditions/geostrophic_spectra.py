"""Initial conditions with prescribed spectra and random phases."""
from __future__ import annotations

from functools import partial
from typing import Callable, TypeVar

import fridom.shallowwater as sw
import numpy as np

T = TypeVar("T", float, np.ndarray)

def geostrophic_energy_spectrum(kx: T, ky: T, d: float = 7, k0: float = 4) -> T:
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
    k0 : float, optional (default=4)
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

def gm_energy_spectrum(kx: T,
                       ky: T,
                       wave_power_law: float = -2,
                       f0: float = 1.0,
                       csqr: float = 1.0) -> T:
    r"""
    Garrett-Munk like energy spectrum.

    Description
    -----------

    We consider a spectral energy density that is motivated by the Garrett-Munk
    spectrum. For that we relate the energy density to the frequency spectrum
    :math:`\omega(k)`:

    .. math::
        S(k) = \omega(k)^d = \sqrt{f_0^2 + c^2 k^2}^d

    where :math:`k = \sqrt{k_x^2 + k_y^2}` is the horizontal wavenumber, :math:`f_0`
    is the Coriolis parameter, :math:`c` is the phase speed, and :math:`d` is the
    power law exponent

    Parameters
    ----------
    kx : float | np.ndarray
        The horizontal wavenumber in the x-direction.
    ky : float | np.ndarray
        The horizontal wavenumber in the y-direction.
    wave_power_law : float, optional (default=-2)
        The power law exponent for the energy spectrum as a function of the
        frequency.
    f0 : float, optional (default=1.0)
        The Coriolis parameter.
    csqr : float, optional (default=1.0)
        The phase speed.

    Returns
    -------
    float | np.ndarray
        The spectral energy density.

    """
    ncp = sw.config.ncp

    # horizontal spectra
    kh = ncp.sqrt(kx**2 + ky**2)

    omega = ncp.sqrt(f0**2 + csqr*kh**2)
    return omega**wave_power_law

class PrescribedSpectraRandomPhase(sw.State):

    """
    Construct a state with prescribed spectral energy density and random phases.

    Parameters
    ----------
    mset : ModelSettings
        The model settings (need to be set up).
    spectral_energy_density : Callable(kx, ky)
        Callable that returns the spectral energy density as a function of the
        wavenumbers `kx` and `ky`.
    mode : str
        The mode of the eigenvector to construct (0 for geostrophic, 1 for wave).
    seed : int
        Seed for the random number generator (for the phase)

    """

    def __init__(self,
                 mset: sw.ModelSettings,
                 spectral_energy_density: Callable[[T, T], T],
                 mode: str,
                 seed: int = 12345,
                 ) -> None:
        super().__init__(mset, is_spectral=False)

        ncp = sw.config.ncp
        grid = mset.grid
        kx, ky = grid.get_mesh(spectral=True)

        # construct the eigenvector
        q = grid.vec_q(s=mode, use_discrete=True)

        # We want to normalize the eigenvector such that the spectral energy
        # density at wavenumber k is 1. Let the spectral energy $E(kx, ky)$ be
        # given at each wave number pair, we are looking for the total spectral
        # energy at wave number k. Hence, we have to integrate over all wave
        # numbers with the same wave number k. For our eigenvector q, the spectral
        # energy $E(kx(k, \theta), ky(k, \theta))$ does not depend on the angle
        # theta. Hence, we can integrate over the angle theta and obtain the
        # spectral energy density is given by $S(k) = 2 \pi k E(k, 0)$.

        # scale the eigenvector such that they have energy 1
        energy_arr = q.spectral_ekin.arr * 2 * ncp.pi * ncp.sqrt(kx**2 + ky**2)
        energy_arr = ncp.where(energy_arr == 0, 1, energy_arr)

        # we further have to integrate over rings with konstant wave number
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

class RandomGeostrophicSpectra(sw.State):

    """
    Random geostrophic state with a given spectral energy density.

    Parameters
    ----------
    mset : ModelSettings
        The model settings (need to be set up).
    seed : int
        Seed for the random number generator (for the phase)
    spectral_energy_density : Callable(kx, ky)
        Callable that returns the spectral energy density as a function of the
        wavenumbers `kx` and `ky`.

    """

    def __init__(self,
                 mset: sw.ModelSettings,
                 spectral_energy_density: Callable = geostrophic_energy_spectrum,
                 seed: int = 12345,
                 ) -> None:
        super().__init__(mset, is_spectral=False)

        self.fields = PrescribedSpectraRandomPhase(
            mset, spectral_energy_density, mode=0, seed=seed).fields

class RandomWaveSpectra(sw.State):

    """
    Random wave state with a given spectral energy density.

    Parameters
    ----------
    mset : ModelSettings
        The model settings (need to be set up).
    seed : int
        Seed for the random number generator (for the phase)
    spectral_energy_density : Callable(kx, ky)
        Callable that returns the spectral energy density as a function of the
        wavenumbers `kx` and `ky`.

    """

    def __init__(self,
                 mset: sw.ModelSettings,
                 spectral_energy_density: Callable = gm_energy_spectrum,
                 seed: int = 12345,
                 ) -> None:
        super().__init__(mset, is_spectral=False)

        wave1 = PrescribedSpectraRandomPhase(
            mset, spectral_energy_density, mode=1, seed=seed)

        wave2 = PrescribedSpectraRandomPhase(
            mset, spectral_energy_density, mode=-1, seed=2*seed + 1)

        self.fields = (wave1 + wave2).fields

class ShallowWaterRandom(sw.State):

    """
    The random shallow water initial condition used in the paper.

    Parameters
    ----------
    mset : ModelSettings
        The model settings (need to be set up).
    amplitude_geo : float, optional (default=0.2)
        The amplitude of the geostrophic component.
    amplitude_wave : float, optional (default=0.1)
        The amplitude of the wave component.
    d : float, optional (default=7)
        The power law exponent for large horizontal wavenumbers (geo).
    k0 : float, optional (default=4)
        The wavenumber with the maximum energy (geo).
    wave_power_law : float, optional (default=-2)
        The power law exponent for the energy spectrum as a function of the
        frequency (wave).
    seed : int
        Seed for the random number generator (for the phase)

    """

    def __init__(self,  # noqa: PLR0913
                 mset: sw.ModelSettings,
                 amplitude_geo: float = 1.0,
                 amplitude_wave: float = 0.1,
                 d: float = 7,
                 k0: float = 4,
                 wave_power_law: float = -2,
                 seed: int = 12345,
                 ) -> None:

        super().__init__(mset, is_spectral=False)

        geo_spectra = partial(geostrophic_energy_spectrum, d=d, k0=k0)
        wav_spectra = partial(
            gm_energy_spectrum, wave_power_law=wave_power_law,
            f0=mset.f0, csqr=mset.csqr)

        z_geo = amplitude_geo * RandomGeostrophicSpectra(
            mset, spectral_energy_density=geo_spectra, seed=seed)

        z_wave = amplitude_wave * RandomWaveSpectra(
            mset, spectral_energy_density=wav_spectra, seed=2*seed + 1)

        self.fields = (z_geo + z_wave).fields
