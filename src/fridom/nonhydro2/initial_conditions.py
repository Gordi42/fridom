r"""
Random-phase eigenmode initial conditions (``nh.initial_conditions``).

Description
-----------
The nonhydrostatic port of the reference
``PrescribedSpectraRandomPhase`` family: states composed of one mode
family's eigenvectors with a prescribed spectral energy density and
Hermitian random phases, on **both** eigenmode tiers:

- fully periodic and walled-vertical (rigid-lid) grids build on the
  analytic operator-sourced
  :class:`~fridom.nonhydro2.eigenmodes.Eigenmodes` (all modes at
  once, vectorized; on a walled vertical the phases key the ``u``
  component's mode lattice, so the buoyancy-top ``m = n`` stratum is
  excluded);
- horizontally walled channels build on the labeled numeric
  :class:`~fridom.nonhydro2.channel_eigenmodes.ChannelEigenmodes`
  (per-plane column synthesis; the ``constraint`` complement is
  never selected).

Surface: :func:`random_state`, :func:`random_vortical` (defaulting
to the horizontal :func:`geostrophic_energy_spectrum`) and
:func:`random_waves` (flat wave spectra, plus the flat Kelvin family
on a channel). Every factory takes a model **or** an eigenmodes
object and returns a :class:`~fridom.nonhydro2.state.State`,
normalized so the largest horizontal-velocity value is one. Spectra
callables receive one wavenumber array per grid axis (grid order,
e.g. ``S(kx, ky, kz)``); the defaults treat the two non-vertical
axes as horizontal (the vertical coordinate is ``"z"``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework2.grid.symbols import ModeChart
from fridom.framework2.model.eigenbasis import channel_random_state
from fridom.framework2.model.eigenstates import (
    geostrophic_energy_spectrum as geostrophic_energy_spectrum,  # noqa: PLC0414 — re-export
)
from fridom.framework2.model.eigenstates import (
    normalize_max_component,
    prescribed_spectra_coefficients,
)
from fridom.nonhydro2.channel_eigenmodes import ChannelEigenmodes
from fridom.nonhydro2.eigenmodes import Eigenmodes, from_model
from fridom.nonhydro2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.framework2.model.model import Model

#: The analytic-tier family vocabulary -> eigenmode branches.
_ANALYTIC_BRANCHES = {"vortical": (0,), "wave": (1, -1)}

#: The canonical vertical coordinate name.
_VERTICAL = "z"

#: The prognostic components of the analytic tier.
_COMPONENTS = ("u", "v", "w", "b")


def flat_spectrum(*wavenumbers: jax.Array) -> jax.Array:  # noqa: ARG001
    """Return a unit spectral energy density (the wave default)."""
    return jnp.asarray(1.0)


def _horizontal_spectrum(
    grid_names: tuple[str, ...],
) -> Callable[..., jax.Array]:
    """Geostrophic spectrum over the non-vertical wavenumbers."""
    positions = tuple(
        i for i, name in enumerate(grid_names) if name != _VERTICAL)

    def density(*wavenumbers: jax.Array) -> jax.Array:
        return geostrophic_energy_spectrum(
            wavenumbers[positions[0]], wavenumbers[positions[1]])

    return density


def _resolve(
    source: Model | Eigenmodes | ChannelEigenmodes,
    at_time: float,
) -> Eigenmodes | ChannelEigenmodes:
    """Accept an eigenmodes object or resolve one from a model."""
    if isinstance(source, Eigenmodes | ChannelEigenmodes):
        return source
    return from_model(source, at_time=at_time)


def random_state(
    source: Model | Eigenmodes | ChannelEigenmodes,
    family: str,
    *,
    spectral_energy_density: Callable[..., jax.Array] | None = None,
    seed: int = 12345,
    at_time: float = 0.0,
) -> State:
    r"""
    Random-phase state of one mode family with prescribed spectra.

    Description
    -----------
    Every mode of the selected family is scaled to the spectral
    energy :math:`S(k)/(\pi k_h)` — the reference's
    :math:`S(k) = 2\pi k\,E(k, 0)` angular convention over the
    horizontal ring measure — and multiplied by a Hermitian
    unit-modulus random phase; the sum synthesizes to a real
    physical state, normalized so the largest horizontal-velocity
    value is one. ``spectral_energy_density`` is called with one
    wavenumber array per grid axis (grid order); on a channel the
    bounded axis carries the effective meridional wavenumber
    :math:`k_y = \pi m / L_y` of each mode's within-family ordinal.

    On the analytic tier the families are ``"vortical"`` and
    ``"wave"`` (both inertia-gravity branches, independent phases);
    on a channel any labeled physical selection of
    ``nh.ChannelEigenmodes.families`` applies (unsigned roots cover
    both signed branches; the ``constraint`` complement is a taught
    error).

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model, or an eigenmodes object of either
        tier.
    family : str
        The mode family selection.
    spectral_energy_density : Callable[..., jax.Array] | None, optional
        ``S(*k)`` over the grid-axis wavenumbers; None selects the
        horizontal :func:`geostrophic_energy_spectrum` for
        ``"vortical"`` and :func:`flat_spectrum` otherwise
        (default: None).
    seed : int, optional
        The PRNG seed — bitwise deterministic (default: 12345).
    at_time : float, optional
        Parameter evaluation time when resolving from a model
        (default: 0.0).

    Returns
    -------
    State
        The normalized random state.
    """
    em = _resolve(source, at_time)
    if spectral_energy_density is None:
        spectral_energy_density = (
            _horizontal_spectrum(tuple(em.grid.names))
            if family == "vortical" else flat_spectrum)
    horizontal = tuple(
        name for name in em.grid.names if name != _VERTICAL)
    if isinstance(em, ChannelEigenmodes):
        fields = channel_random_state(
            em, family, spectral_energy_density, seed=seed,
            horizontal=horizontal)
        return State(fields)
    branches = _ANALYTIC_BRANCHES.get(family)
    if branches is None:
        if family.startswith("kelvin"):
            raise ValueError(
                "no walls, no Kelvin family: Kelvin modes are "
                "trapped at walls the rotation couples to, and the "
                "analytic nonhydro eigenmodes carry only the "
                "vortical and inertia-gravity branches. Build the "
                "model on a horizontally walled channel grid to "
                "get the labeled Kelvin pair")
        raise ValueError(
            f"unknown mode family {family!r}: the analytic "
            "families are 'vortical' and 'wave'")
    fields = prescribed_spectra_coefficients(
        grid=em.grid,
        kit=em.kit,
        chart=ModeChart(em.grid),
        columns=tuple(em.q(s) for s in branches),
        components=_COMPONENTS,
        weights=em._energy_weights(),  # noqa: SLF001 — package-internal
        reference="u",
        horizontal=horizontal,
        spectral_energy_density=spectral_energy_density,
        seed=seed)
    fields = {c: fields[c].retag(em.physical_space(c))
              for c in _COMPONENTS}
    return State(normalize_max_component(fields, ("u", "v")))


def random_vortical(
    source: Model | Eigenmodes | ChannelEigenmodes,
    *,
    spectral_energy_density: Callable[..., jax.Array] | None = None,
    seed: int = 12345,
    at_time: float = 0.0,
) -> State:
    """
    Random vortical state (the ``RandomVorticalSpectra`` port).

    Description
    -----------
    :func:`random_state` on the ``"vortical"`` family with the
    horizontal :func:`geostrophic_energy_spectrum` default.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an eigenmodes object.
    spectral_energy_density : Callable[..., jax.Array] | None, optional
        The prescribed spectrum (default: None, the horizontal
        geostrophic spectrum).
    seed : int, optional
        The PRNG seed (default: 12345).
    at_time : float, optional
        Parameter evaluation time (default: 0.0).

    Returns
    -------
    State
        The normalized random vortical state.
    """
    return random_state(
        source, "vortical",
        spectral_energy_density=spectral_energy_density,
        seed=seed, at_time=at_time)


def random_waves(
    source: Model | Eigenmodes | ChannelEigenmodes,
    *,
    seed: int = 12345,
    at_time: float = 0.0,
) -> State:
    """
    Random flat-spectrum wave state (``RandomWaveFlatSpectra`` port).

    Description
    -----------
    Both inertia-gravity branches under a flat spectrum; on a
    channel the boundary-trapped Kelvin pair joins with its own
    flat spectrum (seeded ``2 * seed + 1000``, the reference's
    offset), and the sum is re-normalized.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an eigenmodes object.
    seed : int, optional
        The PRNG seed (default: 12345).
    at_time : float, optional
        Parameter evaluation time (default: 0.0).

    Returns
    -------
    State
        The normalized random wave state.
    """
    em = _resolve(source, at_time)
    waves = random_state(em, "wave", seed=seed)
    if not isinstance(em, ChannelEigenmodes):
        return waves
    kelvin = random_state(em, "kelvin", seed=2 * seed + 1000)
    combined = waves + kelvin
    return State(normalize_max_component(
        dict(combined.components), ("u", "v")))
