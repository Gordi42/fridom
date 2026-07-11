r"""
Named analytic initial conditions (``nh.initial_conditions``).

Description
-----------
The nonhydrostatic initial-condition factories: plain functions
taking a model (or an eigenmodes object) and returning a
:class:`~fridom.nonhydro2.state.State` the caller assigns via
``model.set_state(...)``.

Random-phase eigenmode states port the reference
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

Named analytic states port the reference initial-condition classes:
:func:`single_wave` and :func:`wave_package` (thin wrappers over the
analytic mode accessor ``em.mode``, the latter with a Gaussian
envelope and a re-projection onto the mode branch),
:func:`kelvin_wave` (the labeled boundary-trapped mode of the walled
channel eigenbasis), :func:`barotropic_jet` / :func:`jet` (unstable
zonal jets plus a single-mode perturbation, geostrophically
projected by default) and :func:`coherent_eddy` (a barotropic
Gaussian streamfunction or vorticity eddy). Wave factories return
``(omega, state)`` like ``em.mode``; profile factories return the
state alone.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.symbols import GridSymbols, ModeChart
from fridom.model.eigenbasis import channel_random_state
from fridom.model.eigenstates import (
    geostrophic_energy_spectrum as geostrophic_energy_spectrum,  # noqa: PLC0414 — re-export
)
from fridom.model.eigenstates import (
    normalize_max_component,
    prescribed_spectra_coefficients,
)
from fridom.nonhydro2.channel_eigenmodes import ChannelEigenmodes
from fridom.nonhydro2.eigenmodes import Eigenmodes, from_model
from fridom.nonhydro2.modules.gaussian_wave_maker import (
    sample_gaussian_mask,
)
from fridom.nonhydro2.state import State
from fridom.nonhydro2.transforms import (
    VorticalProjection,
    mode_projection,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.tensor_product import SpaceLike
    from fridom.model.model import Model

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
    error). On even grid sizes the vortical synthesis draws the
    primary geostrophic column, which includes the
    horizontal-Nyquist steady divergence-free strata; the extra
    internal steady columns of the doubly degenerate strata (the
    overturning and pure-buoyancy modes) are projector-only and
    carry no random content here.

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


# ================================================================
#  Named analytic initial conditions: shared helpers
# ================================================================
def _analytic(
    source: Model | Eigenmodes | ChannelEigenmodes,
    what: str,
    at_time: float,
) -> Eigenmodes:
    """Resolve to the analytic tier or teach the channel error."""
    em = _resolve(source, at_time)
    if isinstance(em, ChannelEigenmodes):
        # a value error (wrong grid topology), not a type error
        raise ValueError(  # noqa: TRY004
            f"{what} builds on the analytic eigenmodes (fully "
            "periodic or walled-vertical grids); this model is a "
            "horizontally walled channel — synthesize labeled "
            "channel modes through the eigenbasis instead, e.g. "
            "nh.eigenbasis(model).mode(family, indices), or "
            "nh.initial_conditions.kelvin_wave for the "
            "boundary-trapped pair")
    return em


def _horizontal(grid: Grid) -> tuple[str, str]:
    """Return the two non-vertical axis names, in grid order."""
    x, y = (name for name in grid.names if name != _VERTICAL)
    return x, y


def _extent(grid: Grid, name: str) -> tuple[float, float]:
    """Physical interval ``(min, max)`` of the axis ``name``."""
    mesh = next(m for m in grid.factors if name in m.names)
    return float(mesh.extent[0]), float(mesh.extent[1])


def _sample(
    grid: Grid,
    space: SpaceLike,
    values: Callable[[dict[str, jax.Array]], jax.Array],
    name: str | None = None,
) -> ScalarField:
    """Sample ``values(coords)`` at a space's own node positions.

    Description
    -----------
    Evaluates the coordinate-dict callable at the target space's
    evaluation nodes (staggered faces or cell centres) through
    ``grid.create_field(init=...)``, stamping the signature with the
    space's non-constant coordinate names (the
    ``sample_gaussian_mask`` precedent).
    """
    names = tuple(
        coordinate for factor in space.factors
        if not isinstance(factor, ConstantSpace)
        for coordinate in factor.names)

    def init(**coords: jax.Array) -> jax.Array:
        return jnp.asarray(values(coords))

    init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(
            coordinate, inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for coordinate in names])
    return grid.create_field(space, init=init, name=name)


# ================================================================
#  Single-mode waves (thin wrappers over the eigenmode machinery)
# ================================================================
def single_wave(
    source: Model | Eigenmodes | ChannelEigenmodes,
    k: Mapping[str, int],
    s: int = 1,
    *,
    phase: float = 0.0,
    at_time: float = 0.0,
) -> tuple[float, State]:
    r"""
    One discrete eigenmode as ``(omega, state)`` (``SingleWave``).

    Description
    -----------
    The thin wrapper over the analytic mode accessor
    ``em.mode(s, k, phase=...)``: the real Hermitian-closed physical
    mode :math:`\mathrm{Re}(q^s(k)\,e^{i(k\cdot x + \mathrm{phase})})`
    with exact discrete dispersion, normalized so the largest
    horizontal-velocity envelope is one. Under the linear model the
    state at time :math:`t` is the same mode at phase
    ``phase + omega * t``. On a walled vertical the ``z`` index is
    the physical vertical mode on the ``0..n`` union lattice.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an analytic eigenmodes object.
    k : Mapping[str, int]
        Axis-keyed integer wavenumber indices (e.g.
        ``{"x": 3, "y": 0, "z": 2}``); a wavenumber of one is a
        wave with one wavelength across the domain.
    s : int, optional
        The mode branch: 0 (geostrophic), +1 or -1
        (inertia-gravity) (default: 1).
    phase : float, optional
        The mode phase shift (default: 0.0).
    at_time : float, optional
        Parameter evaluation time when resolving from a model
        (default: 0.0).

    Returns
    -------
    tuple[float, State]
        The frequency and the single-mode physical state.

    Raises
    ------
    ValueError
        On a horizontally walled channel (labeled modes come from
        ``nh.eigenbasis``), bad indices, or a structurally
        unrepresented mode.
    """
    em = _analytic(source, "single_wave", at_time)
    return em.mode(s, k, phase=phase)


def kelvin_wave(
    source: Model | Eigenmodes | ChannelEigenmodes,
    k: Mapping[str, int],
    *,
    branch: int = 1,
    phase: float = 0.0,
    at_time: float = 0.0,
) -> tuple[float, State]:
    r"""
    One boundary-trapped Kelvin mode of a channel (``KelvinWave``).

    Description
    -----------
    The thin wrapper over the labeled channel eigenbasis accessor
    ``eb.mode("kelvin", k, branch=...)``: the numerically exact
    discrete Kelvin mode trapped at the channel walls, normalized so
    the largest horizontal-velocity envelope is one. ``branch``
    selects the signed frequency branch (the two branches are
    trapped at opposite walls); the bounded-axis entry of ``k`` is
    the within-family mode ordinal and defaults to 0 (the
    fundamental) when absent.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled channel model or its labeled eigenbasis.
    k : Mapping[str, int]
        Axis-keyed indices: integer wavenumbers on the periodic
        axes, the within-family ordinal on the bounded axis
        (optional, default 0).
    branch : int, optional
        The signed Kelvin branch, +1 or -1 (default: 1).
    phase : float, optional
        The mode phase shift (default: 0.0).
    at_time : float, optional
        Parameter evaluation time when resolving from a model
        (default: 0.0).

    Returns
    -------
    tuple[float, State]
        The frequency and the single-mode physical state.

    Raises
    ------
    ValueError
        On a grid without a bounded horizontal axis: Kelvin modes
        are wall-trapped — use :func:`single_wave` on periodic
        grids.
    """
    em = _resolve(source, at_time)
    if not isinstance(em, ChannelEigenmodes):
        # a value error (wrong grid topology), not a type error
        raise ValueError(  # noqa: TRY004
            "no walls, no Kelvin wave: Kelvin modes are trapped at "
            "walls the rotation couples to, and this grid has no "
            "bounded horizontal axis. Build the model on a "
            "horizontally walled channel grid, or use "
            "nh.initial_conditions.single_wave for the plane-wave "
            "modes of the periodic grid")
    indices = dict(k)
    indices.setdefault(em.bounded_axis, 0)
    return em.mode("kelvin", indices, branch=branch, phase=phase)


def wave_package(
    source: Model | Eigenmodes | ChannelEigenmodes,
    k: Mapping[str, int],
    s: int = 1,
    *,
    mask_pos: Mapping[str, float],
    mask_width: Mapping[str, float],
    phase: float = 0.0,
    at_time: float = 0.0,
) -> tuple[float, State]:
    r"""
    Build a Gaussian-enveloped single wave (``WavePackage``).

    Description
    -----------
    The single mode of :func:`single_wave` multiplied by the
    stationary Gaussian envelope

    .. math::
        M(\boldsymbol{x}) =
            \prod_{i} \exp\left(-\frac{(x_i - p_i)^2}{w_i^2}\right)

    over the coordinates named in ``mask_pos`` / ``mask_width``
    (constant along the others), sampled at each component's own
    staggered nodes, and re-projected onto the mode branch ``s`` so
    the package stays polarized. The returned frequency is the
    carrier mode's.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an analytic eigenmodes object.
    k : Mapping[str, int]
        Axis-keyed integer wavenumber indices of the carrier.
    s : int, optional
        The mode branch: 0, +1 or -1 (default: 1).
    mask_pos : Mapping[str, float]
        Envelope centres, keyed by coordinate name; unnamed axes
        are unmasked.
    mask_width : Mapping[str, float]
        Envelope widths; same keys as ``mask_pos``.
    phase : float, optional
        The carrier phase shift (default: 0.0).
    at_time : float, optional
        Parameter evaluation time when resolving from a model
        (default: 0.0).

    Returns
    -------
    tuple[float, State]
        The carrier frequency and the enveloped state.

    Raises
    ------
    ValueError
        On mismatched envelope keys, an envelope coordinate the
        grid does not have, or a horizontally walled channel.
    """
    em = _analytic(source, "wave_package", at_time)
    if set(mask_pos) != set(mask_width):
        raise ValueError(
            f"mask_pos and mask_width must name the same "
            f"coordinates; got mask_pos keys "
            f"{tuple(sorted(mask_pos))} and mask_width keys "
            f"{tuple(sorted(mask_width))}")
    unknown = sorted(set(mask_pos) - set(em.grid.names))
    if unknown:
        raise ValueError(
            f"the wave-package envelope names the coordinate(s) "
            f"{unknown}, which the grid does not have "
            f"(coordinates: {em.grid.names})")
    omega, z = em.mode(s, k, phase=phase)
    masked = {}
    for c in _COMPONENTS:
        mask = sample_gaussian_mask(
            em.grid, z[c].function_space, mask_pos, mask_width)
        masked[c] = z[c] * mask
    return omega, mode_projection(em, s)(State(masked))


# ================================================================
#  Jets and eddies (balanced profile states)
# ================================================================
def barotropic_jet(
    source: Model | Eigenmodes | ChannelEigenmodes,
    *,
    wavenum: int = 5,
    waveamp: float = 0.1,
    jet_width: float = 0.04,
    geo_proj: bool = True,
    at_time: float = 0.0,
) -> State:
    r"""
    Two opposing barotropic zonal jets (``BarotropicJet``).

    Description
    -----------
    The reference barotropic instability setup: the zonal velocity

    .. math::
        u = 2.5 \left(
            \exp\left(-\left(\frac{y - 0.75 L_y}{\sigma L_y \pi}
            \right)^2\right)
            - \exp\left(-\left(\frac{y - 0.25 L_y}{\sigma L_y \pi}
            \right)^2\right)\right)

    sampled on ``u``'s own staggered nodes, with the meridional
    perturbation :math:`v = A \sin(2\pi k_p x / L_x)` on ``v``'s
    nodes; when ``geo_proj`` is set the summed state is projected
    onto the geostrophic (vortical) subspace.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an analytic eigenmodes object.
    wavenum : int, optional
        Zonal wavenumber of the perturbation (default: 5).
    waveamp : float, optional
        Amplitude of the perturbation (default: 0.1).
    jet_width : float, optional
        Relative width of the jets (default: 0.04).
    geo_proj : bool, optional
        Project onto the geostrophic subspace (default: True).
    at_time : float, optional
        Parameter evaluation time when resolving from a model
        (default: 0.0).

    Returns
    -------
    State
        The jet state (assign with ``model.set_state``).
    """
    em = _analytic(source, "barotropic_jet", at_time)
    grid = em.grid
    x, y = _horizontal(grid)
    x0, x1 = _extent(grid, x)
    y0, y1 = _extent(grid, y)
    lx, ly = x1 - x0, y1 - y0
    width = jet_width * ly * jnp.pi

    def u_profile(coords: dict[str, jax.Array]) -> jax.Array:
        yy = coords[y] - y0
        return 2.5 * (jnp.exp(-((yy - 0.75 * ly) / width) ** 2)
                      - jnp.exp(-((yy - 0.25 * ly) / width) ** 2))

    def v_profile(coords: dict[str, jax.Array]) -> jax.Array:
        kx = 2.0 * jnp.pi / lx * wavenum
        return waveamp * jnp.sin(kx * (coords[x] - x0))

    z = State({
        "u": _sample(grid, em.physical_space("u"), u_profile,
                     name="u"),
        "v": _sample(grid, em.physical_space("v"), v_profile,
                     name="v"),
        "w": grid.create_field(em.physical_space("w"), name="w"),
        "b": grid.create_field(em.physical_space("b"), name="b")})
    if geo_proj:
        z = VorticalProjection(em)(z)
    return z


def jet(
    source: Model | Eigenmodes | ChannelEigenmodes,
    *,
    jet_strength: float = 1.0,
    jet_width: float = 0.16,
    pert_strength: float = 0.05,
    pert_wavenum: int = 5,
    geo_proj: bool = True,
    at_time: float = 0.0,
) -> State:
    r"""
    Build a 3-D jet with horizontal and vertical shear (``Jet``).

    Description
    -----------
    The Chouksey et al. (2022) setup: two opposing zonal jets

    .. math::
        u = a \left(
            e^{-(y - 3 L_y / 4)^2 / \sigma^2}
            - e^{-(y - L_y / 4)^2 / \sigma^2}\right)
            \cos\left(\frac{2\pi z}{L_z}\right)

    sampled on ``u``'s own staggered nodes, plus the geostrophic
    single-mode perturbation ``em.mode(0, {x: k_p, y: 0, z: 0})``
    scaled by ``pert_strength``; when ``geo_proj`` is set the summed
    state is projected onto the geostrophic (vortical) subspace.
    For very large jet strengths convective instabilities can occur.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an analytic eigenmodes object.
    jet_strength : float, optional
        Amplitude of the jets (default: 1.0).
    jet_width : float, optional
        Absolute width of the jets (default: 0.16).
    pert_strength : float, optional
        Amplitude of the perturbation (default: 0.05).
    pert_wavenum : int, optional
        Zonal wavenumber of the perturbation (default: 5).
    geo_proj : bool, optional
        Project onto the geostrophic subspace (default: True).
    at_time : float, optional
        Parameter evaluation time when resolving from a model
        (default: 0.0).

    Returns
    -------
    State
        The jet state (assign with ``model.set_state``).
    """
    em = _analytic(source, "jet", at_time)
    grid = em.grid
    x, y = _horizontal(grid)
    y0, y1 = _extent(grid, y)
    z0, z1 = _extent(grid, _VERTICAL)
    ly, lz = y1 - y0, z1 - z0

    def u_profile(coords: dict[str, jax.Array]) -> jax.Array:
        yy = coords[y] - y0
        zz = coords[_VERTICAL] - z0
        jets = (jnp.exp(-((yy - 0.75 * ly) / jet_width) ** 2)
                - jnp.exp(-((yy - 0.25 * ly) / jet_width) ** 2))
        return (jet_strength * jets
                * jnp.cos(2.0 * jnp.pi * zz / lz))

    z = State({
        "u": _sample(grid, em.physical_space("u"), u_profile,
                     name="u"),
        "v": grid.create_field(em.physical_space("v"), name="v"),
        "w": grid.create_field(em.physical_space("w"), name="w"),
        "b": grid.create_field(em.physical_space("b"), name="b")})
    _, pert = em.mode(0, {x: pert_wavenum, y: 0, _VERTICAL: 0})
    z = z + pert_strength * pert
    if geo_proj:
        z = VorticalProjection(em)(z)
    return z


def coherent_eddy(
    source: Model | Eigenmodes | ChannelEigenmodes,
    *,
    pos_x: float = 0.5,
    pos_y: float = 0.5,
    width: float = 0.1,
    amplitude: float = 1.0,
    gauss_field: str = "streamfunction",
    at_time: float = 0.0,
) -> State:
    r"""
    Build a coherent barotropic Gaussian eddy (``CoherentEddy``).

    Description
    -----------
    The Gaussian bump

    .. math::
        G = A \exp\left(
        -\frac{(x - p_x L_x)^2 + (y - p_y L_y)^2}{(\sigma L_x)^2}
        \right)

    (constant in the vertical) prescribes either the streamfunction
    directly (``gauss_field="streamfunction"``, the default) or the
    vorticity (``gauss_field="vorticity"``), whose streamfunction
    follows from the horizontal spectral inversion
    :math:`\hat\psi = \hat\zeta / k_h^2` (zero-mean gauge, discrete
    operator symbols). The velocities are the discrete C-grid curl
    of the corner-sampled streamfunction (``u = \delta_y \psi``,
    ``v = -\delta_x \psi``, exactly divergence-free); the diagnostic
    pressure balances at the first solve.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an analytic eigenmodes object.
    pos_x : float, optional
        Relative zonal position of the eddy (default: 0.5).
    pos_y : float, optional
        Relative meridional position of the eddy (default: 0.5).
    width : float, optional
        Width of the eddy relative to the zonal domain size
        (default: 0.1).
    amplitude : float, optional
        Amplitude of the Gaussian; a negative amplitude flips the
        rotation sense (default: 1.0).
    gauss_field : str, optional
        Which field the Gaussian prescribes: ``"streamfunction"``
        or ``"vorticity"`` (default: "streamfunction").
    at_time : float, optional
        Parameter evaluation time when resolving from a model
        (default: 0.0).

    Returns
    -------
    State
        The eddy state.

    Raises
    ------
    ValueError
        On an unknown ``gauss_field``, a horizontally walled
        channel, or the vorticity inversion on a walled vertical
        (the corner kit needs a fully periodic grid).
    """
    if gauss_field not in {"vorticity", "streamfunction"}:
        raise ValueError(
            f"unknown gauss_field {gauss_field!r}: the Gaussian "
            "prescribes either 'vorticity' or 'streamfunction'")
    em = _analytic(source, "coherent_eddy", at_time)
    grid = em.grid
    x, y = _horizontal(grid)
    x0, x1 = _extent(grid, x)
    y0, y1 = _extent(grid, y)
    lx, ly = x1 - x0, y1 - y0
    u_space = em.physical_space("u")
    v_space = em.physical_space("v")
    corner = u_space.replace(**{y: v_space.factor(y)})

    def bump(coords: dict[str, jax.Array]) -> jax.Array:
        return amplitude * jnp.exp(
            -((coords[x] - x0 - pos_x * lx) ** 2
              + (coords[y] - y0 - pos_y * ly) ** 2)
            / (width * lx) ** 2)

    psi = _sample(grid, corner, bump, name="psi")
    if gauss_field == "vorticity":
        if em._walled:  # noqa: SLF001 — package-internal
            raise ValueError(
                "the spectral vorticity inversion needs a fully "
                "periodic grid (the corner streamfunction has no "
                "trig parity on a walled vertical); prescribe the "
                "streamfunction instead "
                "(gauss_field='streamfunction')")
        kit = GridSymbols(grid, {"psi": corner})
        kh2 = (kit.diff(x, on="psi").magnitude ** 2
               + kit.diff(y, on="psi").magnitude ** 2)
        coeff = kit.forward("psi")(psi)
        inverse = jnp.broadcast_to(
            kh2.inverse().data, coeff.data.shape)
        psi = kit.backward("psi")(
            coeff.with_data(coeff.data * inverse)).real
    return State({
        "u": psi.diff(y).with_metadata(name="u"),
        "v": (-psi.diff(x)).with_metadata(name="v"),
        "w": grid.create_field(em.physical_space("w"), name="w"),
        "b": grid.create_field(em.physical_space("b"), name="b")})
