r"""
Named analytic initial conditions (``sw.initial_conditions``).

Description
-----------
The shallow-water initial-condition factories: plain functions taking
a model (or an eigenmodes object) and returning a
:class:`~fridom.shallowwater2.state.State` the caller assigns via
``model.set_state(...)``.

Random-phase eigenmode states (the reference
``PrescribedSpectraRandomPhase`` family) compose one mode family's
eigenvectors with a prescribed spectral energy density and Hermitian
random phases, on **both** eigenmode tiers:

- fully periodic grids build on the analytic operator-sourced
  :class:`~fridom.shallowwater2.eigenmodes.Eigenmodes` (all modes at
  once, vectorized);
- walled channels build on the labeled numeric
  :class:`~fridom.shallowwater2.channel_eigenmodes.ChannelEigenmodes`
  (per-plane column synthesis).

Surface: :func:`random_state` (the general factory),
:func:`random_vortical` (the ``RandomVorticalSpectra`` equivalent,
defaulting to :func:`geostrophic_energy_spectrum`) and
:func:`random_waves` (the ``RandomWaveFlatSpectra`` equivalent — flat
wave spectra, plus the flat Kelvin family on a channel). Every
factory takes a model **or** an eigenmodes object and returns a
:class:`~fridom.shallowwater2.state.State`, normalized so the
largest horizontal-velocity value is one.

Named analytic states port the reference initial-condition classes:
:func:`single_wave` (one discrete eigenmode, a thin wrapper over
``em.mode``), :func:`jet` (two opposing zonal jets plus a geostrophic
single-mode perturbation), :func:`coherent_eddy` (a Gaussian
streamfunction or vorticity eddy in exact geostrophic balance) and
:func:`equatorial_wave` (the Hermite-Gaussian equatorial modes of
the beta plane). Wave factories return ``(omega, state)`` like
``em.mode``; profile factories return the state alone.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.model._eigenbasis import channel_random_state
from fridom.model.eigenstates import (
    geostrophic_energy_spectrum as geostrophic_energy_spectrum,  # noqa: PLC0414 — re-export
)
from fridom.model.eigenstates import (
    normalize_max_component,
    prescribed_spectra_coefficients,
)
from fridom.model.time_dependent import resolve_at
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.channel_eigenmodes import ChannelEigenmodes
from fridom.shallowwater2.eigenmodes import Eigenmodes, from_model
from fridom.shallowwater2.state import State
from fridom.shallowwater2.transforms import VorticalProjection
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.symbols import GridSymbols, ModeChart

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.model.model import Model
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.tensor_product import SpaceLike

#: The analytic-tier family vocabulary -> eigenmode branches.
_ANALYTIC_BRANCHES = {"vortical": (0,), "wave": (1, -1)}


def flat_spectrum(*wavenumbers: jax.Array) -> jax.Array:  # noqa: ARG001
    """Return a unit spectral energy density (the wave default)."""
    return jnp.asarray(1.0)


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
    :math:`S(k) = 2\pi k\,E(k, 0)` angular convention, so a shell
    of modes realizes the 1-D energy density :math:`S` — and
    multiplied by a Hermitian unit-modulus random phase; the sum
    synthesizes to a real physical state, normalized so the largest
    horizontal-velocity value is one. ``spectral_energy_density``
    is called with one wavenumber array per grid axis (grid order);
    on a channel the bounded axis carries the effective meridional
    wavenumber :math:`k_y = \pi m / L_y` of each mode's
    within-family ordinal.

    On the fully periodic grid the families are ``"vortical"`` and
    ``"wave"`` (both inertia-gravity branches, independent phases);
    on a channel any labeled selection of
    ``sw.ChannelEigenmodes.families`` applies (unsigned roots cover
    both signed branches — the realized real state is closed under
    conjugation, so signed selections spread over the mirror
    branch). On even grid sizes the vortical family includes the
    interpolation-Nyquist steady strata (the divergence-free modes
    that joined the geostrophic column), so a ``"vortical"`` state
    populates the Nyquist planes like every other mode.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model, or an eigenmodes object of either
        tier.
    family : str
        The mode family selection.
    spectral_energy_density : Callable[..., jax.Array] | None, optional
        ``S(*k)`` over the grid-axis wavenumbers; None selects
        :func:`geostrophic_energy_spectrum` for ``"vortical"`` and
        :func:`flat_spectrum` otherwise (default: None).
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
            geostrophic_energy_spectrum if family == "vortical"
            else flat_spectrum)
    if isinstance(em, ChannelEigenmodes):
        fields = channel_random_state(
            em, family, spectral_energy_density, seed=seed,
            horizontal=tuple(em.grid.names))
        return State(fields)
    branches = _ANALYTIC_BRANCHES.get(family)
    if branches is None:
        if family.startswith("kelvin"):
            raise ValueError(
                "no walls, no Kelvin family: Kelvin modes are "
                "boundary-trapped, and the fully periodic "
                "shallow-water eigenmodes carry only the vortical "
                "and inertia-gravity branches. Build the model on "
                "a channel grid (exactly one bounded axis) to get "
                "the labeled Kelvin pair")
        raise ValueError(
            f"unknown mode family {family!r}: the fully periodic "
            "families are 'vortical' and 'wave'")
    fields = prescribed_spectra_coefficients(
        grid=em.grid,
        kit=em._kit,  # noqa: SLF001 — package-internal kit access
        chart=ModeChart(em.grid),
        columns=tuple(em.q(s) for s in branches),
        components=("u", "v", "p"),
        weights=em._energy_weights(),  # noqa: SLF001 — see above
        reference="u",
        horizontal=tuple(em.grid.names),
        spectral_energy_density=spectral_energy_density,
        seed=seed)
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
    :func:`geostrophic_energy_spectrum` default.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an eigenmodes object.
    spectral_energy_density : Callable[..., jax.Array] | None, optional
        The prescribed spectrum (default: None, the geostrophic
        spectrum).
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
            f"{what} builds on the analytic eigenmodes of the "
            "fully periodic grid; this model is a walled channel — "
            "synthesize labeled channel modes through the "
            "eigenbasis instead, e.g. "
            "sw.eigenbasis(model).mode('kelvin', indices, "
            "branch=+1)")
    return em


def _extent(grid: Grid, name: str) -> tuple[float, float]:
    """Physical interval ``(min, max)`` of the axis ``name``."""
    mesh = next(m for m in grid.factors if name in m.names)
    return float(mesh.extent[0]), float(mesh.extent[1])


def _spaces(grid: Grid) -> dict[str, SpaceLike]:
    """Canonical C-grid component spaces of the periodic grid."""
    x, y = grid.names
    return {
        "u": fr.spatial.Staggered(x).resolve(grid),
        "v": fr.spatial.Staggered(y).resolve(grid),
        "p": fr.spatial.Collocated().resolve(grid),
    }


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


def _invert_laplacian(grid: Grid, field: ScalarField) -> ScalarField:
    r"""Return ``psi`` with ``psi_hat = field_hat / k_h^2``.

    Description
    -----------
    The reference streamfunction-from-vorticity inversion, assembled
    from the grid's own operator symbols: the squared staggered-
    derivative magnitudes take the place of the continuous
    :math:`k_h^2`, and ``Symbol.inverse`` maps the :math:`k = 0`
    structural zero to zero (the zero-mean gauge, matching the
    reference's masked division).
    """
    kit = GridSymbols(grid, {"psi": field.function_space.bare})
    x, y = grid.names
    k2 = (kit.diff(x, on="psi").magnitude ** 2
          + kit.diff(y, on="psi").magnitude ** 2)
    coeff = kit.forward("psi")(field)
    inverse = jnp.broadcast_to(
        k2.inverse().data, coeff.data.shape)
    return kit.backward("psi")(
        coeff.with_data(coeff.data * inverse)).real


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
    ``phase + omega * t``.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an analytic eigenmodes object.
    k : Mapping[str, int]
        Axis-keyed integer wavenumber indices (e.g.
        ``{"x": 3, "y": 0}``); a wavenumber of one is a wave with
        one wavelength across the domain.
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
        On a walled channel (labeled modes come from
        ``sw.eigenbasis``), bad indices, or a structurally
        unrepresented mode.
    """
    em = _analytic(source, "single_wave", at_time)
    return em.mode(s, k, phase=phase)


# ================================================================
#  Jets and eddies (balanced profile states)
# ================================================================
def jet(
    source: Model | Eigenmodes | ChannelEigenmodes,
    *,
    wavenum: int = 2,
    waveamp: float = 0.1,
    pos: float = 0.5,
    width: float = 0.1,
    geo_proj: bool = True,
    at_time: float = 0.0,
) -> State:
    r"""
    Build an unstable zonal jet with a perturbation (``Jet``).

    Description
    -----------
    The reference shallow-water jet: the zonal velocity profile

    .. math::
        u = \exp\left(
            -\left(\frac{y - p L_y}{\sigma L_y}\right)^2\right)

    sampled on ``u``'s own staggered nodes, projected onto the
    geostrophic (vortical) subspace when ``geo_proj`` is set and
    normalized so the largest horizontal velocity is one; on top a
    geostrophic single-mode perturbation ``em.mode(0, {x: wavenum,
    y: 0})`` scaled by ``waveamp``.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an analytic eigenmodes object.
    wavenum : int, optional
        Zonal wavenumber of the perturbation (default: 2).
    waveamp : float, optional
        Amplitude of the perturbation (default: 0.1).
    pos : float, optional
        Relative meridional position of the jet (default: 0.5).
    width : float, optional
        Relative width of the jet (default: 0.1).
    geo_proj : bool, optional
        Project the jet onto the geostrophic subspace
        (default: True).
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
    x, y = grid.names
    y0, y1 = _extent(grid, y)
    ly = y1 - y0
    spaces = _spaces(grid)

    def profile(coords: dict[str, jax.Array]) -> jax.Array:
        return jnp.exp(
            -((coords[y] - y0 - pos * ly) / (width * ly)) ** 2)

    z = State({
        "u": _sample(grid, spaces["u"], profile, name="u"),
        "v": grid.create_field(spaces["v"], name="v"),
        "p": grid.create_field(spaces["p"], name="p")})
    if geo_proj:
        z = VorticalProjection(em)(z)
    z = State(normalize_max_component(
        dict(z.components), ("u", "v")))
    _, wave = em.mode(0, {x: wavenum, y: 0})
    return z + waveamp * wave


def coherent_eddy(
    source: Model | Eigenmodes | ChannelEigenmodes,
    *,
    pos_x: float = 0.5,
    pos_y: float = 0.5,
    width: float = 0.1,
    amplitude: float = 1.0,
    gauss_field: str = "vorticity",
    at_time: float = 0.0,
) -> State:
    r"""
    Build a balanced coherent Gaussian eddy (``CoherentEddy``).

    Description
    -----------
    The Gaussian bump

    .. math::
        G = A \exp\left(
        -\frac{(x - p_x L_x)^2 + (y - p_y L_y)^2}{(\sigma L_x)^2}
        \right)

    prescribes either the streamfunction directly
    (``gauss_field="streamfunction"``) or the vorticity
    (``gauss_field="vorticity"``, the default), whose
    streamfunction follows from the spectral inversion
    :math:`\hat\psi = \hat\zeta / k_h^2` (zero-mean gauge, discrete
    operator symbols). The velocities are the discrete C-grid curl
    of the corner-sampled streamfunction (``u = -\delta_y \psi``,
    ``v = \delta_x \psi``, exactly divergence-free) and the pressure
    is the geostrophic ``p = f_0 \psi`` on the cell centres.

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
        Which field the Gaussian prescribes: ``"vorticity"`` or
        ``"streamfunction"`` (default: "vorticity").
    at_time : float, optional
        Parameter evaluation time when resolving from a model
        (default: 0.0).

    Returns
    -------
    State
        The balanced eddy state.

    Raises
    ------
    ValueError
        On an unknown ``gauss_field`` or a walled channel.
    """
    if gauss_field not in {"vorticity", "streamfunction"}:
        raise ValueError(
            f"unknown gauss_field {gauss_field!r}: the Gaussian "
            "prescribes either 'vorticity' or 'streamfunction'")
    em = _analytic(source, "coherent_eddy", at_time)
    grid = em.grid
    x, y = grid.names
    x0, x1 = _extent(grid, x)
    y0, y1 = _extent(grid, y)
    lx, ly = x1 - x0, y1 - y0
    spaces = _spaces(grid)
    corner = spaces["u"].replace(**{y: spaces["v"].factor(y)})

    def bump(coords: dict[str, jax.Array]) -> jax.Array:
        return amplitude * jnp.exp(
            -((coords[x] - x0 - pos_x * lx) ** 2
              + (coords[y] - y0 - pos_y * ly) ** 2)
            / (width * lx) ** 2)

    psi_corner = _sample(grid, corner, bump, name="psi")
    psi_centre = _sample(grid, spaces["p"], bump, name="psi")
    if gauss_field == "vorticity":
        psi_corner = _invert_laplacian(grid, psi_corner)
        psi_centre = _invert_laplacian(grid, psi_centre)
    return State({
        "u": (-psi_corner.diff(y)).with_metadata(name="u"),
        "v": psi_corner.diff(x).with_metadata(name="v"),
        "p": (em.f0 * psi_centre).with_metadata(name="p")})


# ================================================================
#  Equatorial waves on the beta plane
# ================================================================
def _hermite_gaussian(order: int, xs: jax.Array) -> jax.Array:
    r"""Return the Hermite-Gaussian ``H_order(xs) exp(-xs^2 / 2)``.

    The Hermite polynomials follow the recurrence
    :math:`H_m = 2 x H_{m-1} - 2 (m - 1) H_{m-2}` with
    :math:`H_{-1} = 0` and :math:`H_0 = 1`.
    """
    if order < 0:
        return jnp.zeros_like(xs)
    previous = jnp.zeros_like(xs)
    current = jnp.ones_like(xs)
    for m in range(1, order + 1):
        previous, current = current, (
            2.0 * xs * current - 2.0 * (m - 1) * previous)
    return current * jnp.exp(-xs ** 2 / 2.0)


def _beta_plane_parameters(
    model: Model, at_time: float,
) -> tuple[float, float]:
    """Read the validated ``(beta, csqr)`` of a beta-plane model."""
    view = model.parameters
    for name, why in (
        (fr.model.params.CORIOLIS_BETA,
         "the equatorial beta plane (assemble with "
         "fr.model.modules.BetaPlaneCoriolis)"),
        (sw_params.CSQR,
         "a constant squared phase speed (assemble with a "
         "constant-depth DynamicalCore)"),
    ):
        if name not in view:
            raise ValueError(
                f"equatorial waves need {why}: no {name!r} "
                "provider on this model")
    beta = float(resolve_at(view[fr.model.params.CORIOLIS_BETA], at_time))
    csqr = float(resolve_at(view[sw_params.CSQR], at_time))
    if beta <= 0.0 or csqr <= 0.0:
        raise ValueError(
            "the equatorial Rossby radius sqrt(c / beta) needs "
            f"beta > 0 and csqr > 0; got beta={beta}, csqr={csqr}")
    return beta, csqr


def equatorial_wave(
    model: Model,
    longitudinal_mode: int,
    equatorial_mode: int,
    wave_mode: int,
    *,
    phase: float = 0.0,
    equator: float | None = None,
    at_time: float = 0.0,
) -> tuple[float, State]:
    r"""
    Build a Hermite-Gaussian equatorial wave (``EquatorialWave``).

    Description
    -----------
    The analytic equatorial wave of the linearized shallow-water
    equations on the equatorial beta plane :math:`f = \beta \tilde
    y`: the frequency of the ``equatorial_mode``-th meridional mode
    solves the cubic

    .. math::
        \omega_m \left(\omega_m^2 - c^2
            \left(k^2 + (2m + 1)\frac{\beta}{c}\right)\right)
        = k \beta c^2

    (``wave_mode`` indexes the ascending-sorted roots: 0 the
    negative-frequency gravity wave, 1 the Rossby wave, 2 the
    positive-frequency gravity wave), and the meridional structures
    are the Hermite-Gaussian functions of
    :math:`\tilde y = (y - y_0)/R_e` with the equatorial Rossby
    radius :math:`R_e = \sqrt{c / \beta}`, sampled on each
    component's own staggered nodes. The state is normalized so the
    largest horizontal velocity is one and evolves in the linear
    model as the same wave at phase ``phase + omega * t``.

    Parameters
    ----------
    model : Model
        The assembled beta-plane shallow-water model.
    longitudinal_mode : int
        Zonal wavenumber (wavelengths across the domain).
    equatorial_mode : int
        Meridional mode (the Hermite polynomial order, >= 0).
    wave_mode : int
        Root selection: 0 (negative-frequency gravity), 1 (Rossby)
        or 2 (positive-frequency gravity).
    phase : float, optional
        The wave phase shift (default: 0.0).
    equator : float | None, optional
        Physical ``y`` position of the equator; None centres it in
        the domain (default: None).
    at_time : float, optional
        Parameter evaluation time (default: 0.0).

    Returns
    -------
    tuple[float, State]
        The frequency and the normalized wave state.

    Raises
    ------
    ValueError
        On a walled zonal axis, a missing beta-plane / phase-speed
        parameter, or an out-of-range mode selection.
    """
    if equatorial_mode < 0:
        raise ValueError(
            "the equatorial (Hermite) mode is a non-negative "
            f"polynomial order; got {equatorial_mode}")
    if wave_mode not in {0, 1, 2}:
        raise ValueError(
            "wave_mode indexes the three sorted dispersion roots: "
            "0 (negative-frequency gravity), 1 (Rossby), 2 "
            f"(positive-frequency gravity); got {wave_mode}")
    grid = model.grid
    x, y = grid.names
    if not next(m for m in grid.factors if x in m.names).periodic:
        raise ValueError(
            f"equatorial waves travel along the zonal axis {x!r}, "
            "which this grid bounds with walls; equatorial waves "
            "need a periodic zonal axis")
    beta, csqr = _beta_plane_parameters(model, at_time)

    x0, x1 = _extent(grid, x)
    y0, y1 = _extent(grid, y)
    lx = x1 - x0
    speed = csqr ** 0.5
    radius = (speed / beta) ** 0.5
    y_eq = 0.5 * (y0 + y1) if equator is None else float(equator)
    kx = 2.0 * jnp.pi / lx * longitudinal_mode

    m = equatorial_mode
    eigenvalue = (2 * m + 1) / radius ** 2
    coeffs = [1.0, 0.0, -csqr * (float(kx) ** 2 + eigenvalue),
              -float(kx) * beta * csqr]
    omega = float(np.sort(np.roots(coeffs).real)[wave_mode])

    def structure(
        component: str, coords: dict[str, jax.Array],
    ) -> jax.Array:
        y_star = (coords[y] - y_eq) / radius
        if component == "v":
            return _hermite_gaussian(m, y_star)
        plus = _hermite_gaussian(m + 1, y_star)
        minus = _hermite_gaussian(m - 1, y_star)
        sign = 1.0 if component == "u" else -1.0
        scale = speed if component == "u" else csqr
        return 1j * scale / (2.0 * radius) * (
            plus / (omega - kx * speed)
            + sign * 2.0 * m * minus / (omega + kx * speed))

    spaces = _spaces(grid)

    def wave(component: str) -> ScalarField:
        def values(coords: dict[str, jax.Array]) -> jax.Array:
            # e^{i(kx x - phase)}: d/dt state(phase) equals
            # omega * state(phase + pi/2), the em.mode convention
            carrier = jnp.exp(1j * (kx * coords[x] - phase))
            return (structure(component, coords) * carrier).real
        return _sample(grid, spaces[component], values,
                       name=component)

    fields = {c: wave(c) for c in ("u", "v", "p")}
    return omega, State(
        normalize_max_component(fields, ("u", "v")))
