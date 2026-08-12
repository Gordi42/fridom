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
analytic mode accessor ``em.mode``, the latter with a
coordinate-named envelope callable — :func:`gaussian`
builds the common case — a re-projection onto the mode branch, and
optional single-sided ``traveling=`` carriers on bounded axes),
:func:`kelvin_wave` (the labeled boundary-trapped mode of the walled
channel eigenbasis), :func:`barotropic_jet` / :func:`jet` (unstable
zonal jets plus a single-mode perturbation, geostrophically
projected by default) and :func:`coherent_eddy` (a Gaussian
streamfunction or vorticity eddy with a caller-supplied vertical
structure, turned into ``u``, ``v``, ``b`` by geostrophic theory).
Wave factories return ``(omega, state)`` like ``em.mode``; profile
factories return the state alone.

:func:`coherent_eddy` is the one factory that takes the **model**
alone: it needs the component spaces and the Coriolis parameter, not
an eigenbasis, so it serves every grid topology (the horizontally
walled channel included) and builds no eigenmodes.
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.model._eigenbasis import (
    _resolve_mode_family,
    channel_random_state,
)
from fridom.model.eigenstates import (
    geostrophic_energy_spectrum as geostrophic_energy_spectrum,  # noqa: PLC0414 — re-export
)
from fridom.model.eigenstates import (
    normalize_max_component,
    pattern_axes,
    prescribed_spectra_coefficients,
    sample_pattern,
    traveling_carrier,
)
from fridom.model.params import (
    CORIOLIS_F0,
    CORIOLIS_METRIC_RATIO,
    CORIOLIS_ROSSBY,
    SCALING_NONLINEARITY,
)
from fridom.model.shapes import (
    gaussian as gaussian,  # noqa: PLC0414 — re-export
)
from fridom.model.time_dependent import resolve_at
from fridom.nonhydro2.channel_eigenmodes import ChannelEigenmodes
from fridom.nonhydro2.eigenmodes import Eigenmodes, from_model
from fridom.nonhydro2.state import State
from fridom.nonhydro2.transforms import (
    VorticalProjection,
    mode_projection,
)
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
            "nh.eigenbasis(model).mode(family, mode_number), or "
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
    mode_number: Mapping[str, int],
    family: str = "wave+",
    *,
    branch: int | None = None,
    phase: float = 0.0,
    at_time: float = 0.0,
) -> tuple[float, State]:
    r"""
    One discrete eigenmode as ``(omega, state)`` (``SingleWave``).

    Description
    -----------
    The thin wrapper over the analytic mode accessor
    ``em.mode(family, mode_number, phase=...)``: the real Hermitian-closed
    physical mode
    :math:`\mathrm{Re}(q^s(k)\,e^{i(k\cdot x - \mathrm{phase})})`
    with exact discrete dispersion, normalized so the largest
    horizontal-velocity envelope is one. Under the linear model the
    state at time :math:`t` is the same mode at phase
    ``phase + omega * t``, so positive ``omega`` propagates along
    ``+k``: ``"wave+"`` is the positive-frequency branch, moving
    with the wavevector (eastward for positive ``kx``), ``"wave-"``
    the mirror branch, ``"vortical"`` the geostrophic one. On a
    walled vertical the ``z`` mode number is the physical vertical mode
    on the ``0..n`` union lattice.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an analytic eigenmodes object.
    mode_number : Mapping[str, int]
        Axis-keyed integer mode numbers (e.g.
        ``{"x": 3, "y": 0, "z": 2}``); mode number one is a
        wave with one wavelength across the domain.
    family : str, optional
        The labeled mode family: ``"vortical"``, ``"wave+"`` /
        ``"wave-"``, or the unsigned root ``"wave"`` with
        ``branch=`` (default: "wave+").
    branch : int | None, optional
        The signed branch (+1 / -1) of an unsigned family root
        (default: None).
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
        ``nh.eigenbasis``), bad mode numbers, or a structurally
        unrepresented mode.
    """
    em = _analytic(source, "single_wave", at_time)
    return em.mode(family, mode_number, branch=branch, phase=phase)


def kelvin_wave(
    source: Model | Eigenmodes | ChannelEigenmodes,
    mode_number: Mapping[str, int],
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
    ``eb.mode("kelvin", mode_number, branch=...)``: the numerically exact
    discrete Kelvin mode trapped at the channel walls, normalized so
    the largest horizontal-velocity envelope is one. ``branch``
    selects the signed frequency branch (the two branches are
    trapped at opposite walls; ``branch=+1`` is the positive-omega,
    eastward-propagating branch); the bounded-axis entry of ``mode_number`` is
    the within-family mode ordinal and defaults to 0 (the
    fundamental) when absent.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled channel model or its labeled eigenbasis.
    mode_number : Mapping[str, int]
        Axis-keyed mode numbers: integer wavenumbers on the periodic
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
    numbers = dict(mode_number)
    numbers.setdefault(em.bounded_axis, 0)
    return em.mode("kelvin", numbers, branch=branch, phase=phase)


def wave_package(
    source: Model | Eigenmodes | ChannelEigenmodes,
    mode_number: Mapping[str, int],
    family: str = "wave+",
    *,
    branch: int | None = None,
    envelope: Callable[..., jax.Array],
    traveling: Mapping[str, int] | None = None,
    phase: float = 0.0,
    at_time: float = 0.0,
    quadrature: bool = False,
) -> tuple[float, State]:
    r"""
    Build an enveloped single-mode wave packet (``WavePackage``).

    Description
    -----------
    The single carrier mode of :func:`single_wave` multiplied by the
    stationary envelope :math:`E(\boldsymbol{x})` — a callable whose
    signature names the coordinates it varies along (unnamed axes
    stay constant), sampled at each component's own staggered nodes;
    :func:`gaussian` builds the common Gaussian case — and
    re-projected onto the carrier's mode family so the packet stays
    polarized. The returned frequency is the carrier mode's.

    On a bounded (walled) axis a single mode is standing, so an
    envelope localized along it holds both running directions and
    the packet splits into two counter-propagating beams.
    ``traveling`` selects one side instead: per named bounded axis
    the standing carrier is replaced by the running-wave carrier
    whose **envelope drift** (group velocity) has the given sign —
    ``traveling={"z": -1}`` sinks, ``+1`` rises, whatever the phase
    tilt does. On periodic axes the sign of the carrier entry in
    ``mode_number`` already selects the direction. Validity window: the
    envelope should be smooth, several carrier wavelengths wide,
    and well inside the domain (its tails small at the walls).

    ``quadrature=True`` returns the **complex** packet :math:`Q`
    instead of the real one, so the packet can drive a
    :class:`~fridom.model.modules.Source` directly. :math:`Q` is the
    analytic signal of the packet along the mode's own time
    evolution: with :math:`\omega` the returned carrier frequency it
    satisfies, for every :math:`t`,

    .. math::
        \mathrm{Re}\!\left[Q\,e^{-i\omega t}\right]
            = \texttt{wave\_package(..., phase=phase} + \omega t
              \texttt{)}

    (the real packet phase-advanced by :math:`\omega t`, which is the
    single mode's linear time evolution — ``single_wave``: the state
    at time :math:`t` is the same packet at phase
    ``phase + omega * t``). It is built from the phase quadrature pair
    :math:`Q = z(\varphi) + i\,z(\varphi + \pi/2)` (``z`` the real
    packet at ``phase`` :math:`\varphi`), which is exact for every
    tier and path — periodic, standing walled, and ``traveling=`` —
    because the family projection is a real operator and the mode is
    :math:`\mathrm{Re}[W(\boldsymbol{x})\,e^{-i\varphi}]`. Feed it to a
    factor-free harmonic source as

    .. code-block:: python

        omega, packet = nh.initial_conditions.wave_package(
            model, mode_number, envelope=env, quadrature=True)
        src = fr.model.modules.Source(
            "packet", pattern=packet,
            law=fr.model.Harmonic(
                amplitude=A, frequency=omega / (2 * jnp.pi)))

    so the forcing runs at the resonant :math:`\omega`, with the
    frequency now an ordinary sweepable ``source.packet.frequency``
    (deliberate detuning off resonance is a one-line change).

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an analytic eigenmodes object.
    mode_number : Mapping[str, int]
        Axis-keyed integer mode numbers of the carrier.
    family : str, optional
        The carrier's labeled mode family: ``"vortical"``,
        ``"wave+"`` / ``"wave-"``, or the unsigned root ``"wave"``
        with ``branch=`` (default: "wave+").
    branch : int | None, optional
        The signed branch (+1 / -1) of an unsigned family root
        (default: None).
    envelope : Callable[..., jax.Array]
        The coordinate-named envelope callable (e.g.
        ``lambda x, z: ...`` or :func:`gaussian`).
    traveling : Mapping[str, int] | None, optional
        Axis-keyed envelope drift signs (+1 / -1) along bounded
        axes; None keeps the standing carrier (default: None).
    phase : float, optional
        The carrier phase shift (default: 0.0).
    at_time : float, optional
        Parameter evaluation time when resolving from a model
        (default: 0.0).
    quadrature : bool, optional
        Return the complex analytic-signal packet :math:`Q` (for a
        complex :class:`~fridom.model.modules.Source` pattern) instead
        of the real packet (default: False).

    Returns
    -------
    tuple[float, State]
        The carrier frequency and the enveloped state — real for
        ``quadrature=False``, the complex analytic-signal packet
        :math:`Q` for ``quadrature=True``.

    Raises
    ------
    ValueError
        On an envelope coordinate the grid does not have, a
        horizontally walled channel, or a bad ``traveling``
        selection: a periodic axis (use the sign of ``mode_number``), an
        axis the envelope does not name, a drift sign outside
        ``{+1, -1}``, the non-propagating vortical family, or a
        zero carrier mode number along a traveling axis.
    """
    em = _analytic(source, "wave_package", at_time)
    axes = pattern_axes(envelope, tuple(em.grid.names),
                        "wave-package")
    name = _resolve_mode_family(em, family, branch)
    if traveling:
        unnamed = sorted(set(traveling) - set(axes))
        if unnamed:
            raise ValueError(
                f"traveling names the axis/axes {unnamed}, but "
                "the envelope does not vary along them — a "
                "single-sided packet needs a localized envelope "
                "along its traveling axis (declare them in the "
                "envelope signature)")
        if em.families[name] == 0:
            raise ValueError(
                "the vortical family does not propagate "
                "(omega = 0): traveling= applies to the wave "
                "branches")
    project = mode_projection(em, em.families[name])

    def real_packet(ph: float) -> tuple[float, State]:
        """Return the real enveloped, re-projected packet at phase ``ph``."""
        if traveling:
            omega, carrier = traveling_carrier(
                em, name, mode_number, components=_COMPONENTS,
                traveling=traveling, phase=ph)
        else:
            omega, z = em.mode(name, mode_number, phase=ph)
            carrier = {c: z[c] for c in _COMPONENTS}
        enveloped = {
            c: carrier[c] * sample_pattern(
                em.grid, carrier[c].function_space, envelope, axes)
            for c in _COMPONENTS}
        return omega, project(State(enveloped))

    if not quadrature:
        return real_packet(phase)
    # the analytic-signal packet Q = z(phase) + i z(phase + pi/2), so
    # Re[Q e^{-i omega t}] = z(phase + omega t) (the mode's own time
    # evolution) — for a complex, factor-free Source pattern (SRC-D7).
    omega, real = real_packet(phase)
    _, imag = real_packet(phase + 0.5 * jnp.pi)
    return omega, State(
        {c: real[c] + 1j * imag[c] for c in _COMPONENTS})


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
    single-mode perturbation ``em.mode("vortical",
    {x: k_p, y: 0, z: 0})``
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
    _, pert = em.mode(
        "vortical", {x: pert_wavenum, y: 0, _VERTICAL: 0})
    z = z + pert_strength * pert
    if geo_proj:
        z = VorticalProjection(em)(z)
    return z


# ================================================================
#  Coherent eddies (geostrophic streamfunction states)
# ================================================================
def _face_factor(grid: Grid, name: str) -> SpaceLike:
    """Return the factor holding **all** faces of the axis ``name``.

    Description
    -----------
    The staggered face node set of the axis: ``Right`` on a periodic
    mesh (n faces, the wall-free lattice the prognostic ``w`` lives
    on) and ``Outer`` on a bounded one (n + 1 faces, the two boundary
    faces **included** — unlike the model's own wall-normal velocity
    space, which drops them under its Dirichlet tag). Both difference
    onto the n cell centres, which is what the buoyancy needs.
    """
    mesh = next(m for m in grid.factors if name in m.names)
    return mesh.right if mesh.periodic else mesh.outer


def _constant_factor(grid: Grid, name: str) -> SpaceLike:
    """Return the one-DOF broadcast factor of the axis ``name``."""
    mesh = next(m for m in grid.factors if name in m.names)
    return mesh.constant


def _coriolis_parameter(model: Model, at_time: float) -> float:
    r"""
    Read the model's **constant** Coriolis parameter.

    Description
    -----------
    The effective :math:`f` of the assembly, from the variant's own
    primitives in ``model.parameters`` (the nondimensionalization
    re-key, matching :func:`~fridom.nonhydro2.eigenmodes.eigenbasis`):
    the dimensional ``coriolis.f0``, or the nondimensional
    :math:`\varepsilon/\mathrm{Ro}`. A time-dependent leaf is frozen
    at ``at_time``.

    Parameters
    ----------
    model : Model
        The assembled nonhydrostatic model.
    at_time : float
        Parameter evaluation time.

    Returns
    -------
    float
        The constant Coriolis parameter.

    Raises
    ------
    ValueError
        When the model provides no constant rotation (no Coriolis
        module at all, or a beta-plane / chart rotation, whose ``f``
        is a field rather than a scalar).
    """
    params = model.parameters

    def read(name: str) -> float:
        value = resolve_at(params[name], at_time)
        return float(value)

    if (CORIOLIS_ROSSBY in params
            and CORIOLIS_METRIC_RATIO not in params):
        return read(SCALING_NONLINEARITY) / read(CORIOLIS_ROSSBY)
    if CORIOLIS_F0 in params:
        return read(CORIOLIS_F0)
    raise ValueError(
        "a baroclinic eddy needs a constant Coriolis parameter (the "
        "buoyancy is the geostrophic f * d_z psi), but this model "
        "provides none: pass a constant rotation "
        "(nh.FPlaneCoriolis(f0=...) / FPlaneCoriolis(rossby_number="
        "...)) — a beta-plane or chart rotation carries f as a field, "
        "for which no exactly balanced buoyancy exists — or drop "
        "vertical_structure= for the barotropic eddy, which needs no "
        "rotation at all")


def _streamfunction_from_vorticity(
    grid: Grid, zeta: ScalarField,
) -> ScalarField:
    r"""
    Invert the **horizontal** Laplacian, :math:`\nabla_h^2\psi = \zeta`.

    Description
    -----------
    The streamfunction of a prescribed relative vorticity on the
    horizontal corner, in the zero-mean gauge. The spectral route
    assembles :math:`k_h^2` from the grid's own staggered-derivative
    symbols (so the inversion is the exact inverse of the discrete
    ``rel_vort_z`` the velocities go on to carry) and divides,
    :math:`\hat\psi = -\hat\zeta / k_h^2`; ``Symbol.inverse`` maps the
    :math:`k = 0` structural zero to zero.

    The operand is **two-dimensional** — a horizontal corner field
    with a one-DOF (``Constant``) vertical factor — because the
    vertical structure multiplies the streamfunction *after* the
    inversion. The vertical topology therefore plays no part here; a
    walled vertical is served exactly like a periodic one.

    .. note::

        This is the **seam** for the general (walled-horizontal)
        solver. The Fourier route below needs both horizontal axes
        periodic; a horizontally walled domain needs the inversion of
        the discrete corner Laplacian with the wall gauge
        (:math:`\psi` constant along a solid boundary), which is a
        separate solver. Everything above and below this function is
        topology-blind: replacing this body with that solver is the
        whole change.

        Mind the sign at the swap. A general solver is naturally
        written for the positive-definite operator
        :math:`-\nabla_h^2`, so this function is **minus** it:
        ``_streamfunction_from_vorticity(grid, zeta)`` is
        ``-invert_negative_laplacian(grid, zeta, axes=(x, y))``. The
        sign belongs here, with the curl convention it pairs with, not
        inside the solver.

    Parameters
    ----------
    grid : Grid
        The grid mediating the transforms and wavenumbers.
    zeta : ScalarField
        The prescribed relative vorticity on the horizontal corner.

    Returns
    -------
    ScalarField
        The streamfunction on the same space.

    Raises
    ------
    ValueError
        On a horizontally walled grid (the general solver's seam).
    """
    x, y = _horizontal(grid)
    walled = [name for name in (x, y)
              if not next(m for m in grid.factors
                          if name in m.names).periodic]
    if walled:
        raise ValueError(
            "the spectral vorticity inversion needs both horizontal "
            f"axes periodic; this grid bounds {walled!r} — prescribe "
            "the streamfunction instead (gauss_field="
            "'streamfunction'), which is topology-blind, or invert "
            "the vorticity yourself and pass the result")
    kit = GridSymbols(grid, {"psi": zeta.function_space.bare})
    kh2 = (kit.diff(x, on="psi").magnitude ** 2
           + kit.diff(y, on="psi").magnitude ** 2)
    coeff = kit.forward("psi")(zeta)
    inverse = jnp.broadcast_to(kh2.inverse().data, coeff.data.shape)
    return kit.backward("psi")(
        coeff.with_data(-coeff.data * inverse)).real


def coherent_eddy(
    model: Model,
    *,
    pos_x: float = 0.5,
    pos_y: float = 0.5,
    width: float = 0.1,
    amplitude: float = 1.0,
    gauss_field: str = "streamfunction",
    vertical_structure: Callable[[jax.Array], jax.Array] | None = None,
    at_time: float = 0.0,
) -> State:
    r"""
    Build a geostrophic Gaussian eddy (``CoherentEddy``).

    Description
    -----------
    A separable streamfunction :math:`\psi(x, y, z) = G(x, y)\,F(z)`
    turned into ``u``, ``v`` and ``b`` by geostrophic theory. The
    horizontal shape is the Gaussian bump

    .. math::
        G = A \exp\left(
        -\frac{(x - p_x L_x)^2 + (y - p_y L_y)^2}{(\sigma L_x)^2}
        \right)

    which prescribes either the streamfunction directly
    (``gauss_field="streamfunction"``, the default) or the relative
    vorticity (``gauss_field="vorticity"``), whose streamfunction
    follows from the horizontal inversion
    :math:`\nabla_h^2 \psi = \zeta`. The vertical structure
    :math:`F(z)` is the caller's callable, evaluated on the **physical**
    vertical coordinate; the default is the constant 1, the barotropic
    eddy.

    The state is the geostrophic and hydrostatic one,

    .. math::
        u = -\partial_y \psi , \qquad
        v = \partial_x \psi , \qquad
        w = 0 , \qquad
        p = f\,\psi , \qquad
        b = \partial_z p = f\,\partial_z \psi ,

    with :math:`f` the model's constant Coriolis parameter (the
    dimensional ``coriolis.f0``, or the nondimensional
    :math:`\varepsilon/\mathrm{Ro}`). The pressure is not part of the
    returned state -- it is the diagnostic the projection solves for,
    and the first solve reproduces :math:`f\,\psi` interpolated onto
    the cell centres. Neither :math:`N^2` nor the
    aspect ratio :math:`\delta` enters: the nondimensional
    :math:`w`-equation is :math:`\delta^2 \partial_t w = b -
    \partial_z p`, so hydrostatic balance is the plain
    :math:`b = \partial_z p` on every scaling. There is **no**
    projection onto the vortical eigenmodes — see the note below on
    when to add one.

    **Sign convention.** :math:`\psi` is the standard geostrophic
    streamfunction, so a positive ``amplitude`` with
    ``gauss_field="streamfunction"`` is a pressure *high*, i.e. an
    **anticyclone** (clockwise for :math:`f > 0`, negative
    ``rel_vort_z``), while a positive ``amplitude`` with
    ``gauss_field="vorticity"`` prescribes positive ``rel_vort_z``,
    i.e. a **cyclone**. The two branches turn opposite ways for the
    same sign of ``amplitude``, which is the physics
    (:math:`\zeta = \nabla_h^2\psi`), not a convention choice.

    **Staggering.** :math:`G` is sampled on the horizontal corner (the
    ``u`` face in :math:`x` crossed with the ``v`` face in :math:`y`)
    and :math:`F` on the vertical **faces**; the velocities take the
    discrete corner curl of :math:`G` times the face-to-centre
    *interpolant* of :math:`F`, and the buoyancy takes the
    corner-to-centre interpolant of :math:`G` times the face-to-centre
    *difference* of :math:`F`. The two vertical operators are the
    adjacent pair of one staggered lattice, which is what makes the
    result exactly balanced rather than balanced to truncation order:

    - the discrete divergence of :math:`(u, v, w)` is zero to
      machine precision;
    - the model's own tendency, after the pressure projection, is
      zero to machine precision — the state is a discrete steady
      solution of the linear model, hence exactly in the vortical
      (:math:`\omega = 0`) subspace;
    - the *continuum* correspondence holds to second order only: the
      sampled velocity carries :math:`(F(z_{j-1/2}) +
      F(z_{j+1/2}))/2` rather than :math:`F(z_j)`, and the thermal
      wind :math:`f\,\partial_z u = -\partial_y b` closes only up to
      the C-grid interpolation between the ``b`` cell and the ``w``
      face, both :math:`O(\Delta z^2)`.

    .. note::

        **No projection is applied**, and on an f-plane none is
        needed: the construction above is already exactly vortical, so
        ``nh.transforms.VorticalProjection`` is a no-op on it (wave
        energy measured at :math:`10^{-31}` of the total). Project
        afterwards when the balance argument breaks — a beta-plane or
        chart rotation, an immersed or mapped grid, a state you have
        added a perturbation to, or a caller-supplied
        ``vertical_structure`` so steep that it is not resolved
        (whereupon the eddy radiates inertia-gravity waves at the
        amplitude of its own truncation error):

        .. code-block:: python

            z = nh.coherent_eddy(model)
            z = nh.transforms.VorticalProjection.from_model(model)(z)

    Parameters
    ----------
    model : Model
        The assembled nonhydrostatic model. The velocity, buoyancy
        and rotation come from the model itself; no eigenmode
        machinery is built.
    pos_x : float, optional
        Relative zonal position of the eddy (default: 0.5).
    pos_y : float, optional
        Relative meridional position of the eddy (default: 0.5).
    width : float, optional
        Width of the eddy relative to the zonal domain size
        (default: 0.1).
    amplitude : float, optional
        Peak value of the prescribed Gaussian (streamfunction or
        vorticity); a negative amplitude flips the rotation sense
        (default: 1.0).
    gauss_field : str, optional
        Which field the Gaussian prescribes: ``"streamfunction"`` or
        ``"vorticity"`` (default: "streamfunction").
    vertical_structure : Callable[[jax.Array], jax.Array] | None, optional
        The vertical structure :math:`F(z)` multiplying the
        streamfunction, called with the **physical** vertical
        coordinate (the same convention as every other coordinate
        callable in this module) and not renormalized. None is the
        constant 1 — the barotropic eddy, whose ``b`` is identically
        zero and which needs no rotation at all. A structure that is
        not periodic on a periodic vertical carries the jump of its
        own wrap-around into ``b`` (default: None).
    at_time : float, optional
        Parameter evaluation time for a time-dependent rotation
        (default: 0.0).

    Returns
    -------
    State
        The eddy state (assign with ``model.set_state``). ``b`` is
        present iff the model declares it.

    Raises
    ------
    ValueError
        On an unknown ``gauss_field``; on the vorticity branch over a
        horizontally walled grid; on a ``vertical_structure`` without
        a buoyancy module or without a constant Coriolis parameter.

    Examples
    --------
    A surface-intensified anticyclone on a rigid-lid grid:

    .. code-block:: python

        import jax.numpy as jnp
        import fridom.nonhydro2 as nh

        state = nh.coherent_eddy(
            model, width=0.15,
            vertical_structure=lambda z: jnp.exp(z / 0.2))
        model.set_state(state)
    """
    if gauss_field not in {"vorticity", "streamfunction"}:
        raise ValueError(
            f"unknown gauss_field {gauss_field!r}: the Gaussian "
            "prescribes either 'vorticity' or 'streamfunction'")
    if isinstance(model, Eigenmodes | ChannelEigenmodes):
        # a value error (wrong source), not a type error
        raise ValueError(  # noqa: TRY004
            "coherent_eddy takes the model, not an eigenmodes "
            "object: the eddy is an analytic geostrophic state and "
            "builds no eigenbasis at all (the old spelling paid for "
            "a channel eigensolve only to read the grid). Pass the "
            "assembled model, nh.coherent_eddy(model, ...)")
    grid = model.grid
    x, y = _horizontal(grid)
    x0, x1 = _extent(grid, x)
    y0, y1 = _extent(grid, y)
    lx, ly = x1 - x0, y1 - y0
    spaces = {name: model.state[name].function_space.bare
              for name in ("u", "v", "w")}
    b_space = (model.state["b"].function_space.bare
               if "b" in model.state else None)
    if vertical_structure is not None and b_space is None:
        raise ValueError(
            "a vertical_structure makes the eddy baroclinic, whose "
            "buoyancy this model cannot hold: it declares no 'b' — "
            "add a buoyancy module (nh.ConstantStratification(n2=...) "
            "or nh.BuoyancyTracer()), or drop vertical_structure= "
            "for the barotropic eddy")
    # the horizontal corner: u's x face crossed with v's y face, one
    # DOF in the vertical (the shape is horizontal; the structure
    # multiplies it afterwards, so the inversion stays 2-D)
    flat = spaces["u"].replace(**{
        y: spaces["v"].factor(y),
        _VERTICAL: _constant_factor(grid, _VERTICAL)})

    def bump(coords: dict[str, jax.Array]) -> jax.Array:
        return amplitude * jnp.exp(
            -((coords[x] - x0 - pos_x * lx) ** 2
              + (coords[y] - y0 - pos_y * ly) ** 2)
            / (width * lx) ** 2)

    shape = _sample(grid, flat, bump, name="psi")
    if gauss_field == "vorticity":
        shape = _streamfunction_from_vorticity(grid, shape)
    corner = flat.replace(**{
        _VERTICAL: spaces["u"].factor(_VERTICAL)})
    if vertical_structure is None:
        psi = shape.to(corner)
        buoyancy = (None if b_space is None
                    else grid.create_field(b_space, name="b"))
    else:
        centre_z = spaces["u"].factor(_VERTICAL)
        column = _sample(
            grid,
            flat.replace(**{
                x: _constant_factor(grid, x),
                y: _constant_factor(grid, y),
                _VERTICAL: _face_factor(grid, _VERTICAL)}),
            lambda coords: jnp.broadcast_to(
                jnp.asarray(vertical_structure(coords[_VERTICAL])),
                coords[_VERTICAL].shape),
            name="structure")
        # the velocities carry the face-to-centre *interpolant* of F,
        # the buoyancy its face-to-centre *difference*: the adjacent
        # pair of one staggered lattice, for which
        # I(d F) = d(I F) holds identically — which is what makes the
        # discrete hydrostatic balance exact rather than second-order
        psi = shape * column.to(
            column.function_space.bare.replace(**{_VERTICAL: centre_z}))
        buoyancy = (
            _coriolis_parameter(model, at_time)
            * shape.to(b_space.replace(**{
                _VERTICAL: _constant_factor(grid, _VERTICAL)}))
            * column.diff(_VERTICAL)).to(b_space)
    fields = {
        "u": (-psi.diff(y)).with_metadata(name="u"),
        "v": psi.diff(x).with_metadata(name="v"),
        "w": grid.create_field(spaces["w"], name="w")}
    if buoyancy is not None:
        fields["b"] = buoyancy.with_metadata(name="b")
    return State(fields)
