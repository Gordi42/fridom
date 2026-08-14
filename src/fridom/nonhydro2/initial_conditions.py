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
projected by default), :func:`coherent_eddy` (a Gaussian vorticity
or streamfunction eddy, turned into ``u``, ``v``, ``b`` by
geostrophic theory) and :func:`eddy_dipole` (a self-advecting
counter-rotating pair of them, placed by a compass heading). Wave
factories return ``(omega, state)`` like ``em.mode``; profile
factories return the state alone.

The two eddy factories are the ones that take the **model** alone:
they need the component spaces and the Coriolis parameter, not an
eigenbasis, so they serve every grid topology (the horizontally
walled channel included) and build no eigenmodes. They also share
one construction, ``_eddy_state``, which is what keeps their
geostrophy, their staggering and their sign convention from drifting
apart. Both take their vertical profile either as the separable
multiplier ``vertical_structure=`` (the fast path: the sampling and
the elliptic inversion stay two-dimensional) or as callables of the
vertical coordinate on the individual knobs (the general path, which
a depth-varying width forces, since it is not separable at all).
"""
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax
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
from fridom.model.streamfunction import invert_negative_laplacian
from fridom.model.time_dependent import resolve_at
from fridom.nonhydro2.channel_eigenmodes import ChannelEigenmodes
from fridom.nonhydro2.eigenmodes import Eigenmodes, from_model
from fridom.nonhydro2.state import State
from fridom.nonhydro2.transforms import (
    VorticalProjection,
    mode_projection,
)
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.symbols import ModeChart

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

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
    r"""
    Random vortical state (the ``RandomVorticalSpectra`` port).

    Description
    -----------
    :func:`random_state` on the ``"vortical"`` family with the
    horizontal :func:`geostrophic_energy_spectrum` default.

    The default spectrum is **horizontal**
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    :func:`geostrophic_energy_spectrum` is the Masur & Oliver [2020]
    density :math:`S(k_h)`, ported verbatim, and it takes
    :math:`k_h = \sqrt{k_x^2 + k_y^2}` only. **It puts no decay on**
    :math:`k_z`: every vertical mode of a horizontal shell gets the
    same energy, so the vertical structure is white and the field is
    rough at the grid scale in ``z`` (successive levels are nearly
    uncorrelated — that is the spectrum doing what it says, not a
    defect). It is the right default for reproducing the reference,
    and the wrong one if you wanted a vertically smooth eddy field.

    ``spectral_energy_density`` is the knob. It is called with one
    wavenumber array per grid axis, in grid order, so adding a
    vertical roll-off is a one-liner::

        def decaying(kx, ky, kz, kz0=1.0):
            return (nh.geostrophic_energy_spectrum(kx, ky)
                    / (1.0 + (kz / kz0) ** 2) ** 2)

        z = nh.random_vortical(model, spectral_energy_density=decaying)

    Two further consequences worth knowing before you budget a run:

    - **The normalization pins the velocity, not the vorticity.** The
      result is scaled to :math:`\max|u_h| = 1`, so the peak relative
      vorticity comes out at roughly the typical wavenumber of the
      spectrum in your units — several times the peak velocity on a
      :math:`2\pi` box, more on a small one. Set the amplitude from
      the quantity you actually care about.
    - **The synthesis touches the whole lattice.** Every mode of the
      family is built and back-transformed for every component (the
      loop is fully vectorized, but the work is the full spectral
      lattice, and on a small grid tracing and compiling that graph
      dominates it). It is not a cheap call, and it does not get
      cheap by shrinking the grid. A page that just wants *an* eddy
      is far better served by :func:`coherent_eddy` or
      :func:`eddy_dipole`.

    Parameters
    ----------
    source : Model | Eigenmodes | ChannelEigenmodes
        The assembled model or an eigenmodes object.
    spectral_energy_density : Callable[..., jax.Array] | None, optional
        ``S(*k)`` over the grid-axis wavenumbers, in grid order;
        None selects the horizontal geostrophic spectrum, which has
        no vertical decay (default: None).
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



def _invert_horizontal(
    field: ScalarField,
    axes: tuple[str, str],
    vertical: str,
) -> ScalarField:
    r"""
    Solve :math:`-\nabla_h^2 g = f` on a two-dimensional operand.

    Description
    -----------
    :func:`~fridom.model.streamfunction.invert_negative_laplacian`,
    applied one horizontal plane at a time whatever the operand's
    vertical extent. A one-DOF (``Constant``) vertical -- the
    separable eddy, whose structure multiplies :math:`\psi` after the
    inversion -- passes straight through. A vertical the prescribed
    shape genuinely varies along, which a depth-varying width makes
    unavoidable, is mapped over with ``jax.vmap``.

    Both reasons to keep the vertical out of the transform matter,
    and the second is the load-bearing one.

    - **Cost.** A three-dimensional operand pays a vertical transform
      pair that the horizontal Laplacian symbol never reads (measured
      184 times the two-dimensional cost at :math:`512^2 \times 64`).
    - **Existence.** On a rigid lid the corner's vertical face factor
      is the untagged ``Outer`` node set, and a walled horizontal axis
      commits the whole bounded product to the sine family, for which
      no ``Outer`` origin exists: ``spectral_sibling`` tags it
      Dirichlet and the retag changes the degree-of-freedom count
      (``Outer`` is n + 1 nodes, its Dirichlet reading n - 1). Mapping
      never asks, so the branch serves every topology.

    Parameters
    ----------
    field : ScalarField
        The prescribed vorticity, on the horizontal corner.
    axes : tuple[str, str]
        The two horizontal coordinate names.
    vertical : str
        The vertical coordinate name (never transformed).

    Returns
    -------
    ScalarField
        The solution on ``field``'s own function space.
    """
    space = field.function_space.bare
    if space.factor(vertical).is_constant:
        return invert_negative_laplacian(field, axes=axes)
    grid = field.grid
    position = tuple(space.names).index(vertical)
    plane = grid.create_field(
        space.replace(**{vertical: _constant_factor(grid, vertical)}),
        name="zeta")

    def solve(slab: jax.Array) -> jax.Array:
        """Invert one horizontal plane of the operand."""
        inverted = invert_negative_laplacian(
            plane.with_data(jnp.expand_dims(slab, position)),
            axes=axes)
        return jnp.squeeze(inverted.data, position)

    return field.with_data(jax.vmap(
        solve, in_axes=position, out_axes=position)(field.data))


def _at(
    knob: float | Callable[[jax.Array], jax.Array],
    zz: jax.Array,
) -> jax.Array:
    """Evaluate a float-or-callable knob at the vertical coordinate."""
    return knob(zz) if callable(knob) else jnp.asarray(knob)


def _curl(
    grid: Grid,
    psi: ScalarField,
    spaces: Mapping[str, SpaceLike],
    x: str,
    y: str,
    buoyancy: ScalarField | None,
) -> State:
    r"""Return the state of a corner streamfunction and its buoyancy.

    Description
    -----------
    The standard geostrophic pairing :math:`u = -\partial_y \psi`,
    :math:`v = \partial_x \psi`, whose discrete C-grid curl lands
    exactly on the ``u`` and ``v`` spaces and is divergence-free to
    machine precision, with :math:`w \equiv 0`.
    """
    fields = {
        "u": (-psi.diff(y)).with_metadata(name="u"),
        "v": psi.diff(x).with_metadata(name="v"),
        "w": grid.create_field(spaces["w"], name="w")}
    if buoyancy is not None:
        fields["b"] = buoyancy.with_metadata(name="b")
    return State(fields)


def _eddy_state(
    model: Model,
    *,
    what: str,
    shape: Callable[[jax.Array, jax.Array, jax.Array], jax.Array],
    depth_varying: bool,
    gauss_field: str,
    vertical_structure: Callable[[jax.Array], jax.Array] | None,
    at_time: float,
) -> State:
    r"""
    Turn a prescribed horizontal shape into a balanced eddy state.

    Description
    -----------
    The construction shared by :func:`coherent_eddy` and
    :func:`eddy_dipole`, and the whole of the geostrophy. ``shape``
    prescribes either :math:`\psi` or :math:`\zeta` on the horizontal
    corner; the state is the geostrophic, hydrostatic one of
    :func:`coherent_eddy`'s docstring.

    :math:`\psi` is sampled on the vertical **faces** and the
    velocities take its face-to-centre *interpolant* while the
    buoyancy takes its face-to-centre *difference*: the adjacent pair
    of one staggered lattice, for which :math:`I(\delta F) =
    \delta(I F)` holds identically, which is what makes the discrete
    balance exact rather than second-order.

    ``depth_varying`` selects between two spellings of the same
    formula, not two constructions.

    - False -- the shape is horizontal alone. It is sampled once on
      the corner with a one-DOF vertical, and a separable
      ``vertical_structure`` :math:`F(z)` (itself one-dimensional,
      on the faces) carries the whole vertical. Both operators then
      act on :math:`F`, so no three-dimensional array is built before
      the state itself and the elliptic inversion stays a single
      two-dimensional solve.
    - True -- the shape varies with depth (a depth-varying width is
      not separable at all: :math:`A(z)\exp(-r^2/R(z)^2)` is no
      product of a horizontal shape and a vertical profile). It is
      sampled on the corner crossed with the vertical faces, and the
      two operators act on that.

    Parameters
    ----------
    model : Model
        The assembled nonhydrostatic model.
    what : str
        The public factory name, for the taught errors.
    shape : Callable[[jax.Array, jax.Array, jax.Array], jax.Array]
        ``shape(x, y, z)`` on physical coordinates; ``z`` is the
        domain mid-depth when ``depth_varying`` is False (a
        depth-independent shape does not read it).
    depth_varying : bool
        Whether ``shape`` reads its vertical coordinate.
    gauss_field : str
        Which field ``shape`` prescribes.
    vertical_structure : Callable[[jax.Array], jax.Array] | None
        The separable multiplier :math:`F(z)`, or None.
    at_time : float
        Parameter evaluation time for the rotation.

    Returns
    -------
    State
        The eddy state; ``b`` is present iff the model declares it.
    """
    if gauss_field not in {"vorticity", "streamfunction"}:
        raise ValueError(
            f"unknown gauss_field {gauss_field!r}: the Gaussian "
            "prescribes either 'vorticity' or 'streamfunction'")
    if isinstance(model, Eigenmodes | ChannelEigenmodes):
        # a value error (wrong source), not a type error
        raise ValueError(  # noqa: TRY004
            f"{what} takes the model, not an eigenmodes object: the "
            "eddy is an analytic geostrophic state and builds no "
            "eigenbasis at all (the old spelling paid for a channel "
            "eigensolve only to read the grid). Pass the assembled "
            f"model, nh.{what}(model, ...)")
    grid = model.grid
    x, y = _horizontal(grid)
    z0, z1 = _extent(grid, _VERTICAL)
    spaces = {name: model.state[name].function_space.bare
              for name in ("u", "v", "w")}
    b_space = (model.state["b"].function_space.bare
               if "b" in model.state else None)
    baroclinic = depth_varying or vertical_structure is not None
    if baroclinic and b_space is None:
        raise ValueError(
            "a depth-varying streamfunction makes the eddy "
            "baroclinic, whose buoyancy this model cannot hold: it "
            "declares no 'b' — add a buoyancy module "
            "(nh.ConstantStratification(n2=...) or "
            "nh.BuoyancyTracer()), or drop the depth dependence for "
            "the barotropic eddy")
    # the horizontal corner: u's x face crossed with v's y face
    corner = spaces["u"].replace(**{y: spaces["v"].factor(y)})
    centre_z = spaces["u"].factor(_VERTICAL)
    face_z = _face_factor(grid, _VERTICAL)
    flat_z = _constant_factor(grid, _VERTICAL)
    # a depth-independent shape rides on one vertical DOF, which keeps
    # both the sampling and the elliptic inversion two-dimensional
    operand_z = face_z if depth_varying else flat_z
    middle = jnp.asarray(0.5 * (z0 + z1))

    def sampled(coords: dict[str, jax.Array]) -> jax.Array:
        """Evaluate the prescribed shape at a space's own nodes."""
        return shape(coords[x], coords[y],
                     coords.get(_VERTICAL, middle))

    field = _sample(grid, corner.replace(**{_VERTICAL: operand_z}),
                    sampled, name="psi")
    if gauss_field == "vorticity":
        # this curl gives zeta = +laplacian_h psi, so psi is *minus*
        # the inverse of the positive-definite -laplacian_h the shared
        # solver returns; the sign lives here, next to the curl that
        # forces it
        field = -_invert_horizontal(field, (x, y), _VERTICAL)
    if not baroclinic:
        buoyancy = (None if b_space is None
                    else grid.create_field(b_space, name="b"))
        return _curl(grid, field.to(corner), spaces, x, y, buoyancy)
    column = None if vertical_structure is None else _sample(
        grid,
        corner.replace(**{x: _constant_factor(grid, x),
                          y: _constant_factor(grid, y),
                          _VERTICAL: face_z}),
        lambda coords: jnp.broadcast_to(
            jnp.asarray(vertical_structure(coords[_VERTICAL])),
            coords[_VERTICAL].shape),
        name="structure")
    if column is not None and depth_varying:
        # a depth-varying shape times a structure is not separable
        # either, so the two operators must act on the product itself:
        # the discrete difference obeys no product rule
        field, column = field * column, None
    if column is None:
        # psi already lives on the vertical faces
        psi = field.to(corner)
        slope = field.diff(_VERTICAL).to(b_space)
    else:
        # the separable spelling of the same two operators, applied to
        # F alone so the one-dimensional profile stays one-dimensional
        centre_column = column.function_space.bare.replace(
            **{_VERTICAL: centre_z})
        psi = field * column.to(centre_column)
        slope = (field.to(b_space.replace(**{_VERTICAL: flat_z}))
                 * column.diff(_VERTICAL))
    f0 = _coriolis_parameter(model, at_time)
    return _curl(grid, psi, spaces, x, y,
                 (f0 * slope).to(b_space).with_metadata(name="b"))


def coherent_eddy(
    model: Model,
    *,
    pos_x: float | Callable[[jax.Array], jax.Array] = 0.5,
    pos_y: float | Callable[[jax.Array], jax.Array] = 0.5,
    width: float | Callable[[jax.Array], jax.Array] = 0.1,
    amplitude: float | Callable[[jax.Array], jax.Array] = 1.0,
    gauss_field: str = "vorticity",
    vertical_structure: Callable[[jax.Array], jax.Array] | None = None,
    at_time: float = 0.0,
) -> State:
    r"""
    Build a geostrophic Gaussian eddy (``CoherentEddy``).

    Description
    -----------
    A geostrophic streamfunction turned into ``u``, ``v`` and ``b`` by
    geostrophic theory. The horizontal shape is the Gaussian bump

    .. math::
        G = A \exp\left(
        -\frac{(x - p_x L_x)^2 + (y - p_y L_y)^2}{(\sigma L_x)^2}
        \right)

    which prescribes either the relative vorticity
    (``gauss_field="vorticity"``, the default), whose streamfunction
    follows from the horizontal inversion :math:`\nabla_h^2 \psi =
    \zeta`, or the streamfunction directly
    (``gauss_field="streamfunction"``).

    **Depth dependence, two spellings.** ``vertical_structure``
    :math:`F(z)` multiplies the streamfunction, giving the separable
    :math:`\psi = G(x, y) F(z)`; the default is the constant 1, the
    barotropic eddy. Independently, ``pos_x``, ``pos_y``, ``width``
    and ``amplitude`` each accept a **callable of the vertical
    coordinate** instead of a float, which is the general spelling: a
    depth-varying width is not separable at all, since
    :math:`A(z)\exp(-r^2/R(z)^2)` is no product of a horizontal shape
    and a vertical profile. The two agree exactly where they overlap
    (``amplitude=F`` reproduces ``vertical_structure=F`` to round-off),
    and ``vertical_structure`` is the one to reach for when it applies:
    it keeps both the sampling and the elliptic inversion
    two-dimensional, while a callable knob makes them three-dimensional.

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
    ``gauss_field="vorticity"`` prescribes positive ``rel_vort_z``,
    i.e. a **cyclone**, while a positive ``amplitude`` with
    ``gauss_field="streamfunction"`` is a pressure *high*, i.e. an
    **anticyclone** (clockwise for :math:`f > 0`, negative
    ``rel_vort_z``). The two branches turn opposite ways for the same
    sign of ``amplitude``, which is the physics
    (:math:`\zeta = \nabla_h^2\psi`), not a convention choice.

    **The vorticity branch and its gauge.** The inversion is
    :func:`~fridom.model.streamfunction.invert_negative_laplacian`,
    which diagonalizes the *discrete* corner Laplacian on every
    horizontal topology — doubly periodic, a walled channel, a closed
    box, each with or without a rigid lid — so the branch is no more
    restricted than the streamfunction one. Its operand is one
    horizontal plane: the two-dimensional :math:`G` alone under a
    separable structure, and one mapped plane per vertical face
    otherwise. The vertical is never transformed.

    What *is* topology dependent is the gauge, and a caller choosing
    between the two branches should know which one they get:

    - with **at least one walled horizontal axis** the sine basis
      carries no constant mode, the operator has no nullspace, and
      the diagnosed ``rel_vort_z`` is the prescribed Gaussian
      exactly (to round-off, :math:`10^{-13}` relative);
    - on a **fully periodic horizontal** the constant mode is a
      structural zero of the symbol — a periodic domain admits no
      net vorticity, :math:`\int \zeta \, \mathrm{d}A = 0` — so the
      state carries the prescribed Gaussian *minus its domain
      mean*. That offset is the bump's own area fraction
      :math:`\pi \sigma^2 / L_x L_y`, i.e. :math:`\pi \sigma_r^2`
      on a square box for a relative ``width`` :math:`\sigma_r`:
      4.5 percent of the peak at ``width=0.12``, 20 percent at
      ``width=0.25``. Prescribe the streamfunction instead when the
      peak vorticity has to be exact on a periodic grid.

    **Staggering.** :math:`\psi` is sampled on the horizontal corner
    (the ``u`` face in :math:`x` crossed with the ``v`` face in
    :math:`y`) and on the vertical **faces**; the velocities take the
    discrete corner curl times the face-to-centre *interpolant*, and
    the buoyancy the corner-to-centre interpolant times the
    face-to-centre *difference*. The two vertical operators are the
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
        added a perturbation to, or a caller-supplied depth profile
        so steep that it is not resolved (whereupon the eddy radiates
        inertia-gravity waves at the amplitude of its own truncation
        error):

        .. code-block:: python

            z = nh.coherent_eddy(model)
            z = nh.transforms.VorticalProjection.from_model(model)(z)

    Parameters
    ----------
    model : Model
        The assembled nonhydrostatic model. The velocity, buoyancy
        and rotation come from the model itself; no eigenmode
        machinery is built.
    pos_x : float | Callable, optional
        Relative zonal position of the eddy, or a callable of the
        **physical** vertical coordinate (default: 0.5).
    pos_y : float | Callable, optional
        Relative meridional position of the eddy, or a callable of the
        vertical coordinate (default: 0.5).
    width : float | Callable, optional
        Width of the eddy relative to the zonal domain size, or a
        callable of the vertical coordinate (default: 0.1).
    amplitude : float | Callable, optional
        Peak value of the prescribed Gaussian (vorticity or
        streamfunction), or a callable of the vertical coordinate; a
        negative amplitude flips the rotation sense (default: 1.0).
    gauss_field : str, optional
        Which field the Gaussian prescribes: ``"vorticity"`` or
        ``"streamfunction"`` (default: "vorticity").
    vertical_structure : Callable[[jax.Array], jax.Array] | None, optional
        The separable vertical structure :math:`F(z)` multiplying the
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
        On an unknown ``gauss_field``; on an eigenmodes object in
        place of the model; on a depth-varying eddy without a
        buoyancy module or without a constant Coriolis parameter.
        The grid topology is never a reason: both branches serve
        every one.

    Examples
    --------
    A surface-intensified anticyclone on a rigid-lid grid:

    .. code-block:: python

        import jax.numpy as jnp
        import fridom.nonhydro2 as nh

        state = nh.coherent_eddy(
            model, width=0.15, gauss_field="streamfunction",
            vertical_structure=lambda z: jnp.exp(z / 0.2))
        model.set_state(state)

    A cyclone that widens with depth, which no separable structure
    can express:

    .. code-block:: python

        state = nh.coherent_eddy(
            model, width=lambda z: 0.1 + 0.05 * jnp.cos(z))
    """
    grid = model.grid
    x, y = _horizontal(grid)
    x0, x1 = _extent(grid, x)
    y0, y1 = _extent(grid, y)
    lx, ly = x1 - x0, y1 - y0
    knobs = (pos_x, pos_y, width, amplitude)

    def bump(
        xx: jax.Array, yy: jax.Array, zz: jax.Array,
    ) -> jax.Array:
        """Sample the Gaussian bump at its own depth's knobs."""
        radius = _at(width, zz) * lx
        return _at(amplitude, zz) * jnp.exp(
            -((xx - x0 - _at(pos_x, zz) * lx) ** 2
              + (yy - y0 - _at(pos_y, zz) * ly) ** 2) / radius ** 2)

    return _eddy_state(
        model, what="coherent_eddy", shape=bump,
        depth_varying=any(callable(knob) for knob in knobs),
        gauss_field=gauss_field,
        vertical_structure=vertical_structure, at_time=at_time)


# ================================================================
#  Eddy dipole (a self-advecting counter-rotating pair)
# ================================================================
#: Argmax and peak of the streamfunction shape s -> s exp(-s^2).
_PSI_PEAK = (0.7071067811865475, 0.4288819424803534)

#: Argmax and peak of the vorticity shape s -> (1 - exp(-s^2)) / s.
_VORT_PEAK = (1.1209064227785339, 0.6381726863389515)

#: Number of vertical samples the profile checks scan.
_PROBE = 257

#: Slope below which a Newton polish step is skipped (at the peak).
_FLAT = 1e-8

#: Default separation, in widths, per prescribed field.
#:
#: The realized speed of a Gaussian pair depends on ``d / R`` far more
#: strongly than the closed form says, because the pair is not a steady
#: solution and adjusts. These are the ratios that adjust least: the
#: vorticity pair holds together from about three widths out, where its
#: cores stop overlapping, and the streamfunction pair keeps its shape
#: best near 1.4 widths. Leaving ``separation`` unset takes the ratio
#: from here **at every depth**, so a depth-varying width gives
#: self-similar layers, which is what keeps the columns together.
_DEFAULT_RATIO = {"streamfunction": 1.4, "vorticity": 3.0}


def _profile(
    value: float | Callable[[jax.Array], jax.Array],
) -> Callable[[jax.Array], jax.Array]:
    """Return ``value`` as a callable of the vertical coordinate."""
    if callable(value):
        return value

    def constant(zz: jax.Array) -> jax.Array:
        """Broadcast the scalar over the vertical coordinate."""
        return jnp.asarray(float(value)) + jnp.zeros_like(zz)

    return constant


def _shape_of(gauss_field: str, s: jax.Array) -> jax.Array:
    r"""Dimensionless mutual-induction shape :math:`G(d/R)`."""
    if gauss_field == "streamfunction":
        return s * jnp.exp(-s ** 2)
    return -jnp.expm1(-s ** 2) / s


def _shape_slope(gauss_field: str, s: jax.Array) -> jax.Array:
    r"""Return the derivative :math:`G'(s)` of :func:`_shape_of`."""
    decay = jnp.exp(-s ** 2)
    if gauss_field == "streamfunction":
        return decay * (1.0 - 2.0 * s ** 2)
    return (2.0 * s ** 2 * decay - 1.0 + decay) / s ** 2


def _gain(gauss_field: str, radius: jax.Array) -> jax.Array:
    """Return the translation speed per unit amplitude at unit shape."""
    if gauss_field == "streamfunction":
        return 2.0 / radius
    return 0.5 * radius


def _outer_root(gauss_field: str, target: jax.Array) -> jax.Array:
    r"""
    Solve :math:`G(s) = \mathrm{target}` on the outer branch.

    Description
    -----------
    :func:`_shape_of` rises from zero, peaks, and decays, so a
    reachable target has **two** roots. The outer one (:math:`s`
    above the peak, where :math:`G` decreases) is the coherent
    branch: the cores overlap less and the leading-order induction
    formula holds better. Fifty bisections bracket it from
    ``[s_peak, s_hi]`` (an analytic upper bound), then two Newton
    steps polish the value and restore the exact implicit
    derivative that bisection alone would flatten to zero.
    """
    peak_s = (_PSI_PEAK if gauss_field == "streamfunction"
              else _VORT_PEAK)[0]
    if gauss_field == "streamfunction":
        # bounded above by exp(-s^2 / 2) for s at least one
        far = jnp.sqrt(2.0 * jnp.log(1.0 / target))
    else:
        # bounded above by the reciprocal of s
        far = 1.0 / target
    lo = jnp.zeros_like(target) + peak_s
    hi = jnp.maximum(far, 2.0 * peak_s)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        above = _shape_of(gauss_field, mid) > target
        lo = jnp.where(above, mid, lo)
        hi = jnp.where(above, hi, mid)
    s = 0.5 * (lo + hi)
    for _ in range(2):
        slope = _shape_slope(gauss_field, s)
        steep = jnp.abs(slope) > _FLAT
        safe = jnp.where(steep, slope, 1.0)
        step = (_shape_of(gauss_field, s) - target) / safe
        s = s - jnp.where(steep, step, 0.0)
    return s


def _check_positive(name: str, values: jax.Array) -> None:
    """Reject a non-positive profile over the vertical scan."""
    worst = float(jnp.min(values))
    if worst <= 0.0:
        raise ValueError(
            f"{name} must be positive everywhere in the domain, "
            f"got a minimum of {worst:.4g} over the vertical")


def _check_reachable(
    gauss_field: str, target: jax.Array, speed: float,
) -> None:
    """Reject a target speed above the mutual-induction peak."""
    peak_s, peak_g = (_PSI_PEAK if gauss_field == "streamfunction"
                      else _VORT_PEAK)
    worst = float(jnp.max(target))
    if worst > peak_g:
        raise ValueError(
            f"speed={speed:.4g} is unreachable at "
            "match='separation': the mutual induction of a Gaussian "
            f"pair peaks at d = {peak_s:.4g} R, and no separation "
            f"reaches more than {speed * peak_g / worst:.4g} at the "
            "least favourable level. Raise amplitude, widen the "
            "eddies, or use match='amplitude' instead")


def _separation(
    separation: float | Callable[[jax.Array], jax.Array] | None,
    width_of: Callable[[jax.Array], jax.Array],
    gauss_field: str,
) -> Callable[[jax.Array], jax.Array]:
    """Return the separation profile, defaulting to a width ratio."""
    if separation is not None:
        return _profile(separation)
    ratio = _DEFAULT_RATIO[gauss_field]

    def tracking(zz: jax.Array) -> jax.Array:
        """Return the route's default separation at each depth."""
        return ratio * width_of(zz)

    return tracking


def _resolve_knobs(
    gauss_field: str,
    match: str,
    speed: float | None,
    lx: float,
    probe: jax.Array,
    width_of: Callable[[jax.Array], jax.Array],
    amp_of: Callable[[jax.Array], jax.Array],
    sep_of: Callable[[jax.Array], jax.Array],
) -> tuple[Callable[[jax.Array], jax.Array],
           Callable[[jax.Array], jax.Array]]:
    """
    Return the amplitude and separation profiles a target implies.

    Description
    -----------
    Without a ``speed`` the two profiles pass through. With one, the
    ``match`` knob is replaced by the profile that realizes the
    target level by level: the amplitude in closed form (the speed
    is linear in it), the separation through :func:`_outer_root`.
    """
    _check_positive("width", width_of(probe))
    if speed is None:
        _check_positive("separation", sep_of(probe))
        return amp_of, sep_of
    if match == "amplitude":
        _check_positive("separation", sep_of(probe))

        def matched_amplitude(zz: jax.Array) -> jax.Array:
            """Return the amplitude that realizes the target speed."""
            shape = _shape_of(gauss_field, sep_of(zz) / width_of(zz))
            gain = _gain(gauss_field, width_of(zz) * lx)
            return speed / (gain * shape)

        return matched_amplitude, sep_of
    _check_positive("amplitude", amp_of(probe))
    _check_reachable(
        gauss_field,
        speed / (_gain(gauss_field, width_of(probe) * lx)
                 * amp_of(probe)),
        speed)

    def matched_separation(zz: jax.Array) -> jax.Array:
        """Return the separation that realizes the target speed."""
        gain = _gain(gauss_field, width_of(zz) * lx)
        return width_of(zz) * _outer_root(
            gauss_field, speed / (gain * amp_of(zz)))

    return amp_of, matched_separation


def eddy_dipole(
    model: Model,
    *,
    pos_x: float = 0.5,
    pos_y: float = 0.5,
    angle: float = 0.0,
    separation: float | Callable[[jax.Array], jax.Array] | None = None,
    width: float | Callable[[jax.Array], jax.Array] = 0.1,
    amplitude: float | Callable[[jax.Array], jax.Array] = 1.0,
    speed: float | None = None,
    match: str = "amplitude",
    gauss_field: str = "vorticity",
    at_time: float = 0.0,
) -> State:
    r"""
    Build a self-advecting pair of counter-rotating eddies.

    Description
    -----------
    Two Gaussian eddies of amplitude :math:`\pm A`, width :math:`R`
    and centre separation :math:`d`, placed symmetrically about
    ``(pos_x, pos_y)`` on the line perpendicular to the heading
    ``angle``. Each eddy is carried by the other's flow, so the pair
    translates along its perpendicular bisector at a speed set by the
    mutual induction. The pair is the superposition of two
    :func:`coherent_eddy` lobes and shares its construction exactly,
    so it is divergence-free and exactly balanced on every topology.

    **Heading.** ``angle`` is a compass bearing in **degrees**,
    measured clockwise from north: ``0`` is north (:math:`+y`),
    ``90`` east (:math:`+x`), ``180`` south, ``270`` west. The dipole
    travels **toward** the bearing. The counter-clockwise eddy
    (positive relative vorticity) sits on the **left** of the heading
    and the clockwise one on the right, so the jet between the two
    cores points along the heading and carries the pair with it. A
    northward dipole therefore keeps its counter-clockwise eddy on its
    western flank. Which *lobe* that is follows
    :func:`coherent_eddy`'s sign convention rather than a choice made
    here: under ``gauss_field="vorticity"`` the counter-clockwise eddy
    is the :math:`+A` lobe, under ``"streamfunction"`` it is the
    :math:`-A` one, because a streamfunction high is an anticyclone
    (:math:`\zeta = \nabla_h^2\psi`). The factory places the lobes to
    match, so the heading means the same thing on both routes. A
    negative ``amplitude`` swaps the two eddies and reverses the
    travel, exactly like ``angle + 180``.

    **Speed.** For a Gaussian vorticity (``gauss_field="vorticity"``,
    the default; circulation :math:`\Gamma = \pi R^2 A`) the
    Lamb-Oseen induction gives

    .. math::
        U = \frac{\Gamma}{2 \pi d}
            \left(1 - e^{-d^2 / R^2}\right) ,

    which peaks at :math:`d = 1.121 R` and recovers the point-vortex
    :math:`\Gamma / (2 \pi d)` within one percent beyond
    :math:`d = 3 R`. For a Gaussian streamfunction
    (``gauss_field="streamfunction"``) the mutual induction is instead

    .. math::
        U = \frac{2 A d}{R^2}\, e^{-d^2 / R^2} ,

    which peaks at :math:`d = R / \sqrt{2}` and decays faster than
    exponentially beyond it, because a Gaussian streamfunction eddy
    carries **zero** net circulation and so has no far field at all.
    That difference is why the two routes need different separations,
    and why only the vorticity route feels a distant wall. Both
    formulas are the continuum unbounded-domain leading order; on a
    torus the induction is weaker by a factor depending only on
    :math:`d / L` (0.95 at :math:`d = 0.125 L`), and the whole
    steady-translation picture blurs once the cores overlap
    (:math:`d \lesssim R`).

    **Target speed.** ``speed`` sets :math:`U` directly, and it is a
    *nominal* speed: exact at :math:`t = 0` up to the periodic image
    factor, but the Gaussian pair is not a steady dipole and adjusts
    over about one eddy turnover. Since :math:`U` is linear in
    :math:`A` at fixed geometry, the solve costs one division when
    ``match="amplitude"`` (the default): the amplitude follows from
    the requested speed and the given separation, always, uniquely.
    With ``match="separation"`` the separation is solved for instead,
    on the outer branch of the non-monotonic shape function, and the
    request is **unreachable** when it exceeds the peak of that
    function (a taught error) — a dipole cannot translate faster than
    one of its eddies swirls.

    **Depth dependence.** ``width``, ``amplitude`` and ``separation``
    each accept a callable of the vertical coordinate, and the
    construction is :func:`coherent_eddy`'s general (non-separable)
    one, so the thermal-wind buoyancy comes with it and the state
    stays exactly balanced. A depth-varying radius alone shears the
    dipole apart, since :math:`U` depends on :math:`R`. Equalizing the
    closed-form speed is **not** enough to stop that, because the pair
    is not a steady solution and how far its realized speed drifts
    from the closed form depends on :math:`d / R`. What holds the
    columns together is making every level the same shape: leave
    ``separation`` unset so it tracks the width, and let ``speed``
    with ``match="amplitude"`` carry the target.
    ``match="separation"`` serves the other reading of the same
    request, equalizing the closed-form speed at a fixed amplitude by
    tilting the dipole axis, which both leaves the cores misaligned
    from top to bottom and lets :math:`d / R` vary, so the levels
    drift apart anyway.

    On a **rigid lid** the vortical mode's buoyancy has sine parity
    and must vanish at the lids, so a depth profile should have zero
    slope at the top and bottom (:math:`\cos(\pi z / H)` does, a
    linear profile does not).

    Parameters
    ----------
    model : Model
        The assembled nonhydrostatic model.
    pos_x : float, optional
        Relative zonal position of the dipole centre (default: 0.5).
    pos_y : float, optional
        Relative meridional position of the dipole centre
        (default: 0.5).
    angle : float, optional
        Heading in degrees, clockwise from north (default: 0.0).
    separation : float | Callable | None, optional
        Centre separation :math:`d` relative to the zonal domain
        size, or a callable of the vertical coordinate; None takes
        the route's own default ratio of the width at every depth
        (3.0 for ``"vorticity"``, 1.4 for ``"streamfunction"``),
        which keeps a depth-varying dipole self-similar
        (default: None).
    width : float | Callable, optional
        Eddy width :math:`R` relative to the zonal domain size, or a
        callable of the vertical coordinate (default: 0.1).
    amplitude : float | Callable, optional
        Peak of each Gaussian (vorticity or streamfunction), or a
        callable of the vertical coordinate; ignored when ``speed``
        is given with ``match="amplitude"`` (default: 1.0).
    speed : float | None, optional
        Target translation speed; None keeps the given amplitude and
        separation (default: None).
    match : str, optional
        Which knob ``speed`` solves for: ``"amplitude"`` (closed
        form, unique, cores stay aligned) or ``"separation"`` (outer
        branch of a root find, may be unreachable)
        (default: "amplitude").
    gauss_field : str, optional
        Which field each Gaussian prescribes: ``"vorticity"`` or
        ``"streamfunction"`` (default: "vorticity").
    at_time : float, optional
        Parameter evaluation time for a time-dependent rotation
        (default: 0.0).

    Returns
    -------
    State
        The dipole state (assign with ``model.set_state``).

    Raises
    ------
    ValueError
        On an unknown ``gauss_field`` or ``match``, a non-positive
        ``speed``, ``width`` or ``separation``, an eigenmodes object
        in place of the model, a depth-varying dipole on a model with
        no buoyancy or no constant Coriolis parameter, or a
        ``match="separation"`` target speed the geometry cannot
        reach.

    Examples
    --------
    A dipole crossing the domain to the north-east at a known speed:

    .. code-block:: python

        state = nh.eddy_dipole(model, angle=45.0, width=0.06,
                               speed=0.2)
        model.set_state(state)
    """
    if match not in {"amplitude", "separation"}:
        raise ValueError(
            f"unknown match {match!r}: a target speed solves for "
            "either 'amplitude' or 'separation'")
    if speed is not None and speed <= 0.0:
        raise ValueError(
            f"speed must be positive, got {speed}: the heading "
            "already carries the direction (angle + 180 reverses "
            "it)")
    if gauss_field not in _DEFAULT_RATIO:
        raise ValueError(
            f"unknown gauss_field {gauss_field!r}: the Gaussians "
            "prescribe either 'vorticity' or 'streamfunction'")
    grid = model.grid
    x, y = _horizontal(grid)
    x0, x1 = _extent(grid, x)
    y0, y1 = _extent(grid, y)
    z0, z1 = _extent(grid, _VERTICAL)
    lx, ly = x1 - x0, y1 - y0
    heading = jnp.deg2rad(angle)
    # The counter-clockwise eddy belongs on the left of the heading:
    # the jet between the two cores then points along the heading and
    # carries the pair with it. That statement is about the relative
    # vorticity; which *lobe* is counter-clockwise is coherent_eddy's
    # sign convention, under which a streamfunction high is an
    # anticyclone (zeta = laplacian_h psi), so the +A lobe sits on the
    # left under 'vorticity' and on the right under 'streamfunction'.
    lobe = 1.0 if gauss_field == "vorticity" else -1.0
    left = (-lobe * jnp.cos(heading), lobe * jnp.sin(heading))
    # a depth-varying knob is what makes the dipole baroclinic, and
    # only the caller's own knobs count: the default separation
    # follows the width, so it adds no depth dependence of its own
    depth_varying = any(callable(knob)
                        for knob in (width, amplitude, separation))
    width_of = _profile(width)
    amp_of, sep_of = _resolve_knobs(
        gauss_field, match, speed, lx,
        jnp.linspace(z0, z1, _PROBE), width_of,
        _profile(amplitude),
        _separation(separation, width_of, gauss_field))

    def pair(
        xx: jax.Array, yy: jax.Array, zz: jax.Array,
    ) -> jax.Array:
        """Sample the counter-rotating Gaussian pair."""
        radius = width_of(zz) * lx
        offset = 0.5 * sep_of(zz) * lx
        cx, cy = x0 + pos_x * lx, y0 + pos_y * ly
        plus = ((xx - cx - offset * left[0]) ** 2
                + (yy - cy - offset * left[1]) ** 2)
        minus = ((xx - cx + offset * left[0]) ** 2
                 + (yy - cy + offset * left[1]) ** 2)
        return amp_of(zz) * (jnp.exp(-plus / radius ** 2)
                             - jnp.exp(-minus / radius ** 2))

    return _eddy_state(
        model, what="eddy_dipole", shape=pair,
        depth_varying=depth_varying, gauss_field=gauss_field,
        vertical_structure=None, at_time=at_time)
