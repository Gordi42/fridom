r"""
Channel (walled-horizontal) nonhydro eigenmodes: wrapper + labeler.

Description
-----------
The nonhydrostatic face of the dense-column channel eigenbasis
(:func:`fridom.framework2.channel_eigenpairs`) on the rotating
stratified channel — one bounded *horizontal* axis (walls the
rotation couples to), two periodic axes. The engine probes the
projected linearization :math:`S = P L P` (the pressure CONSTRAINT
stage through ``model.constrain``), so every mode plane carries the
**divergence-complement** directions (the discrete pressure-gradient
space, the kernel of the Leray projector ``P``) as extra exact zero
modes next to the physical steady modes. The labeler owns telling
them apart; the family vocabulary per ``(kx, kz)`` plane is:

- **vortical** — the physical steady (thermal-wind balanced)
  columns, :math:`P q = q` and :math:`\omega \approx 0` on the
  f-plane (including the ``kx = 0`` exponential wall modes); under a
  beta-plane ``f(y)`` the ``kx != 0`` branch acquires slow westward
  Rossby frequencies;
- **constraint** — the divergence-complement columns,
  :math:`P q = 0`: non-physical directions a divergence-free state
  never excites. Excluded from every physical projector and from
  completeness sums (the physical families sum to ``P``, not the
  identity);
- **kelvin+ / kelvin-** — the boundary-trapped internal Kelvin pair
  (one per vertical wavenumber), with exactly zero wall-normal
  velocity energy; ``kx != 0``, ``kz != 0`` planes only;
- **wave+ / wave-** — the remaining nonzero branches by frequency
  sign: the Poincaré (inertia-gravity) strata, the vertically
  uniform ``kz = 0`` buoyancy oscillations (``omega = +/- N /
  delta``), and the buoyancy-decoupled inertial strata on the
  ``kz``-Nyquist planes (the interpolation factor ``cos(kz dz / 2)``
  vanishes there, the would-be Kelvin frequency with it — the
  Nyquist Kelvin limit sits in the steady set).

The constraint/vortical split is machine-crisp: the labeler
diagonalizes the Hermitian overlap :math:`A = Q_0^H M P Q_0` of the
zero-frequency columns (:func:`constraint_overlap`, probed through
the public ``model.constrain`` matvec) whose eigenvalues are exactly
0 (complement) or 1 (steady) — measured margins
:math:`\sim 10^{-15}`. The unitary eigenbasis rotation is written
back into ``basis.q``, so the labeled columns are purely steady or
purely complement even where ``eigh`` mixed the degenerate zero
space.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp
import numpy as np

from fridom.framework2.model.eigen_channel import (
    UNLABELED,
    ChannelEigenbasis,
)
from fridom.framework2.model.eigenbasis import (
    ChannelEigenmodesBase,
    recover_crisp_column,
    segment_energy,
    split_frequency_bands,
)
from fridom.nonhydro2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.framework2.model.model import Model

# ================================================================
#  The family label codes
# ================================================================
VORTICAL = 0
KELVIN_PLUS = 1
KELVIN_MINUS = 2
WAVE_PLUS = 3
WAVE_MINUS = 4
CONSTRAINT = 5

#: Family name -> integer label code (the ``labels`` vocabulary).
FAMILIES: Mapping[str, int] = MappingProxyType({
    "vortical": VORTICAL,
    "kelvin+": KELVIN_PLUS,
    "kelvin-": KELVIN_MINUS,
    "wave+": WAVE_PLUS,
    "wave-": WAVE_MINUS,
    "constraint": CONSTRAINT,
})

#: Integer label code -> family name (the reverse of ``FAMILIES``).
FAMILY_NAMES: Mapping[int, str] = MappingProxyType({
    code: name for name, code in FAMILIES.items()})

#: bounded axis name -> the wall-normal velocity component.
_NORMAL_VELOCITY = MappingProxyType({"x": "u", "y": "v"})


# ================================================================
#  The Leray-projector overlap probe (model.constrain)
# ================================================================
def constraint_overlap(
    model: Model,
    basis: ChannelEigenbasis,
    *,
    at_time: float = 0.0,
) -> jax.Array:
    r"""
    Probe the per-plane overlap ``A = Q^H M (P Q)`` of the basis.

    Description
    -----------
    ``P`` is the M-orthogonal Leray projector the pressure CONSTRAINT
    stage realizes, evaluated through the public ``model.constrain``
    matvec — never rebuilt from operator internals. Per eigenvector
    column ``j`` the full-spectrum plane amplitudes ``q[..., j]`` are
    synthesized into real physical states (two per column: the real
    and imaginary parts, so the self-conjugate ``kz`` planes — where
    a real state cannot carry an arbitrary complex amplitude —
    recombine exactly), constrained, and read out per plane with the
    engine's ``rfftn`` convention. On the eigenbasis of ``P L P``
    the restriction of ``A`` to a plane's zero-frequency columns is
    Hermitian with eigenvalues exactly 0 (divergence-complement) or
    1 (physical steady) — the labeler's crisp split.

    A host-side analysis probe (gathers to host, like the engine
    probe itself): ``2 D`` constrained solves.

    Parameters
    ----------
    model : Model
        The assembled channel model the basis was probed on.
    basis : ChannelEigenbasis
        The engine basis (3-D channel: ``omega`` of shape
        ``(n_kx, n_kz, D)``).
    at_time : float, optional
        Evaluation time for time-dependent parameters
        (default: 0.0).

    Returns
    -------
    jax.Array
        The complex overlap planes, shape ``(n_kx, n_kz, D, D)``.
    """
    q = np.asarray(basis.q)
    if q.ndim != 4:  # noqa: PLR2004 — (n_kx, n_kz, D, D) planes
        raise ValueError(
            "constraint_overlap serves the 3-D channel (two periodic "
            "axes + one bounded axis): expected q of shape "
            f"(n_kx, n_kz, D, D), got {q.shape}")
    metric = np.asarray(basis.metric)
    dim = q.shape[-1]
    p_columns = np.stack(
        [_constrained_column(model, basis, q[..., j], at_time)
         for j in range(dim)], axis=-1)
    return jnp.asarray(np.einsum(
        "...di,d,...dj->...ij", np.conj(q), metric, p_columns))


def _constrained_column(
    model: Model,
    basis: ChannelEigenbasis,
    column: np.ndarray,
    at_time: float,
) -> np.ndarray:
    r"""
    Apply the Leray projector to one column's plane amplitudes.

    Description
    -----------
    The plane amplitudes (shape ``(n_kx, n_kz, D)``) are synthesized
    onto the physical grid (inverse FFT of the stored half-spectrum
    slots), split into the real states ``Re(zeta)`` and
    ``Re(i zeta)``, constrained, and read out with ``rfftn``. The
    projector is real and block-diagonal per plane, so the interior
    ``kz`` planes recover ``P q`` from the first probe alone (their
    conjugate mirrors are unstored) while the self-conjugate ``kz``
    planes — where the real synthesis folds the plane onto its
    ``-kx`` mirror — recombine the two probes as ``r1 - i r2``.

    Returns
    -------
    np.ndarray
        The projected plane amplitudes ``P q``, shape
        ``(n_kx, n_kz, D)``.
    """
    template = model.state
    names = model.grid.names
    bounded = names.index(basis.bounded_axis)
    axes = tuple(i for i in range(len(names)) if i != bounded)
    n_kz = column.shape[1]
    n_last = template[basis.components[0]].data.shape[axes[-1]]
    reads = []
    for factor in (1.0, 1j):
        fields = {}
        for name in basis.components:
            seg = column[..., basis.slices[name]]
            shape = template[name].data.shape
            full = np.zeros(shape, dtype=complex)
            key = [slice(None)] * len(shape)
            key[axes[-1]] = slice(0, n_kz)
            # the plane axes keep relative grid order, so moving the
            # segment axis to the bounded slot restores array order
            full[tuple(key)] = np.moveaxis(factor * seg, -1, bounded)
            fields[name] = np.real(np.fft.ifftn(full, axes=axes))
        state = template.replace(**{
            name: template[name].with_data(fields[name])
            for name in basis.components})
        out = model.constrain(state, t=at_time)
        reads.append({
            name: np.moveaxis(
                np.fft.rfftn(np.asarray(out[name].data), axes=axes),
                bounded, -1)
            for name in basis.components})
    r1, r2 = reads
    result = np.zeros_like(column)
    self_conj = [0]
    if n_last % 2 == 0:
        self_conj.append(n_kz - 1)
    for name in basis.components:
        planes = 2.0 * r1[name]
        for ikz in self_conj:
            planes[:, ikz] = r1[name][:, ikz] - 1j * r2[name][:, ikz]
        result[..., basis.slices[name]] = planes
    return result


# ================================================================
#  The labeler
# ================================================================
def label_channel_modes(
    basis: ChannelEigenbasis,
    *,
    overlap: jax.Array,
    zero_tol: float = 1e-8,
    kelvin_tol: float = 1e-20,
    degeneracy_tol: float = 1e-10,
    gap_ratio: float = 10.0,
    constraint_tol: float = 1e-6,
    override: Callable[[ChannelEigenbasis, jax.Array], jax.Array]
    | None = None,
) -> jax.Array:
    r"""
    Label the 3-D channel eigenbasis with the nonhydro families.

    Description
    -----------
    A host-side classifier of the per-``(kx, kz)`` columns:

    1. The zero-frequency columns (``|omega| < zero_tol``) are split
       by the Leray overlap ``A`` (:func:`constraint_overlap`): the
       zero-space restriction of ``A`` is diagonalized, its unitary
       eigenbasis is written back into ``basis.q`` (a rotation
       within the degenerate zero eigenspace — M-orthonormality and
       the eigen-relation are preserved), and eigenvalues within
       ``constraint_tol`` of 0 -> **constraint**, of 1 ->
       **vortical**. Anything non-crisp (never observed; margins
       measured :math:`\sim 10^{-15}`) stays ``UNLABELED``.
    2. On ``kx != 0``, ``kz != 0`` planes, nonzero-frequency columns
       whose wall-normal velocity energy is machine-crisp
       (``< kelvin_tol``) -> **kelvin+/-** by the sign of ``omega``,
       with the shared 2-cluster Gram recovery for near-degenerate
       mixtures. The ``kz = 0`` planes are excluded deliberately:
       their vertically uniform buoyancy oscillations are wall-normal
       free too (``u = v = 0``) and belong to the wave family. On
       the even-``kz`` Nyquist planes buoyancy decouples and the
       Kelvin frequency vanishes — no crisp nonzero column exists
       there and the step finds nothing.
    3. The remaining nonzero columns: the structural wave count is
       ``2 n`` on the ``kz = 0`` planes (``n`` buoyancy strata per
       sign) and ``2 (n - 1)`` elsewhere (the Poincaré strata),
       ``n`` the bounded-axis cell count. An exact count labels
       everything **wave+/-** by sign (the f-plane case); under beta
       the vortical branch acquires slow Rossby frequencies and the
       slow band is split off only across a clean spectral gap
       (``gap_ratio``) — without one the remainder stays
       ``UNLABELED`` and predicates are the primary tool.

    Parameters
    ----------
    basis : ChannelEigenbasis
        The engine basis to classify (3-D channel: ``omega`` of
        shape ``(n_kx, n_kz, D)``).
    overlap : jax.Array
        The Leray overlap planes ``A = Q^H M P Q``, shape
        ``(n_kx, n_kz, D, D)`` (:func:`constraint_overlap`).
    zero_tol : float, optional
        Absolute frequency tolerance for the zero-space test
        (default: 1e-8).
    kelvin_tol : float, optional
        The wall-normal energy bound below which a unit column
        counts as Kelvin (default: 1e-20).
    degeneracy_tol : float, optional
        Relative frequency-cluster width for the Kelvin recovery
        rotation (default: 1e-10).
    gap_ratio : float, optional
        The documented slow/wave spectral-gap factor under beta
        (default: 10.0).
    constraint_tol : float, optional
        Distance of a zero-space overlap eigenvalue from {0, 1}
        beyond which the column stays unlabeled (default: 1e-6).
    override : Callable[[ChannelEigenbasis, jax.Array], jax.Array] \
| None, optional
        Hook applied last: receives the basis and the automatic
        labels, returns adjusted labels (default: None).

    Returns
    -------
    jax.Array
        Integer labels of shape ``omega.shape`` (:data:`FAMILIES`
        codes; unresolved columns are ``UNLABELED``).
    """
    omega = np.asarray(basis.omega)
    if omega.ndim != 3:  # noqa: PLR2004 — (n_kx, n_kz, D) planes
        raise ValueError(
            "the nonhydro channel labeler serves the 3-D channel "
            "(two periodic axes + one bounded axis): expected omega "
            f"of shape (n_kx, n_kz, D), got {omega.shape}")
    overlap_np = np.asarray(overlap)
    if overlap_np.shape != (*omega.shape, omega.shape[-1]):
        raise ValueError(
            "the Leray overlap must hold one (D, D) plane per mode "
            f"plane: expected shape {(*omega.shape, omega.shape[-1])}"
            f", got {overlap_np.shape}")
    q = np.array(basis.q)  # a copy — the labeler rotates columns
    metric = np.asarray(basis.metric)
    normal = _NORMAL_VELOCITY[basis.bounded_axis]
    normal_slice = basis.slices[normal]
    n = normal_slice.stop - normal_slice.start + 1  # cell count

    labels = np.full(omega.shape, UNLABELED, dtype=np.int32)
    for ikx in range(omega.shape[0]):
        for ikz in range(omega.shape[1]):
            _label_plane(
                labels[ikx, ikz], q[ikx, ikz], omega[ikx, ikz],
                overlap_np[ikx, ikz], metric, normal_slice,
                n_wave=2 * n if ikz == 0 else 2 * (n - 1),
                kelvin_allowed=ikx != 0 and ikz != 0,
                zero_tol=zero_tol, kelvin_tol=kelvin_tol,
                degeneracy_tol=degeneracy_tol, gap_ratio=gap_ratio,
                constraint_tol=constraint_tol)
    basis.q = jnp.asarray(q)  # the zero-space split always rotates
    result = jnp.asarray(labels)
    if override is not None:
        result = override(basis, result)
    return result


def _label_plane(
    labels: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    overlap: np.ndarray,
    metric: np.ndarray,
    normal_slice: slice,
    *,
    n_wave: int,
    kelvin_allowed: bool,
    zero_tol: float,
    kelvin_tol: float,
    degeneracy_tol: float,
    gap_ratio: float,
    constraint_tol: float,
) -> None:
    """Label one ``(kx, kz)`` plane in place (``q`` rotates)."""
    zero = np.abs(omega) < zero_tol
    _split_zero_space(labels, q, np.where(zero)[0], overlap,
                      constraint_tol)
    kelvin = np.zeros(omega.shape, dtype=bool)
    if kelvin_allowed:
        energy = segment_energy(q, metric, normal_slice)
        crisp = ~zero & (energy < kelvin_tol)
        for sign, code in ((1, KELVIN_PLUS), (-1, KELVIN_MINUS)):
            cand = crisp & (np.sign(omega) == sign)
            if not cand.any():
                col = recover_crisp_column(
                    q, omega, ~zero & (np.sign(omega) == sign),
                    metric, normal_slice, energy_tol=kelvin_tol,
                    degeneracy_tol=degeneracy_tol)
                if col is None:
                    continue
                cand[col] = True
            labels[cand] = code
            kelvin |= cand
    split_frequency_bands(
        labels, omega, ~zero & ~kelvin, n_fast=n_wave,
        gap_ratio=gap_ratio, slow_code=VORTICAL,
        fast_plus_code=WAVE_PLUS, fast_minus_code=WAVE_MINUS)


def _split_zero_space(
    labels: np.ndarray,
    q: np.ndarray,
    idx: np.ndarray,
    overlap: np.ndarray,
    constraint_tol: float,
) -> None:
    r"""
    Split a plane's zero space into constraint and steady columns.

    Description
    -----------
    Restricted to the zero-frequency columns the Leray overlap is
    Hermitian with eigenvalues exactly 0 (divergence-complement) or
    1 (physical steady): its unitary eigenbasis rotates the columns
    of ``q`` **in place** so each is purely one or the other, and
    the labels follow the eigenvalues. Non-crisp eigenvalues
    (further than ``constraint_tol`` from both 0 and 1) leave their
    columns ``UNLABELED`` — defensive; never observed.
    """
    if idx.size == 0:
        return
    sub = overlap[np.ix_(idx, idx)]
    sub = 0.5 * (sub + np.conj(sub.T))
    evals, evecs = np.linalg.eigh(sub)
    q[:, idx] = q[:, idx] @ evecs
    complement = np.abs(evals) < constraint_tol
    steady = np.abs(evals - 1.0) < constraint_tol
    labels[idx[complement]] = CONSTRAINT
    labels[idx[steady]] = VORTICAL


# ================================================================
#  The labeled channel eigenmode surface
# ================================================================
class ChannelEigenmodes(ChannelEigenmodesBase):

    r"""
    Labeled numeric eigenmodes of the walled nonhydro channel.

    Description
    -----------
    The nonhydrostatic subclass of the shared
    :class:`~fridom.framework2.model.eigenbasis.ChannelEigenmodesBase`
    wrapper: the framework's dense-column channel eigensolve of the
    projected linearization ``P L P``, labeled by
    :func:`label_channel_modes` with the Leray overlap probed
    through ``model.constrain`` (:func:`constraint_overlap`). The
    family vocabulary is the class-level :attr:`families` name ->
    code map (reverse: :attr:`family_names`) — the physical
    families **vortical**, **kelvin+/-**, **wave+/-** plus the
    non-physical **constraint** family (the divergence-complement
    directions, excluded from every physical projector; the
    physical families sum to the Leray projector ``P``, and
    ``constraint`` to its complement ``I - P``).

    The passthrough surface and the ``projector(sel)`` family /
    predicate projections come from the base (the engine path reads
    ``labels``/``q``/``metric`` off this object *after* labeling,
    whose zero-space split and Kelvin degeneracy recovery rotate
    ``basis.q`` in place).

    Parameters
    ----------
    model : Model
        An assembled nonhydro channel model (exactly one bounded
        horizontal axis; the pressure CONSTRAINT stage present).
    at_time : float, optional
        Evaluation time for time-dependent parameters
        (default: 0.0).
    chunk : int | None, optional
        Probe/eigensolve batch size, bounds peak memory
        (default: None).
    zero_tol, kelvin_tol, degeneracy_tol, gap_ratio, \
constraint_tol : float, optional
        The labeler tolerances (see :func:`label_channel_modes`).
    override : Callable[[ChannelEigenbasis, jax.Array], jax.Array] \
| None, optional
        Label-override hook, applied after the automatic pass
        (default: None).
    """

    #: Family name -> integer label code.
    families: ClassVar[Mapping[str, int]] = FAMILIES
    #: Integer label code -> family name.
    family_names: ClassVar[Mapping[int, str]] = FAMILY_NAMES
    #: Vocabulary class wrapping the ``mode()`` states.
    state_class: ClassVar[type] = State
    #: The divergence complement is an engine artifact, not a
    #: physical mode selection.
    nonphysical_families: ClassVar[tuple[str, ...]] = ("constraint",)

    def __init__(
        self,
        model: Model,
        *,
        at_time: float = 0.0,
        chunk: int | None = None,
        zero_tol: float = 1e-8,
        kelvin_tol: float = 1e-20,
        degeneracy_tol: float = 1e-10,
        gap_ratio: float = 10.0,
        constraint_tol: float = 1e-6,
        override: Callable[[ChannelEigenbasis, jax.Array], jax.Array]
        | None = None,
    ) -> None:
        """Solve the constrained eigenproblem and label the families."""

        def labeler(basis: ChannelEigenbasis) -> jax.Array:
            """Probe the Leray overlap, then run the labeler."""
            overlap = constraint_overlap(model, basis,
                                         at_time=at_time)
            return label_channel_modes(
                basis, overlap=overlap, zero_tol=zero_tol,
                kelvin_tol=kelvin_tol, degeneracy_tol=degeneracy_tol,
                gap_ratio=gap_ratio, constraint_tol=constraint_tol,
                override=override)

        super().__init__(model, labeler=labeler, at_time=at_time,
                         chunk=chunk)
