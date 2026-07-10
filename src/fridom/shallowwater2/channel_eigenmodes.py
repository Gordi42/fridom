r"""
Channel (walled) shallow-water eigenmodes: engine wrapper + labeler.

Description
-----------
The shallow-water face of the dense-column channel eigenbasis
(:func:`fridom.framework2.channel_eigenpairs`). The walled-``y``
rotating channel carries three physical mode families per zonal
wavenumber ``kx``:

- **vortical** — :math:`\omega \approx 0` on the f-plane (including
  the two exponential wall modes at ``kx = 0``); under a beta-plane
  ``f(y)`` the branch acquires slow westward Rossby frequencies;
- **kelvin+ / kelvin-** — the boundary-trapped Kelvin pair with
  exactly zero meridional-velocity energy (``kx != 0`` planes only;
  the branch sign is the sign of the engine's ``omega``);
- **wave+ / wave-** — the remaining Poincaré (inertia-gravity)
  branches, split by the sign of ``omega``. Wave excludes Kelvin.

The framework engine stays family-agnostic; this module owns the
physics. :func:`label_channel_modes` classifies the columns of a
:class:`~fridom.framework2.model.eigen_channel.ChannelEigenbasis` —
exact on the f-plane, best-effort (graceful) under beta — and
:class:`ChannelEigenmodes` bundles ``channel_eigenpairs`` with that
labeler behind a passthrough surface for the downstream family
projections.
"""
from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp
import numpy as np

from fridom.framework2.model.eigen_channel import (
    UNLABELED,
    channel_eigenpairs,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.framework2.model.eigen_channel import ChannelEigenbasis
    from fridom.framework2.model.model import Model

# ================================================================
#  The family label codes
# ================================================================
VORTICAL = 0
KELVIN_PLUS = 1
KELVIN_MINUS = 2
WAVE_PLUS = 3
WAVE_MINUS = 4

#: Family name -> integer label code (the ``labels`` vocabulary).
FAMILIES: Mapping[str, int] = MappingProxyType({
    "vortical": VORTICAL,
    "kelvin+": KELVIN_PLUS,
    "kelvin-": KELVIN_MINUS,
    "wave+": WAVE_PLUS,
    "wave-": WAVE_MINUS,
})

#: Integer label code -> family name (the reverse of ``FAMILIES``).
FAMILY_NAMES: Mapping[int, str] = MappingProxyType({
    code: name for name, code in FAMILIES.items()})


# ================================================================
#  The labeler (best-effort family classification)
# ================================================================
def label_channel_modes(
    basis: ChannelEigenbasis,
    *,
    zero_tol: float = 1e-8,
    kelvin_tol: float = 1e-20,
    degeneracy_tol: float = 1e-10,
    gap_ratio: float = 10.0,
    override: Callable[[ChannelEigenbasis, jax.Array], jax.Array]
    | None = None,
) -> jax.Array:
    r"""
    Label the channel eigenbasis columns with the SW mode families.

    Description
    -----------
    A host-side, best-effort classifier of the per-``kx`` columns:

    1. ``|omega| < zero_tol`` -> **vortical** (exact on the f-plane,
       where the family is steady; the ``kx = 0`` wall modes are
       zero-frequency too and land here).
    2. On ``kx != 0`` planes, nonzero-frequency columns whose
       ``v``-segment energy under the metric is machine-crisp
       (``< kelvin_tol``; measured :math:`\sim 10^{-30}` for unit
       columns, on the beta plane as well) -> **kelvin+/-** by the
       sign of ``omega``. If ``eigh`` mixed a Kelvin column into a
       near-degenerate frequency cluster (``|d omega| <
       degeneracy_tol`` relative), no column is crisp; the 2-cluster
       is then rotated by the eigenbasis of its 2x2 ``v``-energy
       Gram (a tiny sub-``eigh``) to expose the ``v``-free
       direction. The rotation is written back into ``basis.q`` (a
       unitary column mixing — M-orthonormality is preserved; the
       eigen-relation residual of the pair changes only at the
       cluster's frequency splitting).
    3. The remaining nonzero columns: with ``n_u`` bounded-axis
       ``u`` nodes the wave (Poincaré) branch structurally holds
       ``2 (n_u - 1)`` columns per plane. If exactly that many
       remain (the f-plane case) they are **wave+/-** by sign.
       Under beta the vortical branch acquires slow Rossby
       frequencies and extra columns remain; they are split from
       the wave band only if a clean spectral gap exists (smallest
       wave ``|omega|`` at least ``gap_ratio`` times the largest
       slow ``|omega|``): the slow band -> **vortical**, the fast
       band -> **wave+/-**. Without a clean gap (or with fewer
       columns than the wave count) the remainder stays
       :data:`~fridom.framework2.model.eigen_channel.UNLABELED` —
       predicates are the primary tool there.

    Parameters
    ----------
    basis : ChannelEigenbasis
        The engine basis to classify (2-D channel: ``omega`` of
        shape ``(n_kx, D)``).
    zero_tol : float, optional
        Absolute frequency tolerance for the vortical/steady test
        (default: 1e-8).
    kelvin_tol : float, optional
        The ``v``-segment energy bound below which a unit column
        counts as Kelvin (default: 1e-20).
    degeneracy_tol : float, optional
        Relative frequency-cluster width for the Kelvin recovery
        rotation (default: 1e-10).
    gap_ratio : float, optional
        The documented slow/wave spectral-gap factor under beta
        (default: 10.0).
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
    if omega.ndim != 2:  # noqa: PLR2004 — (n_kx, D) planes
        raise ValueError(
            "the shallow-water channel labeler serves the 2-D "
            "channel (one periodic axis + one bounded axis): "
            f"expected omega of shape (n_kx, D), got {omega.shape}")
    q = np.array(basis.q)  # a copy — the Kelvin recovery may rotate
    metric = np.asarray(basis.metric)
    n_u = basis.slices["u"].stop - basis.slices["u"].start

    labels = np.full(omega.shape, UNLABELED, dtype=np.int32)
    rotated = False
    for ikx in range(omega.shape[0]):
        rotated |= _label_plane(
            labels[ikx], q[ikx], omega[ikx], metric,
            basis.slices["v"], n_wave=2 * (n_u - 1),
            kelvin_allowed=ikx != 0, zero_tol=zero_tol,
            kelvin_tol=kelvin_tol, degeneracy_tol=degeneracy_tol,
            gap_ratio=gap_ratio)
    if rotated:
        basis.q = jnp.asarray(q)
    result = jnp.asarray(labels)
    if override is not None:
        result = override(basis, result)
    return result


def _label_plane(
    labels: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    metric: np.ndarray,
    v_slice: slice,
    *,
    n_wave: int,
    kelvin_allowed: bool,
    zero_tol: float,
    kelvin_tol: float,
    degeneracy_tol: float,
    gap_ratio: float,
) -> bool:
    """Label one ``kx`` plane in place; return whether ``q`` rotated."""
    zero = np.abs(omega) < zero_tol
    labels[zero] = VORTICAL
    kelvin = np.zeros(omega.shape, dtype=bool)
    rotated = False
    if kelvin_allowed:
        v_energy = _v_energy(q, metric, v_slice)
        crisp = ~zero & (v_energy < kelvin_tol)
        for sign, code in ((1, KELVIN_PLUS), (-1, KELVIN_MINUS)):
            cand = crisp & (np.sign(omega) == sign)
            if not cand.any():
                col = _recover_kelvin(
                    q, omega, ~zero & (np.sign(omega) == sign),
                    metric, v_slice, kelvin_tol=kelvin_tol,
                    degeneracy_tol=degeneracy_tol)
                if col is None:
                    continue
                cand[col] = True
                rotated = True
            labels[cand] = code
            kelvin |= cand
    _label_wave_bands(labels, omega, ~zero & ~kelvin,
                      n_wave=n_wave, gap_ratio=gap_ratio)
    return rotated


def _v_energy(
    q: np.ndarray, metric: np.ndarray, v_slice: slice,
) -> np.ndarray:
    """Per-column ``v``-segment M-energy of the (unit) columns."""
    return np.einsum("ij,i->j", np.abs(q[v_slice, :]) ** 2,
                     metric[v_slice])


def _recover_kelvin(
    q: np.ndarray,
    omega: np.ndarray,
    candidates: np.ndarray,
    metric: np.ndarray,
    v_slice: slice,
    *,
    kelvin_tol: float,
    degeneracy_tol: float,
) -> int | None:
    r"""
    Rotate a near-degenerate 2-cluster to expose a ``v``-free column.

    Description
    -----------
    When a Kelvin column is nearly degenerate with a Poincaré
    column, ``eigh`` may return an arbitrary mixture of the pair;
    neither mixed column is then ``v``-crisp. For each adjacent
    same-sign pair within ``degeneracy_tol`` (relative), the 2x2
    ``v``-energy Gram is diagonalized: a near-zero smallest
    eigenvalue means the 2-space contains a ``v``-free direction,
    and the unitary Gram eigenbasis rotates the two columns of ``q``
    **in place** to expose it (column ``j`` becomes the minimal-
    ``v``-energy combination).

    Returns
    -------
    int | None
        The recovered Kelvin column index, or None if no cluster
        yields a ``v``-free direction.
    """
    cols = np.where(candidates)[0]
    for pos in range(len(cols) - 1):
        j, k = cols[pos], cols[pos + 1]
        scale = max(1.0, abs(omega[j]))
        if abs(omega[k] - omega[j]) > degeneracy_tol * scale:
            continue
        pair = q[:, [j, k]]
        weighted = metric[v_slice, None] * pair[v_slice]
        gram = pair[v_slice].conj().T @ weighted
        evals, evecs = np.linalg.eigh(gram)
        if evals[0] >= kelvin_tol:
            continue
        q[:, [j, k]] = pair @ evecs
        return int(j)
    return None


def _label_wave_bands(
    labels: np.ndarray,
    omega: np.ndarray,
    rest: np.ndarray,
    *,
    n_wave: int,
    gap_ratio: float,
) -> None:
    """Split the remaining columns into slow/vortical and wave+/-."""
    n_slow = int(rest.sum()) - n_wave
    if n_slow < 0:
        return  # unexpected plane structure: stay UNLABELED
    if n_slow > 0:
        mags = np.sort(np.abs(omega[rest]))
        if mags[n_slow] < gap_ratio * mags[n_slow - 1]:
            return  # no clean spectral gap: stay UNLABELED
        slow = rest & (np.abs(omega)
                       < 0.5 * (mags[n_slow - 1] + mags[n_slow]))
        labels[slow] = VORTICAL
        rest = rest & ~slow
    labels[rest & (omega > 0)] = WAVE_PLUS
    labels[rest & (omega < 0)] = WAVE_MINUS


# ================================================================
#  The labeled channel eigenmode surface
# ================================================================
class ChannelEigenmodes:

    r"""
    Labeled numeric eigenmodes of the walled shallow-water channel.

    Description
    -----------
    Bundles the framework's dense-column channel eigensolve
    (:func:`fridom.framework2.channel_eigenpairs`) with the
    shallow-water family labeler (:func:`label_channel_modes`) and
    exposes the labeled basis: ``omega``/``q``/``labels`` per
    ``rfft`` plane of the periodic axis, the segment ``slices`` and
    the diagonal energy ``metric`` (see
    :class:`~fridom.framework2.model.eigen_channel.ChannelEigenbasis`
    for the layout conventions). The family vocabulary is the
    class-level :attr:`families` name -> code map (reverse:
    :attr:`family_names`).

    A host-side analysis object; the downstream family projections
    consume ``labels``/``q``/``metric``.

    Parameters
    ----------
    model : Model
        An assembled shallow-water channel model (exactly one
        bounded axis).
    at_time : float, optional
        Evaluation time for time-dependent parameters
        (default: 0.0).
    chunk : int | None, optional
        Probe/eigensolve batch size, bounds peak memory
        (default: None).
    zero_tol, kelvin_tol, degeneracy_tol, gap_ratio : float, optional
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
        override: Callable[[ChannelEigenbasis, jax.Array], jax.Array]
        | None = None,
    ) -> None:
        """Solve the channel eigenproblem and label the families."""
        self.basis: ChannelEigenbasis = channel_eigenpairs(
            model, at_time=at_time, chunk=chunk)
        self.basis.label_with(partial(
            label_channel_modes, zero_tol=zero_tol,
            kelvin_tol=kelvin_tol, degeneracy_tol=degeneracy_tol,
            gap_ratio=gap_ratio, override=override))

    # ================================================================
    #  Passthrough surface (the labeled basis)
    # ================================================================
    @property
    def omega(self) -> jax.Array:
        """Real frequencies, shape ``(n_kx, D)``, ascending per plane."""
        return self.basis.omega

    @property
    def q(self) -> jax.Array:
        """M-orthonormal eigenvector columns, shape ``(n_kx, D, D)``."""
        return self.basis.q

    @property
    def labels(self) -> jax.Array:
        """Per-column family codes (:attr:`families` vocabulary)."""
        return self.basis.labels

    @property
    def components(self) -> tuple[str, ...]:
        """The stacked-column segment order (``u``, ``v``, ``p``)."""
        return self.basis.components

    @property
    def slices(self) -> Mapping[str, slice]:
        """Per-component segment slices into the stacked ``D`` axis."""
        return self.basis.slices

    @property
    def metric(self) -> jax.Array:
        """The diagonal energy metric ``M``, shape ``(D,)``."""
        return self.basis.metric

    @property
    def periodic_axis(self) -> str:
        """The half-spectrum (``rfft``) periodic axis name."""
        return self.basis.periodic_axis

    @property
    def bounded_axis(self) -> str:
        """The bounded (walled) axis name the columns stack."""
        return self.basis.bounded_axis
