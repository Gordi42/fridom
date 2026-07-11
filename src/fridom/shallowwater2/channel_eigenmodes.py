r"""
Channel (walled) shallow-water eigenmodes: engine wrapper + labeler.

Description
-----------
The shallow-water face of the dense-column channel eigenbasis
(:func:`fridom.channel_eigenpairs`). The walled-``y``
rotating channel carries three physical mode families per zonal
wavenumber ``kx``:

- **vortical** — :math:`\omega \approx 0` on the f-plane (including
  the two exponential wall modes at ``kx = 0``); under a beta-plane
  ``f(y)`` the branch acquires slow westward-propagating Rossby
  frequencies — negative ``omega`` for positive ``kx`` (positive
  ``omega`` propagates eastward);
- **kelvin+ / kelvin-** — the boundary-trapped Kelvin pair with
  exactly zero meridional-velocity energy (``kx != 0`` planes only;
  the branch sign is the sign of the engine's ``omega``):
  ``kelvin+`` is the eastward branch — at the equatorial
  ``f0 = 0`` beta plane the equatorially trapped Kelvin wave —
  and ``kelvin-`` the westward boundary Kelvin wave;
- **wave+ / wave-** — the remaining Poincaré (inertia-gravity)
  branches, split by the sign of ``omega``. Wave excludes Kelvin.

The framework engine stays family-agnostic; this module owns the
physics. :func:`label_channel_modes` classifies the columns of a
:class:`~fridom.model.eigen_channel.ChannelEigenbasis` through the
Kelvin separatrix (exact for any Coriolis profile ``f(y)``,
including the equatorial ``f0 = 0`` beta plane), and
:class:`ChannelEigenmodes` bundles ``channel_eigenpairs`` with that
labeler behind the shared
:class:`~fridom.model._eigenbasis.ChannelEigenmodesBase`
passthrough surface for the downstream family projections.
"""
from __future__ import annotations

from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp
import numpy as np

from fridom.model._eigenbasis import (
    ChannelEigenmodesBase,
    recover_crisp_column,
    segment_energy,
    split_frequency_bands,
)
from fridom.model.eigen_channel import (
    UNLABELED,
    ChannelEigenbasis,
)
from fridom.shallowwater2.state import State

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.model.model import Model

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
       Gram (the shared
       :func:`~fridom.model._eigenbasis.recover_crisp_column`
       helper) to expose the ``v``-free direction. The rotation is
       written back into ``basis.q`` (a unitary column mixing —
       M-orthonormality is preserved; the eigen-relation residual of
       the pair changes only at the cluster's frequency splitting).
    3. The remaining nonzero columns split at the **Kelvin
       separatrix**: the Kelvin eigenspaces are one-dimensional for
       any Coriolis profile ``f(y)``, so no other eigenvalue can
       coincide with a Kelvin frequency. Columns slower than the
       slowest Kelvin branch -> **vortical** (the steady f-plane
       family; slow westward Rossby modes — negative ``omega`` —
       under beta), faster
       columns -> **wave+/-** by frequency sign. The rule is exact
       on the f-plane and for any beta, including the equatorial
       ``f0 = 0`` regime, where it reproduces the adiabatic
       fast--slow prescription (the Rossby--Yanai branch counts as
       fast exactly when ``beta >= 2 k^2``).
    4. On Kelvin-less planes (``kx = 0``) the frequency-band
       fallback applies: with ``n_u`` bounded-axis ``u`` nodes the
       wave branch structurally holds ``2 (n_u - 1)`` columns; an
       exact count labels by sign, extra columns split off across a
       clean spectral gap (``gap_ratio``), and without one the
       remainder stays
       :data:`~fridom.model.eigen_channel.UNLABELED` —
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
        v_energy = segment_energy(q, metric, v_slice)
        crisp = ~zero & (v_energy < kelvin_tol)
        for sign, code in ((1, KELVIN_PLUS), (-1, KELVIN_MINUS)):
            cand = crisp & (np.sign(omega) == sign)
            if not cand.any():
                col = recover_crisp_column(
                    q, omega, ~zero & (np.sign(omega) == sign),
                    metric, v_slice, energy_tol=kelvin_tol,
                    degeneracy_tol=degeneracy_tol)
                if col is None:
                    continue
                cand[col] = True
                rotated = True
            labels[cand] = code
            kelvin |= cand
    rest = ~zero & ~kelvin
    if kelvin.any():
        # The Kelvin separatrix: the Kelvin eigenspaces are strictly
        # one-dimensional for ANY Coriolis profile f(y), so no other
        # eigenvalue can coincide with a Kelvin frequency. Everything
        # slower than the slowest Kelvin branch is therefore the slow
        # (vortical; Rossby under beta) family, everything faster the
        # inertia-gravity branches. Exact on the f-plane (the Poincare
        # minimum sqrt(f^2 + c^2 k^2) exceeds |omega_K| = c k) and for
        # any beta including the equatorial f0 = 0 regime, where the
        # Rossby-Yanai branch classifies fast iff beta >= 2 k^2 (the
        # adiabatic fast-slow mode-mapping prescription).
        separatrix = np.abs(omega[kelvin]).min()
        slow = rest & (np.abs(omega) < separatrix)
        labels[slow] = VORTICAL
        labels[rest & ~slow & (omega > 0)] = WAVE_PLUS
        labels[rest & ~slow & (omega < 0)] = WAVE_MINUS
    else:
        # Kelvin-less planes (kx = 0): the frequency-band fallback
        split_frequency_bands(
            labels, omega, rest, n_fast=n_wave,
            gap_ratio=gap_ratio, slow_code=VORTICAL,
            fast_plus_code=WAVE_PLUS, fast_minus_code=WAVE_MINUS)
    return rotated


# ================================================================
#  The labeled channel eigenmode surface
# ================================================================
class ChannelEigenmodes(ChannelEigenmodesBase):

    r"""
    Labeled numeric eigenmodes of the walled shallow-water channel.

    Description
    -----------
    The shallow-water subclass of the shared
    :class:`~fridom.model._eigenbasis.ChannelEigenmodesBase`
    wrapper: the framework's dense-column channel eigensolve labeled
    by :func:`label_channel_modes`. The family vocabulary is the
    class-level :attr:`families` name -> code map (reverse:
    :attr:`family_names`); the passthrough surface and the
    ``projector(sel)`` family / predicate projections come from the
    base (the engine path reads ``labels``/``q``/``metric`` off this
    object *after* labeling, because the Kelvin degeneracy recovery
    may rotate ``basis.q`` in place).

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
    #: Vocabulary class wrapping the ``mode()`` states.
    state_class: ClassVar[type] = State

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
        super().__init__(
            model,
            labeler=partial(
                label_channel_modes, zero_tol=zero_tol,
                kelvin_tol=kelvin_tol, degeneracy_tol=degeneracy_tol,
                gap_ratio=gap_ratio, override=override),
            at_time=at_time, chunk=chunk)
