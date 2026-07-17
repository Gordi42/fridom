r"""Numeric eigenmodes of the discrete hydrostatic C-grid (HY-D7).

Description
-----------
The hydrostatic "eigenvectors" deliverable of ROADMAP 3.1: the
labeled eigenbasis of the **full** discrete linear operator of the
explicit hydrostatic model on the doubly-periodic flat-bottom grid
(periodic ``x``/``y``, bounded ``z`` with the flat bottom and the free
surface). The prognostic state is ``(u, v, b, ps)`` — ``w`` and
``p_hyd`` are diagnosed inside the linear operator (the S1' DIAGNOSE
stages), so the barotropic free-surface coupling and the baroclinic
hydrostatic coupling are both present in the probed operator.

Free-surface treatment / design decision
-----------------------------------------
Unlike the nonhydro2 walled-vertical grid — where the vertical modes
diagonalize on clean DST/DCT trig tables and the analytic
operator-symbol eigenmodes are *exact* — the hydrostatic free surface
couples the depth-mean divergence to the surface pressure ``ps`` through
the rank-1 barotropic term ``c^2 (1/H) 1 1^T`` (the depth-integrated
gravity-wave restoring). Writing the discrete vertical operator per
horizontal wavenumber as

.. math::

    V = N^2\, C_p M + c^2\, \Pi,
    \qquad C_p = M^{\mathsf T},\quad
    \Pi = \tfrac{1}{n} \mathbf 1 \mathbf 1^{\mathsf T},

with ``M`` the bottom-up ``w``-continuity cumint composed with the
``Outer -> Center`` average and ``C_p`` the top-down half-cell
hydrostatic cumint, one finds ``C_p = M^{\mathsf T}`` **exactly** (the
staggered cumint pair is a transpose pair), so ``V`` is symmetric: the
vertical structure *does* separate cleanly (real dispersion, orthogonal
vertical modes). But the constant vector ``1`` is **not** an eigenvector
of ``M^{\mathsf T} M`` (``M`` is lower-triangular, invertible), so the
barotropic mode is z-constant only to :math:`O(N^2/c^2)` — the free
surface mixes a small baroclinic ``b`` signature into the barotropic
mode and a small depth-mean-velocity signature into the baroclinic
modes. There is therefore **no exact analytic z-constant barotropic
eigenvector**; the exactly-z-constant barotropic triplet with
``omega^2 = f^2 + c^2 k_disc^2`` is the operator *restricted* to the
barotropic subspace (the H2 ``test_barotropic_poincare_dispersion``
oracle, matched to ~1e-3).

The chosen construction is therefore the **exact numeric eigenbasis of
the assembled linear operator**, built by the shared dense-column
channel engine (:func:`fridom.model.eigen_channel.channel_eigenpairs`)
with the bounded axis ``z``: per horizontal wavenumber the operator is a
dense ``D = 3 n_z + 1`` block probed by unit-impulse columns, and the
whitened generalized ``eigh`` under the hydrostatic energy metric
``diag(1, 1, 1/N^2, 1/c^2)`` (the ``ps`` weight depth-integrated to
``H/c^2``, ``hy.energy``) returns real frequencies and **exactly
M-orthonormal** eigenvectors (biorthogonality and the projection
round-trip hold to machine precision, whatever the barotropic mode's
residual b-content). The eigenvalues *are* the discrete dispersion, so
the discrete-symbol formulas match where they are exact — the baroclinic
``m_disc^2 = N^2 k_h^2/(omega^2 - f^2)`` is identical across horizontal
wavenumbers to machine precision, the barotropic branch matches
``f^2 + c^2 k_disc^2`` to the documented z-constant tolerance.

Because the engine builds ``fr.linearize(model)`` first, an implicit /
rigid-lid variant (which declares ``linear_operator_gap``) is refused
by ``require_linear_operator`` — the model carries barotropic physics
outside ``L``, so its eigenmodes would describe a different system.
:func:`from_model` raises that gap before probing.

Labeler
-------
The six-family vocabulary crosses two classifications — geostrophic vs
inertia-gravity, barotropic vs baroclinic:

- **vortical** (``barotropic_vortical`` / ``baroclinic_vortical``) —
  the ``nz + 1`` steady (``omega ~ 0``) geostrophic columns, split into
  the single barotropic (free-surface) column and the ``nz`` baroclinic
  columns by diagonalizing the depth-mean/``ps`` barotropic overlap on
  the degenerate zero space (rotating ``basis.q`` in place, the
  nonhydro2 zero-space-split precedent);
- **wave** (``barotropic_wave+/-`` / ``baroclinic_wave+/-``) — the
  ``2 nz`` inertia-gravity columns, split by the surface-pressure
  energy of each column (the barotropic pair carries ``ps``, the
  baroclinic pairs carry essentially none) and signed by ``omega``.

``VorticalProjection`` / ``WaveProjection`` and the barotropic /
baroclinic splitter live in :mod:`fridom.hydrostatic.transforms`.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar

import jax.numpy as jnp
import numpy as np

from fridom.hydrostatic.state import State
from fridom.model._eigenbasis import (
    ChannelEigenmodesBase,
    segment_energy,
)
from fridom.model.term_predicates import require_linear_operator

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Mapping

    import jax

    from fridom.model.eigen_channel import ChannelEigenbasis
    from fridom.model.model import Model


# ================================================================
#  The family label codes
# ================================================================
BAROTROPIC_VORTICAL = 0
BAROTROPIC_WAVE_PLUS = 1
BAROTROPIC_WAVE_MINUS = 2
BAROCLINIC_VORTICAL = 3
BAROCLINIC_WAVE_PLUS = 4
BAROCLINIC_WAVE_MINUS = 5

#: Family name -> integer label code (the ``labels`` vocabulary).
FAMILIES: Mapping[str, int] = MappingProxyType({
    "barotropic_vortical": BAROTROPIC_VORTICAL,
    "barotropic_wave+": BAROTROPIC_WAVE_PLUS,
    "barotropic_wave-": BAROTROPIC_WAVE_MINUS,
    "baroclinic_vortical": BAROCLINIC_VORTICAL,
    "baroclinic_wave+": BAROCLINIC_WAVE_PLUS,
    "baroclinic_wave-": BAROCLINIC_WAVE_MINUS,
})

#: Integer label code -> family name (the reverse of ``FAMILIES``).
FAMILY_NAMES: Mapping[int, str] = MappingProxyType({
    code: name for name, code in FAMILIES.items()})


# ================================================================
#  The barotropic overlap (depth-mean velocity + surface pressure)
# ================================================================
def barotropic_projector(
    metric: np.ndarray, slices: Mapping[str, slice],
) -> np.ndarray:
    r"""
    Return the M-orthogonal barotropic projector on the column.

    Description
    -----------
    The barotropic subspace is the **depth-mean** horizontal velocity
    plus the surface pressure ``ps`` (buoyancy ``b`` is purely
    baroclinic). On the stacked ``D``-vector this is the block-diagonal
    ``M``-self-adjoint projector: the measure-weighted mean projector
    on each velocity segment (``(P x)_j = sum_i mu_i x_i / sum_i mu_i``,
    which is ``diag(mu)``-self-adjoint and idempotent), the identity on
    the single ``ps`` slot, and zero on the ``b`` segment. Its overlap
    on the degenerate zero space isolates the one barotropic
    geostrophic column; distinct wave frequencies never need it.

    Parameters
    ----------
    metric : np.ndarray
        The diagonal energy metric ``M``, shape ``(D,)``.
    slices : Mapping[str, slice]
        Per-component segment slices into the stacked ``D`` axis.

    Returns
    -------
    np.ndarray
        The ``(D, D)`` barotropic projector matrix.
    """
    dim = metric.shape[0]
    projector = np.zeros((dim, dim), dtype=metric.dtype)
    # u, v, ps are always present in the hydrostatic prognostic state
    for name in ("u", "v"):
        seg = slices[name]
        mu = metric[seg]
        weights = mu / mu.sum()
        projector[seg, seg] = np.ones((mu.size, 1)) @ weights[None, :]
    ps_seg = slices["ps"]
    projector[ps_seg, ps_seg] = 1.0
    return projector


# ================================================================
#  The labeler
# ================================================================
def label_hydrostatic_modes(
    basis: ChannelEigenbasis,
    *,
    zero_tol: float = 1e-8,
    ps_wave_tol: float = 1e-2,
    barotropic_tol: float = 0.5,
) -> jax.Array:
    r"""
    Label the eigenbasis into the six hydrostatic mode families.

    Description
    -----------
    A host-side classifier of the per-``(kx, ky)`` columns:

    1. **Zero space.** Columns with ``|omega| < zero_tol`` are
       geostrophic. The barotropic overlap ``A = Q_0^H M Pi_bt Q_0``
       (:func:`barotropic_projector`) is diagonalized on the zero
       columns; its unitary eigenbasis rotates ``basis.q`` in place
       (a rotation within the exactly-degenerate ``omega = 0``
       eigenspace preserves M-orthonormality and ``L q = 0``), and the
       column of maximal overlap (``> barotropic_tol``) is
       **barotropic_vortical**, the rest **baroclinic_vortical**.
    2. **Wave modes.** The remaining columns are inertia-gravity. Per
       frequency sign the column of maximal surface-pressure energy
       (``segment_energy`` on the ``ps`` slot), if it clears
       ``ps_wave_tol``, is the **barotropic_wave** of that sign; the
       rest are **baroclinic_wave**. On the ``k_h = 0`` plane the waves
       are pure inertial oscillations carrying no ``ps``, so none clear
       the threshold and all are labeled baroclinic.

    Every column receives a label, so the ``vortical`` and ``wave``
    unions (and the ``barotropic`` / ``baroclinic`` unions) are
    complete — the projection round-trip is exact.

    Parameters
    ----------
    basis : ChannelEigenbasis
        The engine basis to classify (``omega`` of shape
        ``(n_kx, n_ky, D)``); its ``q`` is rotated in place on the
        zero-space split.
    zero_tol : float, optional
        Absolute frequency tolerance for the geostrophic test
        (default: 1e-8).
    ps_wave_tol : float, optional
        Surface-pressure energy above which a wave column counts as
        barotropic (default: 1e-2).
    barotropic_tol : float, optional
        Barotropic-overlap eigenvalue above which the maximal
        zero-space column is barotropic (default: 0.5).

    Returns
    -------
    jax.Array
        Integer labels of shape ``omega.shape`` (:data:`FAMILIES`
        codes).
    """
    omega = np.asarray(basis.omega)
    q = np.array(basis.q)  # a copy — the labeler rotates columns
    metric = np.asarray(basis.metric)
    projector = barotropic_projector(metric, basis.slices)
    ps_slice = basis.slices["ps"]

    labels = np.full(omega.shape, BAROCLINIC_VORTICAL, dtype=np.int32)
    for plane in np.ndindex(omega.shape[:-1]):
        _label_plane(
            labels[plane], q[plane], omega[plane], metric, projector,
            ps_slice, zero_tol=zero_tol, ps_wave_tol=ps_wave_tol,
            barotropic_tol=barotropic_tol)
    basis.q = jnp.asarray(q)  # the zero-space split rotates columns
    return jnp.asarray(labels)


def _label_plane(
    labels: np.ndarray,
    q: np.ndarray,
    omega: np.ndarray,
    metric: np.ndarray,
    projector: np.ndarray,
    ps_slice: slice,
    *,
    zero_tol: float,
    ps_wave_tol: float,
    barotropic_tol: float,
) -> None:
    """Label one ``(kx, ky)`` plane in place (``q`` rotates)."""
    zero = np.abs(omega) < zero_tol
    _split_zero_space(labels, q, np.where(zero)[0], metric, projector,
                      barotropic_tol)
    ps_energy = segment_energy(q, metric, ps_slice)
    for sign, wave_code, baro_code in (
            (1, BAROTROPIC_WAVE_PLUS, BAROCLINIC_WAVE_PLUS),
            (-1, BAROTROPIC_WAVE_MINUS, BAROCLINIC_WAVE_MINUS)):
        cand = np.where(~zero & (np.sign(omega) == sign))[0]
        if cand.size == 0:  # pragma: no cover - waves are +/- paired
            continue
        labels[cand] = baro_code
        top = cand[int(np.argmax(ps_energy[cand]))]
        if ps_energy[top] > ps_wave_tol:
            labels[top] = wave_code


def _split_zero_space(
    labels: np.ndarray,
    q: np.ndarray,
    idx: np.ndarray,
    metric: np.ndarray,
    projector: np.ndarray,
    barotropic_tol: float,
) -> None:
    r"""
    Split a plane's zero space into barotropic and baroclinic.

    Description
    -----------
    The Hermitian barotropic overlap ``A = Q_0^H M Pi_bt Q_0`` on the
    degenerate ``omega = 0`` columns is diagonalized; its eigenbasis
    rotates the columns of ``q`` in place (M-orthonormality and
    ``L q = 0`` preserved), the maximal-overlap column (``>
    barotropic_tol`` — one per plane, the free-surface geostrophic
    mode) becomes **barotropic_vortical**, the rest
    **baroclinic_vortical**. A degenerate margin never observed (the
    free surface adds exactly one barotropic geostrophic DOF) leaves
    the plane all-baroclinic.
    """
    if idx.size == 0:  # pragma: no cover - nz+1 geostrophic modes always
        return          # exist per plane (the buoyancy + free-surface set)
    q0 = q[:, idx]
    amat = np.conj(q0.T) @ (metric[:, None] * (projector @ q0))
    amat = 0.5 * (amat + np.conj(amat.T))
    evals, evecs = np.linalg.eigh(amat)
    q[:, idx] = q0 @ evecs
    labels[idx] = BAROCLINIC_VORTICAL
    top = int(np.argmax(evals))
    if evals[top] <= barotropic_tol:  # pragma: no cover - the free
        return  # surface always contributes one barotropic overlap ~1
    labels[idx[top]] = BAROTROPIC_VORTICAL


# ================================================================
#  The labeled hydrostatic eigenmode surface
# ================================================================
class HydrostaticEigenmodes(ChannelEigenmodesBase):

    r"""
    Labeled numeric eigenmodes of the flat-bottom hydrostatic model.

    Description
    -----------
    The hydrostatic subclass of the shared
    :class:`~fridom.model._eigenbasis.ChannelEigenmodesBase`: the
    dense-column eigensolve of the assembled linear operator on the
    bounded vertical, labeled by :func:`label_hydrostatic_modes` into
    the six barotropic/baroclinic geostrophic / inertia-gravity
    families. The passthrough surface (``omega`` / ``q`` / ``labels`` /
    ``slices`` / ``metric``), the ``projector(sel)`` family / predicate
    projections and the ``mode(...)`` single-mode accessor come from the
    base; ``hy.transforms`` builds the public named projections on top.

    Parameters
    ----------
    model : Model
        An assembled explicit hydrostatic model (periodic horizontal,
        bounded vertical; the implicit / rigid-lid variants declare a
        ``linear_operator_gap`` and are refused).
    at_time : float, optional
        Evaluation time for time-dependent parameters (default: 0.0).
    chunk : int | None, optional
        Probe / eigensolve batch size (bounds peak memory)
        (default: None).
    zero_tol, ps_wave_tol, barotropic_tol : float, optional
        The labeler tolerances (see :func:`label_hydrostatic_modes`).
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
        ps_wave_tol: float = 1e-2,
        barotropic_tol: float = 0.5,
    ) -> None:
        """Solve the dense eigenproblem and label the families."""

        def labeler(basis: ChannelEigenbasis) -> jax.Array:
            """Run the hydrostatic family labeler."""
            return label_hydrostatic_modes(
                basis, zero_tol=zero_tol, ps_wave_tol=ps_wave_tol,
                barotropic_tol=barotropic_tol)

        super().__init__(model, labeler=labeler, at_time=at_time,
                         chunk=chunk)


# ================================================================
#  The user surface
# ================================================================
def from_model(
    model: Model, *, at_time: float = 0.0, chunk: int | None = None,
) -> HydrostaticEigenmodes:
    r"""
    Build the numeric eigenmodes of an assembled hydrostatic model.

    Description
    -----------
    The ``hy.eigenmodes.from_model`` surface: refuses a model whose
    modules declare a ``linear_operator_gap`` (the implicit / rigid-lid
    free surface — ``require_linear_operator``) before probing, then
    builds the labeled :class:`HydrostaticEigenmodes`. The engine
    requires exactly one bounded axis (the vertical); a differently
    bounded grid raises the engine's own taught error.

    Parameters
    ----------
    model : Model
        An assembled explicit hydrostatic model.
    at_time : float, optional
        Evaluation time for time-dependent parameters (default: 0.0).
    chunk : int | None, optional
        Probe / eigensolve batch size (default: None).

    Returns
    -------
    HydrostaticEigenmodes
        The labeled hydrostatic eigenmodes.

    Raises
    ------
    LinearOperatorGapError
        If the model carries barotropic physics outside ``L`` (the
        implicit / rigid-lid free surface).
    ValueError
        On a grid that is not singly bounded (the engine's gate).
    """
    require_linear_operator(model, consumer="hy.eigenmodes.from_model")
    return HydrostaticEigenmodes(model, at_time=at_time, chunk=chunk)
