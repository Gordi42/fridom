r"""
Labeled channel eigenmodes: shared wrapper, projections and dispatch.

Description
-----------
The package-shared face of the dense-column channel engine
(:func:`fridom.channel_eigenpairs`). The engine stays
family-agnostic; a model package owns the physics of the columns and
supplies a *labeler*. This module carries everything the packages
share around that split:

- :class:`ChannelEigenmodesBase` — the labeled wrapper base: builds
  the basis, runs the package labeler, exposes the passthrough
  surface (``omega`` / ``q`` / ``labels`` / ``slices`` / ``metric``)
  and the ``projector(sel)`` entry for family / predicate
  projections on physical states;
- the engine projection apply (:func:`family_projection`,
  :func:`predicate_projection`): partial-axis Fourier forward over
  the periodic axes (one or several — the half-spectrum axis is the
  engine's ``rfftn``-halved last periodic axis), the per-plane
  column projection ``Q diag(m) Q^H M z`` under the energy metric,
  and the backward synthesis — jax-traceable and sharding-clean;
- host-side labeling helpers shared by the package labelers
  (:func:`segment_energy`, :func:`recover_crisp_column`,
  :func:`split_frequency_bands`);
- :func:`eigenbasis` — the shared ``fr.eigenbasis(model)`` surface,
  dispatching to the package that owns the model's state vocabulary
  (``sw.eigenbasis`` / ``nh.eigenbasis``).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from importlib import import_module
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from fridom.framework.utils import dtype_comp, dtype_real
from fridom.model.eigen_channel import channel_eigenpairs
from fridom.model.eigenstates import (
    envelope_scale,
    evaluate_frequency_function,
    normalize_max_component,
)
from fridom.model.transforms.projection import (
    EigenFunction,
    EigenProjection,
)
from fridom.model.transforms.signature import StateSignature
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.operators.distributed_contract import (
    resolve_distributed_contraction,
)
from fridom.spatial.operators.fourier import Fourier

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.model.eigen_channel import ChannelEigenbasis
    from fridom.model.model import Model
    from fridom.model.transforms.base import StateTransform
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid


# ================================================================
#  The uniform mode-family surface (both eigenmode tiers)
# ================================================================
class ModeFamilySurface(Protocol):

    """
    What a family-string ``mode`` accessor exposes (both tiers).

    Description
    -----------
    The uniform user surface of the eigenmode engines: the analytic
    (fully periodic) tiers and the numeric channel tiers both carry
    the ``families`` name -> code vocabulary and resolve
    ``mode(family, indices, *, branch=None, phase=0.0)`` requests
    through :func:`_resolve_mode_family`, so the caller never sees
    which engine the topology selected.
    """

    @property
    def families(self) -> Mapping[str, int]:
        """The labeled family vocabulary (name -> code)."""
        ...  # pragma: no cover - protocol stub

    @property
    def nonphysical_families(self) -> tuple[str, ...]:
        """Engine-artifact families excluded from selection."""
        ...  # pragma: no cover - protocol stub


# ================================================================
#  Labeling helpers (host-side; shared by the package labelers)
# ================================================================
def segment_energy(
    q: np.ndarray, metric: np.ndarray, segment: slice,
) -> np.ndarray:
    r"""
    Per-column M-energy of one component segment of unit columns.

    Description
    -----------
    The crispness quantity behind boundary-trapped (Kelvin-type)
    classification: for an M-orthonormal column the energy of the
    wall-normal velocity segment is exactly zero on a Kelvin mode
    and order one otherwise.

    Parameters
    ----------
    q : np.ndarray
        One plane's eigenvector columns, shape ``(D, D)``.
    metric : np.ndarray
        The diagonal energy metric ``M``, shape ``(D,)``.
    segment : slice
        The component segment of the stacked column.

    Returns
    -------
    np.ndarray
        The per-column segment energies, shape ``(D,)``.
    """
    return np.einsum("ij,i->j", np.abs(q[segment, :]) ** 2,
                     metric[segment])


def recover_crisp_column(
    q: np.ndarray,
    omega: np.ndarray,
    candidates: np.ndarray,
    metric: np.ndarray,
    segment: slice,
    *,
    energy_tol: float,
    degeneracy_tol: float,
) -> int | None:
    r"""
    Rotate a near-degenerate 2-cluster to expose a segment-free column.

    Description
    -----------
    When a segment-free (Kelvin-type) column is nearly degenerate
    with another column, ``eigh`` may return an arbitrary mixture of
    the pair; neither mixed column is then crisp. For each adjacent
    candidate pair within ``degeneracy_tol`` (relative), the 2x2
    segment-energy Gram is diagonalized: a near-zero smallest
    eigenvalue means the 2-space contains a segment-free direction,
    and the unitary Gram eigenbasis rotates the two columns of ``q``
    **in place** to expose it (the first column of the pair becomes
    the minimal-energy combination). M-orthonormality is preserved;
    the eigen-relation residual of the pair changes only at the
    cluster's frequency splitting.

    Parameters
    ----------
    q : np.ndarray
        One plane's eigenvector columns, shape ``(D, D)``; rotated
        in place on success.
    omega : np.ndarray
        The plane's frequencies, shape ``(D,)``.
    candidates : np.ndarray
        Boolean candidate-column mask (e.g. one signed branch).
    metric : np.ndarray
        The diagonal energy metric ``M``, shape ``(D,)``.
    segment : slice
        The component segment whose energy must vanish.
    energy_tol : float
        The segment-energy bound below which a column is crisp.
    degeneracy_tol : float
        Relative frequency-cluster width for the recovery rotation.

    Returns
    -------
    int | None
        The recovered column index, or None if no cluster yields a
        segment-free direction.
    """
    cols = np.where(candidates)[0]
    for pos in range(len(cols) - 1):
        j, k = cols[pos], cols[pos + 1]
        scale = max(1.0, abs(omega[j]))
        if abs(omega[k] - omega[j]) > degeneracy_tol * scale:
            continue
        pair = q[:, [j, k]]
        weighted = metric[segment, None] * pair[segment]
        gram = pair[segment].conj().T @ weighted
        evals, evecs = np.linalg.eigh(gram)
        if evals[0] >= energy_tol:
            continue
        q[:, [j, k]] = pair @ evecs
        return int(j)
    return None


def split_frequency_bands(
    labels: np.ndarray,
    omega: np.ndarray,
    rest: np.ndarray,
    *,
    n_fast: int,
    gap_ratio: float,
    slow_code: int,
    fast_plus_code: int,
    fast_minus_code: int,
) -> None:
    r"""
    Split the remaining columns into a slow band and fast branches.

    Description
    -----------
    With ``n_fast`` structurally expected fast (wave) columns among
    the ``rest`` mask: an exact count labels everything fast by
    frequency sign; extra columns (a beta-plane slow Rossby band)
    are split off only across a clean spectral gap (smallest fast
    ``|omega|`` at least ``gap_ratio`` times the largest slow
    ``|omega|``). Without a clean gap — or with fewer columns than
    the fast count — the remainder stays unlabeled: predicates are
    the primary tool there.

    Parameters
    ----------
    labels : np.ndarray
        The plane's integer labels, written in place.
    omega : np.ndarray
        The plane's frequencies, shape ``(D,)``.
    rest : np.ndarray
        Boolean mask of the still-unlabeled nonzero columns.
    n_fast : int
        The structurally expected fast-column count of the plane.
    gap_ratio : float
        The documented slow/fast spectral-gap factor.
    slow_code : int
        The label code of the slow band.
    fast_plus_code : int
        The label code of the positive fast branch.
    fast_minus_code : int
        The label code of the negative fast branch.
    """
    n_slow = int(rest.sum()) - n_fast
    if n_slow < 0:
        return  # unexpected plane structure: stay unlabeled
    if n_slow > 0:
        mags = np.sort(np.abs(omega[rest]))
        if mags[n_slow] < gap_ratio * mags[n_slow - 1]:
            return  # no clean spectral gap: stay unlabeled
        slow = rest & (np.abs(omega)
                       < 0.5 * (mags[n_slow - 1] + mags[n_slow]))
        labels[slow] = slow_code
        rest = rest & ~slow
    labels[rest & (omega > 0)] = fast_plus_code
    labels[rest & (omega < 0)] = fast_minus_code


# ================================================================
#  The labeled wrapper base (packages subclass and label)
# ================================================================
class ChannelEigenmodesBase(ABC):

    r"""
    Labeled numeric eigenmodes of a walled channel model.

    Description
    -----------
    Bundles the framework's dense-column channel eigensolve
    (:func:`fridom.channel_eigenpairs`) with a model
    package's family labeler and exposes the labeled basis:
    ``omega``/``q``/``labels`` per periodic mode plane, the segment
    ``slices`` and the diagonal energy ``metric`` (see
    :class:`~fridom.model.eigen_channel.ChannelEigenbasis`
    for the layout conventions). Subclasses fix the family
    vocabulary through the :attr:`families` name -> code map and
    pass their labeler to this base constructor.

    A host-side analysis object; :meth:`projector` builds the family
    / predicate projections on physical states (the engine path
    reads ``labels``/``q``/``metric`` off this object *after*
    labeling, because label-side degeneracy recovery may rotate
    ``basis.q`` in place).

    Parameters
    ----------
    model : Model
        An assembled channel model (exactly one bounded axis).
    labeler : Callable[[ChannelEigenbasis], jax.Array]
        The package labeler filling the basis label slot.
    at_time : float, optional
        Evaluation time for time-dependent parameters
        (default: 0.0).
    chunk : int | None, optional
        Probe/eigensolve batch size, bounds peak memory
        (default: None).
    """

    @property
    @abstractmethod
    def families(self) -> Mapping[str, int]:
        """Family name -> integer label code (the vocabulary)."""

    #: Vocabulary class wrapping the states :meth:`mode` returns.
    state_class: ClassVar[type] = VectorField

    #: Family names that are engine artifacts, not physical mode
    #: selections (e.g. the nonhydro ``constraint`` complement).
    nonphysical_families: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        model: Model,
        *,
        labeler: Callable[[ChannelEigenbasis], jax.Array],
        at_time: float = 0.0,
        chunk: int | None = None,
    ) -> None:
        """Solve the channel eigenproblem and label the families."""
        self.grid: Grid = model.grid
        self.basis: ChannelEigenbasis = channel_eigenpairs(
            model, at_time=at_time, chunk=chunk)
        self.basis.label_with(labeler)
        self._spaces: Mapping[str, object] = MappingProxyType({
            name: model.state[name].function_space.bare
            for name in self.basis.components})

    # ================================================================
    #  Passthrough surface (the labeled basis)
    # ================================================================
    @property
    def omega(self) -> jax.Array:
        """Real frequencies, shape ``(*modes, D)``, ascending."""
        return self.basis.omega

    @property
    def q(self) -> jax.Array:
        """M-orthonormal eigenvector columns, shape ``(*modes, D, D)``."""
        return self.basis.q

    @property
    def labels(self) -> jax.Array:
        """Per-column family codes (:attr:`families` vocabulary)."""
        return self.basis.labels

    @property
    def components(self) -> tuple[str, ...]:
        """The stacked-column segment order."""
        return self.basis.components

    @property
    def slices(self) -> Mapping[str, slice]:
        """Per-component segment slices into the stacked ``D`` axis."""
        return self.basis.slices

    @property
    def spaces(self) -> Mapping[str, object]:
        """Per-component bare (BC-tagged) physical function spaces."""
        return self._spaces

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

    # ================================================================
    #  Family / predicate projections on physical states
    # ================================================================
    def projector(
        self,
        sel: str | Callable[[jax.Array, jax.Array], jax.Array],
    ) -> StateTransform:
        r"""
        Build a mode-family projection on physical states.

        Description
        -----------
        The selection is either a **family string** — a name from
        :attr:`families`, or an unsigned name covering both signed
        branches (e.g. ``"wave"`` for ``"wave+"`` and ``"wave-"``) —
        or a **predicate** ``(omega, labels) -> bool mask`` over the
        column planes, evaluated once at build time. Predicates are
        the primary tool where the named families blur (a beta-plane
        ``f(y)`` smears the vortical branch into slow Rossby
        frequencies): a frequency-threshold mask like ``|omega| > c``
        is sign-symmetric, hence closed under conjugation, and
        projects exactly.

        Selections that are **not** conjugation-closed (a single
        signed branch, or a sign-asymmetric predicate) act on the
        analytic signal: the stored half-spectrum planes are
        projected as selected while the implied conjugate planes
        carry the conjugate selection, and the real synthesis
        returns the **real part** (the imaginary parts of the
        self-conjugate planes are discarded). Such projections are
        idempotent on the analytic signal but only approximately on
        real states (the self-conjugate planes); closed selections
        are exactly idempotent.

        Parameters
        ----------
        sel : str | Callable[[jax.Array, jax.Array], jax.Array]
            A family name, or a predicate mapping ``(omega,
            labels)`` to a boolean mask of shape ``omega.shape``.

        Returns
        -------
        StateTransform
            The projection acting on physical states.
        """
        if isinstance(sel, str):
            return family_projection(self, sel)
        if callable(sel):
            return predicate_projection(self, sel)
        known = ", ".join(_selection_map(self.families))
        raise TypeError(
            f"projector takes a family name (one of {known}) or a "
            "predicate (omega, labels) -> bool mask; got "
            f"{sel!r}")

    def function(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        sel: str | Callable[[jax.Array, jax.Array], jax.Array],
    ) -> StateTransform:
        r"""
        Apply a scalar function of the linear operator on a selection.

        Description
        -----------
        The general :math:`f(L)` applicator: like :meth:`projector`
        but with the per-plane contraction

        .. math::

            Q\,\mathrm{diag}(f(\omega)\,m)\,Q^H M z

        (``m`` the selection mask), so ``f = 1`` on a selection
        reproduces the corresponding projector exactly. ``sel`` uses
        the projector grammar — a family string or a predicate
        ``(omega, labels) -> bool mask``. ``f`` is evaluated once,
        host-side, on the **real** frequencies of the selected
        columns only (complex return values allowed; the columns
        satisfy ``L q = -i omega q``):
        ``f = lambda w: -1.0 / (1j * w)`` builds :math:`L^{-1}` on
        the selection, ``f = lambda w: -1j * w`` the forward
        operator. A
        singular ``f`` meeting a zero-frequency column (a vortical /
        constraint column; the engine's zeros are numerical, so
        ``f(0)`` is probed on columns with ``|omega| < 1e-8``, the
        labelers' ``zero_tol`` scale) is a taught ``ValueError``,
        never a floored division.

        Real-safety: for a conjugation-closed selection and ``f``
        satisfying :math:`f(-\omega) = \overline{f(\omega)}` (true
        for :math:`-1/(i\omega)` and :math:`-i\omega`) the map sends
        real states to real states — the implied conjugate planes
        carry the conjugate weights, so the real synthesis is exact.
        A non-closed selection or a sign-asymmetric ``f`` acts on
        the analytic signal (real part), as with :meth:`projector`.

        Parameters
        ----------
        f : Callable[[np.ndarray], np.ndarray]
            The scalar spectral function, vectorized over an array
            of real frequencies.
        sel : str | Callable[[jax.Array, jax.Array], jax.Array]
            A family name, or a predicate mapping ``(omega,
            labels)`` to a boolean mask of shape ``omega.shape``.

        Returns
        -------
        StateTransform
            The weighted spectral application on physical states.

        Raises
        ------
        ValueError
            On an unknown family / invalid predicate mask, or if
            ``f`` evaluates non-finite on a selected column.
        """
        return spectral_function(self, f, sel)

    # ================================================================
    #  Mode-indexed single-mode states
    # ================================================================
    def mode(
        self,
        family: str,
        indices: Mapping[str, int],
        *,
        branch: int | None = None,
        phase: float = 0.0,
    ) -> tuple[float, VectorField]:
        r"""
        Return one labeled mode as ``(omega, physical state)``.

        Description
        -----------
        The mode-indexed accessor of the channel eigenbasis:
        ``family`` names a labeled family (a signed name like
        ``"wave+"``, or an unsigned root with ``branch=+1/-1``);
        ``indices`` is an axis-keyed mapping covering every grid
        axis — the periodic axes carry integer wavenumber indices
        (the half-spectrum axis runs ``0..n//2``, full axes take
        any integer modulo ``n``), and the bounded axis carries the
        **within-family mode ordinal**. Ordinals order the family's
        columns of the plane by ascending ``|omega|`` (for the
        Poincaré and Kelvin families this is ascending meridional
        complexity); the ``"vortical"`` family — degenerate at
        ``omega = 0`` on the f-plane — orders by the node count of
        the dominant component segment along the bounded axis.

        The state is the real Hermitian-closed physical mode
        :math:`\mathrm{Re}(q(y)\,e^{i(k\cdot x - \mathrm{phase})})`
        (under the linear model the state at time ``t`` is the same
        mode at phase ``phase + omega * t``, so positive ``omega``
        propagates along ``+k`` — eastward for positive ``kx``),
        normalized so the largest horizontal-velocity amplitude
        (the pointwise oscillation envelope over the ``u`` and
        ``v`` nodes) is one; a mode without horizontal velocity is
        left unnormalized.
        On the self-conjugate planes of the half-spectrum axis a
        signed selection synthesizes the standing (conjugate-mixed)
        real mode.

        Parameters
        ----------
        family : str
            A labeled family name (``self.families``), signed or
            unsigned-with-``branch``.
        indices : Mapping[str, int]
            Axis-keyed mode indices; the bounded axis keys the
            within-family ordinal.
        branch : int | None, optional
            ``+1``/``-1`` selects the signed branch of an unsigned
            family root (default: None).
        phase : float, optional
            The mode phase shift (default: 0.0).

        Returns
        -------
        tuple[float, VectorField]
            The frequency and the single-mode physical state
            (the package's ``state_class``).

        Raises
        ------
        ValueError
            On unknown/nonphysical families, bad indices, or a
            plane holding no (or too few) columns of the family.
        """
        name = _resolve_mode_family(self, family, branch)
        slots, ordinal = _plane_slots(self, indices)
        omega = np.asarray(self.omega)[slots]
        labels = np.asarray(self.labels)[slots]
        q = np.asarray(self.q)[slots]
        cols = _ordered_family_columns(self, name, labels, omega, q)
        if not cols:
            raise ValueError(
                f"no {name!r} column at plane {slots!r}: the family "
                "is structurally absent there (or the labeler left "
                "the plane's columns UNLABELED — inspect eb.labels)")
        if ordinal >= len(cols):
            raise ValueError(
                f"the {name!r} family holds {len(cols)} modes at "
                f"plane {slots!r} (ordinals 0..{len(cols) - 1}); "
                f"got {ordinal}")
        col = cols[ordinal]
        column = jnp.asarray(q[:, col])
        z0 = _synthesize_column(
            self, slots, column * jnp.exp(-1j * float(phase)))
        z1 = _synthesize_column(
            self, slots,
            column * jnp.exp(-1j * (float(phase) + jnp.pi / 2.0)))
        scale = envelope_scale(z0, z1, _horizontal_velocities(self))
        state = self.state_class(
            {c: z0[c] / scale for c in self.components})
        return float(omega[col]), state


# ================================================================
#  The engine path: per-plane column projection
# ================================================================
def _selection_map(
    families: Mapping[str, int],
) -> dict[str, tuple[str, ...]]:
    """Map selection strings to the family names they cover.

    Every family name selects itself; each ``name+``/``name-`` pair
    additionally contributes the unsigned ``name`` selection covering
    both signed branches (conjugation-closed).
    """
    selections: dict[str, tuple[str, ...]] = {}
    for name in families:
        if name and name[-1] in "+-":
            unsigned = name[:-1]
            if unsigned not in selections:
                selections[unsigned] = tuple(
                    n for n in (f"{unsigned}+", f"{unsigned}-")
                    if n in families)
        selections[name] = (name,)
    return selections


def _signature(em: ChannelEigenmodesBase) -> StateSignature:
    """Return the endo signature over the model's component spaces."""
    components = tuple(
        (name, em.spaces[name]) for name in em.components)
    return StateSignature(grid=em.grid, components=components)


def fourier_ops(em: ChannelEigenmodesBase) -> tuple[Fourier, ...]:
    """Per-axis Fourier transforms, the half-spectrum axis first.

    The engine's ``rfftn`` read-out halves the designated periodic
    axis (``em.periodic_axis`` -- the last periodic axis by default,
    a local axis re-designated by the layout when the last one is
    sharded), so the real-to-complex stage must run on that axis:
    applying it first (and its inverse last) reproduces the engine's
    coefficient layout -- full spectra on the remaining periodic
    axes, half spectrum on the designated one.
    """
    periodic = tuple(
        name for name in em.grid.names if name != em.bounded_axis)
    half = em.periodic_axis
    order = (half, *(name for name in periodic if name != half))
    return tuple(Fourier(em.grid, axes=(axis,)) for axis in order)


def _project_engine(
    em: ChannelEigenmodesBase,
    codes: tuple[int, ...],
    state: VectorField,
) -> VectorField:
    r"""
    Project a physical state onto the labeled families in ``codes``.

    Description
    -----------
    The label mask ``labels in codes`` selects the eigenvector
    columns per mode plane; :func:`_project_masked` does the
    transform round-trip and the plane contraction. ``codes`` are
    the package's family-code values (the ``EigenProjection`` mode
    tuple, so same-backend projections merge by code union).
    """
    mask = jnp.isin(em.labels, jnp.asarray(codes, dtype=jnp.int32))
    return _project_masked(em, mask, state)


def _project_masked(
    em: ChannelEigenmodesBase,
    mask: jax.Array,
    state: VectorField,
) -> VectorField:
    r"""
    Apply the per-plane column projector under a boolean mask.

    Description
    -----------
    The masked instance of :func:`_contract_planes`,

    .. math::

        P z = Q\,\mathrm{diag}(m)\,Q^H M z

    with ``m`` the boolean column mask (amplitudes off the selection
    are zeroed exactly).
    """
    weights = jnp.where(mask, 1.0, 0.0)
    return _contract_planes(
        em, state, lambda amp: jnp.where(mask, amp, 0.0), weights)


def _apply_weighted(
    em: ChannelEigenmodesBase,
    weights: jax.Array,
    state: VectorField,
) -> VectorField:
    r"""
    Apply per-plane complex column weights (the ``f(L)`` engine).

    Description
    -----------
    The weighted instance of :func:`_contract_planes`,

    .. math::

        f(L)\, z = Q\,\mathrm{diag}(w)\,Q^H M z

    with ``w = f(omega) * mask`` the complex column weights built by
    :func:`spectral_function` (exact zeros off the selection).
    """
    return _contract_planes(
        em, state, lambda amp: weights * amp, weights)


def _reject_sharded_projection(em: ChannelEigenmodesBase) -> None:
    r"""
    Taught skip on an unsupported sharded-periodic channel layout.

    Description
    -----------
    The multi-device channel projection is served by the fused
    ``jax.shard_map`` lowerings
    (:func:`~fridom.spatial.operators.distributed_contract.resolve_distributed_contraction`)
    on a 1-D device mesh: the 3-D channel (two periodic axes) through
    :class:`~fridom.spatial.operators.distributed_contract.ContractPlan`
    (the engine designates a **local** periodic axis as the half
    (``rfft``) axis, so the sharded axis is the transpose partner), and
    the 2-D channel (one periodic axis) through the transpose pipeline
    of :class:`~fridom.spatial.operators.distributed_contract.Channel2DPlan`
    (it parks the shardedness on the bounded axis so the local ``rfft``
    can run). This taught skip covers the genuinely-unsupported
    remainder those lowerings decline while a periodic axis is still
    sharded: a **non-1-D device mesh** (a pencil decomposition -- one
    transpose can never localize both FFT axes; unreachable today, the
    decomposition builder rejects it) or a hypothetical >3-D channel.
    Left to the plain (GSPMD) transform, those hit the upstream XLA:GPU
    distributed-FFT lowering fault (``complex64`` twiddle constants
    multiplied against ``complex128`` cuFFT data -- the HLO verifier
    rejects the mixed multiply, jax/jaxlib 0.10.x, independent of the
    FFT normalization and not covered by ``multi_output_fusion``), so
    raise a taught error here instead of dying deep in the verifier.
    The single-device path is unaffected -- including a
    ``device_ids=(0,)`` grid on a multi-device host, whose layout
    leaves every axis local. See
    ``design/research/multidevice_test_faults.md``.

    Parameters
    ----------
    em : ChannelEigenmodesBase
        The labeled channel eigenmodes (carries the grid).

    Raises
    ------
    NotImplementedError
        When a periodic axis is sharded and the fused lowerings
        decline (the unsupported remainder above).
    """
    layout = em.grid.decomposition.default_layout
    sharded = tuple(
        name for name in em.grid.names
        if name != em.bounded_axis and not layout.is_local(name))
    if sharded:
        raise NotImplementedError(
            "the channel eigenmode projection cannot run on this grid "
            f"that shards a periodic axis {sharded!r} across devices: "
            "the fused distributed contractions serve the 3-D channel "
            "(two periodic axes) and the 2-D channel (one periodic "
            "axis) on a 1-D device mesh, but this layout is the "
            "unsupported remainder (a non-1-D device mesh, or a >3-D "
            "channel), and the plain GSPMD transform would hit an "
            "upstream XLA:GPU distributed-FFT lowering fault (complex64 "
            "twiddle constants multiplied against complex128 data -- "
            "the HLO verifier rejects the mixed-precision multiply, "
            "jax/jaxlib 0.10.x). Build the channel model on a single "
            "device (Grid(..., device_ids=(0,))) to use the eigenbasis "
            "projections; see "
            "design/research/multidevice_test_faults.md.")


def _contract_planes(
    em: ChannelEigenmodesBase,
    state: VectorField,
    scale: Callable[[jax.Array], jax.Array],
    weights: jax.Array,
) -> VectorField:
    r"""
    Per-plane column contraction ``Q scale(Q^H M z)`` on a state.

    Description
    -----------
    Forward-transforms each component along the periodic axes only
    (partial-axis Fourier, half spectrum on the engine's last
    periodic axis; the bounded axis stays nodal), concatenates the
    component segments into stacked plane columns ``z``, applies
    ``Q scale(amp)`` per plane to the amplitudes ``amp = Q^H M z``
    (``Q = em.q``, ``M`` the diagonal energy metric; ``scale``
    realizes the mask / weight diagonal), splits the segments and
    inverse-transforms. The basis arrays are read off ``em`` here,
    at application time — after labeling, whose degeneracy recovery
    may have rotated ``q`` in place.

    On a grid whose default layout shards a periodic (Fourier) axis
    the plain (GSPMD) transform hits an upstream XLA:GPU
    distributed-FFT lowering fault, so the contraction is routed
    through the fused ``jax.shard_map`` lowering
    (:func:`~fridom.spatial.operators.distributed_contract.resolve_distributed_contraction`),
    which keeps every FFT axis device-local when its transform runs
    and realizes ``Q diag(w) Q^H M`` per shard — bit-for-bit the
    single-device result, ``w`` the explicit complex weight diagonal
    (``scale`` and ``w`` encode the same mask / weight). Both the 3-D
    channel (``ContractPlan``) and the 2-D channel (``Channel2DPlan``,
    the transpose pipeline) are served on a 1-D mesh. The
    single-device and bounded-axis-sharded paths keep the
    ``scale``-driven body below unchanged. The genuinely-unsupported
    remainder (a non-1-D mesh, or a >3-D channel) is rejected upfront
    (:func:`_reject_sharded_projection`).

    The backward half-spectrum synthesis returns the real part: for
    conjugation-closed selections (and conjugation-symmetric
    weights) the result is exactly real up to floating point; a
    non-closed selection acts on the analytic signal (see
    :meth:`ChannelEigenmodesBase.projector`). Sharding-clean: the
    contraction is an einsum of the replicated basis against the
    (decomposition-laid-out) coefficient planes — no host gather.
    """
    plan = resolve_distributed_contraction(
        em.grid, bounded_axis=em.bounded_axis,
        periodic_axis=em.periodic_axis, components=em.components,
        slices=em.slices)
    if plan is not None:
        result = plan.apply(
            {name: state[name] for name in em.components},
            em.q, weights, em.metric)
        return type(state)(result)
    _reject_sharded_projection(em)
    ops = fourier_ops(em)
    bounded = em.grid.names.index(em.bounded_axis)
    coeff = {}
    for name in em.components:
        field = state[name]
        for op in ops:
            field = op.forward(field)
        coeff[name] = field
    z = jnp.concatenate(
        [jnp.moveaxis(coeff[name].data, bounded, -1)
         for name in em.components], axis=-1)
    amp = jnp.einsum("...dj,d,...d->...j", jnp.conj(em.q),
                     em.metric, z)
    out = jnp.einsum("...dj,...j->...d", em.q, scale(amp))
    result = {}
    for name in em.components:
        field = coeff[name].with_data(jnp.moveaxis(
            out[..., em.slices[name]], -1, bounded))
        for op in reversed(ops):
            field = op.backward(field)
        result[name] = state[name].with_data(jnp.real(field.data))
    return type(state)(result)


def family_projection(
    em: ChannelEigenmodesBase,
    selection: str,
    *,
    name: str | None = None,
) -> EigenProjection:
    r"""
    Return the labeled-family projection for a selection string.

    Description
    -----------
    ``selection`` is a family name from ``em.families``, or an
    unsigned name covering both signed branches (which keeps the
    selection conjugation-closed — exactly real and exactly
    idempotent). The family codes become the ``EigenProjection``
    mode tuple, so projections on the same eigenmodes merge by code
    union under ``+`` (and ``.complement`` stays valid).

    Parameters
    ----------
    em : ChannelEigenmodesBase
        The labeled channel eigenmodes.
    selection : str
        The family selection string.
    name : str | None, optional
        A repr label (default: None, ``P[<selection>]``).

    Returns
    -------
    EigenProjection
        The family projection on physical states.
    """
    codes = _family_codes(em, selection)
    return EigenProjection(
        eigenmodes=em,
        modes=codes,
        signature=_signature(em),
        project_fn=_project_engine,
        name=name or f"P[{selection}]")


def _family_codes(
    em: ChannelEigenmodesBase, selection: str,
) -> tuple[int, ...]:
    """Resolve a family selection string to its label codes."""
    selections = _selection_map(em.families)
    families = selections.get(selection)
    if families is None:
        known = ", ".join(selections)
        raise ValueError(
            f"unknown family selection {selection!r}: the channel "
            f"vocabulary is {known} (or pass a predicate "
            "(omega, labels) -> bool mask)")
    return tuple(sorted(em.families[f] for f in families))


def _predicate_mask(
    em: ChannelEigenmodesBase,
    predicate: Callable[[jax.Array, jax.Array], jax.Array],
) -> jax.Array:
    """Evaluate and validate a ``(omega, labels)`` selection mask."""
    mask = jnp.asarray(predicate(em.omega, em.labels))
    if mask.shape != em.omega.shape or mask.dtype != jnp.bool_:
        raise ValueError(
            "a projector predicate must return a boolean mask of "
            f"shape {em.omega.shape} (one flag per column plane); "
            f"got shape {mask.shape}, dtype {mask.dtype}")
    return mask


def predicate_projection(
    em: ChannelEigenmodesBase,
    predicate: Callable[[jax.Array, jax.Array], jax.Array],
    *,
    name: str | None = None,
) -> EigenProjection:
    r"""
    Return the projection selected by a ``(omega, labels)`` predicate.

    Description
    -----------
    The predicate is evaluated **once**, here at build time, on the
    labeled basis; the resulting boolean mask over the column planes
    is captured by the returned projection. Predicates are the
    primary selection tool where the named families blur (beta-plane
    slow modes); sign-symmetric frequency thresholds like
    ``|omega| > c`` are conjugation-closed and project exactly,
    while non-closed masks act on the analytic signal (real part;
    see :meth:`ChannelEigenmodesBase.projector`).

    Parameters
    ----------
    em : ChannelEigenmodesBase
        The labeled channel eigenmodes.
    predicate : Callable[[jax.Array, jax.Array], jax.Array]
        Maps ``(omega, labels)`` to a boolean mask of shape
        ``omega.shape``.
    name : str | None, optional
        A repr label (default: None, the predicate's ``__name__``).

    Returns
    -------
    EigenProjection
        The masked projection on physical states.
    """
    mask = _predicate_mask(em, predicate)

    def project(
        eigenmodes: ChannelEigenmodesBase,
        modes: tuple[int, ...],  # noqa: ARG001 — EigenProjection contract
        state: VectorField,
    ) -> VectorField:
        """Apply the captured predicate mask per plane."""
        return _project_masked(eigenmodes, mask, state)

    label = name or getattr(predicate, "__name__", "predicate")
    return EigenProjection(
        eigenmodes=em,
        modes=(),
        signature=_signature(em),
        project_fn=project,
        name=f"P[{label}]")


def spectral_function(
    em: ChannelEigenmodesBase,
    f: Callable[[np.ndarray], np.ndarray],
    sel: str | Callable[[jax.Array, jax.Array], jax.Array],
    *,
    name: str | None = None,
) -> EigenFunction:
    r"""
    Build the weighted ``f(L)`` application for a mode selection.

    Description
    -----------
    The engine behind :meth:`ChannelEigenmodesBase.function`: the
    selection resolves to a boolean column mask exactly like
    :meth:`ChannelEigenmodesBase.projector`, ``f`` is evaluated once
    host-side on the selected columns' real frequencies
    (:func:`~fridom.model.eigenstates.evaluate_frequency_function`,
    the structural-zero guard included), and the captured complex
    weights drive the same sharded per-plane contraction as the
    projections (:func:`_apply_weighted`).

    Parameters
    ----------
    em : ChannelEigenmodesBase
        The labeled channel eigenmodes.
    f : Callable[[np.ndarray], np.ndarray]
        The scalar spectral function, vectorized over an array of
        real frequencies (complex return values allowed).
    sel : str | Callable[[jax.Array, jax.Array], jax.Array]
        A family name, or a predicate mapping ``(omega, labels)``
        to a boolean mask of shape ``omega.shape``.
    name : str | None, optional
        A repr label (default: None, ``f[<selection>]``).

    Returns
    -------
    EigenFunction
        The weighted application on physical states.
    """
    if isinstance(sel, str):
        codes = _family_codes(em, sel)
        mask = np.isin(np.asarray(em.labels), np.asarray(codes))
        label = sel
    elif callable(sel):
        mask = np.asarray(_predicate_mask(em, sel))
        label = getattr(sel, "__name__", "predicate")
    else:
        known = ", ".join(_selection_map(em.families))
        raise TypeError(
            f"function takes a family name (one of {known}) or a "
            "predicate (omega, labels) -> bool mask; got "
            f"{sel!r}")
    omega = np.asarray(em.omega)
    # the engine's zero-frequency columns are numerically zero
    # (eigh residuals ~1e-14, never exact), so a singular f is
    # probed at omega = 0 explicitly instead of relying on the
    # evaluation guard alone
    zero = mask & (np.abs(np.real(omega)) < ZERO_FREQUENCY_TOL)
    if zero.any():
        with np.errstate(all="ignore"):
            probe = np.asarray(f(np.zeros(1)), dtype=complex)
        if not np.isfinite(probe).all():
            raise ValueError(_describe_nonfinite_columns(em, zero))
    weights = jnp.asarray(evaluate_frequency_function(
        f, omega, mask,
        lambda bad: _describe_nonfinite_columns(em, bad)))

    def apply(
        eigenmodes: ChannelEigenmodesBase, state: VectorField,
    ) -> VectorField:
        """Apply the captured column weights per plane."""
        return _apply_weighted(eigenmodes, weights, state)

    return EigenFunction(
        eigenmodes=em,
        signature=_signature(em),
        apply_fn=apply,
        name=name or f"f[{label}]")


#: Absolute frequency scale below which a channel column counts as
#: structurally zero for the ``function`` guard (the package
#: labelers' ``zero_tol`` default).
ZERO_FREQUENCY_TOL = 1e-8


def _describe_nonfinite_columns(
    em: ChannelEigenmodesBase, bad: np.ndarray,
) -> str:
    """Name the family/plane of the singular ``f(omega)`` columns."""
    labels = np.asarray(em.labels)
    omega = np.asarray(em.omega)
    names = {code: fam for fam, code in em.families.items()}
    idx = tuple(int(i) for i in np.argwhere(bad)[0])
    family = names.get(int(labels[idx]), "UNLABELED")
    return (
        f"function(f, sel): f is non-finite on {int(bad.sum())} "
        f"selected column(s) — e.g. the {family!r} column at plane "
        f"{idx[:-1]!r} with omega = {float(omega[idx]):.6g}. "
        "Structurally zero frequencies are excluded by selection, "
        "never floored: drop the zero-frequency families "
        "(vortical / constraint) from the selection, or pass an f "
        "that is finite there")


# ================================================================
#  Mode indexing and synthesis helpers (the engine tier)
# ================================================================
def _horizontal_velocities(
    em: ChannelEigenmodesBase,
) -> tuple[str, ...]:
    """Return the horizontal-velocity (normalization) components."""
    return tuple(n for n in ("u", "v") if n in em.components)


def _axis_cells(grid: Grid, name: str) -> int:
    """Origin cell count of the mesh factor carrying ``name``."""
    mesh = next(m for m in grid.factors if name in m.names)
    return mesh.n_cells


def _resolve_mode_family(
    em: ModeFamilySurface,
    family: str,
    branch: int | None,
) -> str:
    """Resolve a (family, branch) request to one labeled family."""
    if not isinstance(family, str):
        raise TypeError(
            f"mode() selects by family name, got {family!r}: the "
            "vocabulary is em.families — e.g. mode('vortical', "
            "indices), mode('wave+', indices) or mode('wave', "
            "indices, branch=+1). Integer branches survive only on "
            "the low-level analytic q(s)/omega(s)/projector(s) "
            "surface")
    name = family
    if branch is not None:
        if int(branch) not in (1, -1):
            raise ValueError(
                f"branch selects a signed family branch: +1 or -1, "
                f"got {branch!r}")
        name = f"{family}{'+' if int(branch) > 0 else '-'}"
        if name not in em.families:
            raise ValueError(
                f"{family!r} carries no signed branches "
                f"({name!r} is not a labeled family); pass a "
                "family name without branch=")
    if name in em.nonphysical_families:
        raise ValueError(
            f"{name!r} is not a physical mode family (an engine "
            "artifact — e.g. the divergence-complement constraint "
            "columns); select one of the physical families "
            f"{tuple(n for n in em.families if n not in em.nonphysical_families)!r}")  # noqa: E501
    if name not in em.families:
        known = ", ".join(
            n for n in em.families
            if n not in em.nonphysical_families)
        raise ValueError(
            f"unknown mode family {name!r}: the labeled vocabulary "
            f"is {known}; signed pairs take the signed name or the "
            "unsigned root with branch=+1/-1")
    return name


def _plane_slots(
    em: ChannelEigenmodesBase,
    indices: Mapping[str, int],
) -> tuple[tuple[int, ...], int]:
    """Resolve axis-keyed indices to (plane slots, family ordinal)."""
    names = em.grid.names
    if set(indices) != set(names):
        raise ValueError(
            "mode indices are keyed by the grid axes "
            f"{tuple(names)!r} (the bounded axis "
            f"{em.bounded_axis!r} keys the within-family ordinal); "
            f"got keys {tuple(indices)!r}")
    ordinal = int(indices[em.bounded_axis])
    if ordinal < 0:
        raise ValueError(
            f"the bounded-axis index is the within-family mode "
            f"ordinal (>= 0); got {ordinal}")
    slots = []
    for name in names:
        if name == em.bounded_axis:
            continue
        n = _axis_cells(em.grid, name)
        m = int(indices[name])
        if name == em.periodic_axis:
            if not 0 <= m <= n // 2:
                raise ValueError(
                    f"axis {name!r} stores the Hermitian half "
                    f"spectrum: mode indices run 0..{n // 2}; "
                    f"got {m}")
            slots.append(m)
        else:
            slots.append(m % n)
    return tuple(slots), ordinal


def _node_count(
    column: np.ndarray,
    components: tuple[str, ...],
    slices: Mapping[str, slice],
    metric: np.ndarray,
) -> int:
    """Sign changes of the dominant component segment of a column."""
    best = None
    best_energy = -1.0
    for name in components:
        seg = column[slices[name]]
        energy = float(
            np.sum(np.abs(seg) ** 2 * metric[slices[name]]))
        if energy > best_energy:
            best_energy, best = energy, seg
    magnitude = np.abs(best)
    j = int(np.argmax(magnitude))
    if magnitude[j] == 0.0:
        return 0
    profile = np.real(best * np.conj(best[j]) / magnitude[j])
    keep = np.abs(profile) > 1e-8 * np.abs(profile).max()
    signs = np.sign(profile[keep])
    return int(np.count_nonzero(signs[1:] != signs[:-1]))


def _ordered_family_columns(
    em: ChannelEigenmodesBase,
    family: str,
    labels: np.ndarray,
    omega: np.ndarray,
    q: np.ndarray,
) -> list[int]:
    r"""
    Order one plane's family columns by the mode-ordinal convention.

    Description
    -----------
    Ascending ``|omega|`` (ties by column index) — for Poincaré and
    Kelvin branches that is ascending meridional complexity. The
    ``"vortical"`` family instead orders by the bounded-axis node
    count of the dominant component segment (ties by column index):
    its f-plane columns are an exactly degenerate ``omega = 0``
    cluster where frequency ordering is meaningless, and under beta
    the slow Rossby ``|omega|`` *decreases* with meridional mode.
    """
    cols = np.flatnonzero(labels == em.families[family])
    if family == "vortical":
        metric = np.asarray(em.metric)

        def key(c: int) -> tuple:
            return (_node_count(q[:, c], em.components, em.slices,
                                metric), c)
    else:

        def key(c: int) -> tuple:
            return (abs(float(omega[c])), c)

    return sorted((int(c) for c in cols), key=key)


def _backward_synthesis(
    em: ChannelEigenmodesBase,
    coeffs: Mapping[str, jax.Array],
    templates: Mapping[str, ScalarField],
) -> dict[str, ScalarField]:
    r"""
    Inverse-transform store-frame coefficient columns to real fields.

    Description
    -----------
    The shared backward tail of the channel synthesis features
    (:func:`_synthesize_column`, :func:`channel_random_state`): each
    component's coefficient column is already built in the engine's
    store frame (the periodic axes Fourier-transformed — the
    designated half axis on its ``rfft`` half spectrum, the other
    periodic axis full — the bounded axis nodal), on the coefficient
    space of ``templates[name]``.

    On a grid whose default layout shards a periodic (Fourier) axis
    the plain (GSPMD) transform hits the upstream XLA:GPU
    distributed-FFT lowering fault, so the sharded channel routes the
    inverse through the fused ``jax.shard_map`` synthesis entries
    (:meth:`~fridom.spatial.operators.distributed_contract.ContractPlan.synthesize`
    for the 3-D channel,
    :meth:`~fridom.spatial.operators.distributed_contract.Channel2DPlan.synthesize`
    for the 2-D channel), which keep every FFT axis device-local and
    return the fields on the grid's own layout (no host gather). The
    single-device path — and the genuinely-unsupported remainder the
    fused lowerings decline (a bounded-axis-sharded or non-1-D layout)
    — keep the plain per-axis backward transform below, bit-for-bit
    unchanged; on a sharded remainder that path raises the transform's
    Tier-1 taught error.
    """
    plan = resolve_distributed_contraction(
        em.grid, bounded_axis=em.bounded_axis,
        periodic_axis=em.periodic_axis, components=em.components,
        slices=em.slices)
    if plan is not None:
        return plan.synthesize(coeffs, templates)
    ops = fourier_ops(em)
    fields = {}
    for name in em.components:
        coeff = templates[name]
        for op in ops:
            coeff = op.forward(coeff)
        coeff = coeff.with_data(
            jnp.asarray(coeffs[name]).astype(coeff.data.dtype))
        for op in reversed(ops):
            coeff = op.backward(coeff)
        fields[name] = coeff.real
    return fields


def _synthesize_column(
    em: ChannelEigenmodesBase,
    slots: tuple[int, ...],
    values: jax.Array,
) -> dict[str, ScalarField]:
    r"""
    Real physical fields of one Hermitian-closed plane column.

    Description
    -----------
    Places the stacked column ``values`` on the ``slots`` plane of
    the engine's partial-Fourier coefficient layout and inverse
    transforms (:func:`_backward_synthesis` — the fused distributed
    inverse on a sharded 3-D channel, the plain per-axis transform
    otherwise). On the self-conjugate planes of the half-spectrum
    axis the placement splits into the conjugate pair across the
    full periodic axes (the ``(v/2, conj(v)/2)`` closure), so the
    backward synthesis is exactly real.
    """
    grid = em.grid
    periodic = tuple(
        n for n in grid.names if n != em.bounded_axis)
    half_n = _axis_cells(grid, em.periodic_axis)
    half_slot = slots[periodic.index(em.periodic_axis)]
    self_conj = half_slot == 0 or (half_n % 2 == 0
                                   and half_slot == half_n // 2)
    b_arr = grid.names.index(em.periodic_axis)
    n_b = half_n // 2 + 1
    index: list[object] = [slice(None)] * len(grid.names)
    for name, slot in zip(periodic, slots, strict=True):
        index[grid.names.index(name)] = slot
    partner = list(index)
    for name, slot in zip(periodic, slots, strict=True):
        if name != em.periodic_axis:
            n = _axis_cells(grid, name)
            partner[grid.names.index(name)] = (n - slot) % n
    templates = {
        name: grid.create_field(em.spaces[name], name=name)
        for name in em.components}
    coeffs = {}
    for name in em.components:
        shape = list(templates[name].data.shape)
        shape[b_arr] = n_b
        seg = values[em.slices[name]].astype(dtype_comp())
        data = jnp.zeros(tuple(shape), dtype=dtype_comp())
        if self_conj:
            data = data.at[tuple(index)].add(0.5 * seg)
            data = data.at[tuple(partner)].add(
                0.5 * jnp.conj(seg))
        else:
            data = data.at[tuple(index)].set(seg)
        coeffs[name] = data
    return _backward_synthesis(em, coeffs, templates)


def channel_random_state(
    em: ChannelEigenmodesBase,
    selection: str,
    spectral_energy_density: Callable[..., jax.Array],
    *,
    seed: int,
    horizontal: tuple[str, ...],
) -> dict[str, ScalarField]:
    r"""
    Random-phase family state with a prescribed energy spectrum.

    Description
    -----------
    The engine-tier port of the reference
    ``PrescribedSpectraRandomPhase``: every labeled column of the
    selected family (an unsigned selection covers both signed
    branches) receives the amplitude

    .. math::

        a = \sqrt{\frac{S(k)}{\pi k_h}} \; e^{i\theta}

    with :math:`\theta` uniform random per column (seeded), the
    columns being M-orthonormal (unit energy). :math:`S` is
    evaluated on one wavenumber per grid axis (grid order): the
    periodic axes carry their physical Fourier wavenumbers and the
    bounded axis the **effective meridional wavenumber**
    :math:`k_y = \pi m / L` of the column's within-family ordinal
    ``m`` (the :func:`ChannelEigenmodesBase.mode` ordering).
    :math:`\pi k_h` is the ring measure of the reference's
    :math:`S(k) = 2\pi k\,E(k, 0)` angular convention, with
    :math:`k_h` built from the ``horizontal`` axes. The backward
    synthesis takes the real part (the Hermitian closure — exact
    for conjugation-closed selections), and the result is
    normalized so the largest horizontal-velocity value is one.

    The coefficient assembly is host-built (the replicated basis
    against global plane arrays) and committed to the grid's
    decomposition through the field store; the transforms and any
    downstream projector applications stay sharded.

    Parameters
    ----------
    em : ChannelEigenmodesBase
        The labeled channel eigenmodes.
    selection : str
        A family selection string (unsigned roots cover both
        signed branches).
    spectral_energy_density : Callable[..., jax.Array]
        ``S(*k)`` over the grid-axis wavenumbers, in grid order.
    seed : int
        The PRNG seed (bitwise deterministic).
    horizontal : tuple[str, ...]
        The horizontal axis names entering :math:`k_h`.

    Returns
    -------
    dict[str, ScalarField]
        The normalized real physical components.
    """
    selections = _selection_map(em.families)
    names = selections.get(selection)
    if names is None or any(
            n in em.nonphysical_families
            for n in (selection, *names)):
        known = ", ".join(
            n for n in selections
            if n not in em.nonphysical_families)
        raise ValueError(
            f"unknown or nonphysical family selection "
            f"{selection!r}: the channel vocabulary is {known}")
    grid = em.grid
    omega = np.asarray(em.omega)
    labels = np.asarray(em.labels)
    q_host = np.asarray(em.q)
    plane_shape = omega.shape[:-1]
    length = _axis_length(grid, em.bounded_axis)
    ky = np.zeros(omega.shape)
    selected = np.zeros(omega.shape, dtype=bool)
    for fam in names:
        for plane in np.ndindex(plane_shape):
            cols = _ordered_family_columns(
                em, fam, labels[plane], omega[plane], q_host[plane])
            for ordinal, col in enumerate(cols):
                ky[(*plane, col)] = np.pi * ordinal / length
                selected[(*plane, col)] = True
    kvals = _plane_wavenumbers(em, ndim=omega.ndim)
    kh2 = sum(
        kvals[name] ** 2 for name in horizontal
        if name != em.bounded_axis)
    kh2 = kh2 + (ky ** 2 if em.bounded_axis in horizontal else 0.0)
    kh = np.sqrt(kh2)
    args = tuple(
        jnp.asarray(ky if name == em.bounded_axis
                    else kvals[name])
        for name in grid.names)
    spectra = np.asarray(
        jnp.broadcast_to(spectral_energy_density(*args),
                         omega.shape))
    good = selected & (kh > 0.0)
    denom = np.where(good, np.pi * kh, 1.0)
    amp = np.where(good,
                   np.sqrt(np.where(good, spectra, 0.0) / denom),
                   0.0)
    theta = jax.random.uniform(
        jax.random.key(seed), omega.shape, dtype=dtype_real(),
        maxval=2.0 * jnp.pi)
    gains = jnp.asarray(amp) * jnp.exp(1j * theta)
    z = jnp.einsum("...dj,...j->...d", jnp.asarray(em.q), gains)
    bounded_pos = grid.names.index(em.bounded_axis)
    templates = {
        name: grid.create_field(em.spaces[name], name=name)
        for name in em.components}
    coeffs = {
        name: jnp.moveaxis(
            z[..., em.slices[name]], -1, bounded_pos).astype(dtype_comp())
        for name in em.components}
    fields = _backward_synthesis(em, coeffs, templates)
    return normalize_max_component(
        fields, _horizontal_velocities(em))


def _axis_length(grid: Grid, name: str) -> float:
    """Interval length of the mesh factor carrying ``name``."""
    mesh = next(m for m in grid.factors if name in m.names)
    return float(mesh.dx * mesh.n_cells)


def _plane_wavenumbers(
    em: ChannelEigenmodesBase, *, ndim: int,
) -> dict[str, np.ndarray]:
    """Physical wavenumbers per periodic axis, plane-broadcast."""
    grid = em.grid
    periodic = tuple(
        n for n in grid.names if n != em.bounded_axis)
    kvals = {}
    for i, name in enumerate(periodic):
        n = _axis_cells(grid, name)
        length = _axis_length(grid, name)
        if name == em.periodic_axis:
            modes = np.arange(n // 2 + 1, dtype=float)
        else:
            modes = np.arange(n, dtype=float)
            modes = np.where(modes > n // 2, modes - n, modes)
        k = 2.0 * np.pi * modes / length
        shape = [1] * ndim
        shape[i] = k.size
        kvals[name] = k.reshape(shape)
    return kvals


# ================================================================
#  The shared eigenbasis surface (fr.eigenbasis)
# ================================================================
def eigenbasis(
    model: Model, *, at_time: float = 0.0,
) -> ChannelEigenmodesBase:
    r"""
    Build the labeled numeric eigenbasis of a channel model.

    Description
    -----------
    The shared user surface of the dense-column channel engine:
    dispatches on the model package that owns the model's state
    vocabulary class (``sw.State``, ``nh.State``, ...) and returns
    that package's labeled wrapper — exactly what the package
    surfaces (``sw.eigenbasis`` / ``nh.eigenbasis``) return,
    including their topology gates and taught errors. A model
    assembled without a package core carries a plain ``VectorField``
    state and has no family physics to dispatch to.

    Parameters
    ----------
    model : Model
        The assembled channel model.
    at_time : float, optional
        The clock time at which to freeze time-dependent parameters
        (default: 0.0).

    Returns
    -------
    ChannelEigenmodesBase
        The package-labeled channel eigenmodes.

    Raises
    ------
    ValueError
        If no model package owns the state vocabulary, or the owning
        package exports no ``eigenbasis`` surface; package-side
        topology errors propagate.
    """
    module = type(model.state).__module__
    package, _, _ = module.rpartition(".")
    if (not package.startswith("fridom.")
            or package.startswith(
                ("fridom.framework", "fridom.spatial", "fridom.model"))):
        raise ValueError(
            "fr.model.eigenbasis dispatches on the model package that owns "
            "the state vocabulary; this model's state is "
            f"{type(model.state).__name__!r} from {module!r} — "
            "assemble with a model package core (e.g. "
            "sw.Model / nh.Model) or use the package surface "
            "directly")
    builder = getattr(import_module(package), "eigenbasis", None)
    if builder is None:
        raise ValueError(
            f"the model package {package!r} exports no 'eigenbasis' "
            "surface; build the labeled channel eigenmodes through "
            "the package's own eigenmode module")
    return builder(model, at_time=at_time)
