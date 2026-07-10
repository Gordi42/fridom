r"""
Labeled channel eigenmodes: shared wrapper, projections and dispatch.

Description
-----------
The package-shared face of the dense-column channel engine
(:func:`fridom.framework2.channel_eigenpairs`). The engine stays
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
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from fridom.framework2.grid.operators.fourier import Fourier
from fridom.framework2.model.eigen_channel import channel_eigenpairs
from fridom.framework2.transforms.projection import EigenProjection
from fridom.framework2.transforms.signature import StateSignature

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.framework2.grid.fields.vector_field import VectorField
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.model.eigen_channel import ChannelEigenbasis
    from fridom.framework2.model.model import Model
    from fridom.framework2.transforms.base import StateTransform


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
    (:func:`fridom.framework2.channel_eigenpairs`) with a model
    package's family labeler and exposes the labeled basis:
    ``omega``/``q``/``labels`` per periodic mode plane, the segment
    ``slices`` and the diagonal energy ``metric`` (see
    :class:`~fridom.framework2.model.eigen_channel.ChannelEigenbasis`
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


def _fourier_ops(em: ChannelEigenmodesBase) -> tuple[Fourier, ...]:
    """Per-axis Fourier transforms, the half-spectrum axis first.

    The engine's ``rfftn`` read-out halves the *last* periodic axis
    (``em.periodic_axis``), so the real-to-complex stage must run on
    that axis: applying it first (and its inverse last) reproduces
    the engine's coefficient layout — full spectra on the remaining
    periodic axes, half spectrum on the last one.
    """
    periodic = tuple(
        name for name in em.grid.names if name != em.bounded_axis)
    order = (periodic[-1], *periodic[:-1])
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
    Forward-transforms each component along the periodic axes only
    (partial-axis Fourier, half spectrum on the engine's last
    periodic axis; the bounded axis stays nodal), concatenates the
    component segments into stacked plane columns ``z``, applies

    .. math::

        P z = Q\,\mathrm{diag}(m)\,Q^H M z

    per plane (``Q = em.q``, ``M`` the diagonal energy metric, ``m``
    the column mask), splits the segments and inverse-transforms.
    The basis arrays are read off ``em`` here, at application time —
    after labeling, whose degeneracy recovery may have rotated ``q``
    in place.

    The backward half-spectrum synthesis returns the real part: for
    conjugation-closed masks the result is exactly real up to
    floating point; a non-closed mask acts on the analytic signal
    (see :meth:`ChannelEigenmodesBase.projector`). Sharding-clean:
    the contraction is an einsum of the replicated basis against the
    (decomposition-laid-out) coefficient planes — no host gather.
    """
    ops = _fourier_ops(em)
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
    out = jnp.einsum("...dj,...j->...d", em.q,
                     jnp.where(mask, amp, 0.0))
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
    selections = _selection_map(em.families)
    families = selections.get(selection)
    if families is None:
        known = ", ".join(selections)
        raise ValueError(
            f"unknown family selection {selection!r}: the channel "
            f"vocabulary is {known} (or pass a predicate "
            "(omega, labels) -> bool mask)")
    codes = tuple(sorted(em.families[f] for f in families))
    return EigenProjection(
        eigenmodes=em,
        modes=codes,
        signature=_signature(em),
        project_fn=_project_engine,
        name=name or f"P[{selection}]")


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
    mask = jnp.asarray(predicate(em.omega, em.labels))
    if mask.shape != em.omega.shape or mask.dtype != jnp.bool_:
        raise ValueError(
            "a projector predicate must return a boolean mask of "
            f"shape {em.omega.shape} (one flag per column plane); "
            f"got shape {mask.shape}, dtype {mask.dtype}")

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
            or package.startswith("fridom.framework")):
        raise ValueError(
            "fr.eigenbasis dispatches on the model package that owns "
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
