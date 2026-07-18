r"""
Dense-column channel eigenbasis via per-``kx`` ``eigh(iMS, M)``.

Description
-----------
The **dense-column tier** of the numeric eigenmode machinery: the
eigenbasis of a linearized model on a grid with exactly one bounded
(walled) axis and periodic remaining axes — the rotating channel.

Along the walls no trigonometric basis exists: rotation couples the
components with *opposite* wall parities (the normal velocity vanishes
at the walls, the tangential velocity and pressure do not), so a
sine/cosine column ansatz cannot diagonalize the operator — the
boundary-trapped (Kelvin-type) modes decay exponentially off the wall
instead of oscillating. What survives is translation invariance along
the periodic axes: per periodic wavenumber the operator is a **dense**
:math:`D \times D` block over the stacked bounded-axis columns, with
:math:`D = \sum_c n_c` (the per-component bounded-axis DOF counts —
segments differ; walls are not DOFs).

:func:`channel_eigenpairs` probes those blocks numerically: a unit
impulse at periodic-index 0 has a unit DFT at every periodic mode, so
one linearized-tendency application per (component, bounded-axis node)
fills a full column of every block at once. The response is Fourier
transformed along the periodic axes only (``rfftn`` — the operator is
real, so the half spectrum with :math:`S(-k) = \overline{S(k)}`
suffices). Under the measure-weighted energy metric ``M`` the operator
is skew-adjoint, so :math:`H := iMS` is Hermitian and a whitened
batched ``eigh`` returns real frequencies and ``M``-orthonormal
eigenvectors per mode plane.

The invariance requirement is exactly this: the operator must be
translation-invariant (constant-coefficient) **along the periodic
axes only**. The dense axis is probed numerically with whatever
coefficients the tendency carries, so coefficients may vary
arbitrarily along it — e.g. a beta-plane :math:`f(y) = f_0 + \beta y`
on the walled-``y`` channel (pointwise rotation does no work for any
``f`` profile, so ``H`` stays Hermitian). A coefficient varying along
a *periodic* axis breaks the per-mode block structure and is out of
scope (the Hermiticity assertion is the safety net, not a guarantee).

Coefficients that **enter the energy metric** — the variable-depth
shallow water :math:`c^2(y)` and the meridionally stratified
nonhydro :math:`N^2(y)` — are served through profile-valued metric
weights: ``EnergyMetric.from_model(..., allow_field_weights=True)``
assembles ``diag(c^2, c^2, 1)`` / ``diag(1, 1, dsqr, 1/N^2(y))``,
and :func:`_metric_diagonal` samples each field weight **on the
component's own bounded-axis nodes** through ``.to`` — the identical
sampling the tendency flux uses, which is what keeps ``iMS``
Hermitian for any profile (a varying shallow-water depth
additionally needs the Coriolis module's thickness-weighted
rotation, ``metric_weight="csqr"``).

Constrained models (the nonhydro pressure)
------------------------------------------
A model carrying a ``CONSTRAINT`` stage (the nonhydro pressure
projection) is served by probing the **projected** linearization
:math:`S = P L P` — ``P`` the M-orthogonal Leray projector the
pressure stage realizes, applied through the public
``model.constrain`` matvec: each impulse is projected first (one
extra pressure solve per impulse), then the constrained tendency
``model.tendency(constraints=True)`` supplies :math:`P L`. Probing
:math:`P L` or :math:`L P` alone would **not** be M-skew off the
divergence-free subspace; only the symmetric sandwich keeps
``H = iMS`` Hermitian. The M-orthogonal complement of the
divergence-free subspace (the discrete pressure-gradient directions)
lies in the kernel of both ``P`` factors, so per mode plane it shows
up as **extra exact zero modes** on top of the physical steady
(geostrophic) modes — a labeler's concern, not the basis's.

The framework stays agnostic about the physics of the columns
(vortical / Kelvin / Poincaré families): :class:`ChannelEigenbasis`
carries an **empty** integer label slot that a model package fills
through :meth:`ChannelEigenbasis.label_with`. That agnosticism is
load-bearing for varying coefficients — with a beta-plane ``f(y)``
the vortical branch acquires slow Rossby frequencies and the family
boundaries blur, which is the labeler's concern, not the basis's.

The setup path here may gather to host (a one-off analysis cost); the
downstream projector *application* is the sharding-critical path.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from fridom.framework.utils import dtype_real
from fridom.model.eigen import (
    _metric_weights,
    _rest_background,
)
from fridom.model.energy import EnergyMetric
from fridom.model.stages import StageKind
from fridom.model.term_predicates import linearize
from fridom.spatial.fields.scalar_field import ScalarField

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.model.model import Model
    from fridom.spatial.fields.vector_field import VectorField

# The value of an unset (empty) mode label.
UNLABELED = -1

# Relative Hermiticity tolerance for ``H = iMS`` (skew-adjointness of
# the probed operator under the energy metric).
_HERMITICITY_TOL = 1e-10


class ChannelEigenbasis:

    r"""
    The per-mode dense eigenbasis of a linearized channel model.

    Description
    -----------
    Holds the batched spectrum and eigenvectors produced by
    :func:`channel_eigenpairs`. The leading axes index the periodic
    mode planes (the ``rfftn`` half spectrum of the periodic axes —
    halved along :attr:`periodic_axis`); the trailing axes are the
    stacked bounded-axis column of length ``D``, ordered by
    ``components`` with the per-component :attr:`slices` segments.
    ``q[..., :, j]`` is the M-orthonormal eigenvector for
    ``omega[..., j]`` (``L q = -i omega q``), sorted ascending per
    plane. The operator is real, so the negative-``kx`` planes are the
    conjugates of the stored half spectrum.

    A host-side analysis object: build it once, then hand it to a
    model package's labeler and to the spectral projectors. The
    projector *application* downstream is the sharding-critical path,
    not this object. The basis itself is family-agnostic (labels are
    empty until :meth:`label_with`) — deliberately, because with
    coefficients varying along the bounded axis (a beta-plane
    ``f(y)``) the mode families blur and only a model package can
    judge the boundaries. On a constrained model (the nonhydro
    pressure) the basis diagonalizes the projected linearization
    ``P L P``, so each plane carries the divergence-complement
    directions as extra exact zero modes next to the physical steady
    modes (the labeler owns telling them apart).

    Parameters
    ----------
    omega : jax.Array
        Real frequencies, shape ``(*modes, D)``, ascending per plane.
    q : jax.Array
        Eigenvectors, shape ``(*modes, D, D)``.
    components : tuple[str, ...]
        The prognostic component names, the segment order of the
        stacked column.
    slices : Mapping[str, slice]
        Per-component segment slices into the stacked ``D`` axis.
    metric : jax.Array
        The diagonal energy metric ``M``, shape ``(D,)``: per entry
        the component weight times the bounded-axis measure.
    periodic_axis : str
        The half-spectrum (``rfft``) periodic axis name.
    bounded_axis : str
        The bounded (walled) axis name whose nodes the column stacks.
    hermiticity_error : float
        The measured pre-symmetrization relative residual
        ``max|H - H^H| / max|H|`` of the probed ``H = iMS``.
    """

    def __init__(
        self,
        omega: jax.Array,
        q: jax.Array,
        components: tuple[str, ...],
        slices: Mapping[str, slice],
        metric: jax.Array,
        periodic_axis: str,
        bounded_axis: str,
        hermiticity_error: float,
    ) -> None:
        """Store the batched spectrum, eigenvectors and layout."""
        self.omega = omega
        self.q = q
        self.components = components
        self.slices = MappingProxyType(dict(slices))
        self.metric = metric
        self.periodic_axis = periodic_axis
        self.bounded_axis = bounded_axis
        self.hermiticity_error = hermiticity_error
        self.labels: jax.Array = jnp.full(
            omega.shape, UNLABELED, dtype=jnp.int32)

    # ================================================================
    #  Labeling hook (model packages own the physics)
    # ================================================================
    def label_with(
        self, labeler: Callable[[ChannelEigenbasis], jax.Array],
    ) -> None:
        r"""
        Fill the label slot from a model package's labeler.

        Description
        -----------
        The framework knows nothing about the channel mode families
        (vortical / Kelvin / Poincaré are model physics); a model
        package classifies the columns and writes integer labels
        here. The labeler receives this basis and returns one integer
        label per mode, shape ``omega.shape``; unset entries are
        :data:`UNLABELED`.

        Parameters
        ----------
        labeler : Callable[[ChannelEigenbasis], jax.Array]
            Maps this basis to an integer label array of shape
            ``omega.shape``.
        """
        labels = jnp.asarray(labeler(self))
        if labels.shape != self.omega.shape:
            raise ValueError(
                "the labeler must return one integer label per mode: "
                f"expected shape {self.omega.shape}, got "
                f"{labels.shape}")
        if not jnp.issubdtype(labels.dtype, jnp.integer):
            raise ValueError(
                "mode labels are small integers (an enum per model "
                f"package); the labeler returned dtype {labels.dtype}")
        self.labels = labels

    # ================================================================
    #  Self-consistency diagnostics (validation helpers)
    # ================================================================
    def orthonormality_error(self) -> jax.Array:
        r"""
        Return the max deviation of :math:`q^{H} M q` from the identity.

        Description
        -----------
        The eigenvectors are M-orthonormal per mode plane by
        construction; this is a residual on that property (a 0-d real
        array), near machine epsilon on a well-conditioned metric.

        Returns
        -------
        jax.Array
            ``max |q^H M q - I|`` over all planes (a 0-d array).
        """
        gram = jnp.einsum(
            "...ij,i,...ik->...jk", jnp.conj(self.q), self.metric,
            self.q)
        identity = jnp.eye(gram.shape[-1], dtype=gram.dtype)
        return jnp.max(jnp.abs(gram - identity))


def _designate_half_axis(
    grid: object, periodic_names: tuple[str, ...],
) -> str:
    r"""
    Choose the periodic axis that carries the rfft half spectrum.

    Description
    -----------
    The half-spectrum axis is an engine convention, not physics: any
    periodic axis can carry it (the coefficient frame is free — the
    contracted physical field is invariant). The default is the
    **last** periodic axis (grid order), which keeps single-device
    programs and every already-served multi-device layout
    byte-identical. When the grid's default layout **shards that last
    periodic axis** -- the layout the fused distributed contraction
    cannot serve in the fixed frame, because its local rfft needs real
    data on an unsharded axis -- a local periodic axis is designated
    instead, so the sharded axis carries a full spectrum (the
    transpose partner ``a``) and the rfft runs on the local half axis
    ``b``. All-local (single-device) layouts always keep the default,
    so the pick only ever moves on a layout the fixed frame could not
    serve.

    Parameters
    ----------
    grid : object
        The grid carrying the decomposition and default layout.
    periodic_names : tuple[str, ...]
        The periodic coordinate names, in grid order.

    Returns
    -------
    str
        The designated half (``rfft``) axis name.
    """
    default = periodic_names[-1]
    layout = grid.decomposition.default_layout
    if layout.is_local(default):
        return default
    for name in periodic_names:
        if layout.is_local(name):
            return name
    return default


def channel_eigenpairs(
    model: Model, *, at_time: float = 0.0, chunk: int | None = None,
) -> ChannelEigenbasis:
    r"""
    Solve the per-mode dense channel eigenproblem ``eigh(iMS, M)``.

    Description
    -----------
    The dense-column probe: builds the linear variant
    ``fr.linearize(model)``, fills the per-mode dense blocks ``S`` by
    unit-impulse columns over the bounded-axis nodes (one linearized
    tendency application each, Fourier read-out along the periodic
    axes), forms the Hermitian pencil ``H = iMS`` under the
    measure-weighted energy metric (``fr.EnergyMetric`` times the
    bounded-axis measure), and solves the whitened batched ``eigh``
    per mode plane. Requires exactly one bounded grid factor (all
    others periodic). On a model carrying a CONSTRAINT stage (the 3-D
    nonhydro channel) the probed operator is the projected
    linearization ``P L P`` — each impulse passes through the public
    ``model.constrain`` matvec, then the constrained tendency — so
    the divergence-complement directions appear as extra exact zero
    modes (see the module docstring).

    Coefficients may vary arbitrarily **along the bounded axis** (the
    probe reads whatever the tendency carries — a beta-plane ``f(y)``
    on the walled-``y`` channel works and stays Hermitian); only the
    periodic axes must be constant-coefficient. The energy metric is
    read with ``require_constant_coriolis=False`` and
    ``allow_field_weights=True`` accordingly: a varying
    ``csqr(y)`` / ``N^2(y)`` enters the metric as a profile weight,
    sampled per component on its own bounded-axis nodes (see
    :func:`_metric_diagonal`), and must be strictly positive.

    Parameters
    ----------
    model : Model
        An assembled channel model (one bounded axis; e.g. the walled
        shallow water model).
    at_time : float, optional
        Evaluation time for time-dependent parameters (default: 0.0).
    chunk : int | None, optional
        Batch size for the impulse probe and the plane eigensolve
        (``lax.map`` chunking, bounds peak memory); ``None`` vmaps
        the full batch. On a multi-device grid the probe always runs
        serially with a host gather (the memory-minimal documented
        setup cost), so ``chunk`` then bounds the eigensolve only
        (default: None).

    Returns
    -------
    ChannelEigenbasis
        The per-plane spectrum, M-orthonormal eigenvectors and the
        segment layout.
    """
    bounded = tuple(
        name for mesh in model.grid.factors
        if not getattr(mesh, "periodic", True)
        for name in mesh.names)
    if not bounded:
        raise ValueError(
            "channel_eigenpairs is the dense-column tier for a grid "
            "with exactly one bounded axis; this grid is fully "
            "periodic — use numeric_eigenpairs (the per-mode "
            "translation-invariant probe) instead")
    if len(bounded) > 1:
        raise ValueError(
            "channel_eigenpairs handles exactly one bounded axis "
            f"(the channel); this grid bounds {bounded!r} — a "
            "multi-walled box has no periodic axis to block-"
            "diagonalize over and is out of scope")
    schedule = model._artifacts.schedule  # noqa: SLF001 — host probe
    constrained = bool(schedule.kind_entries(StageKind.CONSTRAINT))

    names = model.grid.names
    bounded_axis = bounded[0]
    bounded_index = names.index(bounded_axis)
    periodic_names = tuple(
        name for name in names if name != bounded_axis)
    half_axis = _designate_half_axis(model.grid, periodic_names)
    # order the Fourier axes so the half (rfft) axis is transformed
    # last: jnp.fft.rfftn halves the last axis in ``axes=``. The
    # default pick is the last periodic axis (grid order), so this is
    # byte-identical to grid order unless a layout re-designation moved
    # the half axis off a sharded last periodic axis.
    ordered_names = (
        *(name for name in periodic_names if name != half_axis),
        half_axis)
    periodic_axes = tuple(names.index(name) for name in ordered_names)

    metric = EnergyMetric.from_model(
        model, at_time=at_time, require_constant_coriolis=False,
        allow_field_weights=True)
    lin = linearize(model)
    prog, base0 = _rest_background(lin, at_time)
    weights = _metric_weights(metric, prog, allow_fields=True)

    slices = _segment_slices(base0, prog, bounded_index)
    metric_diag = _metric_diagonal(
        base0, prog, weights, bounded_axis, bounded_index)
    if not bool(jnp.all(jnp.isfinite(metric_diag)
                        & (metric_diag > 0.0))):
        raise ValueError(
            "the channel energy metric must be positive definite: "
            "a varying weight profile (the shallow-water csqr(y), "
            "the nonhydro N^2(y)) must be finite and strictly "
            "positive on the bounded axis")
    symbol = _probe_block(
        lin, base0, prog, slices, bounded_index, periodic_axes,
        at_time, chunk, constrained=constrained)

    hamiltonian, residual = _hermitian_pencil(symbol, metric_diag)
    hermiticity_error = float(residual)
    if hermiticity_error > _HERMITICITY_TOL:
        raise ValueError(
            "the probed operator is not skew-adjoint under the "
            "energy metric (relative Hermiticity residual "
            f"{hermiticity_error:.2e} of iMS): the linearization "
            "carries a non-conservative term or a non-rest "
            "background, which the channel eigensolve cannot serve")
    omega, q = _generalized_eigh_diag(hamiltonian, metric_diag, chunk)
    return ChannelEigenbasis(
        omega, q, prog, slices, metric_diag,
        periodic_axis=half_axis,
        bounded_axis=bounded_axis,
        hermiticity_error=hermiticity_error)


# ================================================================
#  Segment layout and the diagonal metric
# ================================================================
def _segment_slices(
    base0: VectorField, prog: tuple[str, ...], bounded_index: int,
) -> dict[str, slice]:
    """Map each component to its segment of the stacked column."""
    slices = {}
    offset = 0
    for name in prog:
        n = base0[name].data.shape[bounded_index]
        slices[name] = slice(offset, offset + n)
        offset += n
    return slices


def _metric_diagonal(
    base0: VectorField,
    prog: tuple[str, ...],
    weights: tuple[object, ...],
    bounded_axis: str,
    bounded_index: int,
) -> jax.Array:
    r"""
    Stack the diagonal metric ``M[(c, j)] = w_c(j) \mu_c(j)``.

    Description
    -----------
    Per component the energy weight times the bounded-axis measure on
    the component's own node set (dual cell widths on faces — the
    per-node quadrature the skew-adjointness holds under). The
    uniform periodic-axis measure is a common scalar factor and drops
    out of the pencil.

    A field-valued (profile) weight is sampled **on the component's
    own node set** through ``.to`` — the identical sampling the
    tendency flux uses (``csqr.to(u)`` at the ``u`` faces,
    ``1/N^2`` at the ``b`` cells), which is exactly what keeps
    ``iMS`` Hermitian for a varying profile — and read out along the
    bounded axis (profiles are constant along the periodic axes by
    construction of ``fr.Profile``).
    """
    parts = []
    for name, weight in zip(prog, weights, strict=True):
        mu = _bounded_measure(base0[name], bounded_axis)
        if isinstance(weight, ScalarField):
            sampled = weight.to(base0[name])
            data = np.broadcast_to(np.asarray(sampled.data),
                                   base0[name].data.shape)
            index: list[int | slice] = [0] * data.ndim
            index[bounded_index] = slice(None)
            w = jnp.asarray(data[tuple(index)],
                            dtype=dtype_real()).ravel()
            parts.append(w * mu)
        else:
            parts.append(weight * mu)
    return jnp.concatenate(parts)


def _bounded_measure(
    field: ScalarField, bounded_axis: str,
) -> jax.Array:
    r"""Per-node bounded-axis measure of a component (depth H if constant).

    Description
    -----------
    The quadrature weight per bounded-axis node the metric diagonal
    stacks. A component that is **constant along the bounded axis** (a
    ``fr.Profile`` barotropic field — the hydrostatic ``ps``, which is a
    single depth-integrated DOF) carries no per-cell measure; its energy
    is weighted by the **full extent** ``H`` of the bounded axis (the
    depth integral ``(1/2) H |ps|^2 / c^2`` the free-surface energy pairs
    with the depth-mean divergence). Every genuine nodal/face component
    defers to ``ScalarField.measure`` (the dual-cell quadrature).
    """
    factor = field.function_space.factor(bounded_axis)
    if getattr(factor, "is_constant", False):
        lo, hi = factor.mesh.extent
        return jnp.asarray([float(hi - lo)], dtype=dtype_real())
    return jnp.asarray(
        field.measure(bounded_axis).data, dtype=dtype_real()).ravel()


# ================================================================
#  The dense-column probe
# ================================================================
def _probe_block(
    lin: Model,
    base0: VectorField,
    prog: tuple[str, ...],
    slices: dict[str, slice],
    bounded_index: int,
    periodic_axes: tuple[int, ...],
    at_time: float,
    chunk: int | None,
    *,
    constrained: bool = False,
) -> jax.Array:
    r"""
    Probe the linearized tendency into per-mode dense blocks.

    Description
    -----------
    Applies the linearized tendency to one-hot impulses over the
    bounded-axis nodes at periodic-index 0 (their DFT is unity at
    every periodic mode), transforms the responses along the periodic
    axes only (``rfftn`` half spectrum — the operator is real), and
    stacks the segments, so
    ``S[..., (c', j'), (c, j)] = rfftn(L e_{(c,j)})_{c'}[..., j']``.
    On a ``constrained`` model each impulse is projected first through
    ``model.constrain`` and the tendency runs with
    ``constraints=True``, so the probed operator is the Hermitian
    sandwich :math:`P L P` (one extra pressure solve per impulse).

    Returns
    -------
    jax.Array
        The batched blocks, shape ``(*modes, D, D)``.
    """
    dim = max(s.stop for s in slices.values())
    batch = _impulse_batch(base0, prog, slices, bounded_index, dim)

    def apply_one(arrays: dict[str, jax.Array]) -> dict[str, jax.Array]:
        """Embed one impulse; apply the (projected) linearization."""
        state = base0.replace(**{
            name: base0[name].with_data(arrays[name])
            for name in prog})
        if constrained:
            # S = P L P: project the impulse (one pressure solve),
            # then the constrained tendency supplies P after L
            state = lin.constrain(state, t=at_time)
            out = lin.tendency(state, t=at_time, constraints=True)
        else:
            out = lin.tendency(state, t=at_time, constraints=False)
        return {name: out[name].data for name in prog}

    if base0.grid.decomposition.device_count > 1:
        responses = _gathered_responses(apply_one, batch, prog, dim)
    elif chunk is None:
        responses = jax.vmap(apply_one)(batch)
    else:
        responses = jax.lax.map(apply_one, batch, batch_size=chunk)

    fft_axes = tuple(axis + 1 for axis in periodic_axes)
    rows = []
    for name in prog:
        spec = jnp.fft.rfftn(responses[name], axes=fft_axes)
        spec = jnp.moveaxis(spec, 0, -1)  # (*grid axes, D)
        rows.append(jnp.moveaxis(spec, bounded_index, -2))
    return jnp.concatenate(rows, axis=-2)


def _gathered_responses(
    apply_one: Callable[[dict[str, jax.Array]], dict[str, jax.Array]],
    batch: dict[str, jax.Array],
    prog: tuple[str, ...],
    dim: int,
) -> dict[str, jax.Array]:
    r"""
    Probe the impulses serially with a host gather (multi-device).

    Description
    -----------
    The batched probe cannot thread the impulse axis through the
    sharded storage contract (``store`` attaches the space's
    per-device sharding, which knows nothing about a ``vmap`` batch
    axis), so on a multi-device grid the impulses run one by one on
    the sharded model and each response gathers to host — the
    documented one-off analysis cost of the setup path (the
    downstream projector *application* stays sharded). This serial
    path is already memory-minimal, so ``chunk`` has nothing left to
    bound here.

    Returns
    -------
    dict[str, jax.Array]
        Per-component stacked responses, shape ``(D, *shape_c)``
        (uncommitted host-rebuilt arrays).
    """
    rows = [apply_one({name: batch[name][i] for name in prog})
            for i in range(dim)]
    return {name: jnp.asarray(np.stack(
        [np.asarray(row[name]) for row in rows]))
        for name in prog}


def _impulse_batch(
    base0: VectorField,
    prog: tuple[str, ...],
    slices: dict[str, slice],
    bounded_index: int,
    dim: int,
) -> dict[str, jax.Array]:
    """Build the (D, *shape_c) one-hot probe batch per component."""
    batch = {}
    for name in prog:
        shape = base0[name].data.shape
        nodes = jnp.arange(shape[bounded_index])
        index: list[jax.Array | int] = [0] * len(shape)
        index[bounded_index] = nodes
        arr = jnp.zeros((dim, *shape), dtype=dtype_real())
        batch[name] = arr.at[
            (slices[name].start + nodes, *index)].set(1.0)
    return batch


# ================================================================
#  The Hermitian pencil and its whitened eigensolve
# ================================================================
def _hermitian_pencil(
    symbol: jax.Array, metric_diag: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Form ``H = iMS`` and its symmetrized half with the residual.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The symmetrized ``(H + H^H) / 2`` and the relative
        pre-symmetrization residual ``max|H - H^H| / max|H|``.
    """
    h = 1j * metric_diag[:, None] * symbol
    h_adj = jnp.conj(jnp.swapaxes(h, -1, -2))
    residual = jnp.max(jnp.abs(h - h_adj)) / jnp.max(jnp.abs(h))
    return 0.5 * (h + h_adj), residual


def _generalized_eigh_diag(
    hamiltonian: jax.Array,
    metric_diag: jax.Array,
    chunk: int | None,
) -> tuple[jax.Array, jax.Array]:
    r"""
    Solve ``H q = mu M q`` for a per-row diagonal metric ``M``.

    Description
    -----------
    Whitens the diagonal positive metric (:math:`R = \mathrm{diag}
    \sqrt{M}`), runs a batched ``eigh`` on :math:`R^{-H} H R^{-1}`
    (optionally ``lax.map``-chunked over the mode planes),
    back-substitutes :math:`q = R^{-1}\tilde q`, sets
    :math:`\omega = \mu` (the oceanographic sign convention:
    ``L q = -i omega q``, so a mode ``q e^{i k x}`` evolves as
    :math:`e^{i(kx - \omega t)}` — positive ``omega`` propagates
    along ``+k``), and sorts each plane ascending.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        The frequencies ``(*modes, D)`` and eigenvectors
        ``(*modes, D, D)``.
    """
    inv_sqrt = 1.0 / jnp.sqrt(metric_diag)
    whitened = inv_sqrt[:, None] * hamiltonian * inv_sqrt[None, :]
    if chunk is None:
        mu, q_white = jnp.linalg.eigh(whitened)
    else:
        planes = whitened.reshape(-1, *whitened.shape[-2:])
        mu, q_white = jax.lax.map(
            jnp.linalg.eigh, planes, batch_size=chunk)
        mu = mu.reshape(whitened.shape[:-1])
        q_white = q_white.reshape(whitened.shape)
    omega = mu
    order = jnp.argsort(omega, axis=-1)
    omega = jnp.take_along_axis(omega, order, axis=-1)
    q_white = jnp.take_along_axis(q_white, order[..., None, :], axis=-1)
    return omega, inv_sqrt[:, None] * q_white
