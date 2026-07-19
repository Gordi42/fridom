r"""
Eigenmode-indexed initial-condition machinery (shared primitives).

Description
-----------
The package-shared building blocks behind the eigenmode initial
conditions: the mode-indexed single-mode accessors
(``em.mode(s, indices)`` on the analytic eigenmodes,
``eb.mode(family, indices)`` on the channel eigenbasis) and the
prescribed-spectra random states (``sw.random_state`` /
``nh.random_state``). Everything here is host-side construction
code; the sharded application paths (transforms, projections) are
untouched.

- :func:`geostrophic_energy_spectrum` — the Masur & Oliver [2020]
  geostrophic spectral energy density (the default vortical
  spectrum);
- :func:`coefficient_index` — axis-keyed integer mode indices
  resolved to storage slots on a coefficient space (Fourier
  half/full layouts, trig union-lattice modes);
- :func:`hermitian_mode_data` — a single Hermitian-closed mode
  placement whose backward transform is exactly real;
- :func:`prescribed_spectra_coefficients` — the analytic-tier
  random-phase synthesis (all modes at once, no per-plane loop);
- :func:`synthesize_columns` — the shared backward-synthesis tail of
  the random states and single-mode accessors: the fused distributed
  route (Hermitian half-axis re-expression, no gather) on a sharded
  periodic grid, the replicated / single-device backward otherwise;
- :func:`envelope_scale` / :func:`normalize_max_component` — the
  amplitude conventions of the single-mode and random states;
- :func:`evaluate_frequency_function` — the guarded host-side
  ``f(omega)`` weight evaluation behind ``em.function(f, sel)`` /
  ``eb.function(f, sel)`` on every eigenmode tier.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

from fridom.framework.utils import dtype_comp
from fridom.spatial.fields.storage import (
    factor_axes,
    self_conjugate_axis_indices,
    storage_dtype,
    store,
)
from fridom.spatial.scalars import Scalars
from fridom.spatial.spaces.coefficient import (
    CosineSpace,
    FourierSpace,
    SineSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Iterable, Mapping

    import jax

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.tensor_product import SpaceLike
    from fridom.spatial.symbols import GridSymbols, ModeChart


# ================================================================
#  Spectral energy densities
# ================================================================
def geostrophic_energy_spectrum(
    kx: jax.Array | float,
    ky: jax.Array | float,
    d: float = 7.0,
    k0: float = 2.0,
) -> jax.Array:
    r"""
    Geostrophic spectral energy density (Masur & Oliver [2020]).

    Description
    -----------
    The energy spectrum :math:`S` is given by:

    .. math::

        S = \frac{k^7}{\left(k^2 + a k_0^2\right)^{2b}}

    where :math:`k = \sqrt{k_x^2 + k_y^2}` is the horizontal
    wavenumber and :math:`a` and :math:`b` are constants:

    .. math::

        a = \frac{4}{7}b - 1, \quad b = \frac{7+d}{4}

    with :math:`d` the power-law exponent for large horizontal
    wavenumbers (:math:`S(k) \sim k^{-d}` for :math:`k \to \infty`)
    and :math:`k_0` the wavenumber carrying the maximum energy.

    Parameters
    ----------
    kx : jax.Array | float
        The horizontal wavenumber in the x-direction.
    ky : jax.Array | float
        The horizontal wavenumber in the y-direction.
    d : float, optional
        The large-``k`` power-law exponent (default: 7).
    k0 : float, optional
        The energy-peak wavenumber (default: 2).

    Returns
    -------
    jax.Array
        The spectral energy density.
    """
    kh = jnp.sqrt(jnp.asarray(kx) ** 2 + jnp.asarray(ky) ** 2)
    b = (7.0 + d) / 4.0
    a = (4.0 / 7.0) * b - 1.0
    return kh ** 7 / (kh ** 2 + a * k0 ** 2) ** (2 * b)


# ================================================================
#  Axis-keyed mode indices on coefficient spaces
# ================================================================
def coefficient_index(
    space: SpaceLike,
    indices: Mapping[str, int],
) -> tuple[int, ...] | None:
    r"""
    Resolve axis-keyed mode indices to storage slots on ``space``.

    Description
    -----------
    ``indices`` keys every grid axis by name and gives the integer
    mode along it: on a half-spectrum (real-origin Fourier) factor
    the index must lie in ``0..n//2`` (the stored half); on a
    full-spectrum factor any integer is taken modulo ``n`` (fft
    layout, negative wavenumbers included); on a trig (sine/cosine)
    factor the index is the **physical** mode on the ``0..n`` union
    lattice — a mode the component's own trig family does not hold
    resolves to ``None`` (the component is structurally absent from
    that stratum), never an error.

    Parameters
    ----------
    space : SpaceLike
        The component's coefficient space.
    indices : Mapping[str, int]
        Axis-keyed integer mode indices, one per grid axis.

    Returns
    -------
    tuple[int, ...] | None
        The storage-slot index, or ``None`` when a trig factor
        lacks the requested mode.

    Raises
    ------
    ValueError
        On wrong axis keys or out-of-range mode indices.
    """
    names = space.names
    if set(indices) != set(names):
        raise ValueError(
            "mode indices are keyed by the grid axes "
            f"{tuple(names)!r}; got keys {tuple(indices)!r}")
    slots = []
    for name, (factor, _axis) in zip(
            names, factor_axes(space.bare), strict=True):
        m = int(indices[name])
        if isinstance(factor, FourierSpace):
            n = factor.origin.shape[0]
            if factor.scalars is Scalars.REAL:
                if not 0 <= m <= n // 2:
                    raise ValueError(
                        f"axis {name!r} stores the Hermitian half "
                        f"spectrum: mode indices run 0..{n // 2} "
                        f"(a negative mode is the conjugate of its "
                        f"mirror image); got {m}")
                slots.append(m)
            else:
                slots.append(m % n)
        elif isinstance(factor, SineSpace | CosineSpace):
            union = factor.mesh.n_cells
            if not 0 <= m <= union:
                raise ValueError(
                    f"axis {name!r} holds the trig union modes "
                    f"0..{union}; got {m}")
            slot = m - factor.mode_offset
            if not 0 <= slot < factor.shape[0]:
                return None
            slots.append(slot)
        else:
            # a value error (bad space choice), not a type error
            raise ValueError(  # noqa: TRY004
                f"axis {name!r} of {space!r} is not a coefficient "
                "factor; mode indices resolve on coefficient "
                "spaces only")
    return tuple(slots)


def hermitian_mode_data(
    space: SpaceLike,
    slots: tuple[int, ...],
    value: complex | jax.Array,
) -> jax.Array:
    r"""
    True-shape array carrying one Hermitian-closed mode.

    Description
    -----------
    Places ``value`` at ``slots`` such that the backward transform
    is exactly real: at an interior half-spectrum index the single
    placement suffices (the conjugate mode is implicit in the
    unstored half); on the self-conjugate planes of the halved axis
    the placement splits into the ``(value/2, conj(value)/2)`` pair
    at the index and its full-spectrum mirror — degenerating to
    ``Re(value)`` at a fully self-conjugate index.

    Parameters
    ----------
    space : SpaceLike
        The coefficient space (complex storage required).
    slots : tuple[int, ...]
        The storage-slot index (see :func:`coefficient_index`).
    value : complex | jax.Array
        The complex mode amplitude.

    Returns
    -------
    jax.Array
        The true-shape coefficient array.

    Raises
    ------
    ValueError
        On a real-storage space (no Fourier factor to close).
    """
    dtype = storage_dtype(space)
    if not jnp.issubdtype(dtype, jnp.complexfloating):
        raise ValueError(
            f"hermitian_mode_data needs complex storage; {space!r} "
            "stores real coefficients (no Fourier factor)")
    data = jnp.zeros(space.shape, dtype=dtype)
    value = jnp.asarray(value, dtype=dtype)
    halved = [
        (factor, axis) for factor, axis in factor_axes(space.bare)
        if isinstance(factor, FourierSpace)
        and factor.scalars is Scalars.REAL]
    self_conj = all(
        slots[axis] in self_conjugate_axis_indices(factor)
        for factor, axis in halved)
    if not (halved and self_conj):
        return data.at[slots].set(value)
    partner = list(slots)
    for factor, axis in factor_axes(space.bare):
        if (isinstance(factor, FourierSpace)
                and factor.scalars is Scalars.COMPLEX):
            n = factor.shape[0]
            partner[axis] = (n - slots[axis]) % n
    return (data.at[slots].add(0.5 * value)
            .at[tuple(partner)].add(0.5 * jnp.conj(value)))


# ================================================================
#  Amplitude conventions
# ================================================================
def envelope_scale(
    z0: Mapping[str, ScalarField],
    z1: Mapping[str, ScalarField],
    names: tuple[str, ...],
) -> float:
    r"""
    Largest horizontal-velocity envelope amplitude of a mode pair.

    Description
    -----------
    ``z0`` and ``z1`` are the two phase quadratures of a single
    mode (phases ``phi`` and ``phi + pi/2``), so
    ``z0**2 + z1**2`` is the squared pointwise oscillation
    envelope. The scale is the maximum envelope over the named
    (horizontal-velocity) components on their own nodes —
    phase-independent by construction. Zero (a mode without
    horizontal velocity, e.g. the geostrophic ``k = 0`` mean)
    falls back to ``1.0`` (no normalization).

    Parameters
    ----------
    z0 : Mapping[str, ScalarField]
        The mode at phase ``phi``.
    z1 : Mapping[str, ScalarField]
        The mode at phase ``phi + pi/2``.
    names : tuple[str, ...]
        The horizontal-velocity component names.

    Returns
    -------
    float
        The normalization scale (``1.0`` where zero).
    """
    peak = 0.0
    for name in names:
        env2 = z0[name].data ** 2 + z1[name].data ** 2
        peak = max(peak, float(jnp.max(env2)))
    scale = peak ** 0.5
    return scale if scale > 0.0 else 1.0


def normalize_max_component(
    fields: Mapping[str, ScalarField],
    names: tuple[str, ...],
) -> dict[str, ScalarField]:
    r"""
    Normalize a state so the largest horizontal velocity is one.

    Description
    -----------
    Divides every component by the max over the named
    (horizontal-velocity) components of the pointwise ``|value|``
    on their own staggered nodes; a state with zero horizontal
    velocity everywhere is returned unscaled (the reference
    convention).

    Parameters
    ----------
    fields : Mapping[str, ScalarField]
        The state components.
    names : tuple[str, ...]
        The horizontal-velocity component names.

    Returns
    -------
    dict[str, ScalarField]
        The normalized components (all of ``fields``).
    """
    peak = max(float(jnp.max(jnp.abs(fields[name].data)))
               for name in names)
    scale = peak if peak > 0.0 else 1.0
    return {name: field / scale for name, field in fields.items()}


# ================================================================
#  Guarded f(omega) weights (the em.function / eb.function tiers)
# ================================================================
def evaluate_frequency_function(
    f: Callable[[np.ndarray], np.ndarray],
    omega: np.ndarray,
    selected: np.ndarray,
    describe: Callable[[np.ndarray], str],
) -> np.ndarray:
    r"""
    Evaluate ``f`` on the selected frequencies (structural guard).

    Description
    -----------
    The shared host-side weight builder of the ``function(f, sel)``
    applicators: ``f`` receives the **real** frequencies of the
    selected modes as one array (complex return values are allowed,
    e.g. ``f = lambda w: 1 / (1j * w)``) and is evaluated **only
    there** — structurally absent modes never reach ``f``. A
    non-finite value on any selected mode (a singular ``f`` meeting
    a structurally zero frequency, e.g. ``1/(i omega)`` on a
    vortical selection) raises a ``ValueError`` built by
    ``describe`` — the guard is an error, never a floored division.

    Parameters
    ----------
    f : Callable[[np.ndarray], np.ndarray]
        The scalar spectral function, vectorized over an array of
        real frequencies (scalar returns broadcast).
    omega : np.ndarray
        The real frequencies (imaginary parts, if any, are
        discarded).
    selected : np.ndarray
        Boolean selection mask, shape ``omega.shape``.
    describe : Callable[[np.ndarray], str]
        Builds the guard message from the boolean mask (shape
        ``omega.shape``) of the offending selected modes.

    Returns
    -------
    np.ndarray
        Complex weights of shape ``omega.shape``: ``f(omega)`` on
        the selection, exact zeros elsewhere.

    Raises
    ------
    ValueError
        If ``f`` evaluates non-finite on any selected mode.
    """
    om = np.real(np.asarray(omega))
    sel = np.asarray(selected, dtype=bool)
    weights = np.zeros(om.shape, dtype=complex)
    if not sel.any():
        return weights
    with np.errstate(all="ignore"):
        # a singular f meeting a zero frequency must surface as the
        # taught guard below, not as a numpy floating-point warning
        values = np.broadcast_to(
            np.asarray(f(om[sel]), dtype=complex), om[sel].shape)
    finite = np.isfinite(values)
    if not finite.all():
        bad = np.zeros(om.shape, dtype=bool)
        bad[sel] = ~finite
        raise ValueError(describe(bad))
    weights[sel] = values
    return weights


def resolve_mode_branches(s: int | Iterable[int]) -> tuple[int, ...]:
    """
    Normalize an analytic-tier branch selection to a tuple.

    Description
    -----------
    A single branch or an iterable of distinct branches from the
    analytic mode set ``{0, +1, -1}``; anything else — including
    duplicates, which would double-count their contribution — is a
    taught ``ValueError``.

    Parameters
    ----------
    s : int | Iterable[int]
        The branch selection.

    Returns
    -------
    tuple[int, ...]
        The validated branches, in selection order.

    Raises
    ------
    ValueError
        On branches outside ``{0, +1, -1}``, duplicates, an empty
        iterable, or a non-branch value.
    """
    branches = (s,) if isinstance(s, int) else s
    try:
        branches = tuple(branches)
    except TypeError:
        branches = ()
    good = (branches and len(set(branches)) == len(branches)
            and all(isinstance(b, int) and b in (0, 1, -1)
                    for b in branches))
    if not good:
        raise ValueError(
            "mode branches are 0, +1 or -1 — a single branch or an "
            f"iterable of distinct branches; got {s!r}")
    return branches


def describe_nonfinite_branch(
    s: int, omega: np.ndarray, bad: np.ndarray,
) -> str:
    """Build the analytic-tier structural-zero guard message."""
    magnitude = float(np.abs(np.real(omega))[bad].min())
    return (
        f"function(f, s={s}): f evaluates non-finite on "
        f"{int(bad.sum())} represented mode(s) of the branch "
        f"(|omega| down to {magnitude:.6g}). Structurally zero "
        "frequencies are excluded by selection, never floored: "
        "drop the zero-frequency branch (s = 0) from the "
        "selection, or pass an f that is finite there")


# ================================================================
#  Per-mode operator matrix (the distributed analytic route)
# ================================================================
def assemble_operator_matrix(
    em: object,
    *,
    branches: tuple[int, ...],
    components: tuple[str, ...],
    shape: tuple[int, ...],
    f: Callable[[np.ndarray], np.ndarray] | None = None,
) -> jax.Array:
    r"""
    Assemble the per-mode ``D x D`` operator matrix on a coeff frame.

    Description
    -----------
    The frame-local port of the analytic projector / ``f(L)`` algebra
    for the fused distributed route (the numeric channel's
    ``Q diag(w) Q^H M`` analogue): summing the rank-1 branch terms

    .. math::

        M_{jd} = \sum_{s}\sum_{\text{col}} w_s \; q^s_{\text{col},j}\;
                 \overline{p^s_{\text{col},d}}

    over the branch selection and each branch's internal column family
    (the vortical Nyquist family of an even periodic-vertical grid),
    where ``q`` is the eigenvector column and ``p`` its Rayleigh dual
    (the energy metric folded in). ``f is None`` is the plain projector
    (``w_s = 1``); otherwise ``w_s = f(omega_s)`` on each column's
    **represented** modes (the structural-zero guard of
    :func:`evaluate_frequency_function`, so a singular ``f`` never
    floors -- the rank-1 vanishes at unrepresented modes anyway, but
    ``f`` is not evaluated there). ``em`` is a frame clone
    (``_reframe``) whose ``_columns`` / ``_dual`` / ``omega`` /
    ``_energy_weights`` read the target coefficient frame; the assembly
    is host-built (replicated coefficient basis), the same choice the
    channel contraction makes. On a periodic grid the ``ModeChart`` is
    identity, so no cross-lattice embed is needed.

    Parameters
    ----------
    em : object
        The frame-clone eigenmodes (its symbols on the target frame).
    branches : tuple[int, ...]
        The mode-branch selection.
    components : tuple[str, ...]
        The prognostic component order (the matrix's ``j`` / ``d`` axes).
    shape : tuple[int, ...]
        The coefficient-frame broadcast shape (all components share it).
    f : Callable[[np.ndarray], np.ndarray] | None, optional
        The scalar spectral function ``f(omega)`` (``None`` is the plain
        projector) (default: None).

    Returns
    -------
    jax.Array
        The per-mode matrix, shape ``(*shape, D, D)``, complex.
    """
    dim = len(components)
    dtype = dtype_comp()
    weights = em._energy_weights()  # noqa: SLF001 — frame-clone internals
    mat = jnp.zeros((*shape, dim, dim), dtype=dtype)
    for b in branches:
        omega_real: np.ndarray | None = None
        for col in em._columns(b):  # noqa: SLF001 — frame-clone internals
            p = em._dual(col, b)  # noqa: SLF001 — frame-clone internals
            q_stack = jnp.stack(
                [jnp.broadcast_to(jnp.asarray(col[c].data),
                                  shape).astype(dtype)
                 for c in components], axis=-1)
            p_stack = jnp.stack(
                [jnp.broadcast_to(jnp.asarray(p[c]), shape).astype(dtype)
                 for c in components], axis=-1)
            rank1 = (q_stack[..., :, None]
                     * jnp.conj(p_stack)[..., None, :])
            if f is None:
                mat = mat + rank1
                continue
            if omega_real is None:
                omega_real = np.real(np.asarray(jnp.broadcast_to(
                    jnp.real(jnp.asarray(em.omega(b).data)), shape)))
            norm = sum(
                weights[c] * jnp.abs(jnp.broadcast_to(
                    jnp.asarray(col[c].data), shape)) ** 2
                for c in components)
            w = jnp.asarray(evaluate_frequency_function(
                f, omega_real, np.asarray(norm) != 0,
                lambda bad, b=b, om=omega_real:
                describe_nonfinite_branch(b, om, bad)))
            mat = mat + w[..., None, None] * rank1
    return mat


def _union_frame_shape(space: SpaceLike) -> tuple[int, ...]:
    r"""
    Return the union-lattice shape of a mixed coefficient frame.

    Description
    -----------
    Each trig factor (``Sine`` / ``Cosine``) is widened to its
    ``ModeChart`` union extent ``n_cells + 1`` (the shared ``0..n``
    lattice ``embed`` lands on); Fourier and constant factors keep
    their own extent. This is the broadcast shape of the walled
    per-mode matrix (:func:`assemble_walled_operator_matrix`), whose
    axes are the horizontal Fourier modes and the vertical union
    modes.
    """
    shape: list[int] = []
    for factor in space.bare.factors:
        if isinstance(factor, SineSpace | CosineSpace):
            shape.append(factor.mesh.n_cells + 1)
        else:
            shape.append(factor.shape[0])
    return tuple(shape)


def assemble_walled_operator_matrix(
    em: object,
    *,
    branches: tuple[int, ...],
    components: tuple[str, ...],
    coeff_of: Mapping[str, SpaceLike],
    chart: ModeChart,
    f: Callable[[np.ndarray], np.ndarray] | None = None,
) -> jax.Array:
    r"""
    Assemble the per-mode ``D x D`` matrix on the union trig lattice.

    Description
    -----------
    The walled-vertical (mixed ``Fourier x Fourier x trig``) analogue of
    :func:`assemble_operator_matrix`: the analytic projector / ``f(L)``
    algebra summed as rank-1 branch terms, but every per-component column
    ``q^s`` and dual ``p^s`` is first
    :meth:`~fridom.spatial.symbols.ModeChart.embed`\ -ded from its own
    trig mode lattice (``w`` on DST-I ``1..n-1``, ``b`` on DST-II
    ``1..n``, ``u`` / ``v`` / ``p`` on DCT-II ``0..n-1``) onto the
    shared ``0..n`` union lattice, so the rank-1 outer products align
    per **physical** vertical mode. The horizontal Fourier axes are the
    distributed internal frame's (the sharded axis full spectrum, the
    partner axis complex); the vertical axis is the union lattice. The
    fused region
    (``WalledVerticalTransform.apply_matrix`` in
    ``spatial.operators.distributed_transform``) embeds each transposed
    component onto the same union lattice before the ``D x D``
    contraction and restricts the result back, so this matrix and that
    region compose exactly.

    ``f is None`` is the plain projector (``w_s = 1``); otherwise
    ``w_s = f(omega_s)`` on each column's represented union modes (the
    ``omega`` and the represented-mode mask are themselves embedded from
    ``w``'s frame, matching ``Eigenmodes.function``'s union-lattice
    accumulation). ``em`` is a frame clone (``_reframe``) whose symbols
    read the target coefficient frames; the assembly is host-built
    (replicated), the same choice the periodic matrix and the channel
    contraction make.

    Parameters
    ----------
    em : object
        The frame-clone eigenmodes (its symbols on the target frames).
    branches : tuple[int, ...]
        The mode-branch selection.
    components : tuple[str, ...]
        The prognostic component order (the matrix's ``j`` / ``d`` axes).
    coeff_of : Mapping[str, SpaceLike]
        Per-component internal coefficient (bare) frame.
    chart : ModeChart
        The union-lattice chart aligning the components' trig lattices.
    f : Callable[[np.ndarray], np.ndarray] | None, optional
        The scalar spectral function ``f(omega)`` (``None`` is the plain
        projector) (default: None).

    Returns
    -------
    jax.Array
        The per-mode matrix, shape ``(*union, D, D)``, complex.
    """
    dim = len(components)
    dtype = dtype_comp()
    weights = em._energy_weights()  # noqa: SLF001 — frame-clone internals
    shape = _union_frame_shape(coeff_of[components[0]])

    def embed(data: jax.Array, name: str) -> jax.Array:
        """Broadcast to the component frame, then embed onto the union."""
        per = jnp.broadcast_to(jnp.asarray(data), coeff_of[name].shape)
        return jnp.broadcast_to(chart.embed(per, coeff_of[name]), shape)

    mat = jnp.zeros((*shape, dim, dim), dtype=dtype)
    for b in branches:
        omega_real: np.ndarray | None = None
        for col in em._columns(b):  # noqa: SLF001 — frame-clone internals
            p = em._dual(col, b)  # noqa: SLF001 — frame-clone internals
            q_stack = jnp.stack(
                [embed(col[c].data, c).astype(dtype) for c in components],
                axis=-1)
            p_stack = jnp.stack(
                [embed(p[c], c).astype(dtype) for c in components],
                axis=-1)
            rank1 = (q_stack[..., :, None]
                     * jnp.conj(p_stack)[..., None, :])
            if f is None:
                mat = mat + rank1
                continue
            if omega_real is None:
                omega_real = np.real(np.asarray(embed(
                    jnp.real(jnp.asarray(em.omega(b).data)), "w")))
            norm = sum(
                embed(weights[c] * jnp.abs(jnp.broadcast_to(
                    jnp.asarray(col[c].data), coeff_of[c].shape)) ** 2, c)
                for c in components)
            w = jnp.asarray(evaluate_frequency_function(
                f, omega_real, np.asarray(norm) != 0,
                lambda bad, b=b, om=omega_real:
                describe_nonfinite_branch(b, om, bad)))
            mat = mat + w[..., None, None] * rank1
    return mat


# ================================================================
#  Prescribed-spectra random coefficients (the analytic tier)
# ================================================================
def prescribed_spectra_coefficients(
    *,
    grid: Grid,
    kit: GridSymbols,
    chart: ModeChart,
    columns: tuple[Mapping[str, ScalarField], ...],
    components: tuple[str, ...],
    weights: Mapping[str, float],
    reference: str,
    horizontal: tuple[str, ...],
    spectral_energy_density: Callable[..., jax.Array],
    seed: int,
) -> dict[str, ScalarField]:
    r"""
    Sum random-phase eigenmode columns under a prescribed spectrum.

    Description
    -----------
    The analytic-tier port of the reference
    ``PrescribedSpectraRandomPhase`` semantics, vectorized over all
    modes at once. Per eigenvector column ``q`` (one per branch in
    ``columns``) and per mode ``k``:

    .. math::

        \zeta_c = \sqrt{\frac{S(k)}{\pi k_h \, \|q\|_M^2}}
                  \; \phi(k) \, q_c(k)

    where :math:`S` is the prescribed spectral energy density
    evaluated on the grid wavenumbers (one argument per grid axis,
    in grid order), :math:`\pi k_h` the horizontal ring measure of
    the stored half lattice (the :math:`S(k) = 2\pi k\,E(k, 0)`
    angular convention of the reference: each stored mode carries
    the energy :math:`S/(\pi k_h)`, so a shell of modes realizes
    the 1-D density :math:`S`), :math:`\|q\|_M^2` the eigenvector's
    energy norm, and :math:`\phi` a Hermitian unit-modulus random
    phase (``grid.random.phase``, seeded ``seed + branch``).

    Cross-component mode alignment runs through the ``ModeChart``
    union lattice; the phases and amplitudes are drawn on the
    ``reference`` component's coefficient lattice, so union modes
    that component lacks (the walled-vertical buoyancy-top stratum
    for ``reference="u"``) are excluded. Modes with
    :math:`k_h = 0` or a structurally zero column carry nothing.

    Parameters
    ----------
    grid : Grid
        The grid.
    kit : GridSymbols
        The eigenmode transform kit (per-component coefficient
        spaces and backward transforms).
    chart : ModeChart
        The union-lattice chart of the grid.
    columns : tuple[Mapping[str, ScalarField], ...]
        The coefficient-space eigenvector columns, one per branch
        (e.g. ``(em.q(0),)`` vortical, ``(em.q(1), em.q(-1))``
        wave).
    components : tuple[str, ...]
        The prognostic component names of the columns.
    weights : Mapping[str, float]
        The per-component energy-metric weights.
    reference : str
        The component whose coefficient lattice keys phases and
        amplitudes (canonically ``"u"``).
    horizontal : tuple[str, ...]
        The horizontal axis names entering :math:`k_h`.
    spectral_energy_density : Callable[..., jax.Array]
        ``S(*k)`` over the grid-axis wavenumbers, in grid order.
    seed : int
        The base PRNG seed (branch ``i`` uses ``seed + i``).

    Returns
    -------
    dict[str, ScalarField]
        Real physical fields on the kit's analysis spaces
        (unnormalized).
    """
    coeff_ref = kit.coeff(reference)
    kfields = {name: grid.wavenumbers(coeff_ref, name=name).data
               for name in grid.names}
    kh = jnp.sqrt(sum(kfields[name] ** 2 for name in horizontal))
    spectra = spectral_energy_density(
        *(kfields[name] for name in grid.names))
    total: dict[str, jax.Array] | None = None
    for i, q in enumerate(columns):
        norm = sum(
            chart.embed(
                weights[c] * jnp.abs(q[c].data) ** 2, kit.coeff(c))
            for c in components)
        norm_ref = chart.restrict(norm, coeff_ref)
        good = (kh > 0.0) & (norm_ref > 0.0)
        denom = jnp.where(good, jnp.pi * kh * norm_ref, 1.0)
        amp = jnp.where(good, jnp.sqrt(spectra) / jnp.sqrt(denom),
                        0.0)
        phi = grid.random.phase(coeff_ref, seed=seed + i).data
        gain = chart.embed(
            (amp * phi).astype(dtype_comp()), coeff_ref)
        contribution = {
            c: q[c].data * chart.restrict(gain, kit.coeff(c))
            for c in components}
        total = contribution if total is None else {
            c: total[c] + contribution[c] for c in total}
    return _synthesize_random(grid, kit, columns, components, total)


def _synthesize_random(
    grid: Grid,
    kit: GridSymbols,
    columns: tuple[Mapping[str, ScalarField], ...],
    components: tuple[str, ...],
    total: dict[str, jax.Array],
) -> dict[str, ScalarField]:
    r"""
    Synthesize the summed random coefficient columns to real fields.

    Description
    -----------
    Backward-transforms the per-component gain columns (built on the
    single-device coefficient frame) to real physical fields through
    :func:`synthesize_columns`: the fused distributed route on a sharded
    periodic grid (no gather), the replicated / single-device backward
    otherwise. The template component fields wrap the output.
    """
    templates = {
        c: grid.create_field(kit.backward(c).codomain, name=c)
        for c in components}
    wrap = {c: columns[0][c] for c in components}
    return synthesize_columns(grid, kit, components, total, templates, wrap)


def synthesize_columns(
    grid: Grid,
    kit: GridSymbols,
    components: tuple[str, ...],
    columns: Mapping[str, jax.Array],
    templates: Mapping[str, ScalarField],
    wrap: Mapping[str, ScalarField],
) -> dict[str, ScalarField]:
    r"""
    Backward-synthesize coefficient columns to real fields (no gather).

    Description
    -----------
    The shared synthesis tail of the analytic-tier synthesis-only
    features (the prescribed-spectra random states of
    :func:`prescribed_spectra_coefficients`, the single-mode ``em.mode``
    states). Each ``columns[c]`` is a Hermitian coefficient column on the
    **single-device** frame ``kit.coeff(c)`` -- device-independent and
    replicated, so the draw is deterministic across device counts by
    construction (``grid.random.phase`` / :func:`hermitian_mode_data`
    key on the global storage index, which the single-device frame
    fixes). The backward transform routes:

    - through the fused ``jax.shard_map`` backward half of the
      distributed route on a sharded fully periodic grid -- directly when
      the transpose engine's internal frame coincides with the
      single-device frame (a grid whose sharded axis is not the half
      axis), else through the Hermitian half-axis re-expression
      (:func:`~fridom.model.analytic_distributed.hermitian_reframe`) onto
      the re-designated internal frame. Either way every transform axis
      stays device-local (no all-gather on the IC path);
    - through the plain per-component backward on the replicated
      coefficient columns when no route resolves **or the route cannot
      synthesize** (the walled-vertical route: its internal frame runs
      both periodic axes fully complex, re-designating the Hermitian half
      axis, so the :func:`hermitian_reframe` bridge has no valid source
      half axis and the fused backward has no matching frame -- a non-1-D
      layout also lands here): the coefficient **data** is
      device-invariant, but a field on the default layout carries the
      sharded-axis layout metadata that would trip the Tier-1 transform
      guard, so each column is rebuilt on the bare (unlaid-out)
      coefficient space -- a replicated backward, still device invariant.
      A single-device grid keeps the original path bitwise.

    Parameters
    ----------
    grid : Grid
        The grid.
    kit : GridSymbols
        The eigenmode transform kit (single-device coefficient frame).
    components : tuple[str, ...]
        The component order to synthesize.
    columns : Mapping[str, jax.Array]
        Per-component Hermitian coefficient columns on ``kit.coeff(c)``.
    templates : Mapping[str, ScalarField]
        Per-component output templates (the analysis codomain fields),
        for the fused route's output layout / wrapping.
    wrap : Mapping[str, ScalarField]
        Per-component coefficient-space fields wrapping the columns for
        the replicated / single-device backward (the eigenvector column
        field carrying the frame metadata).

    Returns
    -------
    dict[str, ScalarField]
        The synthesized real component fields.
    """
    from fridom.model.analytic_distributed import (  # noqa: PLC0415 — deferred: avoid an import cycle at module load
        hermitian_reframe,
        resolve_route,
    )
    route = resolve_route(grid, kit._spaces, tuple(components))  # noqa: SLF001 — kit analysis spaces
    if route is not None and route.can_synthesize:
        coeffs = {
            c: (columns[c] if route.coeff_of(c) == kit.coeff(c)
                else hermitian_reframe(
                    columns[c], kit.coeff(c), route.coeff_of(c)))
            for c in components}
        return route.synthesize(coeffs, templates)
    if getattr(grid.decomposition, "device_count", 1) <= 1:
        return {
            c: kit.backward(c)(wrap[c].with_data(columns[c])).real
            for c in components}
    decomposition = grid.decomposition
    return {
        c: kit.backward(c)(bare_coeff_field(
            grid, decomposition, kit.coeff(c), wrap[c], columns[c])).real
        for c in components}


def bare_coeff_field(
    grid: Grid,
    decomposition: object,
    space: SpaceLike,
    template: ScalarField,
    data: jax.Array,
) -> ScalarField:
    """Wrap ``data`` as a replicated (layout-free) coefficient field."""
    stored = store(decomposition, space, data)
    return type(template)(grid, space, stored, template.metadata)
