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
- :func:`envelope_scale` / :func:`normalize_max_component` — the
  amplitude conventions of the single-mode and random states.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_comp
from fridom.framework2.grid.fields.storage import (
    factor_axes,
    self_conjugate_axis_indices,
    storage_dtype,
)
from fridom.framework2.grid.scalars import Scalars
from fridom.framework2.grid.spaces.coefficient import (
    CosineSpace,
    FourierSpace,
    SineSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.framework2.grid.fields.scalar_field import ScalarField
    from fridom.framework2.grid.grid import Grid
    from fridom.framework2.grid.spaces.tensor_product import SpaceLike
    from fridom.framework2.grid.symbols import GridSymbols, ModeChart


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
    return {
        c: kit.backward(c)(
            columns[0][c].with_data(total[c])).real
        for c in total}
