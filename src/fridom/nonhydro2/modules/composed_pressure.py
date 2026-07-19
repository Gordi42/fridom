r"""The composed mapped + immersed (cut-cell metric) pressure solve.

Description
-----------
Mapped + immersed composition plan, stage M2 (decisions MI-D2/D4). On a
grid that declares **both** a ``CoordinateMapping`` with a mapped column
(terrain-following / chart) **and** an
:class:`~fridom.spatial.immersed_domain.ImmersedDomain`, the pressure
projection is the two proven halves composed: the mapped metric SPD
flux operator
(:class:`~fridom.nonhydro2.modules.mapped_pressure.MappedPressureSolver`)
with the immersed open-area fraction inserted as a diagonal face weight
between the exact-transpose corner-hop pairs that carry its SPD license.

This module subclasses :class:`MappedPressureSolver` and threads the
fractions through the flux assembly, the divergence, the velocity
correction, the diagonal / vertical bands, and both preconditioners; the
mapped metric machinery (the ``K = J Bᵀ W B`` coefficients, the
corner-staggered cross hops, the measure-adjoint stretched down-hop) is
inherited unchanged.

Where the fraction enters (MI-D2, the exact-symmetry spelling)
--------------------------------------------------------------
Two open-area fractions weight the operator, both the geometric
min-transfer of :class:`ImmersedDomain`:

- ``alpha_a`` — the **face** fraction on each C-grid flux face
  (``immersed.fraction(face)``, the min of the two adjacent cell
  fractions) — weights each **direct** metric flux leg ``K^{aa} G_a p``
  (and the column leg ``K^{bb} G_b p``), exactly as the flat immersed
  operator weights ``G_a p``.
- ``alpha_corner`` — the **corner** fraction on the cell-corner stagger
  ``Right_a (x) Inner_b`` (``immersed.fraction(corner)``, the min of the
  four cells meeting at the corner) — weights each **cross** leg
  ``K^{ib} G_b p`` *inside* the corner hop, i.e. between the transpose-
  paired ``up`` / ``down`` interpolations.

The corner weight is the spelling the **exact-symmetry gate** selects
(plan §4, the corner-cross subtlety): weighting the whole assembled face
flux by that face's own ``alpha`` would give the ``a``-face cross leg
``alpha_a`` and its base-face transpose partner ``alpha_base`` — two
different scalars on the two halves of one adjoint pair, which breaks
``<q, L p> = <p, L q>`` on genuine partials. A single shared corner
weight rides the corner (a symmetric diagonal at the shared stagger
point), so both halves of the pair carry the same scalar and the pairing
— hence the exact symmetry and the CG license — is preserved. Because
the corner fraction is the min of the four surrounding cells, it is
non-zero only where every cell (hence every face and gradient) feeding
the corner is wet, so the wet-region constant is still an exact
nullspace member (``L(c e) = 0`` at machine zero) and an all-wet chart
(``alpha == 1`` everywhere) reduces **bitwise** to the pure mapped
operator.

The row scale is **unchanged** from the mapped solver: the operator is
self-adjoint under the grid's own computational measure
(``grid.measure``), the very product ``ConjugateGradient`` evaluates
(``krylov`` ``_dot``), so CG's inner product is untouched. The
nullspace is the **wet-region** constant ``e = (theta > 0)``, projected
by the **computational-measure** wet mean (the flat immersed I2 e-form
generalized to the chart: the projector must be orthogonal in the same
product the operator is symmetric in — the computational measure, not
the Jacobian-weighted physical one).

Right-hand side and velocity correction (MI-D4)
-----------------------------------------------
The right-hand side is the ``alpha``-weighted (masked) J-weighted
divergence of the provisional velocity (:meth:`divergence`): the
``a``-face contravariant flux ``J u_a`` weighted by ``alpha_a``, the
column's metric cross weighted by ``alpha_corner``. Masked wall / cut
faces carry ``alpha = 0`` flux, so the right-hand side is compatible
with the wet-region nullspace by telescoping. The velocity correction
(:meth:`velocity_correction`) is the mapped flux-consistent correction
``F/(alpha J)`` gated by the boolean face mask ``alpha > 0``, with the
``alpha J`` and ``alpha_base`` divides double-``jnp.where`` sealed (the
house VJP idiom): the composition ``divergence(velocity_correction(p))``
reproduces :meth:`apply` **exactly**, so the post-projection masked
divergence is the CG residual, not an O(h^2) consistency remainder.

Preconditioners (MI-D3)
-----------------------
``"multigrid"`` (the composed default) re-instantiates this class per
coarsened level (``Grid.coarsened`` propagates both descriptors, so each
level re-derives the metric **and** re-quadratures the fractions), with
the fraction-weighted metric vertical-line smoother and the per-level
wet-mean projection. ``"spectral"`` is the flat separable spectral
inverse at ``alpha``-folded coefficients, **masked onto the wet cells**
(``z = e (x) M(r)``, the immersed IP-D6 correction) — the cheap fallback,
which needs many iterations on a genuine cut chart (reported in the
gates). Both are inherited from the mapped V-cycle / spectral builders
with the fraction inserted.
"""
# Mapped + immersed composition, stage M2 (MI-D2/D3/D4)
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.nonhydro2.modules.mapped_pressure import (
    _LINE_OMEGA,
    MappedPressureSolver,
    _adjacent_face_sum,
    _corner_cross_bracket,
)
from fridom.nonhydro2.modules.pressure import (
    is_fv,
    rediscretize_fv_coarse,
)
from fridom.spatial.operators.krylov import (
    ConjugateGradient,
    _computational_integral,
    _computational_mean,
)
from fridom.spatial.operators.multigrid import (
    MultigridLevel,
    MultigridVCycle,
    VerticalBands,
    VerticalLineJacobi,
)
from fridom.spatial.operators.multigrid_hierarchy import coarsen_levels
from fridom.spatial.operators.staggering import uniform_spacing

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.nonhydro2.modules.mapped_pressure import MetricCache
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.spaces.tensor_product import SpaceLike

#: the smallest immersed quadrature ``order`` a chart admits: below it
#: (``None`` / ``1``) the collocation staircase samples the computational
#: cell centre, which the chart mis-places (§6 seam 2)
_MIN_CHART_ORDER = 2


class ComposedPressureSolver(MappedPressureSolver):

    r"""
    PCG solve of the composed mapped + immersed pressure problem (M2).

    Description
    -----------
    A trace-time object (constructed per projection call, no mutable
    pytree state) extending :class:`MappedPressureSolver`: the mapped
    metric operator with the immersed open-area fractions inserted as
    diagonal face / corner weights (module docstring). It requires a
    grid that declares both a mapped column (``mapping.column_corrections``
    truthy, validated by the base class) and an
    :class:`~fridom.spatial.immersed_domain.ImmersedDomain`. The public
    surface mirrors the mapped and immersed solvers: :meth:`divergence`,
    :meth:`apply`, :meth:`solve`, :meth:`velocity_correction`,
    :meth:`project`.

    Parameters
    ----------
    grid : object
        The grid carrying **both** the coordinate mapping and the
        immersed domain, and the dispatch registry.
    space : SpaceLike
        The (cell-centered / cell-average) pressure space.
    preconditioner : str, optional
        The PCG preconditioner: ``"multigrid"`` (the composed default,
        the fraction-weighted mapped V-cycle with per-level wet-mean
        projection), ``"spectral"`` (the wet-masked ``alpha``-folded flat
        spectral inverse) or ``"none"`` (unpreconditioned CG). A
        stretched base column with ``"spectral"`` raises at construction
        (the base class N1) (default: ``"multigrid"``).
    iterations : int
        The fixed CG iteration budget (forwarded to the base class).
    tolerance : float | None, optional
        The PCG convergence break (forwarded) (default: 1e-8).
    weights : Mapping[str, jax.Array | float] | None, optional
        Per-coordinate physical-axis weights (forwarded) (default: None).
    params : Mapping[str, ScalarField] | None, optional
        Dynamic mapping-parameter fields (forwarded) (default: None).
    single_precision : bool, optional
        Run the spectral preconditioner in single precision (forwarded)
        (default: False).
    multigrid_levels : int | None, optional
        The multigrid depth (forwarded) (default: None).
    multigrid_tridiagonal_method : str, optional
        The vertical-line tridiagonal kernel (forwarded)
        (default: ``"auto"``).
    multigrid_coarsen_vertical : bool, optional
        Whether the V-cycle coarsens the mapped column too (forwarded)
        (default: True).

    Raises
    ------
    ValueError
        If the grid carries no immersed domain (or, via the base class,
        no mapped column / an unknown preconditioner).
    """

    def __init__(
        self,
        grid: object,
        space: SpaceLike,
        *,
        preconditioner: str = "multigrid",
        **kwargs: object,
    ) -> None:
        """Resolve the mapped rows, then fetch the fraction fields."""
        super().__init__(
            grid, space,
            preconditioner=preconditioner,  # type: ignore[arg-type]
            **kwargs)  # type: ignore[arg-type]
        immersed = getattr(grid, "immersed", None)
        if immersed is None:
            raise ValueError(
                "the grid carries no immersed domain; the composed "
                "mapped + immersed pressure solve needs a "
                "Grid(..., mapping=..., immersed=...) — the mapped "
                "MappedPressureSolver applies without a mask")
        # a chart immersed domain must use genuine per-cell quadrature
        # (order >= 2): the collocation staircase (order=None/1) samples
        # the indicator at the *computational* cell centre, which the
        # chart maps to the wrong physical point (M0+M1 §6 seam 2), so a
        # collocation mask on a chart is a silent wrong-geometry mask.
        # A taught error, never a silent unmapped mask.
        if immersed.order is None or immersed.order < _MIN_CHART_ORDER:
            raise NotImplementedError(
                "a chart immersed domain needs genuine per-cell "
                "quadrature (ImmersedDomain(order=q) with q >= 2): the "
                "collocation staircase (order=None/1) samples the "
                "indicator at the computational cell centre, which the "
                "mapped column places at the wrong physical position, so "
                "the wet-region mask would be wrong-geometry on a chart "
                "(mapped + immersed composition plan §6). Pass order>=2.")
        self._immersed = immersed
        # the open-area FACE fraction on each C-grid flux face (the
        # min-transfer of the two adjacent cells, IP-D1): weights each
        # direct metric flux leg through that face
        self._alpha_face: dict[str, ScalarField] = {
            a: immersed.fraction(self._face[a]) for a in self._axes}
        # the CORNER fraction on the cell-corner stagger (the min of the
        # four cells meeting at the corner): weights each cross leg
        # *inside* the transpose-paired corner hop (the exact-symmetry
        # spelling, module docstring)
        self._alpha_corner: dict[str, ScalarField] = {
            a: immersed.fraction(self._corner[a]) for a in self._coupled}
        # the cell volume fraction and the boolean masks
        self._theta: ScalarField = immersed.fraction(self._space)
        self._cell_mask: jax.Array = (
            self._theta.data > 0.0).astype(dtype_real())
        self._fmask: dict[str, ScalarField] = {
            a: self._alpha_face[a].with_data(
                (self._alpha_face[a].data > 0.0).astype(dtype_real()))
            for a in self._axes}
        # the wet-cell indicator e = (theta > 0), the operator's exact
        # nullspace constant, and its computational measure (the product
        # in which the row-scaled operator is symmetric, MI-D2)
        self._wet: ScalarField = self._theta.with_data(self._cell_mask)
        self._wet_measure: jax.Array = jnp.sum(
            _computational_integral(self._wet).data)

    @property
    def _coarsen_vertical(self) -> bool:
        """
        Whether :meth:`_build_vcycle` coarsens the composed column too.

        Description
        -----------
        Overrides the mapped decision to **keep** the semicoarsening
        fallback on a stretched composed base: the eager coarse-mesh
        pre-warm (:meth:`~MappedPressureSolver._prewarm_hierarchy`) that
        lifts the ``MappedIntervalMesh``-ctor jit incompatibility warms
        only the coarse *grids*, but the composed coarse level also
        re-quadratures the immersed **fractions** on those coarse
        spaces (MI-D3) — validating the wet-region fraction
        re-derivation under a coarsened stretched column is a follow-up,
        so a stretched composed base stays horizontally semicoarsened.
        A uniform composed base still takes the full-coarsening default.
        Since ``_prewarm_hierarchy`` reads this same decision, no
        pre-warm fires for a composed solver (byte-identical to before).
        """
        return self._multigrid_coarsen_vertical and not self._stretched_base

    # ================================================================
    #  Cross-term corner chains (fraction inserted at the corner)
    # ================================================================
    def _cross_to_face(self, axis: str, g_b: ScalarField,
                       cache: MetricCache | None = None,
                       ) -> ScalarField:
        """Assemble the ``i``-face cross flux with the corner fraction.

        The mapped :meth:`~MappedPressureSolver._cross_to_face` with
        ``alpha_corner`` multiplied into the corner slope contraction
        (module docstring): the shared corner weight both halves of the
        adjoint pair carry.
        """
        corner = self._up_i[axis](g_b)
        weight = self._alpha_corner[axis] * self._slope(
            corner, axis, cache)
        flux = (corner * weight).retag(self._corner_tagged[axis])
        return self._down_b_hop(axis, flux, cache)

    def _cross_to_column(self, axis: str, v: ScalarField,
                         cache: MetricCache | None = None,
                         ) -> ScalarField:
        """Apply the column-face cross chain with the corner fraction.

        The exact transpose partner of :meth:`_cross_to_face`, carrying
        the **same** ``alpha_corner`` at the shared corner stagger.
        """
        corner = self._up_b[axis](v)
        weight = self._alpha_corner[axis] * self._slope(
            corner, axis, cache)
        return self._down_i[axis](corner * weight)

    # ================================================================
    #  The operator flux assembly (face fraction on the direct legs)
    # ================================================================
    def _fluxes(self, p: ScalarField,
                cache: MetricCache | None = None,
                ) -> dict[str, ScalarField]:
        """Assemble the composed per-axis fluxes ``alpha (x) K^{aj} G_j p``.

        The mapped :meth:`~MappedPressureSolver._fluxes` with the
        **face** fraction ``alpha_a`` weighting each direct leg and the
        **corner** fraction inside each cross leg (the overridden
        :meth:`_cross_to_face` / :meth:`_cross_to_column`).
        """
        grads = {a: self._grad[a](p) for a in self._axes}
        fluxes: dict[str, ScalarField] = {}
        g_b = grads[self._base]
        for a in self._axes:
            if a == self._base:
                continue
            flux = self._alpha_face[a] * (
                grads[a] * self._jacobian(grads[a], cache))
            if a in self._coupled:
                flux = flux - self._cross_to_face(a, g_b, cache)
            fluxes[a] = self._weight(a) * flux
        flux = self._alpha_face[self._base] * (
            g_b * self._column_coefficient(g_b.function_space, cache))
        for a in self._coupled:
            flux = flux - self._weight(a) * self._cross_to_column(
                a, grads[a], cache)
        fluxes[self._base] = flux
        return fluxes

    # ================================================================
    #  Masked divergence (the right-hand side)
    # ================================================================
    def divergence(
        self, vel: Mapping[str, ScalarField],
        cache: MetricCache | None = None,
    ) -> ScalarField:
        r"""Compute the masked, J-weighted divergence (the RHS).

        Description
        -----------
        The mapped :meth:`~MappedPressureSolver.divergence` with the
        open-area fractions inserted: the ``a``-face contravariant flux
        ``J u_a`` weighted by ``alpha_a``, the column's metric cross
        weighted by ``alpha_corner`` (the overridden
        :meth:`_cross_to_column`), and the direct column velocity
        weighted by ``alpha_base``. Masked wall / cut faces carry
        ``alpha = 0`` flux, so the sum against the wet-region constant
        telescopes to zero — the right-hand side is compatible. After
        :meth:`velocity_correction` the *same* divergence of the
        corrected velocity is the CG residual.

        Parameters
        ----------
        vel : Mapping[str, ScalarField]
            The physical velocity components keyed by coordinate name.
        cache : MetricCache | None, optional
            The per-solve metric memo (default: None).

        Returns
        -------
        ScalarField
            The masked J-weighted divergence on the pressure space.
        """
        if set(vel) != set(self._axes):
            raise ValueError(
                f"divergence needs one component per axis "
                f"{self._axes}, got {tuple(sorted(vel))}")
        out = None
        column = self._alpha_face[self._base] * vel[self._base].retag(
            self._face[self._base])
        for a in self._axes:
            if a == self._base:
                continue
            u = vel[a].retag(self._face[a])
            if a in self._coupled:
                column = column - self._cross_to_column(a, u, cache)
            flux = self._alpha_face[a] * (u * self._jacobian(u, cache))
            term = self._div[a](flux.retag(self._tagged[a]))
            out = term if out is None else out + term
        term = self._div[self._base](
            column.retag(self._tagged[self._base]))
        return out + term

    # ================================================================
    #  Velocity correction (boolean-masked, alpha-sealed divides)
    # ================================================================
    def velocity_correction(
        self, p: ScalarField, cache: MetricCache | None = None,
    ) -> dict[str, ScalarField]:
        r"""Derive the flux-consistent, boolean-masked corrections.

        Description
        -----------
        The mapped correction ``F/(alpha J)`` gated by the boolean face
        mask ``m = (alpha > 0)`` (MI-D4): per coupled/plain axis
        ``corr_a = m_a (F_a / (alpha_a J))``; the column reassembles the
        cross contributions ``corr_base = m_base (F_base + sum_i cross_i)
        / alpha_base`` so that the masked :meth:`divergence` of the
        corrections reproduces :meth:`apply` exactly (the projection
        removes exactly the divergence the operator measures). The
        ``alpha J`` and ``alpha_base`` divides are double-``jnp.where``
        sealed (the numerator vanishes wherever the denominator does —
        ``alpha_corner <= alpha_a <= 1`` on every face — so the seal is
        bitwise on wet cells and finite in the closed / padded slots,
        keeping the reverse-mode gradient NaN-free).

        Parameters
        ----------
        p : ScalarField
            The solved pressure.
        cache : MetricCache | None, optional
            The per-solve metric memo (default: None).

        Returns
        -------
        dict[str, ScalarField]
            Per-coordinate corrections to subtract (BC-free).
        """
        fluxes = self._fluxes(p, cache)
        corrections: dict[str, ScalarField] = {}
        column = fluxes[self._base]
        for a in self._axes:
            if a == self._base:
                continue
            corr = self._fmask[a] * self._divide_by_alpha_jacobian(
                fluxes[a], a, cache)
            if a in self._coupled:
                column = column + self._cross_to_column(a, corr, cache)
            corrections[a] = corr
        corrections[self._base] = (
            self._fmask[self._base]
            * self._divide_by_alpha_base(column))
        return corrections

    def _divide_by_alpha_jacobian(
        self, flux: ScalarField, axis: str,
        cache: MetricCache | None = None,
    ) -> ScalarField:
        r"""Divide a composed flux by ``alpha_a J``, VJP-safe.

        The metric quotient ``F_a / (alpha_a J)`` of a coupled/plain
        correction. ``alpha_a J`` is strictly positive on every open,
        valid face and zero in the closed / never-valid padding, where
        the composed flux is itself zero (``alpha_corner <= alpha_a``);
        the double-``jnp.where`` seals the reverse pass while staying
        bitwise on every open face (the house idiom,
        :meth:`~MappedPressureSolver._divide_by_jacobian`).
        """
        jac = self._jacobian(flux, cache)
        denom = self._alpha_face[axis].storage * jac.storage
        bad = denom == 0.0
        safe = jnp.where(bad, 1.0, denom)
        ratio = jnp.where(bad, 0.0, flux.storage / safe)
        return (flux / jac).with_storage(ratio)

    def _divide_by_alpha_base(
        self, column: ScalarField,
    ) -> ScalarField:
        r"""Divide the reassembled column flux by ``alpha_base``, VJP-safe.

        The base-face open-area fraction is strictly positive on every
        open column face and zero in the closed / padded slots, where
        the reassembled column flux is itself zero; the double-
        ``jnp.where`` seals the reverse pass.
        """
        alpha = self._alpha_face[self._base]
        bad = alpha.storage == 0.0
        safe = jnp.where(bad, 1.0, alpha.storage)
        ratio = jnp.where(bad, 0.0, column.storage / safe)
        return column.with_storage(ratio)

    # ================================================================
    #  Wet-region nullspace projection (computational measure)
    # ================================================================
    def _projection(self, f: ScalarField) -> ScalarField:
        r"""Project out the wet-region constant (computational measure).

        ``f - (int_wet f dV_comp / int_wet dV_comp) e`` with
        ``e = (theta > 0)`` the wet-cell indicator — the exact
        ``L``-nullspace constant. The mean is the **computational**
        measure (``_computational_integral``, the product CG's ``_dot``
        evaluates and the composed operator is symmetric in), *not* the
        Jacobian-weighted physical ``integrate`` verb — so the projector
        is orthogonal in CG's geometry and the residual reaches machine
        zero and holds (MI-D2, the I2 e-form generalized to the chart).
        Touches only the wet cells; a compatible right-hand side is a
        fixed point.
        """
        mean = jnp.sum(_computational_integral(self._wet * f).data)
        return f - self._wet * (mean / self._wet_measure)

    # ================================================================
    #  Diagonal / vertical bands (fraction-weighted metric)
    # ================================================================
    def _diagonal_data(
        self, cache: MetricCache | None = None,
    ) -> jax.Array:
        r"""Assemble the composed operator diagonal (storage frame).

        The mapped :meth:`~MappedPressureSolver._diagonal_data` with the
        **face** fraction weighting each diagonal flux leg and the
        **corner** fraction weighting the mixed corner-cross bracket
        (module docstring). Probe-exact against ``sum_c e_c (x) L e_c``.
        """
        storage = self._axis_storage()
        base = self._base
        z_axis = storage[base]
        if self._stretched_base:
            m_cell = self._measure(self._space, cache).data
            m_inner = self._measure(self._face[base], cache).data
        else:
            dz = uniform_spacing(self._space.factor(base))
        diagonal: jax.Array | None = None
        for a in self._axes:
            if a == base:
                kface = self._column_coefficient(self._face[a], cache)
                kdata = self._alpha_face[base].data * kface.data
                if self._stretched_base:
                    leg = -_adjacent_face_sum(
                        kdata / m_inner, z_axis,
                        periodic=False) / m_cell
                else:
                    leg = -_adjacent_face_sum(
                        kdata, z_axis, periodic=False) / (dz * dz)
            else:
                dx = uniform_spacing(self._space.factor(a))
                kface = self._weight(a) * self._metric(
                    self._face[a],
                    f"d{self._mapped}_d{self._base}", cache)
                kdata = self._alpha_face[a].data * kface.data
                leg = -_adjacent_face_sum(
                    kdata, storage[a], periodic=True) / (dx * dx)
            diagonal = leg if diagonal is None else diagonal + leg
        for a in self._coupled:
            dx = uniform_spacing(self._space.factor(a))
            slope = self._metric(
                self._corner[a], f"d{self._mapped}_d{a}", cache)
            sdata = self._alpha_corner[a].data * slope.data
            bracket = _corner_cross_bracket(sdata, storage[a], z_axis)
            if self._stretched_base:
                diagonal = diagonal - self._weight(a) / (
                    2.0 * dx) * bracket / m_cell
            else:
                diagonal = diagonal - self._weight(a) / (
                    2.0 * dx * dz) * bracket
        return diagonal

    def vertical_bands(
        self, cache: MetricCache | None = None,
    ) -> VerticalBands:
        r"""Return the fraction-weighted per-column tridiagonal ``T``.

        The mapped :meth:`~MappedPressureSolver.vertical_bands` with the
        **base** fraction weighting the vertical off-diagonal band
        ``alpha_base K^{bb}`` (the exact vertical off-diagonal of the
        composed operator; the horizontal cross residues couple
        off-column, so the tridiagonal misses nothing on the column).
        ``diag`` is the composed full diagonal (:meth:`_diagonal_data`).
        """
        storage = self._axis_storage()
        base = self._base
        z_axis = storage[base]
        kbb = self._alpha_face[base].data * self._column_coefficient(
            self._face[base], cache).data
        edge = list(kbb.shape)
        edge[z_axis] = 1
        zeros = jnp.zeros(edge, dtype=kbb.dtype)
        if self._stretched_base:
            m_cell = self._measure(self._space, cache).data
            m_inner = self._measure(self._face[base], cache).data
            band = kbb / m_inner
            lower_data = jnp.concatenate(
                [zeros, band], axis=z_axis) / m_cell
            upper_data = jnp.concatenate(
                [band, zeros], axis=z_axis) / m_cell
        else:
            dz = uniform_spacing(self._space.factor(base))
            lower_data = jnp.concatenate([zeros, kbb], axis=z_axis) / (
                dz * dz)
            upper_data = jnp.concatenate([kbb, zeros], axis=z_axis) / (
                dz * dz)
        template = self._grid.create_field(self._space)
        shape = template.data.shape
        diag = template.with_data(
            jnp.broadcast_to(self._diagonal_data(cache), shape))
        lower = template.with_data(jnp.broadcast_to(lower_data, shape))
        upper = template.with_data(jnp.broadcast_to(upper_data, shape))
        return VerticalBands(lower, diag, upper, z_axis)

    # ================================================================
    #  Preconditioners (fraction inserted; wet-mean projection)
    # ================================================================
    def _mean_coefficients(
        self, cache: MetricCache | None = None,
    ) -> dict[str, jax.Array]:
        """Fold the ``alpha``-weighted diagonal coefficients to means.

        The mapped fold with the face fraction inside each coefficient
        (a coarser but wet-aware constant for the masked spectral
        fallback).
        """
        coeffs: dict[str, jax.Array] = {}
        for a in self._axes:
            if a == self._base:
                field = self._column_coefficient(self._face[a], cache)
            else:
                field = self._weight(a) * self._metric(
                    self._face[a],
                    f"d{self._mapped}_d{self._base}", cache)
            field = self._alpha_face[a] * field
            coeffs[a] = jnp.reshape(
                _computational_mean(field).data, ())
        return coeffs

    def _preconditioner(
        self, cache: MetricCache | None = None,
    ) -> Callable[[ScalarField], ScalarField]:
        """Build the wet-masked ``alpha``-folded flat spectral inverse.

        The mapped folded-coefficient spectral inverse (at the
        ``alpha``-weighted means, :meth:`_mean_coefficients`) **masked
        onto the wet cells** (``z = e (x) M(r)``, the IP-D6 correction):
        the residual is wet-supported, so the masked inverse is the
        symmetric ``D M D`` SPD on the wet subspace; the raw unmasked
        inverse would mix its global ``k = 0`` gauge with the
        wet-constant nullspace and destabilize CG.
        """
        base = super()._preconditioner(cache)
        cell_mask = self._cell_mask

        def masked(r: ScalarField) -> ScalarField:
            z = base(r)
            return z.with_data(z.data * cell_mask)

        return masked

    def _build_vcycle(self, cache: MetricCache) -> MultigridVCycle:
        r"""Assemble the composed multigrid V-cycle preconditioner (MI-D3).

        Description
        -----------
        The mapped V-cycle with each level re-instantiated as a
        :class:`ComposedPressureSolver` — ``Grid.coarsened`` propagates
        **both** the mapping and the immersed descriptor, so each coarse
        level re-derives the metric **and** re-quadratures the fractions
        on its own coarse spaces (MG-D6). The smoother is the
        fraction-weighted metric vertical-line Jacobi
        (:meth:`vertical_bands`); every level projects out its **own**
        wet-region constant (the computational-measure wet mean,
        :meth:`_projection`). Static maps only (the base class raises on
        moving geometry).

        Parameters
        ----------
        cache : MetricCache
            The finest level's per-solve metric memo (shared with the
            outer CG operator).

        Returns
        -------
        MultigridVCycle
            The V-cycle callable for the ``preconditioner=`` seam.
        """
        if self._params and any(
                getattr(field, "grid", None) is not None
                for field in self._params.values()):
            raise NotImplementedError(
                "the multigrid preconditioner supports static maps only "
                "(iteration 1): this solve threads dynamic mapping "
                "parameter fields through params= (moving geometry), "
                "which the coarse re-derivation cannot re-bind — use "
                "preconditioner='spectral' with a moving geometry")
        chain = coarsen_levels(
            self._grid, self._space, vertical=self._base,
            coarsen_vertical=self._coarsen_vertical,
            max_levels=self._multigrid_levels,
            rediscretize=(rediscretize_fv_coarse
                          if is_fv(self._space) else None))
        levels: list[MultigridLevel] = []
        for index, (grid, space, transfer) in enumerate(chain):
            if index == 0:
                solver: ComposedPressureSolver = self
                level_cache: MetricCache = cache
            else:
                solver = ComposedPressureSolver(
                    grid, space, iterations=self._iterations,
                    weights=self._weights, params=self._params,
                    single_precision=self._single_precision,
                    preconditioner=(
                        "none" if self._stretched_base else "spectral"))
                level_cache = {}
            smoother = VerticalLineJacobi(
                solver.vertical_bands(level_cache), omega=_LINE_OMEGA,
                method=self._multigrid_tridiagonal_method)
            levels.append(MultigridLevel(
                partial(solver.apply, cache=level_cache),
                smoother, solver._projection, transfer))
        return MultigridVCycle(tuple(levels))

    # ================================================================
    #  The preconditioned, wet-mean-projected solve
    # ================================================================
    def krylov(
        self, cache: MetricCache | None = None,
    ) -> ConjugateGradient:
        """Build the configured PCG solver (wet-mean projected).

        As the mapped :meth:`~MappedPressureSolver.krylov`, but the CG
        nullspace projection is the **wet-region** constant
        (:meth:`_projection`) rather than the global mean, and the
        spectral preconditioner is the wet-masked one
        (:meth:`_preconditioner`).
        """
        if cache is None:
            cache = {}
        if self._preconditioner_kind == "none":
            preconditioner = None
        elif self._preconditioner_kind == "multigrid":
            preconditioner = self._build_vcycle(cache)
        else:
            preconditioner = self._preconditioner(cache)
        return ConjugateGradient(
            partial(self.apply, cache=cache),
            preconditioner=preconditioner,
            iterations=self._iterations,
            tolerance=self._tolerance,
            projection=self._projection)

    def project(
        self, vel: Mapping[str, ScalarField],
        x0: ScalarField | None = None,
    ) -> tuple[ScalarField, dict[str, ScalarField]]:
        """Run the whole masked projection on one shared derivation.

        As the mapped :meth:`~MappedPressureSolver.project`, but the
        returned diagnostic pressure is masked to zero on the dry cells
        (the physical no-pressure-under-topography convention; the
        boolean-masked corrections are already applied, so the mask is
        cosmetic).
        """
        cache: MetricCache = {}
        p = self.solve(self.divergence(vel, cache), x0, cache=cache)
        corr = self.velocity_correction(p, cache)
        return p.with_data(p.data * self._cell_mask), corr
