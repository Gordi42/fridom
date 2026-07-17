r"""The masked (cut-cell) pressure Poisson solve on an immersed grid.

Description
-----------
Immersed-partial-cells plan, stage I2 (decision IP-D6). On a grid
carrying an :class:`~fridom.spatial.immersed_domain.ImmersedDomain`
the incompressibility constraint is the **masked** projection: every
face flux is weighted by the open-area fraction ``alpha_f`` and the
cut-cell Poisson operator is row-scaled by the *full* cell volume, so
it stays symmetric negative-semidefinite in the measure-weighted
product :class:`~fridom.spatial.operators.krylov.ConjugateGradient`
already evaluates. The separable spectral inverse no longer applies
(the coefficients are spatially variable ``{0, alpha, 1}``), so this
module builds the matrix-free operator and wraps the CS-D2
:class:`ConjugateGradient` around it, preconditioned by the flat
spectral inverse and projected onto the wet-region-constant nullspace.

The discrete formulation (plan §2)
----------------------------------
Per cell ``c`` and face ``f`` (any axis): full cell volume ``V_c``,
full face area ``A_f``, center distance ``d_f``, cell volume fraction
``theta_c = fraction(cell space)`` and face-area fraction
``alpha_f = fraction(face space)`` (the min-transfer of I0).

- **Operator** (row-scaled by the full volume, not the wet volume):
  ``L p|_c = (1/V_c) sum_f +/- alpha_f A_f G_f(p)`` with the plain
  two-point gradient ``G_f(p) = (p_R - p_L)/d_f`` (**unweighted**; the
  vertical leg carries the nonhydrostatic ``1/dsqr``). In field
  arithmetic ``L p = sum_axis (alpha_axis (x) G_axis(p)).diff(axis)``:
  the ``FaceDifference`` gradient leg, the open-area multiply, the
  Dirichlet-tagged flux difference (exact-zero wall flux on ``Inner``).
  Then ``<q, L p>_V = - sum_f alpha_f A_f d_f G_f(q) G_f(p)`` — exactly
  symmetric, so CG needs **no** change to its inner product (choosing
  ``1/V`` over the physical ``1/(theta V)`` is a diagonal row scaling
  that keeps the product untouched).
- **Right-hand side**: the same-scaled masked divergence of the
  provisional velocity, ``r|_c = (1/V_c) sum_f +/- alpha_f A_f u*_f``
  (:meth:`divergence`), wet-supported and compatible by telescoping
  (masked wall faces carry zero flux, so ``sum_c V_c r_c = 0``).
- **Velocity correction**: ``u_axis -= m_axis (x) G_axis(p)`` (vertical
  leg ``1/dsqr``-weighted) with ``m_axis`` the **boolean** face mask
  ``alpha_f > 0`` (:meth:`velocity_correction`). Because ``alpha m =
  alpha`` exactly, the masked divergence of the corrected velocity
  equals the CG residual, and no velocity is injected into a closed
  (``alpha = 0``) face.
- **Nullspace**: the physical nullspace is the wet-region constant
  ``e = (theta > 0)``. CG runs with the ``projection=`` hook set to the
  **V-orthogonal** removal of ``e``: ``f - (int_wet V f / int_wet V)
  e``, installed on the right-hand side, every preconditioned residual,
  and the solution. This is a **correction to the plan's IP-D6**
  ``int theta`` mean: subtracting a *global* constant weighted by the
  theta-mean re-introduces incompatibility on genuine partial cells
  (the residual plateaus at ~1e-2, verified). The V-orthogonal ``e``
  form is a no-op on a compatible right-hand side and touches only the
  wet cells, so every CG iterate stays exactly wet-supported — the
  residual reaches machine zero and *holds* (no fixed-iteration
  blow-up). Dry rows of ``L`` are identically zero (a face with
  ``alpha > 0`` has two wet neighbors under the min rule).
- **Preconditioner**: the flat spectral inverse
  (:func:`~fridom.nonhydro2.modules.pressure.build_flat_spectral_solve`
  — the very machinery :class:`SpectralPressureSolver` builds)
  **masked onto the wet cells**, ``z = e (x) M(r)``. Masking is a
  second **correction to the plan's IP-D6**: the *raw* unmasked inverse
  destabilizes the iteration (its global ``k = 0`` gauge mixes with the
  wet-constant nullspace and CG diverges — the hand-rolled prototype
  ran unpreconditioned for exactly this reason). Because the residual
  is wet-supported, the masked ``z = e (x) M r`` is the symmetric
  ``D M D`` (``D = diag e``), SPD on the wet subspace, so PCG converges
  (~15-30 iterations on masked boxes, more on genuine partials as the
  wet region's conditioning degrades); on an all-wet domain ``D = I``,
  the exact inverse, ~1 iteration.

Like :class:`MappedPressureSolver` this is a trace-time object,
constructed per projection call, carrying no mutable pytree state: it
resolves the ``diff`` rows and fetches the (memoized, concrete-only)
fraction fields at construction and holds the live ``dsqr`` leaf, so
the operator is a pure closure safe inside the CG ``lax.scan``.
"""
# Immersed-partial-cells plan, stage I2: masked cut-cell PCG solve
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.nonhydro2.modules.pressure import (
    _dirichlet_mid,
    build_flat_spectral_solve,
)
from fridom.spatial.fields.storage import factor_axes
from fridom.spatial.operators.base import resolve_codomain
from fridom.spatial.operators.krylov import ConjugateGradient
from fridom.spatial.operators.multigrid import VerticalBands
from fridom.spatial.operators.staggering import uniform_spacing

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.operators.base import Operator
    from fridom.spatial.spaces.tensor_product import SpaceLike


# ================================================================
#  Diagonal / band assembly helper (measure-free; storage frame)
# ================================================================
def _adjacent_face_sum(
    data: jax.Array, axis: int, *, periodic: bool,
) -> jax.Array:
    r"""
    Sum the two face fractions adjacent to each cell along ``axis``.

    Description
    -----------
    The ``alpha_{f-} + alpha_{f+}`` cell field of a flux leg: a
    periodic axis wraps (``roll`` — the ``GridTransfer`` neighbor-access
    pattern, GSPMD-lowered), a bounded axis drops the wall faces by
    zero-padding the ``Inner`` face array to the cell count (the
    Neumann boundary rows). ``face c`` lies at ``c + 1/2`` (the right
    face of cell ``c``, the left face of cell ``c + 1``).

    Parameters
    ----------
    data : jax.Array
        The face-fraction array (storage frame).
    axis : int
        The storage-frame axis of the differenced coordinate.
    periodic : bool
        Whether the axis wraps.

    Returns
    -------
    jax.Array
        The adjacent-face sum on the cell axis (length ``n_cells``).
    """
    if periodic:
        return data + jnp.roll(data, 1, axis=axis)
    edge = list(data.shape)
    edge[axis] = 1
    zeros = jnp.zeros(edge, dtype=data.dtype)
    right = jnp.concatenate([data, zeros], axis=axis)
    left = jnp.concatenate([zeros, data], axis=axis)
    return right + left


class ImmersedPressureSolver:

    r"""
    PCG solve of the masked cut-cell pressure Poisson problem (I2).

    Description
    -----------
    A trace-time object (constructed per projection call, no mutable
    state). The public surface mirrors :class:`MappedPressureSolver`:
    :meth:`divergence` (the masked divergence the operator measures),
    :meth:`apply` (the SPD cut-cell operator), :meth:`solve`
    (preconditioned, wet-mean-projected CG), :meth:`velocity_correction`
    (the flux-consistent boolean-masked gradient subtraction), and
    :meth:`project` (all three on one construction).

    Parameters
    ----------
    grid : object
        The grid carrying the immersed domain and dispatch registry.
    space : SpaceLike
        The (cell-centered / cell-average) pressure space.
    vertical : str
        The vertical coordinate name (the ``1/dsqr``-weighted axis).
    dsqr : jax.Array | float
        The live squared-aspect-ratio leaf.
    iterations : int
        The fixed CG iteration budget (``pressure_iterations``).
    single_precision : bool, optional
        Run the spectral *preconditioner* in single precision while the
        CG iterates, the operator and the inner products stay
        ``float64`` (mixed-precision PCG); forwarded to
        :func:`build_flat_spectral_solve` (default: False).

    Raises
    ------
    ValueError
        If the grid carries no immersed domain.
    NotImplementedError
        If the grid also declares a mapped column (mapped + immersed
        is a designed-for composition, plan §6).
    """

    def __init__(
        self,
        grid: object,
        space: SpaceLike,
        *,
        vertical: str,
        dsqr: jax.Array | float,
        iterations: int,
        single_precision: bool = False,
    ) -> None:
        """Resolve the flux rows and fetch the fraction fields."""
        immersed = getattr(grid, "immersed", None)
        if immersed is None:
            raise ValueError(
                "the grid carries no immersed domain; the masked "
                "pressure solve needs a Grid(..., immersed=...) — the "
                "flat SpectralPressureSolver applies otherwise")
        mapping = getattr(grid, "mapping", None)
        if mapping is not None and getattr(
                mapping, "column_corrections", None):
            raise NotImplementedError(
                "mapped + immersed pressure solve is a designed-for "
                "composition (immersed-partial-cells plan §6): this "
                "grid declares both a terrain-following mapped column "
                "and an immersed domain, which iteration 2 does not "
                "support — use one or the other")
        self._grid = grid
        self._space: SpaceLike = space.bare
        self._vertical = vertical
        self._dsqr = dsqr
        self._iterations = iterations
        self._single_precision = bool(single_precision)
        self._immersed = immersed
        self._axes: tuple[str, ...] = self._space.active_axis_names
        self._resolve_flux_rows(grid.dispatch)
        # concrete, memoized fraction fields (I0): open-area fraction
        # on each flux face, cell volume fraction on the pressure cell
        self._alpha: dict[str, ScalarField] = {
            a: immersed.fraction(self._face[a]) for a in self._axes}
        self._theta: ScalarField = immersed.fraction(self._space)
        # the boolean face mask alpha > 0 (the velocity-correction
        # gate — slip-independent, so no velocity enters a closed face)
        self._fmask: dict[str, ScalarField] = {
            a: self._alpha[a].with_data(
                (self._alpha[a].data > 0.0).astype(dtype_real()))
            for a in self._axes}
        # the wet-cell indicator e = (theta > 0): the actual nullspace
        # constant of L (the boolean wet-region, partial cells included),
        # NOT the theta-weighted field. The nullspace projection is the
        # V-orthogonal removal of e; a theta-weighted mean would subtract
        # a global constant with the wrong coefficient and re-introduce
        # incompatibility on genuine partial cells (deviation from the
        # plan's theta-mean, IP-D6 — verified: the theta form plateaus
        # the residual at ~1e-2 on partials). Used too as the diagnostic
        # clean-up mask (dry cells carry no pressure).
        self._cell_mask: jax.Array = (
            self._theta.data > 0.0).astype(dtype_real())
        self._wet: ScalarField = self._theta.with_data(self._cell_mask)
        self._wet_measure: jax.Array = jnp.sum(
            self._wet.integrate().data)

    def _resolve_flux_rows(self, registry: object) -> None:
        """
        Resolve the per-axis gradient/divergence legs (static).

        Description
        -----------
        ``grad_a`` keys on the cell factor (``FaceDifference``,
        ``CellAvg -> face``), ``div_a`` on the Dirichlet-tagged face
        factor (``FluxDifference``, the zero-normal-flux wall closure;
        the tag is the identity on periodic axes).
        """
        self._grad: dict[str, Operator] = {}
        self._face: dict[str, SpaceLike] = {}
        self._tagged: dict[str, SpaceLike] = {}
        self._div: dict[str, Operator] = {}
        for a in self._axes:
            g = registry.resolve("diff", self._space.factor(a))[a]
            face = resolve_codomain(g, self._space)
            tagged = _dirichlet_mid(face, a)
            self._grad[a] = g
            self._face[a] = face
            self._tagged[a] = tagged
            self._div[a] = registry.resolve("diff", tagged.factor(a))[a]

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def axes(self) -> tuple[str, ...]:
        """The pressure space's coordinate names."""
        return self._axes

    @property
    def iterations(self) -> int:
        """The fixed CG iteration count (static)."""
        return self._iterations

    # ================================================================
    #  The operator, its right-hand side, and the velocity update
    # ================================================================
    def _grad_leg(self, p: ScalarField, axis: str) -> ScalarField:
        """Return the gradient leg ``G_axis(p)`` (vertical ``1/dsqr``)."""
        g = self._grad[axis](p)
        if axis == self._vertical:
            g = g / self._dsqr
        return g

    def apply(self, p: ScalarField) -> ScalarField:
        r"""
        Apply the cut-cell elliptic operator ``L p`` (SPD, module §2).

        Description
        -----------
        The flux form ``sum_axis (alpha_axis (x) G_axis(p)).diff(axis)``
        with the open-area-weighted fluxes retagged Dirichlet before
        the flux difference (zero normal flux through the immersed and
        physical walls). ``L`` is exactly symmetric and negative
        semidefinite under the measure-weighted product; its nullspace
        is the wet-region constants (dry rows are identically zero).

        Parameters
        ----------
        p : ScalarField
            The pressure iterate on the solver's space.

        Returns
        -------
        ScalarField
            ``L p`` on the same space.
        """
        out = None
        for a in self._axes:
            flux = self._alpha[a] * self._grad_leg(p, a)
            term = self._div[a](flux.retag(self._tagged[a]))
            out = term if out is None else out + term
        return out

    def divergence(
        self, vel: Mapping[str, ScalarField],
    ) -> ScalarField:
        r"""
        Compute the masked divergence (``(1/V_c) sum_f +/- alpha_f A_f u*_f``).

        Description
        -----------
        The full-volume-scaled masked divergence of the provisional
        velocity — the right-hand side of the projection (module §2):
        after :meth:`velocity_correction` the *same* divergence of the
        corrected velocity equals the CG residual. Each velocity's
        wall Dirichlet tag is stripped onto the BC-free face for the
        open-area multiply and the flux is retagged Dirichlet for the
        flux difference (the adopt-then-strip seam; identities on
        periodic axes).

        Parameters
        ----------
        vel : Mapping[str, ScalarField]
            The provisional velocity components keyed by axis name.

        Returns
        -------
        ScalarField
            The masked divergence on the pressure space.
        """
        if set(vel) != set(self._axes):
            raise ValueError(
                f"divergence needs one component per axis "
                f"{self._axes}, got {tuple(sorted(vel))}")
        out = None
        for a in self._axes:
            u = vel[a].retag(self._face[a])
            flux = self._alpha[a] * u
            term = self._div[a](flux.retag(self._tagged[a]))
            out = term if out is None else out + term
        return out

    def velocity_correction(
        self, p: ScalarField,
    ) -> dict[str, ScalarField]:
        r"""
        Return the boolean-masked gradient corrections (``m G(p)``).

        Description
        -----------
        Derived from the operator's own gradient legs so the projection
        removes exactly the divergence :meth:`divergence` measures
        (``alpha m = alpha``): per axis ``u_axis -= m_axis G_axis(p)``
        with the boolean face mask ``m_axis = (alpha_axis > 0)`` — the
        correction never enters a closed face. Corrections come back
        BC-free; the caller adopts each velocity's own tag.

        Parameters
        ----------
        p : ScalarField
            The solved pressure.

        Returns
        -------
        dict[str, ScalarField]
            Per-axis corrections to subtract.
        """
        return {
            a: self._fmask[a] * self._grad_leg(p, a)
            for a in self._axes}

    # ================================================================
    #  Smoothing surfaces (multigrid, B1)
    # ================================================================
    def _axis_storage(self) -> dict[str, int]:
        """Map each coordinate name to its storage-frame array axis."""
        return {factor.names[0]: axis
                for factor, axis in factor_axes(self._space)}

    def diagonal(self) -> ScalarField:
        r"""
        Return the exact diagonal of ``apply`` on the pressure space.

        Description
        -----------
        The masked cut-cell operator has no cross terms, so its
        diagonal is exactly ``-sum_a (alpha_{f-} + alpha_{f+})/h_a^2``
        (the vertical leg scaled ``1/dsqr``), derived from the open-area
        face fractions ``_alpha`` (:func:`_adjacent_face_sum`, wall
        drops on bounded axes). Dry cells fall out as an exact zero
        diagonal (a face touching a dry cell has ``alpha = 0`` under the
        min rule) — **not** special-cased here; the smoothers guard the
        zero (module docstring, plan §B1).

        Returns
        -------
        ScalarField
            The diagonal field on the pressure space.
        """
        storage = self._axis_storage()
        diagonal: jax.Array | None = None
        for a in self._axes:
            h = uniform_spacing(self._space.factor(a))
            periodic = bool(getattr(
                self._space.factor(a).mesh, "periodic", False))
            weight = (1.0 / self._dsqr) if a == self._vertical else 1.0
            leg = -weight * _adjacent_face_sum(
                self._alpha[a].data, storage[a],
                periodic=periodic) / (h * h)
            diagonal = leg if diagonal is None else diagonal + leg
        template = self._grid.create_field(self._space)
        return template.with_data(
            jnp.broadcast_to(diagonal, template.data.shape))

    def vertical_bands(self) -> VerticalBands:
        r"""
        Return the per-column tridiagonal ``T`` for line relaxation.

        Description
        -----------
        The symmetric tridiagonal of the vertical-line smoother: ``diag``
        is the full operator diagonal (:meth:`diagonal`), and — with no
        cross terms — the off-diagonals are the **exact** vertical flux
        leg ``+alpha_{z-face}/(dsqr dz^2)`` with the Neumann ends zeroed
        (``lower[c]`` couples cell ``c`` to ``c-1``, ``upper[c]`` to
        ``c+1``; symmetric by ``lower[c] == upper[c-1]``). Dry columns
        carry a zero diagonal and zero bands; the smoother substitutes
        ``diag -> 1`` there.

        Returns
        -------
        VerticalBands
            The ``(lower, diag, upper, axis)`` bands on the pressure
            space.

        Raises
        ------
        NotImplementedError
            If the vertical axis is periodic (line smoothing needs a
            bounded, non-cyclic column).
        """
        vertical = self._vertical
        mesh = self._space.factor(vertical).mesh
        if getattr(mesh, "periodic", False):
            raise NotImplementedError(
                "vertical-line bands need a bounded (Neumann) vertical "
                f"column; the {vertical!r} axis is periodic")
        storage = self._axis_storage()
        z_axis = storage[vertical]
        dz = uniform_spacing(self._space.factor(vertical))
        alpha_z = self._alpha[vertical].data
        scale = 1.0 / (self._dsqr * dz * dz)
        edge = list(alpha_z.shape)
        edge[z_axis] = 1
        zeros = jnp.zeros(edge, dtype=alpha_z.dtype)
        lower_data = jnp.concatenate(
            [zeros, alpha_z], axis=z_axis) * scale
        upper_data = jnp.concatenate(
            [alpha_z, zeros], axis=z_axis) * scale
        diag = self.diagonal()
        shape = diag.data.shape
        lower = diag.with_data(jnp.broadcast_to(lower_data, shape))
        upper = diag.with_data(jnp.broadcast_to(upper_data, shape))
        return VerticalBands(lower, diag, upper, z_axis)

    # ================================================================
    #  The preconditioned, wet-mean-projected solve
    # ================================================================
    def _projection(self, f: ScalarField) -> ScalarField:
        r"""
        Project out the wet-region constant (V-orthogonal to ``e``).

        Description
        -----------
        ``f - (int_wet V f / int_wet V) e`` with ``e = (theta > 0)`` the
        wet-cell indicator — the exact ``L``-nullspace constant. This is
        V-orthogonal (idempotent, a no-op on a compatible right-hand
        side) and touches only the wet cells, so every CG iterate stays
        wet-supported: the residual reaches machine zero and *holds*
        there (no fixed-iteration blow-up). The plan's theta-weighted
        ``int theta f / int theta`` subtracts a *global* constant with
        the wrong coefficient and re-introduces incompatibility on
        genuine partial cells — this is the IP-D6 correction.
        """
        mean = jnp.sum((self._wet * f).integrate().data)
        return f - self._wet * (mean / self._wet_measure)

    def _preconditioner(self) -> Callable[[ScalarField], ScalarField]:
        r"""
        Build the wet-masked flat spectral inverse (IP-D6 preconditioner).

        Description
        -----------
        The unmasked spectral inverse ``M`` (the
        :class:`SpectralPressureSolver` machinery) **masked onto the wet
        cells**: ``z = e (x) M(r)``. Because the CG residual is exactly
        wet-supported (``e (x) r = r``), the effective preconditioner is
        the symmetric ``D M D`` (``D = diag(e)``), SPD on the wet
        subspace — where the raw unmasked ``M`` is only semidefinite and
        **destabilizes** the iteration (its global ``k = 0`` gauge mixes
        with the wet-constant nullspace; the hand-rolled prototype ran
        unpreconditioned for exactly this reason). Masking is the IP-D6
        correction that keeps the spectral preconditioner: on an all-wet
        domain ``D = I`` and it is the exact inverse (PCG converges in
        ~1 iteration).
        """
        inverse = build_flat_spectral_solve(
            self._grid, self._space, vertical=self._vertical,
            dsqr=self._dsqr, single_precision=self._single_precision)

        def masked(r: ScalarField) -> ScalarField:
            z = inverse(r)
            return z.with_data(z.data * self._cell_mask)

        return masked

    def krylov(self) -> ConjugateGradient:
        """Build the configured PCG solver (public for diagnostics)."""
        return ConjugateGradient(
            self.apply,
            preconditioner=self._preconditioner(),
            iterations=self._iterations,
            projection=self._projection)

    def solve(
        self, rhs: ScalarField, x0: ScalarField | None = None,
    ) -> ScalarField:
        """
        Solve ``L p = rhs`` by wet-mean-projected preconditioned CG.

        Parameters
        ----------
        rhs : ScalarField
            The masked divergence (:meth:`divergence`).
        x0 : ScalarField | None, optional
            The initial guess; None starts from zeros (default: None).

        Returns
        -------
        ScalarField
            The wet-mean-free pressure on the same space.
        """
        return self.krylov()(rhs, x0)

    def solve_info(
        self, rhs: ScalarField, x0: ScalarField | None = None,
    ) -> tuple[ScalarField, dict[str, object]]:
        """Solve and return ``(pressure, info)`` (the CG residual)."""
        return self.krylov().solve(rhs, x0)

    def project(
        self, vel: Mapping[str, ScalarField],
    ) -> tuple[ScalarField, dict[str, ScalarField]]:
        r"""
        Run the whole masked projection (divergence, solve, correction).

        Description
        -----------
        The velocity corrections are derived from the raw (unmasked)
        solved pressure so the divergence-removal identity holds
        exactly; the returned diagnostic pressure is then masked to
        zero on the dry cells (the physical "no pressure under
        topography" convention — the correction is already applied, so
        the mask is purely cosmetic).

        Parameters
        ----------
        vel : Mapping[str, ScalarField]
            The provisional velocity components (:meth:`divergence`).

        Returns
        -------
        tuple[ScalarField, dict[str, ScalarField]]
            The masked pressure and the per-axis velocity corrections.
        """
        p = self.solve(self.divergence(vel))
        corr = self.velocity_correction(p)
        return p.with_data(p.data * self._cell_mask), corr
