r"""The terrain-following / boundary-fitted pressure Poisson solve.

Description
-----------
Coordinate-systems plan, stage C3 (decision CS-D2). On a grid whose
``CoordinateMapping`` declares a single-base analytic map — a mapped
column ``m = M(b, params)``, e.g. the terrain-following
``zp = z * H(x, y)`` (target b) or the boundary-fitted
``yp = y * Y_N(x)`` (target c) — the coordinates couple and the
separable spectral pressure inverse no longer applies. This module
builds the **flux-form mapped elliptic operator** in computational
coordinates and wraps the CS-D2
:class:`~fridom.spatial.operators.krylov.ConjugateGradient` core
around it, preconditioned by the flat separable spectral inverse.

The coefficient tensor
----------------------
Write ``J = dm/db`` (the column Jacobian) and ``Z_i = dm/dx_i`` (the
slope metric of each coupled coordinate ``x_i``), and let ``W`` be the
diagonal physical-axis weight of the projection (``1/dsqr`` on the
vertical, ``1`` elsewhere). The physical projection operator
``div_phys(W grad_phys p)`` transforms to computational coordinates
as the **J-weighted** flux form

.. math::

    A\,p \;=\; J\,\nabla_{\!phys}\cdot(W\,\nabla_{\!phys}\, p)
    \;=\; \partial_i\bigl(K^{ij}\,\partial_j p\bigr),
    \qquad K \;=\; J\,B^{\top} W B,

with ``B`` the gradient transformation
(:math:`\partial_i|_{phys} = \partial_i - (Z_i/J)\,\partial_b`,
:math:`\partial_m = (1/J)\,\partial_b`), so

.. math::

    K^{ii} = w_i\,J, \quad
    K^{ib} = K^{bi} = -\,w_i\,Z_i, \quad
    K^{bb} = \bigl(\textstyle\sum_i w_i Z_i^2 + w_m\bigr)/J .

The J-weighting (no outer ``1/J``) is deliberate: it makes ``A``
self-adjoint under the grid's own computational measure — exactly the
inner product ``ConjugateGradient`` evaluates through ``integrate`` —
and the matching right-hand side is the J-weighted physical
divergence in flux form,
``rhs = sum_i D_i(J u_i) + D_b(u_m - sum_i Z_i I(u_i))``
(:meth:`MappedPressureSolver.divergence`).

Staggering and the symmetric cross form
---------------------------------------
Diagonal fluxes live on the C-grid faces: ``K^ii G_i p`` on the
``i``-faces, ``K^bb G_b p`` on the column faces. Each cross term is
assembled through the **cell-corner** staggering
(``face_i (x) face_b``): the transverse gradient is interpolated onto
the corners, contracted there with the corner-derived slope metric,
and interpolated onto the flux face. Because the two interpolation
hops of one cross pair are exact transposes of each other
(``C -> R`` / ``R -> C`` on the periodic coupled axis, ``C -> Inner``
/ Dirichlet-tagged ``Inner -> C`` on the bounded column), the two
cross blocks are exact negative-transposes and ``A`` is **exactly
symmetric** — the SPD license CG requires (CS-D2).

The solver is **family-aware** (stage F5): on the FV C-grid the cell
pressure is a ``CellAvg`` average, so the cross-term ``cell -> face``
up-hop resolves the G4 ``("interpolate", CellAvg)`` reconstruction
(``CellAvg -> Right | Inner``) and the ``face -> cell`` down-hop the
``"average"`` reconstruction (``Right | Inner -> CellAvg``, its
Dirichlet ``Inner -> CellAvg`` variant zero-padding the wall face,
:meth:`_to_cell`). Those two-point means are the same numbers as the
nodal ``Center <-> Right`` / ``Center <-> Inner`` interpolations and
exact transposes under the uniform computational cell measure, so the
transpose-pairing — and the exact symmetry and CG license — hold on
both families. On a **uniform** base column no measure weighting
enters the cross hops: the mapped column rides uniform computational
meshes and all geometry lives in the ``K`` coefficients (the
chart-uniform-stencil rule).

On a **stretched** base column (a ``MappedIntervalMesh``, whose base
factor carries a ``coordinate_map``) the physical column measure
diverges from the computational one, and CG's inner product is the
**physical** measure-weighted :math:`L^2` (``krylov.py`` ``_dot``), so
the plain ``0.5/0.5`` base-axis down-hop is no longer the up-hop's
adjoint and ``A`` loses exact symmetry. The base-axis cross down-hop
is then the **measure-weighted adjoint** of the up-hop,
``down_b = diag(1/m_cell) up_b^T diag(m_inner)`` with ``m_cell`` /
``m_inner`` the physical cell / interior-face widths
(:meth:`_down_b_hop`, N2) — restoring exact symmetry
(``design/research/stretched_terrain_combined.md`` §3). The
coupled-axis hops stay uniform-periodic and unweighted; the diagonal
flux legs never touch this hop and stay byte-identical. The separable
spectral preconditioner has no transform to build on a stretched base
mesh, so it is rejected at construction (N1) in favour of the
``preconditioner="none"`` plain-CG correctness stopgap — or the
``preconditioner="multigrid"`` V-cycle, which **does** serve the
stretched column (N3): its vertical-line smoother and point diagonal
build the column tridiagonal from the same ``grid.measure`` widths
(:meth:`vertical_bands`, :meth:`_diagonal_data`), so the band is
self-adjoint under the physical measure (``diag(m_cell) T`` symmetric)
and the ``pre == post`` cycle stays SPD in CG's inner product; the
semicoarsening transfers never touch the (uncoarsened) mapped column,
so their measure-adjoint pair (``R = M_H^{-1} P^\top M_h``) already
carries the stretched vertical unchanged.

The alternative —
evaluating the cross coefficient on the flux faces, which is what the
``physical_diff``-composed ``Div(grad_phys)`` would do — is symmetric
only to O(h^2); per the stage-C3 instruction the exact-SPD corner
form wins, and the velocity update is derived **from the operator
fluxes** (:meth:`MappedPressureSolver.velocity_correction`) so the
projection removes exactly the divergence the operator measures.

Wall closure (the C1 bounded-column decision)
---------------------------------------------
The projection's boundary condition at the mapped boundary is zero
normal **flux**. On the bounded column the flux DOFs live on the
``Inner`` faces (the staggering nonhydro2's wall-normal velocity
already uses): the boundary-face fluxes are structurally absent and
the flux difference back to the centers resolves through the
**Dirichlet-tagged** ``("diff", Inner)`` row, whose ghost fill is the
zero wall flux. The corner cross fluxes are likewise retagged
Dirichlet before the ``Inner -> Center`` interpolation hop, so the
boundary-corner cross flux is dropped — the closure that keeps the
interpolation pair an exact transpose. This resolves the C1
bounded-column caveat by **conforming-BC operands** (flux fields
tagged with the physical zero-flux claim), not by one-sided
interpolation overrides, which would break the transpose pairing and
with it the SPD property.

Sign convention
---------------
``A`` is negative (semi-)definite, matching the flat
``SpectralPressureSolver`` convention (``lap p = div``). CG on the
pair ``(A, M_inv)`` with **both** factors negative definite produces
identically the iterates of standard PCG on ``(-A, -M_inv, -rhs)``
with the solution sign folded back, so no sign flip is needed.

Preconditioner
--------------
The flat separable spectral inverse (CS-D2): the diagonal
coefficients ``K^ii`` and ``K^bb`` are folded to their means (0-d
traced scalars, recomputed per solve — dynamic-parameter ready) and
``Div @ Diag(means) @ Grad`` is inverted by ``SpectralSolve`` on the
Neumann-tagged sibling of the pressure space (the
``SpectralPressureSolver`` seam). Folding the *coefficients* (rather
than the raw ``H``) keeps the slope contribution ``w_i Z_i^2 / J`` in
the column weight; for a constant-``H`` mapping the preconditioner is
the exact inverse and PCG converges in one iteration — the
mapped-flat identity gate. The nullspace of the all-Neumann/periodic
problem is the constants; the solve always projects the mean
(``project_mean=True``).

With ``single_precision=True`` the preconditioner's transform pair
and spectral divide run in ``float32`` / ``complex64`` (mixed-
precision PCG): the preconditioner only shapes the search
directions, so its round-off perturbs the convergence path — the
residual arithmetic, the iterates, and the operator stay
``float64``. The preconditioner application is the dominant cost of
a CG iteration, so this halves most of its bandwidth.

Metric derivation: once per solve, never across steps
-----------------------------------------------------
Every metric coefficient is derived through ``grid.metric`` on the
operand's own space, and the ``params=`` seam threads dynamic
mapping parameters (stage C4) through every derivation (rules
2.3/3.8). Nothing is *cached*: no metric field is ever stored on the
solver, on the grid, or across a step — under a moving geometry
(``MovingGeometry``, where ``H`` / ``Y_N`` change every step) a
retained metric would be silently stale, which is exactly what rules
2.3/3.8 forbid.

The metrics do not, however, depend on the CG iterate: they are
bit-identical across the iterations of one solve, and re-deriving
them inside the operator application made every CG iteration repeat
the same registry ``diff`` / interpolation chains (with their halo
exchanges). They are therefore **derived once per solve** into a
memo dict that is created inside the public entry point, threaded
through the operator applications of that one CG run, and dropped
when it returns — it never touches ``self``, never outlives a trace,
and holds whatever (possibly traced) parameter fields the caller
passed, so a second solve at a moved geometry re-derives everything.
"Derived once per solve" is a trace-structure optimization;
"cached across steps" remains forbidden.
"""
# Coordinate-systems plan, stage C3: mapped PCG pressure solve
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.nonhydro2.modules.pressure import (
    _dirichlet_mid,
    _neumann_sibling,
    is_fv,
    rediscretize_fv_coarse,
)
from fridom.spatial.fields.storage import factor_axes
from fridom.spatial.operators.banded import validate_tridiagonal_method
from fridom.spatial.operators.base import resolve_codomain
from fridom.spatial.operators.composed import (
    Diag,
    Divergence,
    Gradient,
)
from fridom.spatial.operators.krylov import ConjugateGradient
from fridom.spatial.operators.multigrid import (
    MultigridLevel,
    MultigridVCycle,
    VerticalBands,
    VerticalLineJacobi,
)
from fridom.spatial.operators.multigrid_hierarchy import coarsen_levels
from fridom.spatial.operators.spectral_solve import SpectralSolve
from fridom.spatial.operators.staggering import (
    mapped_factor,
    uniform_spacing,
)
from fridom.spatial.spaces.average import AverageSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    import jax

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.operators.base import Operator
    from fridom.spatial.spaces.tensor_product import SpaceLike

    #: the per-solve metric memo: interned (space, metric name) keys
    #: to derived metric fields; created inside one solve, dropped
    #: when it returns, never stored on the solver (module docstring)
    MetricCache = dict[tuple[SpaceLike, str], ScalarField]

#: the accepted ``preconditioner=`` choices (B3/B4; ``"none"`` is the
#: N1 plain-CG correctness stopgap for the stretched base column)
_PRECONDITIONERS = ("spectral", "multigrid", "none")
#: the vertical-line smoother damping of the multigrid V-cycle (B0
#: spike optimum; omega = 1 diverges)
_LINE_OMEGA = 0.8


def _mean_free(field: ScalarField) -> ScalarField:
    """Remove the measure-weighted mean (the constants nullspace)."""
    return field - field.mean()


# ================================================================
#  Diagonal / band assembly helpers (measure-free; storage frame)
# ================================================================
def _adjacent_face_sum(
    data: jax.Array, axis: int, *, periodic: bool,
) -> jax.Array:
    r"""
    Sum the two face coefficients adjacent to each cell along ``axis``.

    Description
    -----------
    The ``K_{f-} + K_{f+}`` cell field of a diagonal flux leg: a
    periodic axis wraps (``roll`` — the ``GridTransfer`` neighbor-access
    pattern, GSPMD-lowered), a bounded axis drops the wall faces by
    zero-padding the ``Inner`` face array to the cell count (the
    Neumann boundary rows). ``face c`` lies at ``c + 1/2`` (the right
    face of cell ``c``, the left face of cell ``c + 1``).

    Parameters
    ----------
    data : jax.Array
        The face-coefficient array (storage frame).
    axis : int
        The storage-frame axis of the coarsened/differenced coordinate.
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


def _corner_cross_bracket(
    slope: jax.Array, axis_a: int, axis_b: int,
) -> jax.Array:
    r"""
    Return the mixed corner difference of a slope (the cross rows).

    Description
    -----------
    The corner cross fluxes couple a cell to its column and transverse
    neighbours; their **diagonal** contribution is the mixed second
    difference of the corner slope ``Z_a`` (on ``Right_a (x) Inner_b``)
    over the two ``a``-faces and two ``b``-faces of the cell,

    .. math::
        Z_a[l, k{-}1] + Z_a[l{-}1, k] - Z_a[l, k] - Z_a[l{-}1, k{-}1],

    with the wall ``b``-faces dropped (the Dirichlet corner closure).
    This is non-zero exactly when ``Z_a`` varies along the column
    (``Z_a = z\,H'`` for a terrain map), so the centered vertical
    average no longer annihilates it (module docstring).

    Parameters
    ----------
    slope : jax.Array
        The corner slope metric ``Z_a`` (storage frame).
    axis_a : int
        The storage-frame axis of the coupled (transverse) coordinate.
    axis_b : int
        The storage-frame axis of the mapped column (vertical) axis.

    Returns
    -------
    jax.Array
        The mixed corner difference on the cell grid.
    """
    edge = list(slope.shape)
    edge[axis_b] = 1
    zeros = jnp.zeros(edge, dtype=slope.dtype)
    onto_right = jnp.concatenate([slope, zeros], axis=axis_b)
    onto_left = jnp.concatenate([zeros, slope], axis=axis_b)
    column = onto_left - onto_right
    return column - jnp.roll(column, 1, axis=axis_a)


class MappedPressureSolver:

    r"""
    PCG solve of the mapped pressure Poisson problem (stage C3).

    Description
    -----------
    A trace-time object (constructed per projection call inside the
    stage, carrying no mutable state): it discovers the grid's mapped
    column from the attached ``CoordinateMapping``, resolves the
    registered ``diff``/``interpolate`` rows of the flux-form
    operator once, and derives every metric coefficient through
    ``grid.metric`` at application (module docstring). The public
    surface mirrors the projection's needs: :meth:`divergence` (the
    J-weighted physical divergence the operator measures),
    :meth:`solve` (preconditioned CG), and
    :meth:`velocity_correction` (the flux-consistent gradient
    subtraction).

    Parameters
    ----------
    grid : object
        The grid carrying the coordinate mapping and the dispatch
        registry.
    space : SpaceLike
        The (cell-centered) pressure/divergence space.
    iterations : int
        The fixed CG iteration count (CS-D2; static). The maximum
        budget when a ``tolerance`` is set.
    tolerance : float | None, optional
        The PCG convergence break forwarded to
        :class:`ConjugateGradient` (the measure-weighted true relative
        residual; masked scan, exact gradient — see its docstring).
        The default ``1e-8`` makes ``iterations`` the maximum budget;
        ``None`` is the opt-out that runs the fixed ``iterations``
        count (default: 1e-8).
    weights : Mapping[str, jax.Array | float] | None, optional
        Per-coordinate physical-axis weights; the mapped column's
        base name keys the weight of its *physical* direction
        (``{"z": 1/dsqr}`` on a terrain-following grid whose base
        vertical coordinate is named ``z``). Missing axes weigh 1
        (default: None).
    params : Mapping[str, ScalarField] | None, optional
        Dynamic mapping-parameter fields threaded through every
        ``grid.metric`` derivation (the stage-C4 seam)
        (default: None).
    single_precision : bool, optional
        Run the spectral *preconditioner* in single precision
        (``float32`` / ``complex64``) while the CG iterates, the
        operator applications and the inner products stay
        ``float64`` — mixed-precision PCG: the preconditioner only
        steers the search directions, so its reduced round-off
        perturbs the convergence path without touching the residual
        arithmetic. Forwarded to the preconditioner's
        :class:`SpectralSolve` (see its ``single_precision`` doc).
        Off by default (default: False).
    preconditioner : str, optional
        The PCG preconditioner (B3/B4): ``"spectral"`` (the flat
        separable spectral inverse at folded coefficients),
        ``"multigrid"`` (the geometric-multigrid V-cycle assembled by
        :meth:`_build_vcycle`, semicoarsening the horizontal axes with
        vertical line smoothing — its vertical bands and diagonal read
        the ``grid.measure`` widths, so it is the **stretched**-column
        preconditioner too, N3), or ``"none"`` (unpreconditioned CG —
        the N1 plain-CG correctness stopgap; slow, not a production
        route). A stretched base column with ``"spectral"`` raises
        ``NotImplementedError`` at construction (N1: the separable
        spectral inverse needs a per-axis transform the mapped mesh
        does not supply). Any other value raises ``ValueError``
        (default: ``"spectral"``).
    multigrid_levels : int | None, optional
        The multigrid depth when ``preconditioner="multigrid"``. ``None``
        (the default) coarsens to the four-cell horizontal floor
        (floor-limited depth, h-independent iteration counts at every
        size); an ``int`` is a **maximum** cap as before. Either way the
        builder floors every horizontal axis at four cells and stops at
        indivisibility, so the realized count is smaller on a small grid
        (a grid too small for any coarsening degrades to a one-level,
        smoothing-only cycle). Ignored for the spectral preconditioner
        (default: None).
    multigrid_tridiagonal_method : str, optional
        The vertical-line tridiagonal kernel of the multigrid smoother,
        forwarded to
        :func:`~fridom.spatial.operators.banded.tridiagonal_solve_along_axis`:
        ``"auto"`` (cuSPARSE on a GPU, parallel cyclic reduction
        elsewhere), ``"cusparse"``, ``"pcr"`` or ``"scan"``. The name is
        validated at construction; the backend requirement of
        ``"cusparse"`` is checked at solve time. Ignored for the
        spectral preconditioner (default: ``"auto"``).
    """

    def __init__(
        self,
        grid: object,
        space: SpaceLike,
        *,
        iterations: int,
        tolerance: float | None = 1e-8,
        weights: Mapping[str, jax.Array | float] | None = None,
        params: Mapping[str, ScalarField] | None = None,
        single_precision: bool = False,
        preconditioner: str = "spectral",
        multigrid_levels: int | None = None,
        multigrid_tridiagonal_method: str = "auto",
    ) -> None:
        """Discover the mapped column and resolve the static rows."""
        if preconditioner not in _PRECONDITIONERS:
            raise ValueError(
                f"preconditioner must be one of {_PRECONDITIONERS}, got "
                f"{preconditioner!r}")
        self._preconditioner_kind = preconditioner
        self._multigrid_levels = multigrid_levels
        self._multigrid_tridiagonal_method = validate_tridiagonal_method(
            multigrid_tridiagonal_method)
        mapping = getattr(grid, "mapping", None)
        if mapping is None:
            raise ValueError(
                "the grid carries no coordinate mapping; the mapped "
                "pressure solve needs a Grid(..., mapping=...) with "
                "a single-base analytic map")
        table = mapping.column_corrections
        if not table:
            raise ValueError(
                "the grid's coordinate mapping declares no "
                "single-base analytic map (no mapped column); the "
                "flat SpectralPressureSolver applies instead")
        columns = set(table.values())
        if len(columns) != 1:
            raise NotImplementedError(
                f"the mapped pressure solve supports exactly one "
                f"mapped column, got {sorted(columns)} "
                "(coordinate-systems plan, stage C3)")
        self._mapped, self._base = next(iter(columns))
        self._grid = grid
        self._space: SpaceLike = space.bare
        # the discretization family of the (cell) pressure space: an
        # average-family (CellAvg) space is the FV C-grid (stage F5),
        # a nodal Center space the point-value C-grid. The two families
        # differ only in the cross-term corner *face -> cell* hop —
        # the FV cell is CellAvg, so that hop is the ``"average"``
        # reconstruction (face -> CellAvg) rather than the nodal
        # ``"interpolate"`` (face -> Center); the *cell -> face* up-hop
        # keys on the cell factor and resolves per family with no
        # branch (the G4 ``("interpolate", CellAvg)`` reconstruction row
        # lands ``CellAvg -> Right | Inner`` exactly as the nodal
        # ``("interpolate", Center)`` lands ``Center -> Right | Inner``)
        self._fv: bool = any(
            isinstance(factor, AverageSpace)
            for factor in self._space.factors)
        self._iterations = iterations
        self._tolerance = tolerance
        self._params = params
        self._single_precision = bool(single_precision)
        axes = self._space.active_axis_names
        if self._base not in axes:
            raise ValueError(
                f"the pressure space resolves {axes}, which lacks "
                f"the mapped column's base coordinate "
                f"{self._base!r}")
        self._axes: tuple[str, ...] = axes
        # a stretched base column (its base factor rides a
        # ``MappedIntervalMesh``, i.e. a non-None ``coordinate_map``)
        # is where the physical and computational column measures
        # diverge: the corner base-axis down-hop needs the measure-
        # weighted adjoint (N2, :meth:`_down_b_hop`) to stay SPD, and
        # the separable spectral preconditioner has no transform to
        # build on (N1, below)
        self._stretched_base: bool = mapped_factor(
            self._space.factor(self._base))
        if self._stretched_base and preconditioner == "spectral":
            raise NotImplementedError(
                f"the spectral preconditioner needs a per-axis "
                f"spectral transform on the mapped column's base "
                f"coordinate {self._base!r}, which its stretched "
                "MappedIntervalMesh does not supply (a coordinate-"
                "mapped mesh carries no spectral basis); run "
                "correctness solves on a stretched column with "
                "preconditioner='none' (the plain, unpreconditioned-CG "
                "stopgap) — see "
                "design/research/stretched_terrain_combined.md (N1)")
        self._coupled: tuple[str, ...] = tuple(
            a for a in axes if a != self._base and a in table)
        weights = dict(weights or {})
        unknown = tuple(k for k in weights if k not in axes)
        if unknown:
            raise ValueError(
                f"unknown weight axes {unknown}; the pressure space "
                f"resolves {axes}")
        self._weights: dict[str, jax.Array | float] = weights
        for a in self._coupled:
            if not getattr(self._space.factor(a).mesh, "periodic",
                           False):
                raise NotImplementedError(
                    f"coupled coordinate {a!r} lives on a bounded "
                    "mesh; the symmetric corner form is derived for "
                    "periodic coupled axes only (stage C3)")
        self._resolve_flux_rows(grid.dispatch)
        self._resolve_corner_rows(grid.dispatch)

    def _resolve_flux_rows(self, registry: object) -> None:
        """
        Resolve the per-axis gradient/divergence legs (static).

        Description
        -----------
        ``G_a`` keys on the center factor, ``D_a`` on the
        Dirichlet-tagged face factor — the zero-normal-flux wall
        closure; the tag is the identity on periodic axes.

        Parameters
        ----------
        registry : object
            The grid's dispatch registry.
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
            self._div[a] = registry.resolve(
                "diff", tagged.factor(a))[a]

    def _resolve_corner_rows(self, registry: object) -> None:
        """
        Resolve the cross-term corner interpolation hops (static).

        Description
        -----------
        Per coupled axis: the corner staggering and the two
        transpose-paired interpolation hops of its cross terms
        (module docstring).

        Parameters
        ----------
        registry : object
            The grid's dispatch registry.
        """
        self._corner: dict[str, SpaceLike] = {}
        self._corner_tagged: dict[str, SpaceLike] = {}
        self._up_i: dict[str, Operator] = {}
        self._down_i: dict[str, Operator] = {}
        self._up_b: dict[str, Operator] = {}
        self._down_b: dict[str, Operator] = {}
        face_b = self._face[self._base]
        for a in self._coupled:
            up_i = registry.resolve(
                "interpolate", face_b.factor(a))[a]
            corner = resolve_codomain(up_i, face_b)
            corner_tagged = _dirichlet_mid(corner, self._base)
            self._up_i[a] = up_i
            self._down_i[a] = self._to_cell(
                registry, corner.factor(a), a)
            self._up_b[a] = registry.resolve(
                "interpolate", self._face[a].factor(self._base),
            )[self._base]
            self._down_b[a] = self._to_cell(
                registry, corner_tagged.factor(self._base), self._base)
            self._corner[a] = corner
            self._corner_tagged[a] = corner_tagged

    def _to_cell(
        self, registry: object, factor: SpaceLike, axis: str,
    ) -> Operator:
        r"""
        Resolve the cross-term face -> cell staggering hop (family).

        Description
        -----------
        The down leg of a corner cross chain lands the flux back on
        the pressure cell. On the nodal C-grid the cell is
        ``Center`` and the hop is the ``"interpolate"`` row
        (``Right | Inner -> Center``); on the FV C-grid (stage F5)
        the cell is ``CellAvg`` and the hop is the ``"average"``
        reconstruction (``Right | Inner -> CellAvg``, the
        ``f.to`` direction). The two are exact transposes of the
        matching ``cell -> face`` up-hop under the (uniform,
        computational) cell measure — the FV ``CellAvg <-> Right``
        two-point-mean pair and the walled ``CellAvg <-> Inner``
        pair (the ``Inner(DIRICHLET) -> CellAvg`` variant zero-pads
        the wall face) — so the corner cross blocks stay exact
        negative-transposes and ``A`` stays exactly symmetric on
        both families (module docstring). This resolves the **plain**
        (unweighted, ``0.5/0.5``) reconstruction on any mesh; on a
        uniform base column it is already the up-hop's adjoint and no
        measure weighting enters (the geometry lives entirely in the
        ``K`` coefficients). On a **stretched** base column the
        base-axis down-hop is wrapped in the measure-weighted adjoint
        of the up-hop downstream (:meth:`_down_b_hop`, N2); the plain
        row resolved here is its unweighted core.

        Parameters
        ----------
        registry : object
            The grid's dispatch registry.
        factor : SpaceLike
            The (bare, nodal) corner face factor to reconstruct.
        axis : str
            The coordinate the hop reduces along.

        Returns
        -------
        Operator
            The bound face -> cell reconstruction/interpolation.
        """
        kind = "average" if self._fv else "interpolate"
        return registry.resolve(kind, factor)[axis]

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def base(self) -> str:
        """The mapped column's base (computational) coordinate."""
        return self._base

    @property
    def mapped(self) -> str:
        """The mapped physical coordinate name."""
        return self._mapped

    @property
    def coupled(self) -> tuple[str, ...]:
        """The coordinates coupled through mapping parameters."""
        return self._coupled

    @property
    def axes(self) -> tuple[str, ...]:
        """The pressure space's coordinate names."""
        return self._axes

    @property
    def iterations(self) -> int:
        """The fixed CG iteration count (static)."""
        return self._iterations

    @property
    def tolerance(self) -> float | None:
        """The optional PCG convergence break (None = fixed count)."""
        return self._tolerance

    # ================================================================
    #  Metric coefficients (derived once per solve, never cached)
    # ================================================================
    def _metric(self, space: SpaceLike, name: str,
                cache: MetricCache | None = None) -> ScalarField:
        """
        Derive one mapping metric on ``space`` (params-ready).

        Description
        -----------
        With a ``cache`` — the per-solve memo the public entry
        points create and thread through one CG run (module
        docstring) — the derivation of a ``(space, name)`` pair is
        performed once and reused; the spaces are interned, so the
        key is an identity key. Without one the metric is derived
        on the spot. The memo is **per solve**, never per solver
        and never across steps: it only removes the repetition of a
        derivation whose inputs (the grid's mapping, the caller's
        ``params=`` fields) are fixed for the duration of the call.

        Parameters
        ----------
        space : SpaceLike
            The querying (staggered) space.
        name : str
            The metric name (e.g. ``"dzp_dz"``).
        cache : MetricCache | None, optional
            The per-solve memo; None derives without memoizing
            (default: None).

        Returns
        -------
        ScalarField
            The metric field on ``space``.
        """
        if cache is None:
            return self._grid.metric(space, name, params=self._params)
        key = (space, name)
        field = cache.get(key)
        if field is None:
            field = self._grid.metric(
                space, name, params=self._params)
            cache[key] = field
        return field

    def _measure(self, space: SpaceLike,
                 cache: MetricCache | None = None) -> ScalarField:
        """
        Derive the base-axis physical measure on ``space`` (memoized).

        Description
        -----------
        The stretched base column's physical widths (``grid.measure``
        of the base factor): the primal cell widths on a cell space,
        the dual center-to-center widths on the interior-face space —
        the two diagonal factors of the measure-weighted down-hop
        (:meth:`_down_b_hop`, N2). Static mesh geometry (no ``params=``
        seam), threaded through the per-solve memo like the chart
        metrics so the reduction's weights are derived once per solve
        rather than once per CG iteration; ``grid.measure`` also holds
        its own concrete memo (module docstring on the metric memo).

        Parameters
        ----------
        space : SpaceLike
            The querying (staggered) space.
        cache : MetricCache | None, optional
            The per-solve memo; None derives without memoizing
            (default: None).

        Returns
        -------
        ScalarField
            The base-axis measure field on ``space``.
        """
        if cache is None:
            return self._grid.measure(space, self._base)
        key = (space, f"measure:{self._base}")
        field = cache.get(key)
        if field is None:
            field = self._grid.measure(space, self._base)
            cache[key] = field
        return field

    def _weight(self, axis: str) -> jax.Array | float:
        """Return the physical-axis weight of ``axis`` (default 1)."""
        return self._weights.get(axis, 1.0)

    def _jacobian(self, field: ScalarField,
                  cache: MetricCache | None = None) -> ScalarField:
        """Derive the column Jacobian ``J = dm/db`` on the field."""
        return self._metric(field.function_space,
                            f"d{self._mapped}_d{self._base}", cache)

    def _slope(self, field: ScalarField, axis: str,
               cache: MetricCache | None = None) -> ScalarField:
        """Derive the slope metric ``Z_i = dm/dx_i`` on the field."""
        return self._metric(field.function_space,
                            f"d{self._mapped}_d{axis}", cache)

    def _column_coefficient(
        self, space: SpaceLike,
        cache: MetricCache | None = None,
    ) -> ScalarField:
        r"""``K^bb = (sum_i w_i Z_i^2 + w_m) / J`` on ``space``."""
        inv_j = self._metric(
            space, f"d{self._base}_d{self._mapped}", cache)
        coeff = self._weight(self._base) * inv_j
        for a in self._coupled:
            slope = self._metric(
                space, f"d{self._mapped}_d{a}", cache)
            coeff = coeff + self._weight(a) * (slope * slope) * inv_j
        return coeff

    # ================================================================
    #  Cross-term corner chains (the transpose-paired hops)
    # ================================================================
    def _cross_to_face(self, axis: str, g_b: ScalarField,
                       cache: MetricCache | None = None,
                       ) -> ScalarField:
        """
        Assemble the ``i``-face cross flux ``I_b(Z_i * I_i(g_b))``.

        Description
        -----------
        Interpolate the column gradient onto the corners along
        ``axis``, contract with the corner slope metric, retag the
        corner flux Dirichlet (the wall closure, module docstring),
        and interpolate onto the ``axis`` faces along the column.

        Parameters
        ----------
        axis : str
            The coupled coordinate.
        g_b : ScalarField
            The column gradient (on the column faces).
        cache : MetricCache | None, optional
            The per-solve metric memo (default: None).

        Returns
        -------
        ScalarField
            The cross flux on the ``axis`` face space.
        """
        corner = self._up_i[axis](g_b)
        flux = corner * self._slope(corner, axis, cache)
        flux = flux.retag(self._corner_tagged[axis])
        return self._down_b_hop(axis, flux, cache)

    def _down_b_hop(self, axis: str, flux: ScalarField,
                    cache: MetricCache | None = None) -> ScalarField:
        r"""
        Land the corner cross flux on the cell (base-axis down-hop).

        Description
        -----------
        The down leg of :meth:`_cross_to_face` along the mapped
        column's base axis. On a **uniform** base column the plain
        ``"average"`` reconstruction (:meth:`_to_cell`) is already the
        exact transpose of the ``cell -> face`` up-hop, so ``A`` is
        symmetric and the hop is applied unchanged (**bitwise** — the
        F5 chart-uniform-stencil path). On a **stretched** base column
        (:attr:`_stretched_base`) the physical column measure diverges
        from the computational one, so the ``0.5/0.5`` down-hop is no
        longer the up-hop's adjoint under CG's measure-weighted inner
        product (``krylov.py`` ``_dot``) and the operator loses exact
        symmetry (record §3). The measure-weighted adjoint of the
        up-hop restores it (N2):

        .. math::
            \mathrm{down}_b \;=\;
            \operatorname{diag}(1/m_{\text{cell}})\;
            \mathrm{up}_b^{\top}\;
            \operatorname{diag}(m_{\text{inner}}),

        with ``m_cell`` the physical stretched cell widths of the base
        axis and ``m_inner`` its physical center-to-center face
        measures (both :meth:`_measure`). Because the plain down-hop
        **is** ``up_b^T`` (the zero-padded two-point mean), this is the
        weighted reduction ``(1/m_cell) * down_b(m_inner * flux)`` — no
        matrices are formed. The coupled-axis down-hop
        (:meth:`_cross_to_column` via :attr:`_down_i`) is uniform-
        periodic and needs no weighting; the diagonal flux legs never
        touch this hop, so they stay byte-identical across the change.

        Parameters
        ----------
        axis : str
            The coupled coordinate whose corner cross flux is landed.
        flux : ScalarField
            The Dirichlet-tagged corner cross flux (interior-face along
            the base axis).
        cache : MetricCache | None, optional
            The per-solve metric/measure memo (default: None).

        Returns
        -------
        ScalarField
            The cross flux on the ``axis`` face space.
        """
        if not self._stretched_base:
            return self._down_b[axis](flux)
        m_inner = self._measure(flux.function_space, cache)
        reduced = self._down_b[axis](flux * m_inner)
        # scale by the reciprocal cell measure. A field divide
        # ``reduced / m_cell`` is a storage-frame quotient and the base
        # measure's bounded-axis ghost slots are exactly zero, so it
        # would be a masked singularity: the forward stays finite (the
        # never-valid padding is re-synced) but ``1/0`` in the padding
        # poisons the reverse gradient (AGENTS.md differentiability
        # policy). The physical widths are strictly positive, so the
        # reciprocal built on the true region is finite and the padding
        # multiplies cleanly to zero — no divide, no guard needed.
        m_cell = self._measure(reduced.function_space, cache)
        return reduced * m_cell.with_data(1.0 / m_cell.data)

    def _cross_to_column(self, axis: str, v: ScalarField,
                         cache: MetricCache | None = None,
                         ) -> ScalarField:
        """
        Apply the column-face cross chain ``I_i(Z_i * I_b(v))``.

        Description
        -----------
        The exact transpose partner of :meth:`_cross_to_face`, and
        the shared chain of the divergence's slope term and the
        column velocity correction (the projection-consistency
        identity relies on one linear chain serving both).

        Parameters
        ----------
        axis : str
            The coupled coordinate.
        v : ScalarField
            A field on the ``axis`` face space (a gradient, a
            velocity component, or a correction).
        cache : MetricCache | None, optional
            The per-solve metric memo (default: None).

        Returns
        -------
        ScalarField
            The chain's output on the column face space (BC-free).
        """
        corner = self._up_b[axis](v)
        return self._down_i[axis](
            corner * self._slope(corner, axis, cache))

    # ================================================================
    #  The operator, its right-hand side, and the velocity update
    # ================================================================
    def _fluxes(self, p: ScalarField,
                cache: MetricCache | None = None,
                ) -> dict[str, ScalarField]:
        """Assemble the per-axis pressure fluxes ``K^{aj} G_j p``."""
        grads = {a: self._grad[a](p) for a in self._axes}
        fluxes: dict[str, ScalarField] = {}
        g_b = grads[self._base]
        for a in self._axes:
            if a == self._base:
                continue
            flux = grads[a] * self._jacobian(grads[a], cache)
            if a in self._coupled:
                flux = flux - self._cross_to_face(a, g_b, cache)
            fluxes[a] = self._weight(a) * flux
        flux = g_b * self._column_coefficient(
            g_b.function_space, cache)
        for a in self._coupled:
            flux = flux - self._weight(a) * self._cross_to_column(
                a, grads[a], cache)
        fluxes[self._base] = flux
        return fluxes

    def apply(self, p: ScalarField,
              cache: MetricCache | None = None) -> ScalarField:
        r"""
        Apply the mapped elliptic operator ``A p``.

        Description
        -----------
        The flux-form ``sum_a D_a(K^{aj} G_j p)`` with the fluxes
        retagged Dirichlet on bounded axes before the flux
        difference (zero normal flux through the mapped boundary,
        module docstring). ``A`` is exactly symmetric and negative
        semidefinite under the computational measure; its nullspace
        is the constants.

        Parameters
        ----------
        p : ScalarField
            The pressure iterate on the solver's space.
        cache : MetricCache | None, optional
            The per-solve metric memo threaded by :meth:`krylov`;
            None derives the metrics for this application only
            (default: None).

        Returns
        -------
        ScalarField
            ``A p`` on the same space.
        """
        fluxes = self._fluxes(p, cache)
        out = None
        for a in self._axes:
            term = self._div[a](fluxes[a].retag(self._tagged[a]))
            out = term if out is None else out + term
        return out

    def divergence(
        self, vel: Mapping[str, ScalarField],
        cache: MetricCache | None = None,
    ) -> ScalarField:
        r"""
        Compute the J-weighted divergence the operator measures.

        Description
        -----------
        The flux form
        ``sum_i D_i(J u_i) + D_b(u_m - sum_i Z_i I(u_i))``
        of ``J div_phys(u)``: the slope terms transform the physical
        column velocity into the contravariant boundary-conforming
        flux, and the Dirichlet-tagged column difference closes the
        walls with zero normal flux. This — not the computational
        ``Divergence()`` — is the right-hand side of the projection:
        after :meth:`velocity_correction` the *same* divergence of
        the updated velocity equals the CG residual.

        Parameters
        ----------
        vel : Mapping[str, ScalarField]
            The physical velocity components keyed by coordinate
            name (the mapped column's base keys the component along
            the mapped physical direction).
        cache : MetricCache | None, optional
            The per-solve metric memo (:meth:`project` shares one
            across divergence, solve and correction); None derives
            the metrics for this call only (default: None).

        Returns
        -------
        ScalarField
            The J-weighted divergence on the pressure space.
        """
        if set(vel) != set(self._axes):
            raise ValueError(
                f"divergence needs one component per axis "
                f"{self._axes}, got {tuple(sorted(vel))}")
        out = None
        column = vel[self._base].retag(self._face[self._base])
        for a in self._axes:
            if a == self._base:
                continue
            u = vel[a].retag(self._face[a])
            if a in self._coupled:
                column = column - self._cross_to_column(a, u, cache)
            term = self._div[a](
                (u * self._jacobian(u, cache)).retag(self._tagged[a]))
            out = term if out is None else out + term
        term = self._div[self._base](
            column.retag(self._tagged[self._base]))
        return out + term

    def velocity_correction(
        self, p: ScalarField, cache: MetricCache | None = None,
    ) -> dict[str, ScalarField]:
        r"""
        Derive the flux-consistent velocity corrections per axis.

        Description
        -----------
        Derived from the operator's own fluxes so the projection
        removes exactly the divergence :meth:`divergence` measures:
        per coupled/plain axis ``u_i -= F_i / J`` (continuum:
        ``w_i d p/d x_i`` at constant physical coordinates), and
        along the column ``u_m -= F_b + sum_i Z_i I(F_i / J)``
        (continuum: ``w_m dp/dm`` — the cross contributions cancel
        pointwise). Corrections come back BC-free; the caller adopts
        each velocity's own tag (the flat path's retag discipline).

        Parameters
        ----------
        p : ScalarField
            The solved pressure.
        cache : MetricCache | None, optional
            The per-solve metric memo (default: None).

        Returns
        -------
        dict[str, ScalarField]
            Per-coordinate corrections to subtract.
        """
        fluxes = self._fluxes(p, cache)
        corrections: dict[str, ScalarField] = {}
        column = fluxes[self._base]
        for a in self._axes:
            if a == self._base:
                continue
            corr = self._divide_by_jacobian(fluxes[a], cache)
            if a in self._coupled:
                column = column + self._cross_to_column(a, corr, cache)
            corrections[a] = corr
        corrections[self._base] = column
        return corrections

    def _divide_by_jacobian(
        self, flux: ScalarField, cache: MetricCache | None = None,
    ) -> ScalarField:
        r"""
        Divide a physical flux by the column Jacobian, VJP-safe.

        Description
        -----------
        The metric quotient ``F_i / J`` of a plain/coupled velocity
        correction (:meth:`velocity_correction`). ``J = dm/db`` is
        strictly positive on every valid cell but zero-filled in the
        never-valid storage padding, where the raw divide is a sealed
        ``inf`` the caller's retag discards in the primal — yet whose
        reverse VJP is singular (``0 * inf -> NaN``, the masked-
        singularity poison the differentiability policy names). The
        double-``jnp.where`` seals the reverse pass while staying
        bitwise identical on every valid cell (``bad`` covers only the
        ``J == 0`` padding), so the forward projection is untouched.

        Parameters
        ----------
        flux : ScalarField
            The physical flux to rescale (the numerator).
        cache : MetricCache | None, optional
            The per-solve metric memo (default: None).

        Returns
        -------
        ScalarField
            The BC-free correction ``flux / J`` on the divide's bare
            space, finite (0) in the never-valid padding.
        """
        jac = self._jacobian(flux, cache)
        bad = jac.storage == 0.0
        safe = jnp.where(bad, 1.0, jac.storage)
        ratio = jnp.where(bad, 0.0, flux.storage / safe)
        # the divide fixes the result's structure (bare space, merged
        # halo validity); its raw quotient data is discarded for the
        # guarded ratio, so the singular divide-VJP is never built.
        return (flux / jac).with_storage(ratio)

    # ================================================================
    #  Smoothing surfaces (multigrid, B1)
    # ================================================================
    def _axis_storage(self) -> dict[str, int]:
        """Map each coordinate name to its storage-frame array axis."""
        return {factor.names[0]: axis
                for factor, axis in factor_axes(self._space)}

    def _diagonal_data(
        self, cache: MetricCache | None = None,
    ) -> jax.Array:
        r"""
        Assemble the exact operator diagonal as a storage array.

        Description
        -----------
        The diagonal of :meth:`apply` derived analytically from the
        face coefficient fields: per axis the ``-(K_{f-} + K_{f+})/h^2``
        second-difference pattern of the diagonal flux leg (``K^{aa} =
        w_a J`` on the transverse faces, ``K^{bb}`` on the column
        faces, with the Neumann wall drops on the bounded column),
        **plus** the corner cross rows — the slope's mixed corner
        difference (:func:`_corner_cross_bracket`), which contribute to
        the diagonal because the column slope ``Z_a`` varies along the
        column (module docstring). Broadcast to the cell shape by the
        caller.

        Parameters
        ----------
        cache : MetricCache | None, optional
            The per-solve metric memo threaded through the derivations
            (default: None).

        Returns
        -------
        jax.Array
            The diagonal in the storage frame (broadcastable over the
            cell space).
        """
        storage = self._axis_storage()
        base = self._base
        z_axis = storage[base]
        # the base-axis (vertical) spacing enters as a scalar ``dz`` on
        # a uniform column and as the physical ``grid.measure`` fields
        # on a stretched one (N3): the base ``diff`` legs divide the
        # gradient by the dual center-to-center width ``m_inner`` and
        # the divergence by the primal cell width ``m_cell``, so the
        # column leg diagonal is ``-(K^bb_f / m_inner_f)`` summed over
        # the cell's two faces, over ``m_cell`` (and the cross bracket's
        # vertical scale is ``1 / m_cell`` in place of ``1 / dz``). The
        # uniform branch stays **bitwise** the phase-B scalar path.
        if self._stretched_base:
            m_cell = self._measure(self._space, cache).data
            m_inner = self._measure(self._face[base], cache).data
        else:
            dz = uniform_spacing(self._space.factor(base))
        diagonal: jax.Array | None = None
        for a in self._axes:
            if a == base:
                kface = self._column_coefficient(self._face[a], cache)
                if self._stretched_base:
                    leg = -_adjacent_face_sum(
                        kface.data / m_inner, z_axis,
                        periodic=False) / m_cell
                else:
                    leg = -_adjacent_face_sum(
                        kface.data, z_axis, periodic=False) / (dz * dz)
            else:
                dx = uniform_spacing(self._space.factor(a))
                kface = self._weight(a) * self._metric(
                    self._face[a],
                    f"d{self._mapped}_d{self._base}", cache)
                leg = -_adjacent_face_sum(
                    kface.data, storage[a], periodic=True) / (dx * dx)
            diagonal = leg if diagonal is None else diagonal + leg
        for a in self._coupled:
            dx = uniform_spacing(self._space.factor(a))
            slope = self._metric(
                self._corner[a], f"d{self._mapped}_d{a}", cache)
            bracket = _corner_cross_bracket(
                slope.data, storage[a], z_axis)
            if self._stretched_base:
                diagonal = diagonal - self._weight(a) / (
                    2.0 * dx) * bracket / m_cell
            else:
                diagonal = diagonal - self._weight(a) / (
                    2.0 * dx * dz) * bracket
        return diagonal

    def diagonal(
        self, cache: MetricCache | None = None,
    ) -> ScalarField:
        r"""
        Return the exact diagonal of ``apply`` on the pressure space.

        Description
        -----------
        The operator's diagonal (:meth:`_diagonal_data`) as a field on
        :attr:`_space`, for the damped point-Jacobi smoother
        (:class:`~fridom.spatial.operators.multigrid.DampedJacobi`). It
        is probe-exact: ``sum_c e_c (x) A e_c`` matches it to roundoff
        (the corner cross rows included). Derived from the same
        per-solve metric machinery as :meth:`apply`; nothing is stored.

        Parameters
        ----------
        cache : MetricCache | None, optional
            The per-solve metric memo (default: None).

        Returns
        -------
        ScalarField
            The diagonal field on the pressure space.
        """
        template = self._grid.create_field(self._space)
        data = jnp.broadcast_to(
            self._diagonal_data(cache), template.data.shape)
        return template.with_data(data)

    def vertical_bands(
        self, cache: MetricCache | None = None,
    ) -> VerticalBands:
        r"""
        Return the per-column tridiagonal ``T`` for line relaxation.

        Description
        -----------
        The per-column tridiagonal of the vertical-line smoother
        (:class:`~fridom.spatial.operators.multigrid.VerticalLineJacobi`):
        ``diag`` is the **full** operator diagonal (:meth:`diagonal`,
        so the horizontal stiffness enters as a stronger diagonal),
        while ``lower``/``upper`` are the **vertical flux leg only** —
        the column coefficient ``K^{bb}`` at the ``z``-faces, with the
        Neumann ends zeroed. ``T`` deliberately excludes the
        horizontal-flux cross residues: they couple off-column and are
        asymmetric per column (``K^{bb}`` already carries the
        slope-squared stiffness).

        On a **uniform** column the off-diagonal is ``+K^{bb}_f / dz^2``
        and ``T`` is Euclidean-symmetric by construction (``lower[c] ==
        upper[c-1]``). On a **stretched** column (N3) it is the exact
        vertical restriction of the measure-weighted operator,
        ``upper[c] = K^{bb}_f / (m_inner_f\, m_cell_c)`` and the
        sub-face partner for ``lower``: Euclidean-asymmetric, but
        **self-adjoint under the physical measure** — ``m_cell_{c-1}
        upper[c-1] == m_cell_c lower[c]``, i.e. ``diag(m_cell) T`` is
        symmetric — which is the inner product CG evaluates
        (``krylov`` ``_dot``), so the ``pre == post`` V-cycle stays SPD
        under it. The batched Thomas solve needs no symmetry, and the
        full diagonal keeps every column diagonally dominant either way.

        Parameters
        ----------
        cache : MetricCache | None, optional
            The per-solve metric memo (default: None).

        Returns
        -------
        VerticalBands
            The ``(lower, diag, upper, axis)`` bands on the pressure
            space.
        """
        storage = self._axis_storage()
        base = self._base
        z_axis = storage[base]
        kbb = self._column_coefficient(self._face[base], cache).data
        edge = list(kbb.shape)
        edge[z_axis] = 1
        zeros = jnp.zeros(edge, dtype=kbb.dtype)
        # the vertical off-diagonals of the operator's column
        # tridiagonal. On a uniform column the constant ``dz`` (bitwise
        # the phase-B scalar path); on a stretched column the
        # measure-weighted band ``upper[c] = K^bb_f / (m_inner_f
        # m_cell_c)`` (and ``lower[c]`` the sub-face partner). That
        # band is Euclidean-asymmetric (``lower[c] != upper[c-1]``) but
        # **self-adjoint under the physical measure**: ``m_cell_{c-1}
        # upper[c-1] == m_cell_c lower[c] == K^bb_f / m_inner_f``, i.e.
        # ``diag(m_cell) T`` is symmetric. That is exactly the inner
        # product CG uses (krylov ``_dot``), so ``T^{-1}`` is
        # measure-self-adjoint and the ``pre == post`` V-cycle stays
        # symmetric under it (N3). The Thomas kernel needs no symmetry;
        # the full diagonal keeps every column diagonally dominant.
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
    #  The preconditioned solve
    # ================================================================
    def _mean_coefficients(
        self, cache: MetricCache | None = None,
    ) -> dict[str, jax.Array]:
        """Fold the diagonal coefficients to their (0-d) means."""
        coeffs: dict[str, jax.Array] = {}
        for a in self._axes:
            if a == self._base:
                field = self._column_coefficient(self._face[a], cache)
            else:
                field = self._weight(a) * self._metric(
                    self._face[a],
                    f"d{self._mapped}_d{self._base}", cache)
            coeffs[a] = jnp.reshape(field.mean().data, ())
        return coeffs

    def _preconditioner(
        self, cache: MetricCache | None = None,
    ) -> Callable[[ScalarField], ScalarField]:
        """
        Build the flat spectral inverse at folded coefficients.

        Description
        -----------
        ``Div @ Diag(mean K) @ Grad`` inverted by ``SpectralSolve``
        on the Neumann-tagged sibling of the pressure space (the
        ``SpectralPressureSolver`` seam); negative definite like
        ``A`` (the consistent sign pair, module docstring), exact
        for constant metrics. The means are recomputed per solve —
        dynamic parameters trace through.

        Parameters
        ----------
        cache : MetricCache | None, optional
            The per-solve metric memo; the folded coefficients read
            the very metrics the operator applications reuse
            (default: None).

        Returns
        -------
        Callable[[ScalarField], ScalarField]
            The preconditioner ``M_inv``.
        """
        solve_space = _neumann_sibling(self._space)
        grad_block = Gradient().expand(solve_space, self._grid)
        axes = solve_space.active_axis_names
        mid = tuple(
            _dirichlet_mid(space, axis)
            for axis, space in zip(
                axes, grad_block.codomains(solve_space),
                strict=True))
        div_block = Divergence().expand(mid, self._grid)
        diag = Diag(self._mean_coefficients(cache), axes=axes)
        lap = (div_block @ diag @ grad_block).scalar()
        solve = SpectralSolve(lap, self._grid, solve_space,
                              single_precision=self._single_precision)
        if solve_space is self._space:
            return solve.solve

        def apply(r: ScalarField) -> ScalarField:
            return solve.solve(r.retag(solve_space)).retag(r)

        return apply

    def _build_vcycle(self, cache: MetricCache) -> MultigridVCycle:
        r"""
        Assemble the semicoarsened multigrid V-cycle preconditioner.

        Description
        -----------
        The ``preconditioner="multigrid"`` seam (B3, MG-D4/D6/D8): the
        horizontal axes semicoarsen (never the mapped column ``base``),
        each level re-instantiates this solver class on its coarse grid
        (re-discretization of the metrics on the coarse spaces), the
        transfers are the order-2 ``GridTransfer`` pair, the smoother is
        vertical-line Jacobi at :data:`_LINE_OMEGA`, and every level
        projects out the constants (mean-free). The finest level re-uses
        ``self`` and the shared ``cache``; the coarse levels each carry
        their own per-solve metric memo, re-derived on the coarse grid.

        Static maps only (iteration 1): a moving geometry threads
        grid-bound parameter fields through ``params=``, which the
        coarse re-derivation cannot re-bind, so this raises.

        Parameters
        ----------
        cache : MetricCache
            The finest level's per-solve metric memo (shared with the
            outer CG operator).

        Returns
        -------
        MultigridVCycle
            The V-cycle callable for the ``preconditioner=`` seam.

        Raises
        ------
        NotImplementedError
            If ``params`` carry grid-bound field data (moving geometry).
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
            max_levels=self._multigrid_levels,
            rediscretize=(rediscretize_fv_coarse
                          if is_fv(self._space) else None))
        levels: list[MultigridLevel] = []
        for index, (grid, space, transfer) in enumerate(chain):
            if index == 0:
                solver = self
                level_cache: MetricCache = cache
            else:
                # the coarse level supplies only its operator, bands
                # and diagonal — its own preconditioner is never built,
                # so ``"none"`` sidesteps the N1 spectral-on-stretched
                # construction raise a stretched coarse column would
                # otherwise trigger (the vertical stays mapped at every
                # level, MG-D4). On a uniform column this stays the
                # phase-B ``"spectral"`` default (behaviour unchanged).
                solver = MappedPressureSolver(
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
                smoother, _mean_free, transfer))
        return MultigridVCycle(tuple(levels))

    def krylov(
        self, cache: MetricCache | None = None,
    ) -> ConjugateGradient:
        """
        Build the configured PCG solver (public for diagnostics).

        Description
        -----------
        The returned solver's operator closes over a **per-solve**
        metric memo — a fresh dict per ``krylov()`` call unless the
        caller shares one (:meth:`project`) — so the metrics are
        derived once instead of once per CG iteration; they cannot
        go stale, since the memo dies with the returned solver and
        never reaches ``self`` (module docstring). The returned
        object is therefore, like the solver itself, valid for the
        single trace it was built in. The ``preconditioner`` knob
        selects the flat spectral inverse or the multigrid V-cycle
        (:meth:`_build_vcycle`).

        Parameters
        ----------
        cache : MetricCache | None, optional
            An existing per-solve memo to share; None creates a
            fresh one (default: None).

        Returns
        -------
        ConjugateGradient
            PCG on ``apply`` with the configured preconditioner, the
            constants-nullspace projection, and the optional
            convergence ``tolerance``.
        """
        if cache is None:
            cache = {}
        if self._preconditioner_kind == "none":
            # the N1 plain-CG stopgap: unpreconditioned CG (identity
            # M_inv). ``A`` is negative definite, but the CG recurrence
            # on ``(A, identity)`` gives the same iterates as standard
            # CG on the SPD ``-A`` (the sign folds through the alpha/beta
            # ratios), so no sign flip is needed. Slow (no spectral
            # compression), for correctness runs on a stretched column,
            # not production.
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
            project_mean=True)

    def solve(
        self, rhs: ScalarField, x0: ScalarField | None = None,
        cache: MetricCache | None = None,
    ) -> ScalarField:
        """
        Solve ``A p = rhs`` by preconditioned CG.

        Parameters
        ----------
        rhs : ScalarField
            The J-weighted divergence (:meth:`divergence`).
        x0 : ScalarField | None, optional
            The initial guess; None starts from zeros
            (default: None).
        cache : MetricCache | None, optional
            The per-solve metric memo to share; None creates a fresh
            one for this solve (default: None).

        Returns
        -------
        ScalarField
            The mean-free pressure on the same space.
        """
        return self.krylov(cache)(rhs, x0)

    def project(
        self, vel: Mapping[str, ScalarField],
        x0: ScalarField | None = None,
    ) -> tuple[ScalarField, dict[str, ScalarField]]:
        """
        Run the whole projection on one shared metric derivation.

        Description
        -----------
        The projection's three halves — the J-weighted divergence,
        the PCG solve, and the flux-consistent velocity corrections
        — read the *same* metric coefficients. This entry point
        derives them once (one per-solve memo, created here and
        dropped on return; module docstring) and threads that memo
        through all three, so a step's trace carries a single
        derivation chain instead of one per operator application.
        Under a moving geometry the next step builds a new solver
        with the new parameter fields and derives everything afresh.

        The optional ``x0`` warm-starts the PCG from the previous
        step's solved potential (:meth:`solve`); the stopping test is
        RHS-relative, so a good guess saves achieved iterations while
        the returned solution stays mean-free and start-independent.

        Parameters
        ----------
        vel : Mapping[str, ScalarField]
            The physical velocity components (:meth:`divergence`).
        x0 : ScalarField | None, optional
            The warm-start initial guess for the solve; None starts
            from zeros (default: None).

        Returns
        -------
        tuple[ScalarField, dict[str, ScalarField]]
            The mean-free pressure and the per-coordinate velocity
            corrections to subtract (:meth:`velocity_correction`).
        """
        cache: MetricCache = {}
        p = self.solve(self.divergence(vel, cache), x0, cache=cache)
        return p, self.velocity_correction(p, cache)
