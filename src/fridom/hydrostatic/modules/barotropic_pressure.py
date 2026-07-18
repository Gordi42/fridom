r"""The terrain-following barotropic (surface-pressure) Helmholtz solve.

Description
-----------
Multigrid-generalization plan, decision GM-D1 (option 1, volume-exact)
and GM-D2. On a terrain-following (sigma) grid the physical column depth
``H(x, y) = \int J\,\mathrm{d}z`` (``J`` the column Jacobian) varies
horizontally, so the implicit free surface's barotropic Helmholtz
operator becomes **variable-coefficient** and leaves the separable
:class:`~fridom.spatial.operators.spectral_solve.SpectralSolve` fast
path (the flat :meth:`ImplicitFreeSurface._solve`). This module builds
the SPD flux-form operator by explicit per-axis field arithmetic and
wraps a fixed-iteration
:class:`~fridom.spatial.operators.krylov.ConjugateGradient` around it —
the terrain analogue of the immersed
:meth:`ImplicitFreeSurface._solve_immersed` and of the nonhydrostatic
:class:`~fridom.nonhydro2.modules.mapped_pressure.MappedPressureSolver`.

The operator (volume-exact form)
--------------------------------
With ``dt' = ctx.stage_dt``, ``g = c^2 / H_{ref}`` (``H_{ref}`` the
constant vertical mesh extent), ``H_a(x, y) = \int J\,\mathrm{d}z`` on
the a-face, and ``T^*`` the raw transport divergence
``\int[\partial_x(Ju) + \partial_y(Jv)]\,\mathrm{d}z``:

.. math::

    A\,p_s = \varepsilon\, p_s
        - dt'^2\, g \sum_a \nabla_a\!\cdot(H_a\, \nabla_a p_s),
    \qquad
    \text{rhs} = \varepsilon\, p_s^{n} - dt'\, g\, T^{*} .

``A`` is SPD in the **plain 2-D measure product** for ``\varepsilon > 0``
(PSD with the constant nullspace at ``\varepsilon = 0``); it conserves
the plain ``\int p_s\,\mathrm{d}A`` (the barotropic volume) to round-off
by construction, and carries **no** ``1/H(x, y)`` division anywhere — the
right-hand side uses the raw ``T^*``, not the normalized depth-mean
divergence, so the guarded-division autodiff hazard of the flat depth
mean never enters this path.

The velocity correction (:meth:`ImplicitFreeSurface._barotropic_solve`)
stays the z-uniform ``u \leftarrow u - dt'\,\nabla_h p_s`` — with these
definitions its raw transport divergence cancels the right-hand-side
divergence **exactly** (gate GB-1). The operator's discrete gradient /
divergence legs are therefore the same registry ``diff`` rows the
correction's ``p_s.diff(a)`` resolves, keyed for the wall closure exactly
as the mapped solver's flux legs (GM-D6): the divergence consumes the
Dirichlet-tagged face, so a walled horizontal axis carries zero
barotropic transport through the wall and a periodic axis is untouched.

Preconditioner (GM-D2, Phase B)
-------------------------------
The flat separable spectral inverse at the **mean depth**: the folded
coefficient ``c^2 \bar H / H_{ref}`` (``\bar H`` the domain-mean physical
cell depth) reduces the variable-coefficient operator to its
constant-coefficient sibling, inverted by
:class:`SpectralSolve` on the Neumann-tagged wall sibling. On a flat
(``a = 0``) chart the preconditioner is the exact inverse and PCG
converges in one iteration; the ``\varepsilon = 0`` rigid lid is singular
at ``k = 0`` and the ``where_zero`` mean gauge handles it.

Metric derivation: once per solve, never across steps
-----------------------------------------------------
Every depth field (``H_a``, ``H_{cell}``) is derived through
``grid.metric`` per :meth:`solve` call, never cached on the solver or
across a step — the mapped-solver discipline (rules 2.3 / 3.8). Terrain
charts in the hydrostatic model are static, so no dynamic-parameter
threading is needed here; the solver is a plain trace-time object
(constructed per solve inside the CONSTRAINT stage), not a pytree leaf.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom as fr
from fridom.hydrostatic.modules.terrain import jacobian_name
from fridom.spatial.bc import BC
from fridom.spatial.operators.base import Identity, resolve_codomain
from fridom.spatial.operators.composed import Diag, Divergence, Gradient
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.krylov import ConjugateGradient
from fridom.spatial.operators.spectral_solve import SpectralSolve
from fridom.spatial.spaces.nodal import NodalSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid
    from fridom.spatial.operators.base import Operator
    from fridom.spatial.spaces.tensor_product import SpaceLike


# ================================================================
#  Wall-closure space siblings (the mapped-pressure precedent,
#  reproduced locally so the hydrostatic package carries no
#  nonhydro2 dependency — the "duplicate small builders" rule)
# ================================================================
def _mean_free(field: ScalarField) -> ScalarField:
    """Remove the plain measure-weighted mean (the constants gauge)."""
    return field - field.mean()


def _neumann_sibling(space: SpaceLike) -> SpaceLike:
    r"""Return the Neumann-tagged wall sibling of a solve space.

    Description
    -----------
    The trig-transform rows of a walled grid key on the BC-tagged
    origin, and the surface-pressure parity at a rigid horizontal wall
    is Neumann (the parity-even ``Div @ Diag @ Grad`` chain, the
    Cosine-II basis). Every bounded nodal factor is swapped for its
    ``BC.NEUMANN`` sibling (same mesh, node set and shape); periodic
    factors pass through, so on a doubly-periodic grid the sibling is
    the space itself (the interned no-retag fast path). The barotropic
    surface pressure is always a nodal ``Profile``, so only the nodal
    branch is reachable (no FV cell-average factor).

    Parameters
    ----------
    space : SpaceLike
        The bare surface-pressure solve space.

    Returns
    -------
    SpaceLike
        The interned sibling (``space`` itself when nothing is bounded).
    """
    replacements: dict[str, object] = {}
    for factor in space.factors:
        if getattr(factor.mesh, "periodic", True):
            continue
        if isinstance(factor, NodalSpace):
            replacements[factor.names[0]] = factor.mesh.nodal(
                factor.node_set, bc=BC.NEUMANN)
    if not replacements:
        return space
    return space.replace(**replacements)


def _dirichlet_mid(space: SpaceLike, axis: str) -> SpaceLike:
    r"""Declare the Dirichlet parity of one gradient-output space.

    Description
    -----------
    The wall-normal gradient of an even (Neumann) pressure is odd, i.e.
    it vanishes **at** the wall — the Dirichlet claim on the staggered
    faces. Declaring it keys the divergence legs on the BC-structured
    rows (zero normal barotropic transport through the wall); periodic
    factors pass through untouched.

    Parameters
    ----------
    space : SpaceLike
        One per-axis gradient codomain (bare).
    axis : str
        The gradient axis of this component.

    Returns
    -------
    SpaceLike
        The interned sibling with the axis factor Dirichlet-tagged
        (``space`` itself on a periodic axis).
    """
    factor = space.factor(axis)
    if (isinstance(factor, NodalSpace)
            and not getattr(factor.mesh, "periodic", True)
            and factor.bc.is_free):
        return space.replace(**{axis: factor.mesh.nodal(
            factor.node_set, bc=BC.DIRICHLET)})
    return space


class BarotropicPressureSolver:

    r"""PCG solve of the terrain barotropic Helmholtz problem (GM-D1/D2).

    Description
    -----------
    A trace-time object (constructed per solve inside the free surface's
    CONSTRAINT stage, carrying no mutable state and not a pytree leaf):
    it resolves the per-axis gradient / divergence legs of the
    volume-exact flux-form operator once, derives the physical face
    depths ``H_a`` through ``grid.metric`` at :meth:`solve` (never cached
    across steps), and inverts the operator by preconditioned CG. The
    public surface mirrors :class:`MappedPressureSolver`:
    :meth:`operator` (the SPD apply closure), :meth:`preconditioner`
    (the flat mean-depth spectral inverse), :meth:`krylov` and
    :meth:`solve`.

    Parameters
    ----------
    grid : Grid
        The bound terrain grid (carries the mapping / dispatch registry).
    space : SpaceLike
        The (bare, 2-D ``Profile``) surface-pressure solve space.
    column : tuple[str, str]
        The ``(mapped, base)`` terrain column
        (:func:`~fridom.hydrostatic.modules.terrain.discover_column`).
    vertical : str
        The vertical coordinate the column depth integrates over.
    epsilon : float
        The free-surface knob: ``> 0`` the backward-Euler free surface
        (non-singular), ``== 0`` the rigid lid (the singular Poisson,
        the plain-mean nullspace gauge, GM-D7).
    inv_depth : float
        The reciprocal reference depth ``1/H_{ref}`` (the constant
        vertical mesh extent).
    iterations : int
        The fixed PCG iteration budget (the maximum under a tolerance).
    tolerance : float | None
        The PCG convergence break forwarded to
        :class:`ConjugateGradient`; ``None`` runs the fixed count.
    """

    def __init__(
        self,
        grid: Grid,
        space: SpaceLike,
        column: tuple[str, str],
        vertical: str,
        *,
        epsilon: float,
        inv_depth: float,
        iterations: int,
        tolerance: float | None,
    ) -> None:
        """Resolve the static flux legs; see the class docstring."""
        self._grid = grid
        self._space: SpaceLike = space.bare
        self._column = column
        self._vertical = vertical
        self._epsilon = float(epsilon)
        self._inv_depth = inv_depth
        self._iterations = iterations
        self._tolerance = tolerance
        self._jname = jacobian_name(column)
        self._axes: tuple[str, ...] = self._space.active_axis_names
        registry = grid.dispatch
        # the 3-D collocated space fixes the per-axis face on which the
        # column-integrated depth H_a lands (the same face the C-grid
        # gradient of ps reaches, so coeff * grad ps is a same-space
        # multiply and the exact-cancellation identity holds, GB-1)
        coll = fr.spatial.Collocated().resolve(grid)
        self._grad: dict[str, Operator] = {}
        self._face: dict[str, SpaceLike] = {}
        self._tagged: dict[str, SpaceLike] = {}
        self._div: dict[str, Operator] = {}
        self._face3: dict[str, SpaceLike] = {}
        for a in self._axes:
            g = registry.resolve("diff", self._space.factor(a))[a]
            face = resolve_codomain(g, self._space)
            tagged = _dirichlet_mid(face, a)
            self._grad[a] = g
            self._face[a] = face
            self._tagged[a] = tagged
            self._div[a] = registry.resolve("diff", tagged.factor(a))[a]
            g3 = registry.resolve("diff", coll.factor(a))[a]
            self._face3[a] = resolve_codomain(g3, coll)
        self._coll: SpaceLike = coll

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def axes(self) -> tuple[str, ...]:
        """The surface-pressure solve axes (the horizontal coordinates)."""
        return self._axes

    @property
    def epsilon(self) -> float:
        """The free-surface knob (``0`` rigid lid, ``> 0`` free surface)."""
        return self._epsilon

    @property
    def iterations(self) -> int:
        """The fixed PCG iteration budget (the max under a tolerance)."""
        return self._iterations

    @property
    def tolerance(self) -> float | None:
        """The optional PCG convergence break (None = fixed count)."""
        return self._tolerance

    # ================================================================
    #  Physical column depths (derived per solve, never cached)
    # ================================================================
    def _face_depth(self, axis: str) -> ScalarField:
        r"""Return ``H_a = \int J\,\mathrm{d}z`` on the a-face.

        Description
        -----------
        The column-integrated Jacobian on the 3-D a-face reduced onto
        the 2-D ``Profile`` face (the plain vertical integral is the
        physical column extent, rules 2.7). It lands on the exact face
        the C-grid gradient ``p_s.diff(a)`` reaches, so the operator's
        ``coeff * grad`` multiply is a same-space product.

        Parameters
        ----------
        axis : str
            The horizontal coordinate whose face depth is derived.

        Returns
        -------
        ScalarField
            ``H_a`` on the 2-D a-face.
        """
        jac = self._grid.metric(self._face3[axis].bare, self._jname)
        return Integral()[self._vertical](jac)

    def _mean_depth(self) -> jax.Array:
        r"""Return ``\bar H``, the domain-mean physical cell depth.

        Description
        -----------
        The in-trace plain measure-weighted mean of the cell depth
        ``H_{cell} = \int J\,\mathrm{d}z`` on the ``p_s`` cell — a 0-d
        traced scalar, the folded coefficient of the flat mean-depth
        preconditioner. Recomputed per solve (dynamic-safe).

        Returns
        -------
        jax.Array
            The 0-d mean physical cell depth.
        """
        jac = self._grid.metric(self._coll.bare, self._jname)
        h_cell = Integral()[self._vertical](jac)
        return jnp.reshape(h_cell.mean().data, ())

    # ================================================================
    #  The operator, its preconditioner, and the preconditioned solve
    # ================================================================
    def _apply(
        self, ps: ScalarField, coeff: dict[str, ScalarField],
        dt: object,
    ) -> ScalarField:
        r"""Apply ``A p_s`` given the per-solve face coefficients.

        Description
        -----------
        The SPD flux form
        ``\varepsilon p_s - dt'^2 \sum_a D_a(coeff_a\,G_a p_s)`` with the
        flux retagged Dirichlet on a bounded axis before the flux
        difference (zero normal barotropic transport through the wall;
        the retag is the identity on periodic axes).

        Parameters
        ----------
        ps : ScalarField
            The surface-pressure iterate on the solve space.
        coeff : dict[str, ScalarField]
            The per-axis face coefficient ``H_a\,g`` (:meth:`operator`).
        dt : object
            The stage increment ``dt' = ctx.stage_dt``.

        Returns
        -------
        ScalarField
            ``A p_s`` on the same space.
        """
        out = self._epsilon * ps
        for a in self._axes:
            flux = coeff[a] * self._grad[a](ps)
            out = out - (dt ** 2) * self._div[a](
                flux.retag(self._tagged[a]))
        return out

    def operator(
        self, *, csqr: object, dt: object,
    ) -> Callable[[ScalarField], ScalarField]:
        r"""Return the SPD apply closure ``A`` (public for diagnostics).

        Description
        -----------
        Derives the per-axis face coefficient ``H_a\,g`` once
        (``g = c^2 / H_{ref}``) and closes over it, so a whole CG run
        reuses one metric derivation instead of re-deriving ``H_a`` each
        application.

        Parameters
        ----------
        csqr : object
            The live squared-phase-speed leaf ``c^2``.
        dt : object
            The stage increment ``dt' = ctx.stage_dt``.

        Returns
        -------
        Callable[[ScalarField], ScalarField]
            The field-to-field operator ``A``.
        """
        g = csqr * self._inv_depth
        coeff = {a: self._face_depth(a) * g for a in self._axes}
        return lambda ps: self._apply(ps, coeff, dt)

    def _flat_spectral(
        self, coeff: object, dt: object,
    ) -> Callable[[ScalarField], ScalarField]:
        r"""Build the flat ``(eps - dt'^2 coeff\,\nabla^2)`` spectral solve.

        Description
        -----------
        The constant-coefficient sibling of the operator inverted by
        :class:`SpectralSolve` on the Neumann-tagged wall sibling of the
        solve space (the identity on a doubly-periodic grid): ``-dt'^2
        coeff`` folds into the ``Div @ Diag @ Grad`` diagonal, plus the
        static ``epsilon`` identity, with the ``where_zero`` mean gauge
        for the ``epsilon = 0`` singular ``k = 0`` mode. On a walled grid
        the incoming field is retagged onto the sibling and the solution
        retagged back.

        Parameters
        ----------
        coeff : object
            The folded scalar coefficient ``c^2 \bar H / H_{ref}``.
        dt : object
            The stage increment ``dt' = ctx.stage_dt``.

        Returns
        -------
        Callable[[ScalarField], ScalarField]
            The field-to-field spectral inverse on the solve space.
        """
        space = self._space
        solve_space = _neumann_sibling(space)
        neg = -(dt ** 2) * coeff
        grad_block = Gradient().expand(solve_space, self._grid)
        axes = solve_space.active_axis_names
        mid = tuple(
            _dirichlet_mid(sib, axis)
            for axis, sib in zip(
                axes, grad_block.codomains(solve_space), strict=True))
        div_block = Divergence().expand(mid, self._grid)
        diag = Diag(dict.fromkeys(axes, neg), axes=axes)
        laplacian = (div_block @ diag @ grad_block).scalar()
        helmholtz = laplacian + Identity() * float(self._epsilon)
        solve = SpectralSolve(
            helmholtz, self._grid, solve_space, where_zero=0.0)
        if solve_space is space:
            return solve.solve

        def apply(field: ScalarField) -> ScalarField:
            return solve.solve(field.retag(solve_space)).retag(field)

        return apply

    def preconditioner(
        self, *, csqr: object, dt: object,
    ) -> Callable[[ScalarField], ScalarField]:
        r"""Return the flat mean-depth spectral preconditioner ``M_inv``.

        Description
        -----------
        The flat spectral inverse (:meth:`_flat_spectral`) at the folded
        coefficient ``c^2 \bar H / H_{ref}`` — the exact operator inverse
        on a flat (``a = 0``) chart, an approximate inverse whose residual
        the CG iteration removes on steep terrain.

        Parameters
        ----------
        csqr : object
            The live squared-phase-speed leaf ``c^2``.
        dt : object
            The stage increment ``dt' = ctx.stage_dt``.

        Returns
        -------
        Callable[[ScalarField], ScalarField]
            The preconditioner ``M_inv``.
        """
        hbar = self._mean_depth()
        return self._flat_spectral(csqr * hbar * self._inv_depth, dt)

    def krylov(
        self, *, csqr: object, dt: object,
    ) -> ConjugateGradient:
        r"""Build the configured PCG solver (public for diagnostics).

        Description
        -----------
        PCG on the SPD :meth:`operator` with the flat mean-depth
        :meth:`preconditioner`, the reused ``pressure_iterations`` /
        ``pressure_tolerance`` budget, and — for the ``epsilon = 0``
        rigid lid — the plain-mean nullspace projection (GM-D7; the RHS
        is orthogonal to constants by telescoping of ``T^*``).
        ``epsilon > 0`` is non-singular and passes no projection.

        Parameters
        ----------
        csqr : object
            The live squared-phase-speed leaf ``c^2``.
        dt : object
            The stage increment ``dt' = ctx.stage_dt``.

        Returns
        -------
        ConjugateGradient
            The preconditioned CG solver.
        """
        projection = _mean_free if self._epsilon == 0.0 else None
        return ConjugateGradient(
            self.operator(csqr=csqr, dt=dt),
            preconditioner=self.preconditioner(csqr=csqr, dt=dt),
            iterations=self._iterations,
            tolerance=self._tolerance,
            projection=projection)

    def solve(
        self, rhs: ScalarField, x0: ScalarField | None = None,
        *, csqr: object, dt: object,
    ) -> ScalarField:
        r"""Solve ``A p_s = rhs`` by preconditioned CG.

        Parameters
        ----------
        rhs : ScalarField
            The barotropic right-hand side on the ``p_s`` cell.
        x0 : ScalarField | None, optional
            The initial guess (the previous ``p_s`` for a warm start);
            None starts from zeros. CG projects the guess, so a
            non-mean-free ``x0`` is safe under the rigid-lid mean gauge
            (default: None).
        csqr : object
            The live squared-phase-speed leaf ``c^2``.
        dt : object
            The stage increment ``dt' = ctx.stage_dt``.

        Returns
        -------
        ScalarField
            The surface pressure ``p_s^{n+1}`` on ``rhs``'s space.
        """
        return self.krylov(csqr=csqr, dt=dt)(rhs, x0)
