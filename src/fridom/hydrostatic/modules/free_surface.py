r"""Free-surface modules: the barotropic (surface-pressure) evolution.

Description
-----------
The free-surface family (HY-D3) owns the surface pressure ``ps = g*eta``
(2D, on the constant-along-z ``Profile("x", "y")`` space — the
barotropic mode) and **both sides of the barotropic coupling**: the
``-c^2\nabla_h\cdot\bar u`` gravity term that evolves ``ps`` and the
``-\nabla_h p_s`` force it exerts on the momentum. The hydrostatic core
reads only the baroclinic ``p_hyd``, so an implicit variant's force
cannot double-count with the constraint's own velocity correction
(H3 refactor).

Two variants live here:

- ``ExplicitFreeSurface`` — the correctness oracle (the Oceananigans
  ``ExplicitFreeSurface`` analogue): ``ps`` is a PROGNOSTIC field and
  the coupling is two **linear tendency terms**, the adjoint C-grid
  pair

  .. math::

      \partial_t p_s = -c^2\, \nabla_h\cdot\bar{\boldsymbol{u}},
      \qquad
      \partial_t \boldsymbol{u}_h = -\nabla_h p_s ,

  with ``c^2 = g H`` and the depth mean ``\bar u = (1/H)\int u\,dz``.
  Under the energy weight ``1/c^2`` on ``ps`` (integrated over the full
  depth) the pair is exactly skew-adjoint, so the linear barotropic
  gravity wave conserves energy to round-off. CFL-limited by
  ``sqrt(c^2)``.

- ``ImplicitFreeSurface`` — the workhorse (HY-D4): a **CONSTRAINT-stage
  2D projection**. After the advance it reads the post-advance velocity
  and solves a horizontal Helmholtz problem for ``ps^{n+1}``, then
  applies the ``-dt\,\nabla_h p_s`` correction to ``(u, v)``:

  .. math::

      (\varepsilon - dt'^2\, \nabla_h\!\cdot(c^2 \nabla_h))\, p_s^{n+1}
        = \varepsilon\, p_s^{n} - dt'\, c^2\, \nabla_h\cdot\bar u^{*},
      \\
      \boldsymbol{u}^{n+1}(z) = \boldsymbol{u}^{*}(z)
        - dt'\, \nabla_h p_s^{n+1} ,

  with ``dt' = ctx.stage_dt``. ``epsilon=1`` is the backward-Euler
  linear free surface (the pyOM/Oceananigans choice; symbol
  ``1 + c^2 dt'^2 k_disc^2``, non-singular); ``epsilon=0`` is the rigid
  lid (the singular Poisson with the ``where_zero`` mean gauge,
  enforcing a divergence-free depth mean). The barotropic gravity
  coupling leaves the ``linear=True`` terms, so this variant declares a
  ``linear_operator_gap`` (HY-D7).
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.hydrostatic.params import CSQR
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.operators.base import Identity
from fridom.spatial.operators.composed import Diag, Divergence, Gradient
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.spectral_solve import SpectralSolve

if TYPE_CHECKING:  # pragma: no cover
    from fridom.model.context import StepContext
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.tensor_product import SpaceLike


def _validate_horizontal(
    horizontal: tuple[str, str],
) -> tuple[str, str]:
    """Return the validated (zonal, meridional) coordinate names.

    Parameters
    ----------
    horizontal : tuple[str, str]
        The candidate (zonal, meridional) names.

    Returns
    -------
    tuple[str, str]
        The validated pair.

    Raises
    ------
    TypeError
        If the pair is not two distinct strings.
    """
    horizontal = tuple(horizontal)
    if (len(horizontal) != 2  # noqa: PLR2004 — zonal + meridional
            or not all(isinstance(name, str) for name in horizontal)
            or horizontal[0] == horizontal[1]):
        raise TypeError(
            "horizontal names the (zonal, meridional) coordinates: "
            f"two distinct strings, got {horizontal!r}")
    return horizontal


class _FreeSurfaceBase(fr.model.Module):

    r"""Shared surface-pressure plumbing for the free-surface variants.

    Description
    -----------
    Holds the geometry names, the depth-mean divisor ``1/H`` (frozen at
    bind from the mesh extent), the velocity references, the ``c^2``
    parameter reference and the depth-mean divergence
    ``\nabla_h\cdot\bar u`` — everything the explicit and implicit
    variants share. Not jaxified / never instantiated directly; the
    concrete variants are the pytree citizens.

    Parameters
    ----------
    vertical : str, optional
        The vertical coordinate the depth mean reduces over
        (default: ``"z"``).
    horizontal : tuple[str, str], optional
        The (zonal, meridional) coordinate names of the horizontal
        divergence (default: ``("x", "y")``).
    """

    def __init__(
        self,
        *,
        vertical: str = "z",
        horizontal: tuple[str, str] = ("x", "y"),
    ) -> None:
        """Store the geometry names; see the class docstring."""
        self._vertical = vertical
        self._horizontal = _validate_horizontal(horizontal)
        # the total vertical measure (the depth H); the exact divisor
        # of ScalarField.mean, frozen at bind (host constant, not
        # traced) so the depth mean stays tracer-compatible
        self._inv_depth: float = 1.0

    def bind(self, table: object) -> None:
        """Freeze the reciprocal depth ``1/H`` (the depth-mean divisor).

        Description
        -----------
        The halo-tracer has no ``.mean`` (it is host-side quadrature
        sugar), so the barotropic divergence uses ``integrate(z) *
        (1/H)`` instead — with ``H`` the total vertical extent. The
        cell measures tile the domain, so ``sum(measure) == H`` (a
        telescoping edge difference), keeping the depth mean
        measure-exact and the energy pairing skew to machine
        precision. ``H`` is read from the mesh geometry (no field
        materialization), so assembling a second model on the same
        already-frozen grid does not re-fetch a halo-shaped measure
        array — the grid reuse the D4 preset test needs.

        Parameters
        ----------
        table : object
            The binding table (carries the grid).

        Raises
        ------
        ValueError
            If the grid has no vertical coordinate.
        """
        grid = table.grid
        for mesh in grid.factors:
            if self._vertical in mesh.names:
                lo, hi = mesh.extent
                self._inv_depth = 1.0 / float(hi - lo)
                return
        raise ValueError(  # pragma: no cover — z-less grid fails at the
            # core's vertical declarations (w on Outer(z)) first
            f"the grid has no {self._vertical!r} coordinate for the "
            "free-surface depth mean")

    # ================================================================
    #  References (shared by both variants)
    # ================================================================
    @property
    def field_references(self) -> tuple[fr.model.FieldReference, ...]:
        """The barotropic solve rides the declared horizontal velocities."""
        zonal, meridional = self._horizontal
        return (
            fr.model.FieldReference(
                "u", hint="the barotropic divergence reads the "
                          "horizontal velocity (hy.HydrostaticCore "
                          f"declares {zonal}/{meridional} as u/v)"),
            fr.model.FieldReference(
                "v", hint="the barotropic divergence reads the "
                          "horizontal velocity (hy.HydrostaticCore)"),
        )

    parameter_references = (
        fr.model.ParameterReference(
            CSQR, hint="the squared phase speed c^2 = g*H is provided "
                       "by hy.HydrostaticCore(csqr=...)"),
    )

    # ================================================================
    #  The depth-mean divergence (the C-grid barotropic divergence)
    # ================================================================
    def _depth_mean_div(self, state: object) -> ScalarField:
        r"""Return ``\nabla_h\cdot\bar u`` on the ``ps`` cell.

        Description
        -----------
        The depth-mean horizontal divergence: reduce the *collocated*
        divergence ``d_x u + d_y v`` over ``z`` (the z-reduction and
        the horizontal derivative commute, so this equals
        ``d_x\bar u + d_y\bar v``) and divide by the total depth ``H``.
        The ``Integral`` runs on the same cell-centred z-factor the
        DIAGNOSE cumint reduces, keeping the barotropic gravity pair
        exactly adjoint to the ``-\nabla_h p_s`` momentum forcing under
        the depth-integrated energy metric.

        Parameters
        ----------
        state : object
            The current state (reads ``u`` and ``v``).

        Returns
        -------
        ScalarField
            The depth-mean divergence on the ``Profile`` cell.
        """
        zonal, meridional = self._horizontal
        div_h = state["u"].diff(zonal) + state["v"].diff(meridional)
        return Integral()[self._vertical](div_h) * self._inv_depth


@partial(jaxify, dynamic=())
class ExplicitFreeSurface(_FreeSurfaceBase):

    r"""Declares ``ps``; the two linear barotropic terms (H3).

    Description
    -----------
    The explicit (oracle) free surface. ``ps`` is a PROGNOSTIC field on
    ``Profile("x", "y")`` and the barotropic coupling is the adjoint
    C-grid pair of linear terms: ``d_t ps = -c^2 (d_x ubar + d_y vbar)``
    (the depth-mean divergence lands on the ``ps`` cell) and
    ``d_t u = -d_x ps`` / ``d_t v = -d_y ps`` (the surface-pressure
    gradient reaches the velocity faces through the ConstantSpace
    broadcast in ``.to``). Under the ``1/c^2`` energy weight integrated
    over the full depth, the pair is exactly skew-adjoint (the H2
    energy gate).

    Parameters
    ----------
    vertical : str, optional
        The vertical coordinate name the depth mean reduces over
        (default: ``"z"``).
    horizontal : tuple[str, str], optional
        The (zonal, meridional) coordinate names of the horizontal
        divergence (default: ``("x", "y")``).
    """

    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """The surface pressure ``ps`` (2D, constant along z)."""
        zonal, meridional = self._horizontal
        return (
            fr.model.FieldDeclaration(
                "ps", space=fr.spatial.Profile(zonal, meridional),
                lifecycle=fr.model.Lifecycle.PROGNOSTIC,
                long_name="Surface pressure (g*eta)",
                units="m^2/s^2"),
        )

    @fr.model.term(advances=("ps",), linear=True)
    def gravity(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``d_t ps = -c^2 (d_x ubar + d_y vbar)`` (depth-mean divergence).

        The depth means ``ubar = u.mean(z)`` / ``vbar = v.mean(z)`` are
        the measure-exact ``integrate(z) / H`` reductions onto the
        barotropic ``ConstantSpace`` factor; their horizontal
        divergence lands on the ``ps`` cell (the adjoint of the
        ``-grad ps`` momentum forcing below).
        """
        csqr = ctx.params[CSQR]
        return {"ps": -(csqr * self._depth_mean_div(state))}

    @fr.model.term(advances=("u", "v"), linear=True)
    def pressure_gradient(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""``d_t u = -d_x ps``, ``d_t v = -d_y ps`` (barotropic force).

        The surface pressure ``ps`` (constant along z) lifts onto the
        velocity faces: a single staggered difference broadcast onto
        the velocity's vertical layout through the ConstantSpace
        broadcast in ``.to`` and retagged onto the velocity (identity
        on periodic axes; adopts the wall-normal Dirichlet tag on a
        walled axis). The adjoint of the ``gravity`` term.
        """
        zonal, meridional = self._horizontal
        u, v = state["u"], state["v"]
        ps = state["ps"]
        return {
            "u": (-ps.diff(zonal).to(u)).retag(u),
            "v": (-ps.diff(meridional).to(v)).retag(v),
        }


@partial(jaxify, dynamic=())
class ImplicitFreeSurface(_FreeSurfaceBase):

    r"""Declares ``ps``; the CONSTRAINT-stage 2D projection (HY-D4).

    Description
    -----------
    The backward-Euler implicit free surface. After the advance a
    CONSTRAINT stage (write set ``{u, v, ps}``) reads the post-advance
    velocity ``u^*`` and solves a horizontal Helmholtz problem for
    ``ps^{n+1}`` on the 2D ``Profile`` space, then applies the
    ``-dt'\,\nabla_h ps`` correction to ``(u, v)`` (the transforms drop
    the ConstantSpace z factor — a natural 2D spectral solve). Running
    after the advance, it composes with **any** stepper (the pyOM
    quasi-AB2 + implicit surface reproduced by ``AdamBashforth(order=2,
    eps=0.1)`` alone); under IMEX it runs after the mixing solve.

    ``epsilon`` is a **static** constructor argument resolved at
    assembly — **not a sweepable parameter** (it selects the field
    lifecycle, so it cannot be an ``update_parameters`` leaf): ``> 0``
    makes ``ps`` PROGNOSTIC (the free surface, carried across steps),
    ``== 0`` makes it DIAGNOSTIC (the rigid lid, recomputed each step).

    The barotropic gravity coupling leaves the ``linear=True`` terms,
    so the module declares a ``linear_operator_gap`` (HY-D7): a
    ``require_linear_operator`` consumer (eigenmodes, projections,
    optimal balance, IMEX-by-linearity) refuses this model and points
    at ``hy.ExplicitFreeSurface``.

    Parameters
    ----------
    epsilon : float, optional
        The pyOM free-surface knob (static, not sweepable): ``1.0`` is
        the backward-Euler linear free surface (symbol
        ``1 + c^2 dt'^2 k_disc^2``, non-singular); ``0.0`` is the rigid
        lid (the singular Poisson, ``where_zero`` mean gauge). Must be
        ``>= 0`` (default: 1.0).
    vertical : str, optional
        The vertical coordinate name the depth mean reduces over
        (default: ``"z"``).
    horizontal : tuple[str, str], optional
        The (zonal, meridional) coordinate names of the horizontal
        divergence (default: ``("x", "y")``).

    Raises
    ------
    ValueError
        If ``epsilon`` is not a non-negative real number.
    """

    #: the honesty gate (HY-D7): this variant's barotropic coupling is
    #: NOT in L (``fr.model.require_linear_operator``)
    linear_operator_gap = (
        "the barotropic gravity coupling (the -grad(ps) momentum force "
        "and the ps = g*eta free-surface update) lives inside the "
        "CONSTRAINT-stage 2D Helmholtz projection, NOT in a linear=True "
        "term, so the linear operator L has NO surface-pressure mode at "
        "all. Assemble hy.ExplicitFreeSurface instead (the same "
        "barotropic physics as explicit linear terms, L complete) if "
        "you need eigenmodes, projections, optimal balance or "
        "IMEX-by-linearity")

    def __init__(
        self,
        *,
        epsilon: float = 1.0,
        vertical: str = "z",
        horizontal: tuple[str, str] = ("x", "y"),
    ) -> None:
        """Validate ``epsilon``; store the geometry names."""
        super().__init__(vertical=vertical, horizontal=horizontal)
        if (isinstance(epsilon, bool)
                or not isinstance(epsilon, int | float)
                or epsilon < 0):
            raise ValueError(
                "epsilon is the pyOM free-surface knob: a non-negative "
                f"real (1.0 free surface, 0.0 rigid lid), got "
                f"{epsilon!r}")
        self._epsilon = float(epsilon)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def epsilon(self) -> float:
        """The static free-surface knob (``0`` rigid lid, ``>0`` free)."""
        return self._epsilon

    # ================================================================
    #  Field declarations (lifecycle depends on epsilon)
    # ================================================================
    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """``ps`` — PROGNOSTIC iff ``epsilon > 0``, else DIAGNOSTIC.

        Description
        -----------
        A free surface (``epsilon > 0``) carries ``ps`` across steps
        (PROGNOSTIC, on the stepper ring buffers as a constant-along-z
        ``Profile`` field). The rigid lid (``epsilon == 0``) has no
        surface memory — ``ps`` is recomputed from the divergence every
        step (DIAGNOSTIC, owned by this module so the CONSTRAINT stage
        may write it).
        """
        zonal, meridional = self._horizontal
        lifecycle = (fr.model.Lifecycle.PROGNOSTIC if self._epsilon > 0
                     else fr.model.Lifecycle.DIAGNOSTIC)
        return (
            fr.model.FieldDeclaration(
                "ps", space=fr.spatial.Profile(zonal, meridional),
                lifecycle=lifecycle,
                long_name="Surface pressure (g*eta)",
                units="m^2/s^2"),
        )

    # ================================================================
    #  The barotropic projection CONSTRAINT stage (S4)
    # ================================================================
    @property
    def extra_halo(self) -> HaloSpec:
        """Exempt the (global, spectral) 2D solve from the halo trace.

        Description
        -----------
        The CONSTRAINT wraps raw ``Fourier``/``.data`` arrays inside
        :class:`SpectralSolve`; it declares its FD-stencil halo here
        (one cell per horizontal coordinate — the staggered ``diff`` of
        the divergence RHS and the gradient correction) rather than
        being halo-traced (the nonhydro projection precedent, V-N2).
        """
        return HaloSpec(dict.fromkeys(self._horizontal, 1))

    @property
    def stages(self) -> tuple[fr.model.Stage, ...]:
        """The barotropic solve: replace ``u, v`` and write ``ps``."""
        return (
            fr.model.Stage(kind=fr.model.StageKind.CONSTRAINT,
                     fn="_barotropic_solve", name="barotropic_solve"),
        )

    def _barotropic_solve(
        self, state: object, ctx: StepContext,
    ) -> dict[str, object]:
        r"""Solve the Helmholtz ``ps^{n+1}``; correct ``(u, v)``.

        Description
        -----------
        With ``dt' = ctx.stage_dt`` and the post-advance depth-mean
        divergence ``\nabla_h\cdot\bar u^*``:

        .. math::

            (\varepsilon - dt'^2\,\nabla_h\!\cdot(c^2\nabla_h))\,
                p_s^{n+1}
              = \varepsilon\, p_s^{n} - dt'\, c^2\,
                \nabla_h\cdot\bar u^{*}, \\
            u^{n+1}(z) = u^{*}(z) - dt'\, \partial_x p_s^{n+1}, \quad
            v^{n+1}(z) = v^{*}(z) - dt'\, \partial_y p_s^{n+1}.

        The correction is z-uniform (the ConstantSpace ``ps`` gradient
        broadcast onto every level), so its depth-mean divergence
        exactly cancels the RHS divergence — for ``epsilon == 0`` the
        corrected depth-mean flow is non-divergent to machine precision.
        """
        csqr = ctx.params[CSQR]
        dt = ctx.stage_dt
        zonal, meridional = self._horizontal
        u, v = state["u"], state["v"]
        div_bar = self._depth_mean_div(state)
        # RHS: eps * ps_old - dt' * c^2 * div(ubar*)  (eps=0 drops ps)
        rhs = self._epsilon * state["ps"] - dt * csqr * div_bar
        ps_new = self._solve(rhs, csqr=csqr, dt=dt)
        # z-uniform correction: broadcast the ConstantSpace ps gradient
        # onto the velocity faces (the same C-grid diff the solve uses)
        grad_x = ps_new.diff(zonal).to(u).retag(u)
        grad_y = ps_new.diff(meridional).to(v).retag(v)
        return {
            "u": u - dt * grad_x,
            "v": v - dt * grad_y,
            "ps": ps_new,
        }

    def _solve(
        self, rhs: ScalarField, *, csqr: object, dt: object,
    ) -> ScalarField:
        r"""Invert ``(eps - dt'^2 div(c^2 grad))`` on the ``ps`` cell.

        Description
        -----------
        Builds the honest discrete C-grid ``Div @ Diag @ Grad`` pair on
        the 2D ``Profile`` solve space (the ConstantSpace z is dropped
        by the transform), folds the traced ``-dt'^2 c^2`` into the
        diagonal so the operator's own eigenvalue symbol is
        ``dt'^2 c^2 k_disc^2``, adds the static ``epsilon`` identity,
        and inverts through :class:`SpectralSolve`. For ``epsilon > 0``
        the symbol ``epsilon + dt'^2 c^2 k_disc^2`` is non-singular; for
        ``epsilon == 0`` the sole structural zero at ``k = 0`` is the
        Poisson null space, gauged by ``where_zero`` (the mean gauge).

        Parameters
        ----------
        rhs : ScalarField
            The right-hand side on the 2D ``Profile`` cell.
        csqr : object
            The live squared-phase-speed leaf ``c^2``.
        dt : object
            The stage increment ``dt' = ctx.stage_dt``.

        Returns
        -------
        ScalarField
            The surface pressure ``ps^{n+1}`` on ``rhs``'s space.
        """
        grid: Grid = rhs.grid
        solve_space: SpaceLike = rhs.function_space.bare
        axes = solve_space.active_axis_names
        # fold -dt'^2 c^2 into the metric between grad and div, so the
        # honest discrete symbol of Div @ Diag @ Grad is +dt'^2 c^2 k^2
        neg = -(dt**2) * csqr
        grad_block = Gradient().expand(solve_space, grid)
        mid = grad_block.codomains(solve_space)
        div_block = Divergence().expand(mid, grid)
        diag = Diag(dict.fromkeys(axes, neg), axes=axes)
        laplacian = (div_block @ diag @ grad_block).scalar()
        # + epsilon * I: 1 (Helmholtz, non-singular) or 0 (Poisson)
        helmholtz = laplacian + Identity() * float(self._epsilon)
        solve = SpectralSolve(helmholtz, grid, solve_space,
                              where_zero=0.0)
        return solve.solve(rhs)
