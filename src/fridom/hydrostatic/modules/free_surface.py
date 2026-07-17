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

Three variants live here:

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

- ``SplitExplicitFreeSurface`` — the wave-resolving workhorse (H6, the
  frozen §5.4 design). ``ps`` and the barotropic transports ``U, V``
  (2D, constant along z; **no** Velocity role) are PROGNOSTIC. A
  module-owned ADVANCE stage subcycles the 2D barotropic
  shallow-water pair with a ``lax.scan`` over ``N`` forward-backward
  substeps (``dtau = 2 dt / N``), reads the slow (baroclinic) forcing
  ``G`` off the substage-start depth mean (buffered in own AUX by a
  SELF_UPDATE snapshot) or off ``ctx``'s per-treatment tendency sums,
  and commits the Shchepetkin-McWilliams (2005) time-averaged
  barotropic state. A CONSTRAINT stage then replaces the depth mean of
  ``u, v`` with the averaged ``U/H, V/H``. The barotropic coupling
  leaves the ``linear=True`` terms (it lives in the subcycle), so this
  variant also declares a ``linear_operator_gap`` (HY-D7); and because
  a per-stage-projected subcycle has no production precedent, it
  refuses a non-multistep outer driver (IMEX-RK / RK) at assembly
  (§5.4 "multistep outer drivers only").
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.framework.utils import dtype_real, jaxify
from fridom.hydrostatic.params import CSQR
from fridom.model.errors import AssemblyError
from fridom.model.terms import Treatment
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.operators.base import Identity, resolve_codomain
from fridom.spatial.operators.composed import Diag, Divergence, Gradient
from fridom.spatial.operators.integrate import Integral
from fridom.spatial.operators.krylov import ConjugateGradient
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
        # the immersed descriptor (None off a cut-cell grid), captured
        # at bind: an id-hashable static aux (like MaskState's), read to
        # switch the barotropic reductions onto their masked forms. The
        # per-column transport depths are materialized in-trace from it
        # (memoized concrete-only, like grid._measures) — a jax array
        # cannot ride a dynamic=() pytree's static aux
        self._immersed: object | None = None

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
        self._immersed = getattr(grid, "immersed", None)
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

        On an immersed grid this is the **transport-divergence** form
        (IP-D9): the horizontal fluxes are fraction-weighted before the
        vertical integral, so the quantity is
        ``(1/H)[d_x\!\int\alpha_x u\,dz + d_y\!\int\alpha_y v\,dz]`` —
        the volume-conserving barotropic divergence ``(1/H)\nabla\cdot U``
        with ``U`` the wet transport and ``H`` the **reference** depth
        (the mesh extent, still the scalar ``self._inv_depth``). The
        variable *column* depth enters the implicit operator's
        coefficient, not here (the mass-leak trap is the wet-depth-mean
        ``\nabla\cdot(U/H_{col})``, which is **not** volume-conserving —
        avoided). With ``\alpha == 1`` the multiply is the identity, so
        an all-wet immersed grid is byte-identical to the unimmersed
        form.

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
        u, v = state["u"], state["v"]
        if self._immersed is None:
            div_h = u.diff(zonal) + v.diff(meridional)
            return Integral()[self._vertical](div_h) * self._inv_depth
        alpha_x = self._immersed.fraction(u.function_space)
        alpha_y = self._immersed.fraction(v.function_space)
        div_h = ((alpha_x * u).diff(zonal)
                 + (alpha_y * v).diff(meridional))
        return Integral()[self._vertical](div_h) * self._inv_depth

    # ================================================================
    #  Immersed barotropic reductions (IP-D9; no-ops off a cut cell)
    # ================================================================
    def _transport_depth(self, field: ScalarField) -> ScalarField:
        r"""Return the wet transport depth ``H = \int\alpha\,dz`` on a face.

        Description
        -----------
        The per-column wet transport depth of a velocity component: the
        vertical integral of its **face** fraction (``alpha_x`` for
        ``u``, ``alpha_y`` for ``v`` — NOT the cell fraction ``theta``:
        transport-depth consistency is the mass-leak trap). Lands on the
        component's ``Profile`` face (constant along z). On an all-wet
        grid it is the reference depth ``H``; on a land column it is 0.
        """
        alpha = self._immersed.fraction(field.function_space)
        return Integral()[self._vertical](alpha)

    @staticmethod
    def _guarded_inverse(depth: ScalarField) -> ScalarField:
        """Return ``1/depth`` (a land column ``depth == 0`` maps to 0)."""
        d = depth.data
        wet = d > 0.0
        return depth.with_data(jnp.where(wet, 1.0 / jnp.where(wet, d, 1.0),
                                         0.0))

    def _wet_depth_mean(self, field: ScalarField) -> ScalarField:
        r"""Return the wet-depth mean ``(1/H)\int\alpha\,q\,dz`` of a face.

        Description
        -----------
        The barotropic velocity of a velocity component: the wet
        transport ``\int\alpha q\,dz`` divided by the per-column wet
        depth ``H`` (guarded on land columns). Off an immersed grid this
        is the ordinary ``field.mean(z)``.
        """
        if self._immersed is None:
            return field.mean(self._vertical)
        alpha = self._immersed.fraction(field.function_space)
        transport = Integral()[self._vertical](alpha * field)
        return transport * self._guarded_inverse(
            self._transport_depth(field))

    def _face_wet_mask(self, field: ScalarField) -> ScalarField:
        r"""Return the boolean open-face mask ``alpha > 0`` of a velocity face.

        The slip-independent correction gate (plan §2): a barotropic
        pressure-gradient correction never enters a closed
        (``alpha == 0``) face. A concrete ``{0, 1}`` field multiplied
        onto the z-uniform correction — field arithmetic (not ``.data``)
        so it survives the halo trace of the ``linear=True`` explicit
        term as well as the halo-exempt implicit CONSTRAINT.
        """
        alpha = self._immersed.fraction(field.function_space)
        return alpha.with_data((alpha.data > 0.0).astype(dtype_real()))


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
        walled axis). The adjoint of the ``gravity`` term. On an
        immersed grid the force is gated by the boolean open-face mask
        ``alpha > 0`` (IP-D9), so no barotropic force enters a closed
        face (a no-op when ``alpha == 1`` everywhere).
        """
        zonal, meridional = self._horizontal
        u, v = state["u"], state["v"]
        ps = state["ps"]
        grad_u = ps.diff(zonal).to(u)
        grad_v = ps.diff(meridional).to(v)
        if self._immersed is not None:
            grad_u = grad_u * self._face_wet_mask(u)
            grad_v = grad_v * self._face_wet_mask(v)
        return {
            "u": (-grad_u).retag(u),
            "v": (-grad_v).retag(v),
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

    **Immersed grids (IP-D9).** On a grid carrying an immersed domain
    the depth-varying operator ``eps eta - dt'^2 div(c^2 H_tilde grad)``
    (with the per-column normalized transport depth ``H_tilde``) is no
    longer separable, so the solve flips from the flat
    :class:`SpectralSolve` to a fixed-iteration
    :class:`ConjugateGradient` (:meth:`_solve_immersed`): the SPD
    flux-form operator, the flat mean-depth spectral inverse masked onto
    the wet columns as preconditioner, and the V-orthogonal wet-column
    projection for the rigid lid. The unimmersed path is unchanged
    (:meth:`_solve`).

    Parameters
    ----------
    epsilon : float, optional
        The pyOM free-surface knob (static, not sweepable): ``1.0`` is
        the backward-Euler linear free surface (symbol
        ``1 + c^2 dt'^2 k_disc^2``, non-singular); ``0.0`` is the rigid
        lid (the singular Poisson, ``where_zero`` mean gauge). Must be
        ``>= 0`` (default: 1.0).
    pressure_iterations : int, optional
        The fixed PCG iteration budget of the immersed barotropic solve
        (mirrors ``nh.Model(pressure_iterations=...)``); consumed only
        on an immersed grid — the flat spectral solve is exact and
        iterates nothing. Must be ``>= 1`` (default: 30).
    pressure_tolerance : float | None, optional
        An optional PCG convergence break forwarded to the immersed
        barotropic :class:`ConjugateGradient` (the measure-weighted
        true relative residual; masked scan, exact gradient — see its
        docstring). ``pressure_iterations`` becomes the maximum budget;
        ``None`` runs the fixed count (default: None).
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
        pressure_iterations: int = 30,
        pressure_tolerance: float | None = None,
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
        if (isinstance(pressure_iterations, bool)
                or not isinstance(pressure_iterations, int)
                or pressure_iterations < 1):
            raise ValueError(
                "pressure_iterations is the fixed PCG budget of the "
                "immersed barotropic solve (a positive int, consumed "
                f"only on an immersed grid), got {pressure_iterations!r}")
        self._epsilon = float(epsilon)
        self._pressure_iterations = int(pressure_iterations)
        self._pressure_tolerance = pressure_tolerance

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def epsilon(self) -> float:
        """The static free-surface knob (``0`` rigid lid, ``>0`` free)."""
        return self._epsilon

    @property
    def pressure_iterations(self) -> int:
        """The fixed PCG budget of the immersed barotropic solve."""
        return self._pressure_iterations

    @property
    def pressure_tolerance(self) -> float | None:
        """The optional PCG convergence break (None = fixed count)."""
        return self._pressure_tolerance

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
        """The barotropic solve: replace ``u, v`` and write ``ps``.

        Description
        -----------
        With a free surface (``epsilon > 0``) the stage claims
        ``advances=("ps",)``: the solve is the *only* thing
        integrating the PROGNOSTIC ``ps`` forward, so the claim is
        what satisfies the D1.4 coverage lint (spec 5.4 amendment).
        The velocity writes stay unclaimed — they are a correction,
        not an advance. Under the rigid lid ``ps`` is DIAGNOSTIC and
        nothing is claimed.
        """
        advances = ("ps",) if self._epsilon > 0 else ()
        return (
            fr.model.Stage(kind=fr.model.StageKind.CONSTRAINT,
                     fn="_barotropic_solve", name="barotropic_solve",
                     advances=advances),
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
        div_bar = self._depth_mean_div(state)  # masked on a cut cell
        # RHS: eps * ps_old - dt' * c^2 * div(ubar*)  (eps=0 drops ps)
        rhs = self._epsilon * state["ps"] - dt * csqr * div_bar
        if self._immersed is None:
            ps_new = self._solve(rhs, csqr=csqr, dt=dt)
        else:
            ps_new = self._solve_immersed(rhs, state, csqr=csqr, dt=dt)
        # z-uniform correction: broadcast the ConstantSpace ps gradient
        # onto the velocity faces (the same C-grid diff the solve uses).
        # On an immersed grid the boolean open-face mask keeps the
        # correction out of closed faces (alpha m = alpha, so the
        # corrected wet transport still matches the operator solve).
        grad_x = ps_new.diff(zonal).to(u)
        grad_y = ps_new.diff(meridional).to(v)
        if self._immersed is not None:
            grad_x = grad_x * self._face_wet_mask(u)
            grad_y = grad_y * self._face_wet_mask(v)
        return {
            "u": u - dt * grad_x.retag(u),
            "v": v - dt * grad_y.retag(v),
            "ps": ps_new,
        }

    def _flat_spectral(
        self, solve_space: SpaceLike, grid: Grid, *,
        csqr: object, dt: object,
    ) -> SpectralSolve:
        r"""Build the flat (mean-depth) ``(eps - dt'^2 div(c^2 grad))`` solve.

        Description
        -----------
        The honest discrete C-grid ``Div @ Diag @ Grad`` pair on the 2D
        ``Profile`` solve space (the ConstantSpace z is dropped by the
        transform), with ``-dt'^2 c^2`` folded into the diagonal so the
        operator's own symbol is ``dt'^2 c^2 k_disc^2``, plus the static
        ``epsilon`` identity, inverted through :class:`SpectralSolve`.
        On an unimmersed grid it is the whole solve; on an immersed grid
        it is the constant-coefficient (``H_tilde == 1``) preconditioner
        of the variable-coefficient PCG.
        """
        axes = solve_space.active_axis_names
        neg = -(dt**2) * csqr
        grad_block = Gradient().expand(solve_space, grid)
        mid = grad_block.codomains(solve_space)
        div_block = Divergence().expand(mid, grid)
        diag = Diag(dict.fromkeys(axes, neg), axes=axes)
        laplacian = (div_block @ diag @ grad_block).scalar()
        # + epsilon * I: 1 (Helmholtz, non-singular) or 0 (Poisson)
        helmholtz = laplacian + Identity() * float(self._epsilon)
        return SpectralSolve(helmholtz, grid, solve_space, where_zero=0.0)

    def _solve(
        self, rhs: ScalarField, *, csqr: object, dt: object,
    ) -> ScalarField:
        r"""Invert ``(eps - dt'^2 div(c^2 grad))`` on the ``ps`` cell.

        Description
        -----------
        The flat spectral solve (:meth:`_flat_spectral`): for
        ``epsilon > 0`` the symbol ``epsilon + dt'^2 c^2 k_disc^2`` is
        non-singular; for ``epsilon == 0`` the sole structural zero at
        ``k = 0`` is the Poisson null space, gauged by ``where_zero``
        (the mean gauge). The unimmersed constant-depth path.

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
        solve_space: SpaceLike = rhs.function_space.bare
        return self._flat_spectral(
            solve_space, rhs.grid, csqr=csqr, dt=dt).solve(rhs)

    def _solve_immersed(
        self, rhs: ScalarField, state: object, *,
        csqr: object, dt: object,
    ) -> ScalarField:
        r"""Invert the variable-coefficient barotropic Helmholtz (IP-D9).

        Description
        -----------
        On an immersed grid the depth-varying operator is
        ``eps eta - dt'^2 div(c^2 H_tilde grad eta)`` with the
        normalized transport depth ``H_tilde_a = H_a / H`` on the
        a-face (``H_a = int alpha_a dz`` the wet transport depth, ``H``
        the reference depth) — derived from the discrete barotropic
        continuity + momentum so that the ``u^{n+1} = u* - dt' grad ps``
        correction is exactly volume-consistent, and identical to
        :meth:`_solve` when ``H_tilde == 1``. The variable coefficient
        breaks separability, so it flips to the CS-D2 / IP-D6 route: the
        SPD flux-form operator built by explicit field arithmetic, a
        fixed-iteration :class:`ConjugateGradient`, the flat mean-depth
        spectral inverse (:meth:`_flat_spectral`) **masked onto the wet
        columns** as the preconditioner, and — for ``epsilon == 0`` (the
        singular rigid lid) — the V-orthogonal wet-column-mean nullspace
        projection. ``epsilon > 0`` is non-singular (no projection). On
        an all-wet grid the preconditioner is the exact inverse and the
        PCG converges in ~1 iteration.

        Parameters
        ----------
        rhs : ScalarField
            The masked barotropic right-hand side on the ``ps`` cell.
        state : object
            The current state (supplies the velocity face fractions and
            the cell fractions for the wet-column indicator).
        csqr : object
            The live squared-phase-speed leaf ``c^2``.
        dt : object
            The stage increment ``dt' = ctx.stage_dt``.

        Returns
        -------
        ScalarField
            The wet-column surface pressure ``ps^{n+1}``.
        """
        grid: Grid = rhs.grid
        solve_space: SpaceLike = rhs.function_space.bare
        axes = solve_space.active_axis_names
        registry = grid.dispatch
        zonal, meridional = self._horizontal
        vel = {zonal: state["u"], meridional: state["v"]}
        # per-axis gradient/divergence legs (doubly-periodic horizontal:
        # no wall Dirichlet tags) and the per-column coefficient
        # c^2 H_tilde_a = c^2 (1/H) H_a on the a-face
        grad: dict[str, object] = {}
        div: dict[str, object] = {}
        coeff: dict[str, ScalarField] = {}
        for a in axes:
            g = registry.resolve("diff", solve_space.factor(a))[a]
            face = resolve_codomain(g, solve_space)
            grad[a] = g
            div[a] = registry.resolve("diff", face.factor(a))[a]
            coeff[a] = self._transport_depth(vel[a]) * (
                csqr * self._inv_depth)

        def apply(ps: ScalarField) -> ScalarField:
            out = self._epsilon * ps
            for a in axes:
                flux = coeff[a] * grad[a](ps)
                out = out - (dt**2) * div[a](flux)
            return out

        spectral = self._flat_spectral(
            solve_space, grid, csqr=csqr, dt=dt)
        theta_col = Integral()[self._vertical](
            self._immersed.fraction(state["p_hyd"].function_space))
        cell_mask = (theta_col.data > 0.0).astype(dtype_real())
        wet = rhs.with_data(cell_mask)
        wet_measure = jnp.sum(wet.integrate().data)

        def precondition(r: ScalarField) -> ScalarField:
            z = spectral.solve(r)
            return z.with_data(z.data * cell_mask)

        def wet_projection(f: ScalarField) -> ScalarField:
            mean = jnp.sum((wet * f).integrate().data) / wet_measure
            return f - wet * mean

        # eps > 0 is non-singular (no projection); eps == 0 (rigid lid)
        # removes the V-orthogonal wet-column-constant nullspace
        projection = wet_projection if self._epsilon == 0.0 else None
        cg = ConjugateGradient(
            apply, preconditioner=precondition,
            iterations=self._pressure_iterations,
            tolerance=self._pressure_tolerance, projection=projection)
        ps_new = cg(rhs)
        return ps_new.with_data(ps_new.data * cell_mask)


# ================================================================
#  Split-explicit free surface (H6, the frozen §5.4 design)
# ================================================================
def _barotropic_zero(field: ScalarField) -> ScalarField:
    """Return a zero twin of a barotropic field (stored array).

    Zeros are built on the *stored* array (halos included, trivially
    valid), never as ``0.0 * field`` — the schedule ``zero_like``
    convention (no NaN poisoning, no arithmetic pass).
    """
    return type(field)(
        field.grid, field.function_space,
        jnp.zeros_like(field._data),  # noqa: SLF001 — plumbing-constructor seam
        field.metadata)


def _sm2005_weights(
    substeps: int, p: int, q: int, r: float,
) -> tuple[float, ...]:
    r"""Return the normalized SM2005 averaging weights (host-side).

    Description
    -----------
    The Shchepetkin-McWilliams (2005) barotropic averaging shape
    function, evaluated at the substep times
    :math:`\tau_m = m\,(2/N)`, :math:`m = 1 \ldots N` — the
    :math:`\tau \in [0, 2]` window discretized by ``substeps`` intervals
    (so :math:`\tau = 1` is one baroclinic step). With
    :math:`\tau_0 = (p+2)(p+q+2) / ((p+1)(p+q+1))` and
    :math:`x = \tau/\tau_0`,

    .. math::

        A(\tau) = x^p\,(1 - x^q) - r\,x ,

    clamped to its positive central lobe and normalized to sum 1. The
    unit sum makes the barotropic mean conserved (a weighted average of
    substep states whose means each drift only by a periodic divergence,
    which sums to zero); the discrete first moment
    :math:`\sum_m \tau_m\,a_m \approx 1` centers the average at
    :math:`\tau = 1`. This is the Oceananigans normalization (weights
    from the shape function over ``substeps`` points, normalized to
    sum 1); the discrete first moment converges to 1 as ``substeps``
    grows (measured 0.996 at N=8, 1.002 at N=16, 0.999 at N=64).

    Parameters
    ----------
    substeps : int
        The subcycle length ``N`` (the window discretization).
    p : int
        The rising-edge exponent (Oceananigans default 2).
    q : int
        The falling-edge exponent (Oceananigans default 4).
    r : float
        The small linear correction (Oceananigans default 0.18927).

    Returns
    -------
    tuple[float, ...]
        The ``substeps`` normalized weights (leading/trailing zeros
        outside the positive lobe), summing to 1.

    Raises
    ------
    ValueError
        If no substep falls in the shape function's positive lobe.
    """
    m = np.arange(1, substeps + 1, dtype=np.float64)
    tau = m * (2.0 / substeps)
    tau0 = (p + 2) * (p + q + 2) / ((p + 1) * (p + q + 1))
    x = tau / tau0
    shape = x**p * (1.0 - x**q) - r * x
    weights = np.where(shape > 0.0, shape, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        raise ValueError(
            "the SM2005 averaging window has no positive weight "
            f"(substeps={substeps}, filter=({p}, {q}, {r})): the shape "
            "function is non-positive at every substep. Use more "
            "substeps (>= ~4) or the default filter (2, 4, 0.18927)")
    return tuple(float(w) for w in weights / total)


@partial(jaxify, dynamic=())
class SplitExplicitFreeSurface(_FreeSurfaceBase):

    r"""Declares ``ps, U, V``; the barotropic subcycle + correction (H6).

    Description
    -----------
    The split-explicit free surface (the frozen §5.4 design). The
    surface pressure ``ps`` and the barotropic transports ``U, V``
    (2D, constant along z, on the C-grid faces — ``U`` staggered along
    the zonal axis, ``V`` along the meridional, both ``ConstantSpace``
    in z) are PROGNOSTIC. ``U, V`` carry **no** Velocity role (they are
    diagnostically-slaved transports; ``table.velocity()`` keeps
    returning the baroclinic trio ``u, v, w``).

    The module owns three stages, all halo-trace exempt (``extra_halo``,
    the V-N2 precedent — the subcycle's C-grid stencils and the depth
    means are declared, not traced):

    - **SELF_UPDATE** (S1) snapshots the substage-start depth-mean
      velocity ``(\bar u, \bar v)`` into the own-AUX buffers
      ``ubar_prev, vbar_prev`` — the V-H4 increment reference.
    - **ADVANCE** (S3') subcycles the 2D barotropic shallow-water pair
      with a ``lax.scan`` over ``N`` forward-backward substeps of size
      ``dtau = 2 dt / N``:

      .. math::

          p_s \leftarrow p_s - d\tau\, c^2\,
              (\partial_x \bar u + \partial_y \bar v), \\
          \bar u \leftarrow \bar u
              - d\tau\, \partial_x p_s + d\tau\, G_u ,

      (and likewise for ``\bar v``) — the exact linearization of
      ``ExplicitFreeSurface``'s two terms, plus the slow forcing ``G``.
      The forward-backward pair (new ``p_s`` in the velocity update) is
      stable to a wider CFL than forward-forward. The
      Shchepetkin-McWilliams (2005)-weighted average of the substep
      trajectory (``filter=(p, q, r)``, host-computed at construction,
      static under the scan) is committed as ``ps, U, V``.
    - **CONSTRAINT** (S4) replaces the depth mean of ``u, v`` with the
      averaged ``U/H, V/H`` (a z-uniform correction, the same broadcast
      as the implicit variant), injecting the resolved barotropic mode.

    **Slow forcing** ``G`` (the depth mean of the baroclinic tendencies
    driving the barotropic momentum, held fixed across the subcycle):

    - ``forcing="increment"`` (default, V-H4):
      ``G = (\bar u^* - \bar u_{\rm start}) / dt`` with ``\bar u^*`` the
      post-advance depth mean and ``\bar u_{\rm start}`` the SELF_UPDATE
      snapshot — automatically consistent with the outer scheme's
      weights, warm-up row and any implicit-mixing increment (they are
      all in ``\bar u^*``).
    - ``forcing="tendency_sums"``: ``G`` is the depth mean of ``ctx``'s
      per-treatment tendency sums (EXPLICIT, plus IMPLICIT when the
      scheme populates it) — the raw first-order variant.

    Because a per-stage-projected barotropic subcycle has no production
    precedent, ``bind`` refuses a non-multistep outer driver (an RK /
    IMEX-RK stepper, whose ``supports_split_advance`` is ``False``) with
    a taught assembly error (§5.4 "multistep outer drivers only").

    The barotropic coupling leaves the ``linear=True`` terms (it lives
    in the subcycle and the correction), so the module declares a
    ``linear_operator_gap`` (HY-D7).

    The integrator statics (``substeps``, the ``filter`` triple and the
    ``forcing`` choice) are encoded into the ADVANCE stage's attribution
    name, so they join the restart fingerprint through the schedule
    (V-H3): a snapshot written by one configuration refuses to resume a
    structurally different one.

    Parameters
    ----------
    substeps : int, optional
        The subcycle length ``N``: the number of forward-backward
        substeps discretizing the SM2005 averaging window
        ``\tau \in [0, 2]``, so the physical substep is
        ``dtau = 2 dt / N`` and the barotropic CFL is
        ``sqrt(c^2)\,2 dt / (N dx)``. Must be ``>= 2`` (default: 16).
    filter : tuple[int, int, float], optional
        The SM2005 shape-function parameters ``(p, q, r)``; the
        Oceananigans defaults ``(2, 4, 0.18927)`` (default:
        ``(2, 4, 0.18927)``).
    forcing : {"increment", "tendency_sums"}, optional
        The slow-forcing convention (default: ``"increment"``).
    vertical : str, optional
        The vertical coordinate the depth mean reduces over
        (default: ``"z"``).
    horizontal : tuple[str, str], optional
        The (zonal, meridional) coordinate names (default:
        ``("x", "y")``).

    Raises
    ------
    ValueError
        If ``substeps < 2``, the ``filter`` is malformed or yields no
        positive averaging weight, or ``forcing`` is unknown.
    """

    #: the honesty gate (HY-D7): this variant's barotropic coupling is
    #: NOT in L (``fr.model.require_linear_operator``)
    linear_operator_gap = (
        "the barotropic gravity coupling (the surface-pressure update "
        "and the -grad(ps) momentum force) lives inside the "
        "split-explicit ADVANCE-stage subcycle and the depth-mean "
        "CONSTRAINT, NOT in a linear=True term, so the linear operator "
        "L has NO surface-pressure mode at all. Assemble "
        "hy.ExplicitFreeSurface instead (the same barotropic physics as "
        "explicit linear terms, L complete) if you need eigenmodes, "
        "projections, optimal balance or IMEX-by-linearity")

    def __init__(
        self,
        *,
        substeps: int = 16,
        filter: tuple[int, int, float] = (2, 4, 0.18927),  # noqa: A002 — §5.4 keyword
        forcing: str = "increment",
        vertical: str = "z",
        horizontal: tuple[str, str] = ("x", "y"),
    ) -> None:
        """Validate the integrator statics; store the geometry names."""
        super().__init__(vertical=vertical, horizontal=horizontal)
        if (isinstance(substeps, bool) or not isinstance(substeps, int)
                or substeps < 2):  # noqa: PLR2004 — a subcycle needs >= 2
            raise ValueError(
                "substeps is the barotropic subcycle length N (>= 2, "
                "discretizing the tau in [0, 2] averaging window), got "
                f"{substeps!r}")
        try:
            p, q, r = filter
        except (TypeError, ValueError):
            raise ValueError(
                "filter is the SM2005 shape triple (p, q, r), e.g. the "
                f"default (2, 4, 0.18927), got {filter!r}") from None
        if (isinstance(p, bool) or isinstance(q, bool)
                or not isinstance(p, int) or not isinstance(q, int)
                or p < 1 or q < 1 or isinstance(r, bool)
                or not isinstance(r, int | float)):
            raise ValueError(
                "filter is the SM2005 shape triple (p, q, r) with p, q "
                f"positive integers and r a real, got {filter!r}")
        if forcing not in ("increment", "tendency_sums"):
            raise ValueError(
                "forcing is the barotropic slow-forcing convention: "
                "'increment' (the V-H4 default) or 'tendency_sums', got "
                f"{forcing!r}")
        self._substeps = substeps
        self._filter = (int(p), int(q), float(r))
        self._forcing = forcing
        # host-side SM2005 weights: a hashable tuple, static under the
        # scan and treedef-safe (the module is a dynamic=() pytree)
        self._weights = _sm2005_weights(substeps, int(p), int(q),
                                        float(r))

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def substeps(self) -> int:
        """The static subcycle length ``N`` (fingerprinted)."""
        return self._substeps

    @property
    def sm_filter(self) -> tuple[int, int, float]:
        """The static SM2005 ``(p, q, r)`` triple (fingerprinted)."""
        return self._filter

    @property
    def forcing(self) -> str:
        """The static slow-forcing convention (fingerprinted)."""
        return self._forcing

    @property
    def weights(self) -> tuple[float, ...]:
        """The normalized SM2005 averaging weights (host, sum 1)."""
        return self._weights

    # ================================================================
    #  Bind: the multistep-only guard (§5.4)
    # ================================================================
    def bind(self, table: object) -> None:
        """Freeze ``1/H`` and refuse a non-multistep outer driver.

        Description
        -----------
        Runs the base depth-mean freeze, then the §5.4 assembly guard:
        a per-stage-projected barotropic subcycle has no production
        precedent under an RK / IMEX-RK outer driver, so the split
        free surface is a **multistep-only** stage. The stepper's
        ``supports_split_advance`` capability (set by ``AdamBashforth``
        and ``IMEXMultistep``, false for the RK and exponential
        families) is read off the bind table; a non-multistep driver is
        a taught assembly error, never a silent mis-integration.

        Parameters
        ----------
        table : object
            The binding table (carries the grid and the outer stepper).

        Raises
        ------
        AssemblyError
            If the assembly's outer stepper is not a multistep driver.
        """
        super().bind(table)
        stepper = getattr(table, "time_stepper", None)
        if not getattr(stepper, "supports_split_advance", False):
            raise AssemblyError(
                f"SplitExplicitFreeSurface refuses the outer stepper "
                f"{type(stepper).__name__}: the split-explicit "
                "barotropic subcycle is a "
                "multistep-only stage (§5.4 'multistep outer drivers "
                "only') — a per-stage-projected subcycle under an "
                "RK / IMEX-RK driver has no production precedent. "
                "Assemble with AdamBashforth or an IMEX multistep "
                "scheme (fr.model.time_steppers.CNAB2 / SBDF2), or use "
                "hy.ImplicitFreeSurface (which composes with any "
                "stepper)")

    # ================================================================
    #  Field declarations
    # ================================================================
    def _transport_space(
        self, staggered: str, collocated: str,
    ) -> object:
        """Return the C-grid transport pattern (2D, constant in z).

        ``staggered`` lands on the face (``Dof.STAGGERED``),
        ``collocated`` on the centre, every other coordinate (the
        vertical) on ``Dof.CONSTANT`` — exactly the space the depth mean
        ``u.mean(z)`` and ``ps.diff(staggered)`` land on (verified at
        assembly through the ``replace`` space check).
        """
        return fr.spatial.SpacePattern.create(
            default=fr.spatial.Dof.CONSTANT,
            tags={staggered: fr.spatial.Dof.STAGGERED,
                  collocated: fr.spatial.Dof.COLLOCATED})

    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """``ps`` (centre), ``U, V`` (faces) PROGNOSTIC; the AUX buffers.

        Description
        -----------
        ``ps`` on ``Profile(zonal, meridional)`` (the barotropic mode);
        ``U`` on the zonal face, ``V`` on the meridional face (the
        constant-along-z transports, **no** Velocity role); and the two
        own-AUX depth-mean buffers ``ubar_prev, vbar_prev`` (the V-H4
        increment reference, snapshotted every substage by the
        SELF_UPDATE stage — zero-initialized, overwritten before read).
        """
        zonal, meridional = self._horizontal
        return (
            fr.model.FieldDeclaration(
                "ps", space=fr.spatial.Profile(zonal, meridional),
                lifecycle=fr.model.Lifecycle.PROGNOSTIC,
                long_name="Surface pressure (g*eta)",
                units="m^2/s^2"),
            fr.model.FieldDeclaration(
                "U", space=self._transport_space(zonal, meridional),
                lifecycle=fr.model.Lifecycle.PROGNOSTIC,
                long_name="Barotropic zonal transport", units="m^2/s"),
            fr.model.FieldDeclaration(
                "V", space=self._transport_space(meridional, zonal),
                lifecycle=fr.model.Lifecycle.PROGNOSTIC,
                long_name="Barotropic meridional transport",
                units="m^2/s"),
            fr.model.FieldDeclaration(
                "ubar_prev",
                space=self._transport_space(zonal, meridional),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                long_name="Substage-start depth-mean zonal velocity",
                units="m/s"),
            fr.model.FieldDeclaration(
                "vbar_prev",
                space=self._transport_space(meridional, zonal),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                long_name="Substage-start depth-mean meridional velocity",
                units="m/s"),
        )

    # ================================================================
    #  Stages (halo-trace exempt: the C-grid stencils are declared)
    # ================================================================
    @property
    def extra_halo(self) -> HaloSpec:
        """Declare the C-grid stencil halo; exempt the stages from trace.

        One cell per horizontal coordinate — the staggered ``diff`` of
        the barotropic divergence and pressure gradient inside the
        subcycle. Declaring it (the V-N2 precedent) exempts the module's
        stages from the halo trace, so the depth-mean reductions
        (``u.mean(z)``, halo-tracer-incompatible sugar) and the scan
        run on real fields.
        """
        return HaloSpec(dict.fromkeys(self._horizontal, 1))

    @property
    def _advance_name(self) -> str:
        """The ADVANCE attribution key that carries the statics (V-H3).

        Encoding ``substeps``/``filter``/``forcing`` into the stage name
        threads them into the schedule token and thus the restart
        fingerprint: changing any of them changes the fingerprint.
        """
        p, q, r = self._filter
        r_tag = format(r, ".6g").replace(".", "p").replace("-", "m")
        return (f"barotropic_advance__n{self._substeps}"
                f"__sm{p}_{q}_{r_tag}__{self._forcing}")

    @property
    def stages(self) -> tuple[fr.model.Stage, ...]:
        """SELF_UPDATE snapshot, ADVANCE subcycle, CONSTRAINT correction."""
        return (
            fr.model.Stage(
                kind=fr.model.StageKind.SELF_UPDATE,
                fn="_snapshot_barotropic", name="barotropic_snapshot",
                reads=("u", "v")),
            fr.model.Stage(
                kind=fr.model.StageKind.ADVANCE,
                fn="_barotropic_subcycle", name=self._advance_name,
                advances=("ps", "U", "V")),
            fr.model.Stage(
                kind=fr.model.StageKind.CONSTRAINT,
                fn="_correct_depth_mean", name="barotropic_correction"),
        )

    def _snapshot_barotropic(
        self, state: object, ctx: StepContext,  # noqa: ARG002
    ) -> dict[str, object]:
        r"""Buffer the substage-start depth mean into own AUX (S1).

        ``ubar_prev = \bar u`` / ``vbar_prev = \bar v`` — the V-H4
        increment reference, read at S3'. SELF_UPDATE runs first in the
        substage (S1), so the snapshot is the depth mean **before** the
        primary advance touches ``u, v``. On an immersed grid the mean
        is the wet-depth mean ``(1/H_col)\int\alpha q\,dz`` (IP-D9), so
        the increment forcing stays transport-depth consistent.
        """
        return {
            "ubar_prev": self._wet_depth_mean(state["u"]),
            "vbar_prev": self._wet_depth_mean(state["v"]),
        }

    def _barotropic_subcycle(
        self, state: object, ctx: StepContext,
    ) -> dict[str, object]:
        r"""Subcycle the barotropic pair; commit the SM2005 average (S3').

        Description
        -----------
        Reads the slow forcing ``G`` (increment or tendency-sums), takes
        ``N`` forward-backward substeps of size ``dtau = 2 dt / N`` from
        the prognostic barotropic state ``(ps, U/H, V/H)``, and commits
        the SM2005 time-averaged ``ps, U, V``. The primary advance (S3)
        does not touch ``ps, U, V`` (they carry no tendency term), so the
        prognostic ``U, V`` read here are the previous step's committed
        transports and ``U/H`` is the substage-start barotropic velocity.
        """
        csqr = ctx.params[CSQR]
        dt = ctx.stage_dt
        zonal, meridional = self._horizontal
        g_u, g_v = self._slow_forcing(state, ctx, dt)
        ps0 = state["ps"]
        immersed = self._immersed is not None
        if immersed:
            # per-column transport depths H_u, H_v on the U/V faces; the
            # barotropic velocity is U/H_col, the transport U = H_col ubar
            # (transport-depth consistent), and the ps forward step is the
            # volume-conserving transport divergence (1/H) div(H ubar).
            # A land-column face (H == 0) is closed (open-face gate).
            depth_u = self._transport_depth(state["u"])
            depth_v = self._transport_depth(state["v"])
            inv_u = self._guarded_inverse(depth_u)
            inv_v = self._guarded_inverse(depth_v)
            fmask_u = depth_u.with_data(
                (depth_u.data > 0.0).astype(dtype_real()))
            fmask_v = depth_v.with_data(
                (depth_v.data > 0.0).astype(dtype_real()))
            ubar0 = state["U"] * inv_u
            vbar0 = state["V"] * inv_v
        else:
            inv_h = self._inv_depth
            ubar0 = state["U"] * inv_h
            vbar0 = state["V"] * inv_h
        dtau = 2.0 * dt / self._substeps
        weights = jnp.asarray(self._weights, dtype=dtype_real())

        def body(
            carry: tuple, weight: object,
        ) -> tuple[tuple, None]:
            ps_c, ub_c, vb_c, aps, au, av = carry
            # forward: ps from the OLD barotropic-velocity divergence
            if immersed:
                div = self._inv_depth * (
                    (depth_u * ub_c).diff(zonal)
                    + (depth_v * vb_c).diff(meridional))
            else:
                div = ub_c.diff(zonal) + vb_c.diff(meridional)
            ps_n = ps_c - dtau * csqr * div
            # backward: velocity from the NEW ps + the slow forcing
            grad_u = ps_n.diff(zonal).retag(ub_c)
            grad_v = ps_n.diff(meridional).retag(vb_c)
            if immersed:
                grad_u = grad_u * fmask_u
                grad_v = grad_v * fmask_v
            ub_n = ub_c - dtau * grad_u + dtau * g_u
            vb_n = vb_c - dtau * grad_v + dtau * g_v
            return ((ps_n, ub_n, vb_n, aps + weight * ps_n,
                     au + weight * ub_n, av + weight * vb_n), None)

        init = (ps0, ubar0, vbar0, _barotropic_zero(ps0),
                _barotropic_zero(ubar0), _barotropic_zero(vbar0))
        (_ps, _ub, _vb, ps_avg, ubar_avg, vbar_avg), _ = jax.lax.scan(
            body, init, weights)
        if immersed:
            return {"ps": ps_avg, "U": ubar_avg * depth_u,
                    "V": vbar_avg * depth_v}
        inv_h = self._inv_depth
        return {"ps": ps_avg, "U": ubar_avg / inv_h,
                "V": vbar_avg / inv_h}

    def _slow_forcing(
        self, state: object, ctx: StepContext, dt: object,
    ) -> tuple[object, object]:
        r"""Return the depth-mean slow forcing ``(G_u, G_v)`` on the faces.

        The increment form (default) reads the post-advance depth mean
        against the SELF_UPDATE snapshot:
        ``G = (\bar u^* - \bar u_{\rm start}) / dt`` — scheme-consistent
        by construction. The tendency-sums form reads the per-treatment
        summed tendencies from ``ctx`` (EXPLICIT plus IMPLICIT where the
        scheme populated the forward apply) and takes their depth mean.
        """
        if self._forcing == "increment":
            g_u = (self._wet_depth_mean(state["u"])
                   - state["ubar_prev"]) / dt
            g_v = (self._wet_depth_mean(state["v"])
                   - state["vbar_prev"]) / dt
            return g_u, g_v
        sums = ctx.tendency_sums
        du = sums[Treatment.EXPLICIT]["u"]
        dv = sums[Treatment.EXPLICIT]["v"]
        try:
            implicit = sums[Treatment.IMPLICIT]
        except KeyError:
            implicit = None
        if implicit is not None:
            du = du + implicit["u"]
            dv = dv + implicit["v"]
        return self._wet_depth_mean(du), self._wet_depth_mean(dv)

    def _correct_depth_mean(
        self, state: object, ctx: StepContext,  # noqa: ARG002
    ) -> dict[str, object]:
        r"""Replace the depth mean of ``u, v`` with ``U/H, V/H`` (S4).

        The z-uniform correction ``u(z) \leftarrow u(z) - (\bar u -
        U/H)`` broadcasts the barotropic increment onto every level (the
        ConstantSpace lift in ``.to``, the implicit variant's pattern),
        so the corrected depth mean is exactly ``U/H`` while the
        baroclinic shear is untouched.
        """
        u, v = state["u"], state["v"]
        if self._immersed is None:
            vertical = self._vertical
            inv_h = self._inv_depth
            du = (u.mean(vertical) - state["U"] * inv_h).to(u).retag(u)
            dv = (v.mean(vertical) - state["V"] * inv_h).to(v).retag(v)
            return {"u": u - du, "v": v - dv}
        # immersed: set the wet-depth mean to the barotropic velocity
        # U/H_col (per-column, transport-depth consistent); the z-uniform
        # correction is gated to the open faces so it never enters a
        # closed face (a land column stays 0: H == 0 -> U/H == 0).
        inv_u = self._guarded_inverse(self._transport_depth(u))
        inv_v = self._guarded_inverse(self._transport_depth(v))
        du = (self._wet_depth_mean(u) - state["U"] * inv_u).to(u).retag(u)
        dv = (self._wet_depth_mean(v) - state["V"] * inv_v).to(v).retag(v)
        return {
            "u": u - du * self._face_wet_mask(u),
            "v": v - dv * self._face_wet_mask(v),
        }
