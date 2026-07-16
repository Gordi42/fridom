r"""Free-surface modules: the barotropic (surface-pressure) evolution.

Description
-----------
``ExplicitFreeSurface`` is the correctness-oracle free-surface variant
(the Oceananigans ``ExplicitFreeSurface`` analogue, HY-D3): it
declares the surface pressure ``ps = g*eta`` as a PROGNOSTIC field on
the constant-along-z ``Profile("x", "y")`` space (collocated in the
horizontal, one degree of freedom in the vertical — the barotropic
mode) and contributes the single linear term

.. math::

    \partial_t p_s = -c^2\, \nabla_h\cdot\bar{\boldsymbol{u}}, \qquad
    \bar{\boldsymbol{u}} = \frac{1}{H}\int_{-H}^{0}
        \boldsymbol{u}_h \, dz,

the shallow-water gravity term acting on the **depth mean** of the
horizontal velocity (``c^2 = g H``). The depth mean is the exact
measure-weighted ``u.mean("z")`` (``integrate("z")`` divided by the
total depth), so ``\int_{-H}^{0} u\, dz = H \bar u`` holds to
machine precision.

The gradient/divergence legs are the adjoint C-grid pair: the core's
pressure gradient ``-\nabla_h p_s`` staggers the cell-centred ``ps``
onto the ``u``/``v`` faces, and this divergence stages the face-valued
depth mean back onto the ``ps`` cell. Under the energy metric weight
``1/c^2`` on ``ps`` — integrated over the full depth (``hy.energy``) —
the pair is exactly skew-adjoint, so the linear barotropic gravity
wave conserves energy to machine round-off (the energy gate). It is
CFL-limited by ``sqrt(c^2)``; the implicit and split-explicit
variants (HY-D3) relax that limit and are built in later stages.
"""
from __future__ import annotations

from functools import partial

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.hydrostatic.params import CSQR
from fridom.spatial.operators.integrate import Integral


@partial(jaxify, dynamic=())
class ExplicitFreeSurface(fr.model.Module):

    r"""Declares ``ps``; the explicit barotropic gravity term.

    Parameters
    ----------
    vertical : str, optional
        The vertical coordinate name the depth mean reduces over
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
        """Store the geometry names of the barotropic solve."""
        horizontal = tuple(horizontal)
        if (len(horizontal) != 2  # noqa: PLR2004 — zonal + meridional
                or not all(isinstance(name, str) for name in horizontal)
                or horizontal[0] == horizontal[1]):
            raise TypeError(
                "horizontal names the (zonal, meridional) coordinates: "
                f"two distinct strings, got {horizontal!r}")
        self._vertical = vertical
        self._horizontal = horizontal
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

    @property
    def field_references(self) -> tuple[fr.model.FieldReference, ...]:
        """The depth mean rides the declared horizontal velocities."""
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

    @fr.model.term(advances=("ps",), linear=True)
    def gravity(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``d_t ps = -c^2 (d_x ubar + d_y vbar)`` (depth-mean divergence).

        The depth means ``ubar = u.mean(z)`` / ``vbar = v.mean(z)`` are
        the measure-exact ``integrate(z) / H`` reductions onto the
        barotropic ``ConstantSpace`` factor; their horizontal
        divergence lands on the ``ps`` cell (the adjoint of the core's
        ``-grad ps`` momentum forcing).
        """
        csqr = ctx.params[CSQR]
        zonal, meridional = self._horizontal
        # depth-mean divergence = (1/H) * integral_z(d_x u + d_y v);
        # reduce the *collocated* divergence field (the z-reduction and
        # the horizontal derivative commute, so this equals
        # d_x ubar + d_y vbar) -- the Integral runs on the same
        # cell-centred z-factor the DIAGNOSE cumint reduces, keeping
        # the barotropic gravity pair exactly adjoint to the core's
        # -grad ps momentum forcing under the depth-integrated energy
        # metric
        div_h = state["u"].diff(zonal) + state["v"].diff(meridional)
        div_bar = Integral()[self._vertical](div_h) * self._inv_depth
        return {"ps": -(csqr * div_bar)}
