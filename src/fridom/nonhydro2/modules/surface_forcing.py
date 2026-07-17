r"""
Nonhydrostatic surface forcing: ``WindStress`` and ``SurfaceBuoyancyFlux``.

Description
-----------
The model-package wrappers that own the oceanographic sign conventions
(BF-D4), built on the generic ``fr.modules.BoundaryFlux`` machinery. The
generic module keeps the axis-direction flux convention (a positive flux
transports the quantity in ``+coord``); these wrappers give users the
physical signs — a positive wind stress accelerates the surface flow in
its own direction, a positive buoyancy flux adds buoyancy to the
wall-adjacent water — and reuse the shared wall-weight builder, flux
profiles, and bind-time validation rather than duplicating them.

- ``WindStress(tau_x, tau_y, coord, side, scale)`` is **one** module with
  terms on ``u`` and ``v``: a positive ``tau_x`` accelerates the surface
  flow in ``+x``. The kinematic stress :math:`\tau/\rho_0` (model units)
  enters the wall-adjacent cell as :math:`\partial_t u = s(t)\,\tau_x\,W`,
  which for a right/top wall is exactly the generic
  ``BoundaryFlux("u", coord, side, flux=-tau_x)`` (internally
  :math:`q = -\tau`). The scale is published as
  ``wind_stress.<coord>_<side>.scale``.
- ``SurfaceBuoyancyFlux(q, coord, side, scale)`` is a thin subclass of
  ``BoundaryFlux`` on ``b``: a positive ``q`` is a buoyancy *gain* at the
  wall, so internally ``flux = sign(side) * q`` (``-q`` at a right/top
  wall). The scale is published as
  ``surface_buoyancy_flux.<coord>_<side>.scale``.

Both scale names carry the ``(coord, side)`` so instances at opposite
walls coexist (heating the top and cooling the bottom — the
Rayleigh-Benard idealization; symmetric wind stress at two walls
likewise), while duplicates on one ``(coord, side)`` still collide (the
declaration-collision error — intended). The bespoke class prefix keeps
``model.update_parameters`` names matching the class the user
instantiated.
"""
from __future__ import annotations

import inspect
import numbers
from functools import partial
from typing import TYPE_CHECKING

from fridom.framework.utils import jaxify
from fridom.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.model.module import Module
from fridom.model.modules.boundary_flux import (
    _SIGN,
    BoundaryFlux,
    build_wall_weight,
    check_forced_field,
    check_tangential_flux,
    check_walled_coord,
    flux_declaration,
    reject_chart_grid,
)
from fridom.model.parameters import ParameterDeclaration, leaf
from fridom.model.params import ParamName
from fridom.model.terms import TendencyTerm, Treatment
from fridom.spatial.space_patterns import Profile

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.model.time_dependent import TimeDependent
    from fridom.spatial.fields.scalar_field import ScalarField

_VELOCITY_HINT = ("the velocities are declared by the dynamical core "
                  "(nh.DynamicalCore declares u, v, w)")


def _signed_flux(factor: float, flux: float | Callable) -> float | Callable:
    """Return ``factor * flux``, preserving a callable's coordinate names.

    A number is scaled directly; a tangential profile callable is
    wrapped so it still varies along the coordinates its signature names
    (the flux-profile normalization inspects that signature).
    """
    if callable(flux):
        def scaled(**coords: object) -> object:
            return factor * flux(**coords)
        scaled.__signature__ = inspect.signature(flux)
        return scaled
    return factor * flux


@partial(jaxify, dynamic=("scale",))
class WindStress(Module):

    r"""
    Kinematic wind stress at a wall: forces ``u`` and ``v`` together.

    Description
    -----------
    One module contributing, in the wall-adjacent cell,

    .. math::
        \partial_t u = s(t)\,\tau_x(\boldsymbol{x}_\parallel)\,W ,
        \qquad
        \partial_t v = s(t)\,\tau_y(\boldsymbol{x}_\parallel)\,W ,

    with :math:`W = 1/\Delta n` the shared wall weight. A positive
    ``tau_x`` accelerates the surface flow in ``+x`` (a positive
    ``tau_y`` in ``+y``) regardless of the wall — the oceanographic
    convention (BF-D4). The stresses are kinematic (:math:`\tau/\rho_0`,
    model units), numbers or callables of the tangential coordinate
    names. The scale is published as the dynamic-leaf parameter
    ``wind_stress.<coord>_<side>.scale`` (``fr.Ramp``- /
    ``TimeDependent``-capable), so ``model.update_parameters`` sweeps it
    without re-assembly and instances at opposite walls coexist. For a
    right/top wall the ``u`` term equals
    ``fr.modules.BoundaryFlux("u", coord, side, flux=-tau_x)``.

    Parameters
    ----------
    tau_x : float | Callable, optional
        Kinematic zonal stress; a positive value accelerates ``+x``
        (default: 0.0).
    tau_y : float | Callable, optional
        Kinematic meridional stress; a positive value accelerates ``+y``
        (default: 0.0).
    coord : str, optional
        The bounded axis the stress enters through (default: "z").
    side : str, optional
        The wall, ``"left"`` or ``"right"`` (default: "right", the top).
    scale : float | fr.Ramp, optional
        The shared time scale :math:`s(t)` (default: 1.0).
    """

    def __init__(
        self,
        tau_x: float | Callable = 0.0,
        tau_y: float | Callable = 0.0,
        coord: str = "z",
        side: str = "right",
        scale: float | TimeDependent = 1.0,
    ) -> None:
        """Freeze the stresses and geometry; store the scale leaf."""
        if side not in _SIGN:
            raise ValueError(
                f"WindStress side must be 'left' or 'right', got "
                f"{side!r}")
        for name, value in (("tau_x", tau_x), ("tau_y", tau_y)):
            if not (isinstance(value, numbers.Number)
                    or callable(value)):
                raise TypeError(
                    f"WindStress {name} must be a number or a "
                    f"tangential profile callable, got {value!r}")
        self._tau_x: float | Callable = tau_x
        self._tau_y: float | Callable = tau_y
        self._coord: str = coord
        self._side: str = side
        self.scale = leaf(scale)
        tag = f"{coord}_{side}"
        self._weight_name: str = f"windstress_{tag}_weight"
        self._taux_name: str = f"windstress_{tag}_taux"
        self._tauy_name: str = f"windstress_{tag}_tauy"
        self._scale_name: ParamName = ParamName(
            f"wind_stress.{tag}.scale", units="n/a",
            hint="provided by the nh.WindStress instance at the "
                 f"{coord} {side} wall")

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def coord(self) -> str:
        """The bounded axis the stress enters through."""
        return self._coord

    @property
    def side(self) -> str:
        """The forced wall (``"left"`` or ``"right"``)."""
        return self._side

    @property
    def scale_parameter(self) -> ParamName:
        """The dotted name of the provided scale parameter."""
        return self._scale_name

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_references(self) -> tuple[FieldReference, ...]:
        """The checked claims on the forced velocities."""
        return (FieldReference("u", hint=_VELOCITY_HINT),
                FieldReference("v", hint=_VELOCITY_HINT))

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The shared wall weight and the two stress profiles."""
        return (
            FieldDeclaration(
                self._weight_name, space=Profile(self._coord),
                lifecycle=Lifecycle.AUXILIARY,
                default=self._wall_weight_default,
                long_name="Wind-stress wall weight", units="1/m"),
            flux_declaration(self._taux_name, self._tau_x,
                             long_name="Kinematic zonal wind stress"),
            flux_declaration(self._tauy_name, self._tau_y,
                             long_name="Kinematic meridional wind stress"),
        )

    @property
    def parameter_declarations(
        self,
    ) -> tuple[ParameterDeclaration, ...]:
        """Publish the live shared scale leaf."""
        return (
            ParameterDeclaration(
                self._scale_name, attr="scale", units="n/a",
                doc="wind-stress time scale"),
        )

    def _wall_weight_default(
        self, grid: object, space: object,
    ) -> ScalarField:
        r"""Owner-method default: ``1/\Delta n`` at the wall-adjacent cell."""
        return build_wall_weight(
            grid, space, self._coord, self._side, self._weight_name)

    # ================================================================
    #  Bind-time validation (reuses the BoundaryFlux taught errors)
    # ================================================================
    def bind(self, table: object) -> None:
        """Validate the wall, the stresses, and the forced velocities."""
        grid = table.grid
        reject_chart_grid(grid, "WindStress")
        check_walled_coord(grid, self._coord, "WindStress")
        check_tangential_flux(grid, self._coord, self._tau_x, "WindStress")
        check_tangential_flux(grid, self._coord, self._tau_y, "WindStress")
        check_forced_field(table, "u", self._coord, self._side,
                           "WindStress")
        check_forced_field(table, "v", self._coord, self._side,
                           "WindStress")

    # ================================================================
    #  The wind-stress term (u and v together; the wrapper's own sign)
    # ================================================================
    def tendency_terms(self) -> tuple[TendencyTerm, ...]:
        """One term forcing u and v at the wall."""
        return (
            TendencyTerm(
                name="wind_stress", fn=self._stress_term,
                treatment=Treatment.EXPLICIT, advances=("u", "v")),
        )

    def _stress_term(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``du/dt = s tau_x W``; ``dv/dt = s tau_y W`` (wrapper sign).

        The stress drives the flow in its own direction on either wall,
        so the term carries no side sign (unlike the generic
        ``BoundaryFlux``); on a right/top wall the ``u`` term coincides
        with ``BoundaryFlux("u", coord, side, flux=-tau_x)``. The scale
        is read from ``ctx.params``; the weight and stress profiles ride
        onto the velocity faces with ``.to``.
        """
        scale = ctx.params[self._scale_name]
        u, v = state["u"], state["v"]
        weight = state[self._weight_name]
        tau_x = state[self._taux_name]
        tau_y = state[self._tauy_name]
        return {
            "u": scale * tau_x.to(u) * weight.to(u),
            "v": scale * tau_y.to(v) * weight.to(v),
        }


class SurfaceBuoyancyFlux(BoundaryFlux):

    r"""
    Surface buoyancy flux on ``b``: a positive ``q`` is a wall gain.

    Description
    -----------
    A thin subclass of ``fr.modules.BoundaryFlux`` on the buoyancy
    field: a positive ``q`` adds buoyancy to the wall-adjacent water (a
    surface *gain* — heating the top), the oceanographic convention
    (BF-D4). Internally the generic flux is ``sign(side) * q`` (``-q`` at
    a right/top wall), so the wall-adjacent tendency is
    :math:`\partial_t b = s(t)\,q\,W` with :math:`W = 1/\Delta n`. The
    generic machinery (wall weight, flux profile, bind-time taught
    errors) is inherited; the scale is published under the wrapper's own
    name ``surface_buoyancy_flux.<coord>_<side>.scale``, so instances at
    opposite walls coexist (heating the top and cooling the bottom)
    while duplicates on one wall still collide.

    Parameters
    ----------
    q : float | Callable
        The buoyancy flux; a positive value is a gain at the wall.
        A number, or a callable of the tangential coordinate names.
    coord : str, optional
        The bounded axis the flux enters through (default: "z").
    side : str, optional
        The wall, ``"left"`` or ``"right"`` (default: "right", the top).
    scale : float | fr.Ramp, optional
        The time scale :math:`s(t)` (default: 1.0).
    """

    def __init__(
        self,
        q: float | Callable,
        coord: str = "z",
        side: str = "right",
        scale: float | TimeDependent = 1.0,
    ) -> None:
        """Map the physical gain ``q`` to the generic flux ``sign * q``."""
        if side not in _SIGN:
            raise ValueError(
                f"SurfaceBuoyancyFlux side must be 'left' or 'right', "
                f"got {side!r}")
        if not (isinstance(q, numbers.Number) or callable(q)):
            raise TypeError(
                f"SurfaceBuoyancyFlux q must be a number or a "
                f"tangential profile callable, got {q!r}")
        super().__init__(
            "b", coord, side,
            flux=_signed_flux(_SIGN[side], q), scale=scale)

    def _make_scale_name(
        self, field: str, coord: str, side: str,  # noqa: ARG002
    ) -> ParamName:
        """Publish the scale under the wrapper's own class name (BF-D4)."""
        return ParamName(
            f"surface_buoyancy_flux.{coord}_{side}.scale", units="n/a",
            hint="provided by the nh.SurfaceBuoyancyFlux instance at "
                 f"the {coord} {side} wall")
