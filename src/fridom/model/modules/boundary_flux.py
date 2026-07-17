r"""
Prescribed boundary-flux forcing: ``BoundaryFlux`` (BF-D1/BF-D2).

Description
-----------
Owning design record: ``design/plans/active/boundary_forcing_plan.md``.
A prescribed boundary *flux* (wind stress, surface heat/buoyancy flux —
the ocean-forcing cases) never touches the halo: the zero-gradient wall
fill makes the interior operators produce no spurious boundary flux, and
the physical flux is injected as a tendency contribution in the
wall-adjacent cell, scaled by face-area over cell-volume. For a
PROGNOSTIC ``field`` on the bounded axis ``coord`` at ``side`` in
{"left", "right"} this module adds

.. math::
    \partial_t \phi \leftarrow \partial_t \phi
        + \operatorname{sign}(\text{side})\; s(t)\;
          F(\boldsymbol{x}_{\parallel})\; W(x_{\text{coord}})

with :math:`\operatorname{sign} = +1` at a left wall and :math:`-1` at a
right wall (the Oceananigans convention: a positive flux transports the
quantity in the ``+coord`` direction, so a positive flux at a right wall
is a loss), :math:`F` the spatial flux pattern over the tangential
coordinates, and :math:`W` the wall weight — :math:`1/\Delta n` in the
wall-adjacent cell row (read from the grid measure, so stretched meshes
are handled) and ``0`` elsewhere.

Both :math:`F` and :math:`W` are assembly-materialized AUXILIARY
``fr.Profile`` fields (R2): a constant flux is a one-DOF ``fr.Profile()``,
a callable flux varies along the tangential coordinates its signature
names, and :math:`W` is an index-based indicator over ``fr.Profile(coord)``
built by an unbound owner-method default — never a float-equality
coordinate test. Both move onto the forced field with ``.to`` in the
term (a pure broadcast on the shared cell-centre nodes along ``coord``,
since staggered-normal fields are excluded by a bind-time taught error),
so the term is pure field arithmetic with no ``extra_halo``.

The scale :math:`s(t)` is a provided dynamic-leaf parameter
``boundary_flux.<field>.<coord>_<side>.scale`` (the ``Relaxation.rate``
pattern), so ``model.update_parameters`` sweeps it — a plain float, an
``fr.Ramp``, or any ``TimeDependent`` curve — without re-assembly, and
the term reads it from ``ctx.params`` (never the clock).

The wall-weight builder, the flux-profile declaration, and the bind-time
validation are module-level helpers (``build_wall_weight``,
``flux_declaration``, ``reject_chart_grid`` / ``check_walled_coord`` /
``check_tangential_flux`` / ``check_forced_field``) so the model-package
wrappers that own the physical sign conventions
(``nonhydro2.WindStress`` / ``SurfaceBuoyancyFlux``, BF-D4) reuse them
rather than duplicating.
"""
from __future__ import annotations

import inspect
import numbers
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

from fridom.framework.utils import jaxify, modify_array
from fridom.model.declarations import (
    FieldDeclaration,
    FieldReference,
    Lifecycle,
)
from fridom.model.module import Module
from fridom.model.parameters import ParameterDeclaration, leaf
from fridom.model.params import ParamName
from fridom.model.terms import TendencyTerm, Treatment
from fridom.spatial.bc import BC
from fridom.spatial.space_patterns import Profile
from fridom.spatial.spaces.average import CellAvg
from fridom.spatial.spaces.nodal import NodeSet

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    from fridom.model.time_dependent import TimeDependent
    from fridom.spatial.fields.scalar_field import ScalarField

#: the side vocabulary and the wall-adjacent cell index it selects
_SIDES: dict[str, int] = {"left": 0, "right": -1}

#: the tendency sign per side (positive flux transports in +coord)
_SIGN: dict[str, float] = {"left": 1.0, "right": -1.0}

_FIELD_HINT = ("the forced field must be declared by another module "
               "(e.g. a dynamical core registers the velocities, a "
               "stratification module registers b); check the "
               "BoundaryFlux field-name spelling")


# ================================================================
#  Shared helpers (reused by the model-package wrappers, BF-D4)
# ================================================================
def _coordinate_names(fn: Callable) -> tuple[str, ...]:
    """Return the coordinate names a flux callable varies along."""
    return tuple(inspect.signature(fn).parameters)


def _coord_axis(space: object, coord: str) -> int:
    """Return the array axis of the factor carrying ``coord``."""
    for axis, factor in enumerate(space.factors):
        if coord in factor.names:
            return axis
    # unreachable: bind validates the coordinate against the grid
    raise ValueError(  # pragma: no cover
        f"coordinate {coord!r} matches no factor of {space!r}")


def _is_collocated_along(space: object, coord: str) -> bool:
    """Whether ``space`` is cell-centred (primal) along ``coord``.

    The wall weight lives on the primal cell centres along ``coord``
    (``Center`` on a nodal grid, ``CellAvg`` on the FV default); a
    forced field sharing that placement takes the weight as a pure
    ``.to`` broadcast. A face placement (``Inner``/``Outer``/... or
    ``FaceAvg``) is the wall-normal velocity — its wall faces are not
    DOFs (taught error b).
    """
    factor = space.factor(coord)
    node_set = getattr(factor, "node_set", None)
    if node_set is not None:
        return node_set is NodeSet.CENTER
    return isinstance(factor, CellAvg)


def build_wall_weight(
    grid: object, space: object, coord: str, side: str, name: str,
) -> ScalarField:
    r"""Materialize the wall weight ``1/\Delta n`` at the wall cell.

    Builds a true-shape indicator that is ``1`` at the wall-adjacent
    cell row (index ``0`` for a left wall, ``-1`` for a right wall) and
    ``0`` elsewhere — an index test, never a float-equality coordinate
    test — then divides by the grid measure so the nonzero entry is
    :math:`1/\Delta n`, the actual wall-cell width (stretched meshes
    included). The indicator and the measure are fields on the same
    ``Profile(coord)`` space, so the division shards consistently under
    decomposition. Shared by ``BoundaryFlux`` and the model-package
    wrappers.
    """
    axis = _coord_axis(space, coord)
    where = tuple(
        _SIDES[side] if a == axis else slice(None)
        for a in range(len(space.shape)))
    indicator = modify_array(jnp.zeros(space.shape), where, 1.0)
    weight = (grid.create_field(space, data=indicator)
              / grid.measure(space, coord))
    return weight.with_metadata(name=name)


def flux_declaration(
    name: str, flux: float | Callable, *, long_name: str,
) -> FieldDeclaration:
    """Declare a flux pattern as an AUXILIARY profile field.

    A number becomes a constant fill on the one-DOF ``fr.Profile()``; a
    callable becomes the coordinate default of a ``fr.Profile`` over
    exactly the tangential coordinates its signature names (the
    ``Relaxation`` profile normalization). Shared by ``BoundaryFlux``
    and the model-package wrappers.
    """
    if callable(flux):
        space = Profile(*_coordinate_names(flux))
        default: float | Callable = flux
    else:
        space = Profile()
        default = float(flux)
    return FieldDeclaration(
        name, space=space, lifecycle=Lifecycle.AUXILIARY,
        default=default, long_name=long_name, units="n/a")


def reject_chart_grid(grid: object, owner: str) -> None:
    """Refuse a chart/mapped grid (iteration-1 scope, BF-D2 d)."""
    chart = grid.chart_coords
    if chart is not None:
        raise ValueError(
            f"{owner} computes the wall weight 1/Delta n from 1-D "
            "measures, but this grid carries an embedding chart on "
            f"{chart}: the metric-aware A_face/V_cell weight is "
            "designed-for but not implemented (iteration 1). Use a "
            "flat Cartesian grid")


def check_walled_coord(grid: object, coord: str, owner: str) -> None:
    """Require ``coord`` to be a bounded (walled) axis of the grid (a)."""
    mesh = next(
        (m for m in grid.factors if coord in m.names), None)
    walls = sorted(
        name for m in grid.factors
        if not getattr(m, "periodic", False) for name in m.names)
    if mesh is None:
        raise ValueError(
            f"{owner} coord={coord!r} is not a coordinate of the grid "
            f"(coordinates: {grid.names}); the walled axes are {walls}")
    if getattr(mesh, "periodic", False):
        raise ValueError(
            f"{owner} coord={coord!r} is periodic: a boundary flux "
            "enters through a bounded (walled) axis; the walled axes "
            f"are {walls}")


def check_tangential_flux(
    grid: object, coord: str, flux: float | Callable, owner: str,
) -> None:
    """Reject a flux callable naming the normal or a missing coord."""
    if not callable(flux):
        return
    names = _coordinate_names(flux)
    if coord in names:
        raise ValueError(
            f"the {owner} flux callable names the normal coordinate "
            f"{coord!r}: the flux lives on the wall face and varies "
            "only along the tangential coordinates")
    unknown = sorted(set(names) - set(grid.names))
    if unknown:
        raise ValueError(
            f"the {owner} flux callable names the coordinate(s) "
            f"{unknown}, which the grid does not have (coordinates: "
            f"{grid.names})")


def check_forced_field(
    table: object, field: str, coord: str, side: str, owner: str,
) -> None:
    """Require a PROGNOSTIC, cell-centred, non-DIRICHLET field (b, c).

    Rejects a non-PROGNOSTIC field, a field staggered (wall-normal)
    along ``coord``, and a field whose resolved BC on ``(coord, side)``
    is DIRICHLET (the wall value is pinned, so a flux cannot be
    prescribed).
    """
    record = table[field]
    if record.lifecycle is not Lifecycle.PROGNOSTIC:
        raise ValueError(
            f"{owner} forces {field!r}, which is "
            f"{record.lifecycle.name}: only PROGNOSTIC fields are "
            "advanced from tendencies")
    if not _is_collocated_along(record.space, coord):
        raise ValueError(
            f"{owner} forces {field!r}, which is staggered along "
            f"{coord!r} (its wall faces are not DOFs): prescribing "
            "wall-normal flow is an open-boundary condition, out of "
            "scope. Force a cell-centred field (a tracer, buoyancy, or "
            "a tangential velocity)")
    component = record.space.factor(coord).bc.components[_SIDES[side]]
    if component is BC.DIRICHLET:
        raise ValueError(
            f"{owner} forces {field!r} at the {coord} {side} wall, but "
            "its resolved boundary condition there is DIRICHLET: the "
            "wall value is pinned, so a flux cannot be prescribed. Use "
            "the NEUMANN sibling, or fr.modules.Relaxation to nudge "
            "toward a target")


@partial(jaxify, dynamic=("scale",))
class BoundaryFlux(Module):

    r"""
    Inject a prescribed boundary flux as a wall-adjacent tendency.

    Description
    -----------
    Contributes the tendency term

    .. math::
        \partial_t \phi \leftarrow \partial_t \phi
            + \operatorname{sign}(\text{side})\; s(t)\;
              F(\boldsymbol{x}_{\parallel})\; W(x_{\text{coord}})

    for the PROGNOSTIC ``field`` on the bounded axis ``coord`` at
    ``side`` (``sign`` ``+1`` left / ``-1`` right). The wall weight
    :math:`W` is ``1/\Delta n`` in the wall-adjacent cell (from the
    grid measure — stretched meshes included) and ``0`` elsewhere; the
    flux :math:`F` is a number (constant) or a callable of the
    tangential coordinate names. On ``CellAvg`` spaces the term is
    exactly the flux-divergence contribution of a prescribed face flux,
    so the discrete budget
    :math:`\mathrm{d}/\mathrm{d}t \int \phi\,\mathrm{d}V =
    \operatorname{sign}\, s \int_{\text{wall}} F\,\mathrm{d}A` holds to
    rounding.

    The scale :math:`s(t)` is published as the dynamic-leaf parameter
    ``boundary_flux.<field>.<coord>_<side>.scale``, so
    ``model.update_parameters`` sweeps it (a float, an ``fr.Ramp``, or
    any ``TimeDependent`` curve) without re-assembly.

    Parameters
    ----------
    field : str
        The PROGNOSTIC field name to force; unknown names raise a
        taught assembly error.
    coord : str
        The bounded (non-periodic) axis the flux enters through.
    side : str
        The wall, ``"left"`` or ``"right"``.
    flux : float | Callable, optional
        The flux pattern :math:`F`: a number (constant), or a callable
        of the tangential coordinate names (naming ``coord`` is a
        taught error — the flux lives on the wall face) (default: 1.0).
    scale : float | fr.Ramp, optional
        The time scale :math:`s(t)` shared by the wall (a float or any
        ``TimeDependent`` curve for a spun-up forcing) (default: 1.0).
    """

    def __init__(
        self,
        field: str,
        coord: str,
        side: str,
        *,
        flux: float | Callable = 1.0,
        scale: float | TimeDependent = 1.0,
    ) -> None:
        """Freeze the target/geometry; store the scale leaf."""
        if not isinstance(field, str) or not field:
            raise TypeError(
                f"BoundaryFlux field must be a non-empty name, got "
                f"{field!r}")
        if not isinstance(coord, str) or not coord:
            raise TypeError(
                f"BoundaryFlux coord must be a non-empty coordinate "
                f"name, got {coord!r}")
        if side not in _SIDES:
            raise ValueError(
                f"BoundaryFlux side must be 'left' or 'right', got "
                f"{side!r}")
        if not (isinstance(flux, numbers.Number) or callable(flux)):
            raise TypeError(
                f"BoundaryFlux flux must be a number or a profile "
                f"callable of the tangential coordinate names, got "
                f"{flux!r}")
        self._field: str = field
        self._coord: str = coord
        self._side: str = side
        self._flux: float | Callable = flux
        self.scale = leaf(scale)
        tag = f"{coord}_{side}"
        self._weight_name: str = f"bflux_{field}_{tag}_weight"
        self._flux_name: str = f"bflux_{field}_{tag}_flux"
        self._scale_name: ParamName = ParamName(
            f"boundary_flux.{field}.{tag}.scale", units="n/a",
            hint="provided by the fr.modules.BoundaryFlux instance "
                 f"forcing {field!r} at the {coord} {side} wall")

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def field(self) -> str:
        """The forced PROGNOSTIC field name."""
        return self._field

    @property
    def coord(self) -> str:
        """The bounded axis the flux enters through."""
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
        """The checked claim on the forced field."""
        return (FieldReference(self._field, hint=_FIELD_HINT),)

    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """The AUXILIARY wall-weight and flux profiles."""
        return (
            FieldDeclaration(
                self._weight_name, space=Profile(self._coord),
                lifecycle=Lifecycle.AUXILIARY,
                default=self._wall_weight_default,
                long_name=f"Wall weight on {self._field}",
                units="1/m"),
            flux_declaration(
                self._flux_name, self._flux,
                long_name=f"Boundary flux on {self._field}"),
        )

    @property
    def parameter_declarations(
        self,
    ) -> tuple[ParameterDeclaration, ...]:
        """Publish the live scale leaf under the instance's name."""
        return (
            ParameterDeclaration(
                self._scale_name, attr="scale", units="n/a",
                doc=f"boundary-flux scale on {self._field} at the "
                    f"{self._coord} {self._side} wall"),
        )

    def _wall_weight_default(
        self, grid: object, space: object,
    ) -> ScalarField:
        r"""Owner-method default: ``1/\Delta n`` at the wall-adjacent cell."""
        return build_wall_weight(
            grid, space, self._coord, self._side, self._weight_name)

    # ================================================================
    #  Bind-time validation (taught errors a-d)
    # ================================================================
    def bind(self, table: object) -> None:
        """Validate the coordinate, the field, and the BC (a)-(d)."""
        grid = table.grid
        owner = type(self).__name__
        reject_chart_grid(grid, owner)                          # (d)
        check_walled_coord(grid, self._coord, owner)            # (a)
        check_tangential_flux(grid, self._coord, self._flux, owner)
        check_forced_field(
            table, self._field, self._coord, self._side, owner)  # (b,c)

    # ================================================================
    #  The boundary-flux term
    # ================================================================
    def tendency_terms(self) -> tuple[TendencyTerm, ...]:
        """One term forcing the field at its wall."""
        return (
            TendencyTerm(
                name="boundary_flux", fn=self._flux_term,
                treatment=Treatment.EXPLICIT,
                advances=(self._field,)),
        )

    def _flux_term(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``d field/dt += sign * s(t) * F(x_tang) * W(x_coord)``.

        The flux and weight profiles are moved onto the forced field's
        own space with ``.to`` — a pure broadcast on the shared
        cell-centre nodes along ``coord`` — so the term is pure field
        arithmetic and needs no ``extra_halo``. The scale ``s(t)`` is
        read from ``ctx.params`` (already resolved at the stage clock
        time), never from the clock directly.
        """
        scale = ctx.params[self._scale_name]
        field = state[self._field]
        flux = state[self._flux_name].to(field)
        weight = state[self._weight_name].to(field)
        source = _SIGN[self._side] * scale * flux * weight
        return {self._field: source}
