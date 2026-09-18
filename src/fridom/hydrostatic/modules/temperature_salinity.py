r"""Seawater buoyancy: prognostic ``T`` and ``S`` with an equation of state.

Description
-----------
``TemperatureSalinity`` is the buoyancy formulation of a realistic
ocean (hydrostatic plan §7, the ``SeawaterBuoyancy`` of the buoyancy-
slot decision): it declares the two prognostic tracers

- ``T`` — conservative temperature :math:`\Theta` [degC],
- ``S`` — absolute salinity :math:`S_A` [g/kg],

both ``TRACER + ADVECTED``, so the assembled advection scheme
transports them and every ``TRACER``-selecting closure
(``hy.VerticalMixing(kb=...)``, the harmonic / biharmonic diffusion
family) mixes them with no further wiring — exactly like the ``b`` of
``hy.BuoyancyTracer``. The buoyancy itself becomes a **diagnosed**
field,

.. math::

    b = -g\,\frac{\rho(\Theta, S_A, d) -
        \rho(\Theta_\mathrm{ref}, S_\mathrm{ref}, d)}{\rho_0} ,

recomputed from the current ``T``, ``S`` and the geopotential depth
``d`` of each cell in a ``DIAGNOSE`` stage ordered **before** the
core's ``diagnose_p_hyd``, which integrates it into the hydrostatic
pressure unchanged: the core neither knows nor cares that ``b`` is no
longer a tracer. The density is an :mod:`fridom.hydrostatic.eos`
object — ``hy.LinearEOS`` (tunable ``eos.alpha`` / ``eos.beta``),
``hy.RoquetEOS`` (simplified second order, cabbeling +
thermobaricity) or ``hy.TEOS10EOS`` (the 55-term TEOS-10 polynomial).
The anomaly is taken against a reference parcel *at the same depth*
(see the EOS module), so no pure-depth density ever reaches the
pressure gradient.

**Dimensional only.** Temperature, salinity and an equation of state
are physical quantities with no nondimensional variant: the module is
fixed to the dimensional scaling variant and references the core's
``hydrostatic.gravity``, so a nondimensional assembly is refused at
assembly with the taught mixed-variant / missing-gravity error.

**Reductions.** ``constant_salinity=`` / ``constant_temperature=``
(the Oceananigans spelling) replace one tracer by a constant: the
field is not declared and the step carries one tracer fewer.

**Depth.** ``d = z_surface - z`` with ``z`` the physical height of the
cell: the vertical nodes on a flat or stretched column, the mapped
column position (at the geometry the state carries) on a terrain /
z* grid. ``z_surface`` defaults to the upper bound of the vertical
mesh axis, which is where both shipped column mappings (sigma
``zp = z H``, z*) keep the resting surface; pass ``surface=`` for a
mapping that does not.

**Families and masks.** On the finite-volume family ``T`` / ``S`` are
cell means and the EOS is applied to the means — a second-order
approximation of the mean buoyancy, the convention of every
finite-volume ocean model. On an immersed grid the diagnosed ``b`` is
zeroed on dry cells (their ``T = S = 0`` would otherwise read as very
light fresh water).

The stage works on raw arrays (a pointwise evaluation with no
stencil), so the module declares a zero ``extra_halo`` instead of
being halo-traced — the ``MaskState`` precedent.
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.hydrostatic.eos import EquationOfState, LinearEOS
from fridom.hydrostatic.params import EOS_ALPHA, EOS_BETA, GRAVITY
from fridom.model.modules.moving_geometry import mapping_params
from fridom.spatial.decomposition.halo import HaloSpec

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.model.params import ParamName
    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.fields.vector_field import VectorField

#: the closed vocabulary of tunable EOS coefficients -> bound names
_TUNABLE: dict[str, ParamName] = {"alpha": EOS_ALPHA, "beta": EOS_BETA}
_TUNABLE_DOC = {
    "alpha": "thermal expansion coefficient of the equation of state",
    "beta": "haline contraction coefficient of the equation of state",
}


def _column(grid: object, vertical: str) -> tuple[str, str] | None:
    """Return the ``(mapped, base)`` column on ``vertical``, or None.

    The non-raising twin of ``terrain.discover_column``: a grid without
    a mapping, or whose mapping carries no single-base column on the
    vertical axis (a purely horizontal chart), reads its depth from the
    vertical nodes themselves.
    """
    mapping = getattr(grid, "mapping", None)
    if mapping is None:
        return None
    column = mapping.column_corrections.get(vertical)
    if column is None or column[1] != vertical:
        return None
    return column


@partial(jaxify, dynamic=("alpha", "beta"))
class TemperatureSalinity(fr.model.Module):

    r"""
    Prognostic ``T`` / ``S``; buoyancy diagnosed by an equation of state.

    Description
    -----------
    The buoyancy-slot module of a realistic ocean
    (``hy.Model(buoyancy=hy.TemperatureSalinity(...))``): declares the
    tracers ``T`` [degC] and ``S`` [g/kg] and the DIAGNOSTIC buoyancy
    ``b`` [m/s^2], which a ``DIAGNOSE`` stage fills from the equation
    of state ahead of the core's hydrostatic-pressure integral. Like
    every prognostic field the tracers start at **zero** — fresh water
    at the freezing point, not an ocean — so set **both** with
    ``model.set_fields(T=..., S=...)``.

    The bound diagnostics (``model.diagnostics``):

    - ``b_total`` — the buoyancy of the **current** ``T`` / ``S`` (the
      stored ``b`` is the value the last substage diagnosed). There is
      no background to add: this formulation's buoyancy is total.
    - ``density`` — the in-situ density :math:`\rho(\Theta, S_A, d)`.
    - ``potential_density`` — :math:`\rho(\Theta, S_A, 0)`, the
      density referenced to the surface (the isopycnals to plot).

    Parameters
    ----------
    eos : EquationOfState | None, optional
        The equation of state, ``hy.LinearEOS(...)`` /
        ``hy.RoquetEOS(...)`` / ``hy.TEOS10EOS()``; ``None`` is the
        default ``hy.LinearEOS()`` (default: None).
    constant_salinity : float | None, optional
        Replace the salinity tracer by this constant [g/kg]: ``S`` is
        not declared (default: None).
    constant_temperature : float | None, optional
        Replace the temperature tracer by this constant [degC]: ``T``
        is not declared (default: None).
    family : str | None, optional
        The discretization family of the tracers and ``b``; ``None``
        defers to the grid default, which ``hy.Model`` sets from the
        core (default: None).
    vertical : str, optional
        The vertical coordinate name (default: ``"z"``).
    surface : float | None, optional
        The physical height of the resting surface [m]; ``None`` is
        the upper bound of the vertical mesh axis (default: None).

    Raises
    ------
    TypeError
        If ``eos`` is not an ``EquationOfState``, both constants are
        given (no tracer left), or a depth-dependent EOS declares
        tunable coefficients.
    """

    #: fr.scaling seam: T, S and an EOS are dimensional physics
    scaling_variant = "dimensional"

    def __init__(
        self,
        eos: EquationOfState | None = None,
        *,
        constant_salinity: float | None = None,
        constant_temperature: float | None = None,
        family: str | None = None,
        vertical: str = "z",
        surface: float | None = None,
    ) -> None:
        """Store the EOS, the reductions and the tunable leaves."""
        if eos is None:
            eos = LinearEOS()
        if not isinstance(eos, EquationOfState):
            raise TypeError(
                "TemperatureSalinity eos= takes an equation-of-state "
                "object — hy.LinearEOS(alpha=..., beta=...), "
                "hy.RoquetEOS(...) or hy.TEOS10EOS() — got "
                f"{eos!r}")
        if constant_salinity is not None and (
                constant_temperature is not None):
            raise TypeError(
                "TemperatureSalinity got both constant_salinity= and "
                "constant_temperature=: no tracer would be left and "
                "the buoyancy would be a constant — keep at least one "
                "of T / S prognostic (or use buoyancy=None for a "
                "constant-density model)")
        tunable = eos.tunable
        unknown = sorted(set(tunable) - set(_TUNABLE))
        if unknown or (tunable and eos.uses_depth):
            raise TypeError(
                f"{type(eos).__name__} declares the tunable "
                f"coefficients {sorted(tunable)}: the bound vocabulary "
                f"is {sorted(_TUNABLE)}, and only a depth-independent "
                "equation of state may be tunable")
        self._eos: EquationOfState = eos
        self._tunable: tuple[str, ...] = tuple(
            name for name in _TUNABLE if name in tunable)
        self.alpha = (fr.model.leaf(tunable["alpha"])
                      if "alpha" in tunable else None)
        self.beta = (fr.model.leaf(tunable["beta"])
                     if "beta" in tunable else None)
        self._constant_salinity: float | None = (
            None if constant_salinity is None
            else float(constant_salinity))
        self._constant_temperature: float | None = (
            None if constant_temperature is None
            else float(constant_temperature))
        self._family = family
        self._vertical = vertical
        self._surface: float | None = (
            None if surface is None else float(surface))
        # bind-captured geometry
        self._column: tuple[str, str] | None = None
        self._coords: tuple[str, ...] = ()
        self._immersed: object = None

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def eos(self) -> EquationOfState:
        """The equation of state."""
        return self._eos

    @property
    def tracers(self) -> tuple[str, ...]:
        """The declared prognostic tracer names (``T`` and/or ``S``)."""
        names = []
        if self._constant_temperature is None:
            names.append("T")
        if self._constant_salinity is None:
            names.append("S")
        return tuple(names)

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """``T`` / ``S`` (tracers) and the DIAGNOSTIC buoyancy ``b``.

        ``b`` carries **no** role: it is neither
        advected nor mixed (its sources are), and a ``TRACER``-
        selecting closure must not target it.
        """
        space = fr.spatial.Collocated(family=self._family)
        declarations = []
        if self._constant_temperature is None:
            declarations.append(fr.model.FieldDeclaration.tracer(
                "T", space=space,
                long_name="Conservative temperature", units="degC",
                nc_attrs={"standard_name":
                          "sea_water_conservative_temperature"}))
        if self._constant_salinity is None:
            declarations.append(fr.model.FieldDeclaration.tracer(
                "S", space=space,
                long_name="Absolute salinity", units="g/kg",
                nc_attrs={"standard_name":
                          "sea_water_absolute_salinity"}))
        declarations.append(fr.model.FieldDeclaration(
            "b", space=space,
            lifecycle=fr.model.Lifecycle.DIAGNOSTIC,
            long_name="Buoyancy", units="m/s^2"))
        return tuple(declarations)

    @property
    def parameter_references(
        self,
    ) -> tuple[fr.model.ParameterReference, ...]:
        """The dimensional gravity of the core."""
        return (fr.model.ParameterReference(
            GRAVITY,
            hint="the buoyancy of seawater is -g (rho - rho_ref)/rho0 "
                 "with the gravitational acceleration of the "
                 "DIMENSIONAL hydrostatic core, hy.Core(gravity=...); "
                 "temperature, salinity and an equation of state have "
                 "no nondimensional variant"),)

    @property
    def parameter_declarations(
        self,
    ) -> tuple[fr.model.ParameterDeclaration, ...]:
        """``eos.alpha`` / ``eos.beta`` of a tunable equation of state."""
        units = {"alpha": "1/K", "beta": "kg/g"}
        return tuple(
            fr.model.ParameterDeclaration(
                _TUNABLE[name], attr=name, units=units[name],
                doc=_TUNABLE_DOC[name])
            for name in self._tunable)

    # ================================================================
    #  Bind: the geometry the depth is read from
    # ================================================================
    def bind(self, table: object) -> None:
        """Capture the vertical column, the surface height, the mask.

        Raises
        ------
        ValueError
            If ``vertical`` is not a coordinate of the grid.
        """
        grid = table.grid
        self._coords = tuple(grid.names)
        if self._vertical not in self._coords:
            raise ValueError(
                f"TemperatureSalinity vertical={self._vertical!r} is "
                f"not a grid coordinate (coordinates: {self._coords}); "
                "name the vertical axis the core integrates along")
        self._column = _column(grid, self._vertical)
        self._immersed = getattr(grid, "immersed", None)
        if self._surface is None:
            for mesh in grid.factors:
                if self._vertical in mesh.names:
                    self._surface = float(mesh.extent[1])

    @property
    def extra_halo(self) -> HaloSpec:
        """Exempt the pointwise EOS evaluation from the halo trace.

        The stage reads ``T`` / ``S`` as raw arrays (a zero-stencil
        evaluation the ``HaloTracer`` cannot follow), so it declares
        its zero FD-stencil halo here — the ``MaskState`` precedent.
        """
        return HaloSpec(dict.fromkeys(self._coords, 0))

    # ================================================================
    #  The DIAGNOSE stage
    # ================================================================
    @property
    def stages(self) -> tuple[fr.model.Stage, ...]:
        """``diagnose_b``, ordered before the core's ``diagnose_p_hyd``.

        ``order=-1`` sorts it ahead of the core's order-0 DIAGNOSE
        stages whatever the module tuple position, so the hydrostatic
        integral of every substage reads the buoyancy of that
        substage's ``T`` / ``S``.
        """
        return (
            fr.model.Stage(kind=fr.model.StageKind.DIAGNOSE,
                           fn="_diagnose_b", name="diagnose_b",
                           order=-1, writes=("b",)),
        )

    def _depth(self, state: VectorField, like: ScalarField) -> object:
        """Geopotential depth of ``like``'s nodes (array; 0 if unused)."""
        if not self._eos.uses_depth:
            return 0.0
        grid = like.grid
        space = like.function_space
        if self._column is None:
            height = grid.evaluation_nodes(space, self._vertical)
        else:
            height = grid.evaluation_nodes(
                space, self._column[0],
                params=mapping_params(state, grid))
        return self._surface - height.data

    def _evaluate(
        self,
        state: VectorField,
        params: Mapping[str, object],
        fn: Callable,
        *,
        surface: bool = False,
    ) -> tuple[ScalarField, object]:
        """Apply an EOS method to the state; return (carrier, data)."""
        carrier = state[self.tracers[0]]
        temperature = (state["T"].data
                       if self._constant_temperature is None
                       else self._constant_temperature)
        salinity = (state["S"].data
                    if self._constant_salinity is None
                    else self._constant_salinity)
        depth = 0.0 if surface else self._depth(state, carrier)
        coefficients = ({name: params[_TUNABLE[name]]
                         for name in self._tunable} or None)
        data = fn(temperature, salinity, depth, coefficients)
        if self._immersed is not None:
            mask = self._immersed.mask(carrier.function_space)
            data = data * mask.data
        return carrier, data

    def _buoyancy(
        self, state: VectorField, params: Mapping[str, object],
    ) -> tuple[ScalarField, object]:
        """Return the EOS buoyancy of the state's ``T`` / ``S``."""
        gravity = params[GRAVITY]
        eos = self._eos

        def fn(temperature: object, salinity: object, depth: object,
               coefficients: Mapping[str, object] | None) -> object:
            return eos.buoyancy(temperature, salinity, depth,
                                gravity=gravity,
                                coefficients=coefficients)

        return self._evaluate(state, params, fn)

    def _diagnose_b(self, state, ctx) -> dict:  # noqa: ANN001
        r"""``b = -g (rho(T, S, d) - rho(T_ref, S_ref, d)) / rho0``."""
        _, data = self._buoyancy(state, ctx.params)
        return {"b": state["b"].with_data(data)}

    # ================================================================
    #  Bound diagnostics (model.diagnostics)
    # ================================================================
    @property
    def diagnostics(self) -> dict[str, Callable]:
        """``b_total``, ``density`` and ``potential_density``."""
        return {"b_total": self._b_total,
                "density": self._density,
                "potential_density": self._potential_density}

    def _b_total(
        self, state: VectorField, params: Mapping[str, object],
    ) -> ScalarField:
        """Total buoyancy of the current ``T`` / ``S`` [m/s^2]."""
        carrier, data = self._buoyancy(state, params)
        return carrier.new_quantity(
            data, name="b_total", long_name="Total buoyancy",
            units="m/s^2")

    def _density(
        self, state: VectorField, params: Mapping[str, object],
    ) -> ScalarField:
        """In-situ density of the current ``T`` / ``S`` [kg/m^3]."""
        carrier, data = self._evaluate(
            state, params, self._eos.density)
        return carrier.new_quantity(
            data, name="density", long_name="In-situ density",
            units="kg/m^3",
            nc_attrs={"standard_name": "sea_water_density"})

    def _potential_density(
        self, state: VectorField, params: Mapping[str, object],
    ) -> ScalarField:
        """Density referenced to the surface (``d = 0``) [kg/m^3]."""
        carrier, data = self._evaluate(
            state, params, self._eos.density, surface=True)
        return carrier.new_quantity(
            data, name="potential_density",
            long_name="Potential density (surface referenced)",
            units="kg/m^3",
            nc_attrs={"standard_name": "sea_water_potential_density"})
