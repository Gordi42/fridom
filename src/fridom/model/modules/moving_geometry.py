r"""
Moving-geometry modules: prescribed mapping motion + ALE terms.

Description
-----------
Coordinate-systems plan, stage C4 (decision CS-D4). Two modules make
a grid's ``CoordinateMapping`` parameters move in time:

- :class:`MovingGeometry` **owns the mapping-parameter fields as
  model state**: one AUXILIARY field per parameter (named exactly
  after it — ``H``, ``YN``, ...) plus its time derivative
  (``H_dot``, ...), both re-evaluated every substage from a
  user-supplied schedule callable (``H(t)``, ``Y_N(x, t)``) through
  the trace-friendly clock. Geometry consumers — the mapped pressure
  solve, the ``physical_diff`` composites, the advection corrections
  — read the CURRENT values by threading these state fields through
  the ``grid.metric(..., params=...)`` seam (see
  :func:`mapping_params`); metrics are recomputed from the passed
  values at every query and never cached (grid rules 2.3/3.8).
- :class:`MeshVelocityCorrection` is the **dedicated, optional ALE
  tendency module** (CS-D4): it adds the mesh-velocity correction to
  every prognostic field it is configured for. Omitting it from the
  module list is the off switch — the owner explicitly wants runs
  without it, trading physical correctness during the motion
  knowingly; see the class docstring for the caveat.

Treedef discipline (grid follow-ups, item 12): the parameter fields
ride the jitted ``scan`` carry and are **replaced every substage**
through ``ScalarField.with_data``, which claims zero ghost validity —
exactly the state a freshly materialized field claims — so the carry
treedef is step-stationary by construction (the PROGNOSTIC-field
discipline applied to geometry state); the consumption-side ghost
cache never mutates carried fields since the item-12 root-cause fix.
"""
# Coordinate-systems plan, stage C4: dynamic metrics + optional ALE
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.model.declarations import (
    FieldDeclaration,
    Lifecycle,
)
from fridom.model.module import Module
from fridom.model.stages import Stage, StageKind
from fridom.model.terms import TendencyTerm, Treatment
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.space_patterns import Profile

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.spatial.fields.scalar_field import ScalarField
    from fridom.spatial.grid import Grid


def mapping_params(
    state: Mapping[str, ScalarField], grid: Grid,
) -> dict[str, ScalarField] | None:
    """
    Collect the dynamic mapping-parameter fields from the state.

    Description
    -----------
    The one discovery convention of stage C4: dynamic geometry
    parameters are state fields **named exactly after the mapping
    parameters** (declared by :class:`MovingGeometry`). Geometry
    consumers call this per evaluation and thread the result through
    the ``params=`` seams (``grid.metric``, the mapped pressure
    solver, ``MappedDerivative.with_params``); parameters without a
    state field keep their static declaration defaults, and a model
    without a dynamic-geometry module gets ``None`` — the exact
    static code path.

    Parameters
    ----------
    state : Mapping[str, ScalarField]
        The model state (must support ``in``).
    grid : Grid
        The grid whose mapping declares the parameter names.

    Returns
    -------
    dict[str, ScalarField] | None
        The current parameter fields by name, or None when the grid
        has no mapping or no parameter rides the state.
    """
    mapping = getattr(grid, "mapping", None)
    if mapping is None:
        return None
    params = {name: state[name] for name in mapping.param_names
              if name in state}
    return params or None


# ================================================================
#  MovingGeometry: schedule-driven mapping parameters
# ================================================================
class MovingGeometry(Module):

    r"""
    Prescribed time dependence of the mapping parameters.

    Description
    -----------
    Owns one AUXILIARY field per scheduled mapping parameter (named
    after the parameter) and its time derivative (``<name>_dot``),
    both written every substage by a SELF_UPDATE stage — the closed
    stage vocabulary's designated slot for time-dependent geometry —
    from the schedule callable evaluated at the substage's traced
    clock time. The time derivative comes from **autodiff of the
    schedule** (one ``jax.jvp`` in ``t`` yields value and derivative
    together), consistent with how the mapping derives coordinate
    tangents; a schedule without genuine time dependence yields an
    exactly zero ``_dot`` field, so frozen motion reproduces the
    static run.

    Schedule callables are static descriptors (identity-hashed aux
    data); only their *values* are traced — geometry sweeps through
    jit compile once. Every field is replaced via ``with_data``
    (zero claimed ghost validity), keeping the scan-carry treedef
    step-stationary (module docstring).

    Parameters
    ----------
    schedules : Mapping[str, Callable]
        Mapping parameter name -> schedule callable. Parameters of
        the callable name grid coordinates plus the time argument
        (``lambda t: ...``, ``lambda x, t: ...``); the coordinate
        set must not exceed the coordinates the mapping declares
        for that parameter.
    time : str, optional
        The schedule callables' time argument name (default:
        ``"t"``).
    """

    def __init__(
        self,
        schedules: Mapping[str, Callable],
        *,
        time: str = "t",
    ) -> None:
        """Validate and store the schedules (static descriptors)."""
        schedules = dict(schedules)
        if not schedules:
            raise ValueError(
                "MovingGeometry needs at least one parameter "
                "schedule ({name: callable})")
        if not isinstance(time, str) or not time:
            raise TypeError(
                f"time must be the schedules' time argument name "
                f"(a non-empty string), got {time!r}")
        coords: dict[str, tuple[str, ...]] = {}
        for name, fn in schedules.items():
            if not isinstance(name, str) or not callable(fn):
                raise TypeError(
                    f"schedules map parameter names to callables, "
                    f"got {name!r}: {fn!r}")
            args = tuple(inspect.signature(fn).parameters)
            if time not in args:
                raise ValueError(
                    f"the schedule for {name!r} does not take the "
                    f"time argument {time!r} (arguments: {args}); a "
                    "static parameter belongs in the mapping's "
                    "params= declaration, not in MovingGeometry")
            coords[name] = tuple(a for a in args if a != time)
        self._schedules: dict[str, Callable] = schedules
        self._time: str = time
        self._coords: dict[str, tuple[str, ...]] = coords

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def param_names(self) -> tuple[str, ...]:
        """The scheduled mapping-parameter names."""
        return tuple(self._schedules)

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_declarations(self) -> tuple[FieldDeclaration, ...]:
        """Per parameter: the value field and its time derivative.

        Both live on a ``Profile`` over the schedule's coordinate
        names — cell centers along the named coordinates, constant
        elsewhere — exactly the base layout the mapping's metric
        derivation aligns supplied parameter fields to, so the
        dynamic pipeline reproduces the static-default pipeline
        bitwise at equal values (the frozen-motion gate).
        """
        declarations: list[FieldDeclaration] = []
        for name, coords in self._coords.items():
            declarations.append(FieldDeclaration(
                name, space=Profile(*coords),
                lifecycle=Lifecycle.AUXILIARY,
                default=_value_default(name),
                long_name=f"Mapping parameter {name}",
                units="n/a"))
            declarations.append(FieldDeclaration(
                f"{name}_dot", space=Profile(*coords),
                lifecycle=Lifecycle.AUXILIARY,
                default=_dot_default(name),
                long_name=f"Mapping parameter {name} time "
                          "derivative",
                units="n/a"))
        return tuple(declarations)

    # ================================================================
    #  Bind-time validation (taught errors)
    # ================================================================
    def bind(self, table) -> None:  # noqa: ANN001
        """Validate the schedules against the grid's mapping.

        Raises
        ------
        ValueError
            If the grid carries no ``CoordinateMapping``, a schedule
            names an undeclared mapping parameter, or a schedule's
            coordinate dependence exceeds the parameter's declared
            coordinates.
        """
        grid = table.grid
        mapping = getattr(grid, "mapping", None)
        if mapping is None:
            raise ValueError(
                "MovingGeometry drives mapping parameters, but the "
                "grid carries no coordinate mapping; build the grid "
                "with Grid(..., mapping=CoordinateMapping(...))")
        declared = mapping.param_coords
        for name, coords in self._coords.items():
            if name not in declared:
                raise ValueError(
                    f"schedule {name!r} names no mapping parameter; "
                    f"the mapping declares {tuple(declared)} "
                    "(schedules are keyed by the params= names)")
            extra = tuple(c for c in coords
                          if c not in declared[name])
            if extra:
                raise ValueError(
                    f"the schedule for {name!r} varies along "
                    f"{extra}, but the mapping declares {name!r} on "
                    f"coordinates {declared[name]}; align the "
                    "schedule signature (or the mapping default)")

    # ================================================================
    #  The SELF_UPDATE stage (S1, per substage)
    # ================================================================
    #: the update is pointwise (schedule at the fields' own nodes,
    #: zero stencil) but reads the traced clock — a jax scalar the
    #: halo tracer cannot follow (V-N2): declare the zero substitute
    extra_halo = HaloSpec({})

    @property
    def stages(self) -> tuple[Stage, ...]:
        """The per-substage geometry update (SELF_UPDATE, S1)."""
        reads = tuple(self._schedules) + tuple(
            f"{name}_dot" for name in self._schedules)
        return (
            Stage(kind=StageKind.SELF_UPDATE, fn="_update_geometry",
                  name="moving_geometry", reads=reads),
        )

    def _update_geometry(self, state, ctx) -> dict:  # noqa: ANN001
        """Re-evaluate every schedule at the substage clock time.

        SELF_UPDATE runs first in every substage (S1), so all
        geometry consumers of the substage — terms, the projection
        constraint — see parameter values and mesh velocities at the
        stage time, consistent with ``eval_params``.
        """
        # ctx.clock is the Clock in-run, a bare stage-time scalar in
        # dry-run/tendency contexts (the schedule.context idiom)
        time = getattr(ctx.clock, "time", ctx.clock)
        out: dict[str, object] = {}
        for name in self._schedules:
            field = state[name]
            value, dot = self._sample(
                name, field.grid, field.function_space, time)
            out[name] = field.with_data(value)
            out[f"{name}_dot"] = state[f"{name}_dot"].with_data(dot)
        return out

    def _sample(
        self,
        name: str,
        grid,  # noqa: ANN001
        space,  # noqa: ANN001
        time,  # noqa: ANN001
    ) -> tuple[jax.Array, jax.Array]:
        """
        Evaluate one schedule and its time derivative at ``time``.

        Description
        -----------
        One ``jax.jvp`` in ``t`` (unit tangent) at the field's own
        evaluation nodes yields the value and the mesh-velocity
        ingredient together — the single code path shared by the
        in-run update and the allocation defaults, so the two agree
        bitwise at equal times.

        Parameters
        ----------
        name : str
            The parameter name.
        grid : Grid
            The grid supplying the evaluation nodes.
        space : SpaceLike
            The parameter field's own (laid-out) space.
        time : jax.Array | float
            The (traced) evaluation time.

        Returns
        -------
        tuple[jax.Array, jax.Array]
            Value and time derivative, broadcast to the space shape.
        """
        schedule = self._schedules[name]
        coords = {c: grid.evaluation_nodes(space, c).data
                  for c in self._coords[name]}
        t = jnp.asarray(time, dtype=dtype_real())

        def at_time(tt: jax.Array) -> jax.Array:
            return jnp.asarray(
                schedule(**coords, **{self._time: tt}))

        value, dot = jax.jvp(at_time, (t,),
                             (jnp.ones((), t.dtype),))
        shape = space.shape
        return (jnp.broadcast_to(value, shape),
                jnp.broadcast_to(dot, shape))


def _value_default(name: str) -> Callable:
    """Build the unbound owner-method default of a value field."""
    def _default(self, grid, space):  # noqa: ANN001, ANN202
        value, _ = self._sample(name, grid, space, 0.0)
        return grid.create_field(space, data=value, name=name)
    return _default


def _dot_default(name: str) -> Callable:
    """Build the unbound owner-method default of a ``_dot`` field."""
    def _default(self, grid, space):  # noqa: ANN001, ANN202
        _, dot = self._sample(name, grid, space, 0.0)
        return grid.create_field(space, data=dot,
                                 name=f"{name}_dot")
    return _default


# ================================================================
#  MeshVelocityCorrection: the optional ALE tendency (CS-D4)
# ================================================================
class MeshVelocityCorrection(Module):

    r"""
    Mesh-velocity (ALE) corrections on a moving mapped grid.

    Description
    -----------
    **Derivation.** Let the mapping move the physical image of the
    computational nodes, :math:`x_{\rm phys} = X(\xi, t)` (in stage
    C4 only the mapped column coordinate moves: for
    ``z = sigma * H(t)`` the nodes move vertically with
    :math:`\dot z = (\partial z/\partial H)\,\dot H`). A field
    prognosed **at fixed computational nodes**,
    :math:`F(\xi, t) = f(X(\xi, t), t)`, obeys by the chain rule

    .. math::

        \left.\frac{\partial F}{\partial t}\right|_\xi
        \;=\;
        \left.\frac{\partial f}{\partial t}\right|_{x}
        \;+\; \dot X \cdot \nabla_{\!\rm phys} f .

    The physics modules supply the physical tendency
    :math:`\partial f/\partial t|_x` (their spatial operators are
    physical derivatives of the current fields), so this module
    **adds** the transport of the field past the moving nodes,

    .. math::

        \dot m \, \frac{\partial f}{\partial m}
        \;=\;
        \Bigl(\sum_p \frac{\partial M}{\partial p}\,\dot p\Bigr)
        \frac{1}{J}\frac{\partial f}{\partial b} ,

    per configured prognostic field: :math:`\dot p` are the
    ``<p>_dot`` fields owned by :class:`MovingGeometry`,
    :math:`\partial M/\partial p` the mapping's parameter
    sensitivities (``d<mapped>_d<p>`` metrics), and
    :math:`(1/J)\,\partial_b` the registered ``physical_diff``
    composite along the column — all derived from the **current**
    parameter values through the ``params=`` seam, every step,
    nothing cached. Geometry-agnostic by construction: the vertical
    terrain-following ``H(t)`` case and the horizontal
    boundary-fitted ``Y_N(x, t)`` case run through the identical
    metric machinery.

    **The off switch and its caveat (CS-D4).** The correction lives
    in this dedicated module precisely so that omitting it from the
    module list turns it off. Running WITHOUT it while the geometry
    moves is **physically wrong during the motion**: fields then
    stay frozen at the computational nodes and are silently advected
    along with the moving mesh (a spurious transport of order
    ``x_grid_dot``). That trade is deliberate and owner-sanctioned
    (2026-07-12) for experiments that accept the transient error;
    make it knowingly.

    Parameters
    ----------
    fields : tuple[str, ...] | None, optional
        The prognostic fields to correct; None corrects every
        PROGNOSTIC field (default: None).
    """

    def __init__(self, fields: tuple[str, ...] | None = None,
                 ) -> None:
        """Store the configured field selection."""
        if fields is not None:
            fields = tuple(fields)
            if not fields or not all(
                    isinstance(name, str) for name in fields):
                raise TypeError(
                    "fields= selects prognostic field names (a "
                    f"non-empty tuple of strings), got {fields!r}")
        self._fields: tuple[str, ...] | None = fields
        self._mapped: str = ""
        self._base: str = ""
        self._driven: tuple[str, ...] = ()
        self._coords: tuple[str, ...] = ()

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def fields(self) -> tuple[str, ...] | None:
        """The corrected fields (resolved at bind when None)."""
        return self._fields

    @property
    def driven_params(self) -> tuple[str, ...]:
        """The moving mapping parameters found at bind."""
        return self._driven

    # ================================================================
    #  Bind-time validation (taught errors)
    # ================================================================
    def bind(self, table) -> None:  # noqa: ANN001
        """Resolve the column, moving parameters, and field set.

        Raises
        ------
        ValueError
            If the grid has no mapped column (no single-base
            analytic map), no ``<p>_dot`` field rides the state (no
            MovingGeometry in the module list), or a configured
            field is not PROGNOSTIC.
        NotImplementedError
            If the mapping declares more than one mapped column
            (stage C4 mirrors the stage-C3 solver support).
        """
        grid = table.grid
        mapping = getattr(grid, "mapping", None)
        table_names = {record.name for record in table}
        corrections = (mapping.column_corrections
                       if mapping is not None else {})
        if not corrections:
            raise ValueError(
                "MeshVelocityCorrection needs a grid whose "
                "CoordinateMapping declares a single-base analytic "
                "map (a mapped column); on a flat grid there is no "
                "mesh to move")
        columns = set(corrections.values())
        if len(columns) != 1:
            raise NotImplementedError(
                f"MeshVelocityCorrection supports exactly one "
                f"mapped column, got {sorted(columns)} "
                "(coordinate-systems plan, stage C4)")
        self._mapped, self._base = next(iter(columns))
        self._driven = tuple(
            name for name in mapping.param_names
            if f"{name}_dot" in table_names)
        if not self._driven:
            raise ValueError(
                "no mapping parameter carries a <name>_dot state "
                "field: MeshVelocityCorrection corrects for moving "
                "geometry, which fr.model.modules.MovingGeometry "
                "owns — add it to the module list (a static mapped "
                "grid needs no ALE terms)")
        if self._fields is None:
            self._fields = tuple(
                record.name for record in table
                if record.lifecycle is Lifecycle.PROGNOSTIC)
        else:
            for name in self._fields:
                if name not in table_names:
                    raise ValueError(
                        f"MeshVelocityCorrection corrects {name!r}, "
                        "which no module declares")
                if (table[name].lifecycle
                        is not Lifecycle.PROGNOSTIC):
                    raise ValueError(
                        f"MeshVelocityCorrection corrects {name!r}, "
                        f"which is {table[name].lifecycle.name}: "
                        "only PROGNOSTIC fields are advanced from "
                        "tendencies")
        self._coords = tuple(grid.names)

    # ================================================================
    #  The correction term
    # ================================================================
    #: the metric coefficients multiply through ``grid.metric``
    #: derivations the halo tracer cannot follow (V-N2): declare the
    #: stencil substitute (diff + interpolate hops: depth 2)
    @property
    def extra_halo(self) -> HaloSpec:
        """Two halo cells per coordinate (diff + interp chains)."""
        return HaloSpec(dict.fromkeys(self._coords, 2))

    def tendency_terms(self) -> tuple[TendencyTerm, ...]:
        """One linear term correcting every configured field."""
        return (
            TendencyTerm(
                name="mesh_velocity", fn=self._correct,
                treatment=Treatment.EXPLICIT,
                advances=self._fields, linear=True),
        )

    def _correct(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""``df/dt += m_dot * df/dm`` for every configured field.

        The column derivative is the registry-resolved
        ``physical_diff`` composite bound to the CURRENT parameter
        fields (``with_params``), interpolated back onto the field's
        own staggering; the mesh velocity contracts the mapping's
        parameter sensitivities with the ``<p>_dot`` state fields at
        the same nodes.
        """
        grid = state[self._fields[0]].grid
        registry = grid.dispatch
        params = mapping_params(state, grid)
        out: dict[str, object] = {}
        for name in self._fields:
            f = state[name]
            dfdm = self._column_derivative(f, registry, params)
            mdot = self._mesh_velocity(
                state, grid, dfdm.function_space, params)
            out[name] = (mdot * dfdm).retag(f)
        return out

    def _column_derivative(
        self,
        f: ScalarField,
        registry,  # noqa: ANN001
        params: dict | None,
    ) -> ScalarField:
        """``(1/J) df/db`` interpolated back onto ``f``'s nodes."""
        base = self._base
        builder = registry.resolve(
            "physical_diff", f.function_space.bare.factor(base))
        d = builder.with_params(params)[base](f)
        interp = registry.resolve(
            "interpolate",
            d.function_space.bare.factor(base))[base]
        return interp(d)

    def _mesh_velocity(
        self,
        state,  # noqa: ANN001
        grid,  # noqa: ANN001
        space,  # noqa: ANN001
        params: dict | None,
    ) -> ScalarField:
        r"""``m_dot = sum_p (dm/dp) p_dot`` at ``space``'s nodes."""
        total = None
        for p in self._driven:
            sensitivity = grid.metric(
                space, f"d{self._mapped}_d{p}", params=params)
            term = sensitivity * state[f"{p}_dot"].to(sensitivity)
            total = term if total is None else total + term
        return total
