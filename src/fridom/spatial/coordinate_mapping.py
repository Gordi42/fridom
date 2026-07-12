r"""
``CoordinateMapping``: grid-attached coordinate-transform declaration.

Description
-----------
Owning class doc: ``design/specs/grid/classes/grid.md``; rules in
``design/specs/grid/02_rules.md`` section 3.8. A **static
descriptor** of terrain-following / curvilinear coordinate maps and
the single owner of the metric *derivation*: it holds callables and
static parameter defaults only — no arrays, no fields — and derives
every metric per staggered space on demand at trace time, exactly
like ``grid.evaluation_nodes`` (rules section 2.7).

Three declaration forms (stage C1 of the coordinate-systems plan):

- **analytic maps** — one callable per mapped physical coordinate,
  taking base coordinates and named parameter fields by keyword
  (``maps={"z": lambda sigma, H: sigma * H}``). Metric names:
  ``d<p>_d<q>`` (the map derivative with respect to coordinate
  ``q``, parameter fields chain-ruled in) and, for single-base
  columns, the reciprocal ``d<base>_d<p>``.
- **supplied metrics** — ``metrics={"name": callable}`` for cases
  with no closed form; callables take coordinates and parameters by
  keyword and are evaluated at the requested space's nodes.
- **embedding form (CS-D1)** — a chart into ambient coordinates
  (``chart={"X": lambda lon, lat: (x, y, z)}``); the induced metric
  ``g_ij = dX/du_i . dX/du_j`` is derived by jax autodiff of the
  chart callable at the requested space's evaluation nodes. Metric
  names: ``g_<u><v>``, ``inv_g_<u><v>``, ``sqrt_g``. Derivation is
  per chart, so a future multi-chart atlas stays additive.

Derivatives of the map/chart with respect to their *coordinate*
arguments are exact (``jax.jvp``); derivatives of *parameter fields*
are discrete — the registry ``("diff", space)`` operators plus
``.to`` interpolation onto the requested staggered space — so the
metrics every module sees are staggered-consistent by construction
(one owner; H at u-, v-, w-points never derived per module). The
``params=`` overload of :meth:`CoordinateMapping.metric` replaces
the static defaults with caller-supplied dynamic fields (CS-D4):
values enter as traced arrays, never as static hash keys, so
sweeping parameter values through jit compiles once.
"""
# Coordinate-systems plan, stage C1: CoordinateMapping + grid.metric
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from fridom.framework.utils import dtype_real
from fridom.spatial.fields.metadata import FieldMetadata
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.fields.storage import store
from fridom.spatial.spaces.average import AverageSpace
from fridom.spatial.spaces.coefficient import CoefficientSpace
from fridom.spatial.spaces.constant import ConstantSpace
from fridom.spatial.spaces.tensor_product import (
    TensorProductSpace,
)

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.spatial.grid import Grid
    from fridom.spatial.spaces.function_space import (
        FunctionSpace,
    )
    from fridom.spatial.spaces.tensor_product import SpaceLike


class _Declared:

    """
    One declared callable, split into coordinate and parameter args.

    Parameters
    ----------
    fn : Callable
        The declared callable (map, chart, or supplied metric).
    param_names : tuple[str, ...]
        The mapping's declared parameter names; signature arguments
        in this set are parameters, all others base coordinates.
    """

    def __init__(self, fn: Callable[..., object],
                 param_names: tuple[str, ...]) -> None:
        """Split the signature into coordinate and parameter args."""
        self.fn: Callable[..., object] = fn
        self.order: tuple[str, ...] = tuple(
            inspect.signature(fn).parameters)
        self.params: tuple[str, ...] = tuple(
            name for name in self.order if name in param_names)
        self.coords: tuple[str, ...] = tuple(
            name for name in self.order if name not in param_names)


# ================================================================
#  Derivation context (one grid.metric query)
# ================================================================
class _Derivation:

    """
    Per-query materialization context of one ``metric`` call.

    Description
    -----------
    Supplies the derivation recipes with broadcast-ready arrays on
    the requested space: coordinate arrays from
    ``grid.evaluation_nodes``, parameter values/derivatives through
    the field pipeline — materialize (default callable) or align
    (supplied field), differentiate via the registry ``diff``
    operators, interpolate onto the requested staggering via ``.to``
    — memoized for the duration of the query only. Nothing survives
    the call (rules 3.8: no operator, grid, or mapping caches
    metrics).

    Parameters
    ----------
    mapping : CoordinateMapping
        The owning declaration.
    grid : Grid
        The bound grid.
    space : SpaceLike
        The laid-out requested space.
    explicit : Mapping[str, ScalarField]
        Caller-supplied parameter fields (may be empty).
    """

    def __init__(self, mapping: CoordinateMapping, grid: Grid,
                 space: SpaceLike,
                 explicit: Mapping[str, ScalarField]) -> None:
        """Bind the query context; nothing is materialized yet."""
        self._mapping = mapping
        self._grid = grid
        self._space = space
        self._explicit = explicit
        self._bases: dict[str, ScalarField] = {}

    def coord(self, name: str) -> jax.Array:
        """Node coordinates of ``name``, broadcastable on the space."""
        return self._grid.evaluation_nodes(self._space, name).data

    def param_value(self, name: str) -> jax.Array:
        """Return the parameter at the requested space's nodes."""
        return self._at_space(self._base(name)).data

    def param_tangent(self, name: str,
                      wrt: str) -> jax.Array | None:
        """
        Discrete ``d<param>/d<wrt>`` at the requested space's nodes.

        Description
        -----------
        None when the parameter does not depend on ``wrt`` (a
        structural zero of the chain rule). Otherwise the registry
        ``diff`` derivative of the base parameter field,
        interpolated onto the requested staggering — the discrete
        chain-rule ingredient of the map/chart tangents.

        Parameters
        ----------
        name : str
            The parameter name.
        wrt : str
            The coordinate to differentiate along.

        Returns
        -------
        jax.Array | None
            The tangent array, or None for a structural zero.
        """
        if wrt not in self._mapping._param_coords[name]:  # noqa: SLF001
            return None
        base = self._base(name)
        return self._at_space(base.diff(wrt)).data

    # ------------------------------------------------------------
    #  Parameter field pipeline
    # ------------------------------------------------------------
    def _base(self, name: str) -> ScalarField:
        """Return the aligned base field (memoized per query)."""
        field = self._bases.get(name)
        if field is None:
            supplied = self._explicit.get(name)
            if supplied is None:
                field = self._default_base(name)
            else:
                field = self._aligned_explicit(name, supplied)
            self._bases[name] = field
        return field

    def _default_base(self, name: str) -> ScalarField:
        """
        Materialize the static default onto collocated factors.

        Description
        -----------
        The base lives at the cell centers of every coordinate the
        parameter depends on — as ``CellAvg`` when the requested
        factor is in the average family (the collocation values
        coincide, and the FV conversion/derivative rows then join
        the staggerings), as ``Center`` otherwise.

        Parameters
        ----------
        name : str
            The parameter name.

        Returns
        -------
        ScalarField
            The materialized default on the aligned base space.
        """
        coords = self._mapping._param_coords[name]  # noqa: SLF001
        factors: list[FunctionSpace] = []
        for factor in self._space.factors:
            if any(n in coords for n in factor.names):
                attr = ("cell_avg"
                        if isinstance(factor, AverageSpace)
                        else "center")
                try:
                    factors.append(getattr(factor.mesh, attr))
                except (AttributeError, ValueError,
                        NotImplementedError) as exc:
                    raise NotImplementedError(
                        f"the default for parameter {name!r} "
                        f"materializes on cell centers, but "
                        f"{factor.mesh!r} has no Center space; pass "
                        "an explicit field via params=") from exc
            else:
                factors.append(factor.mesh.constant)
        space = _product(factors, self._space.layout)
        default = self._mapping._params[name]  # noqa: SLF001
        return self._grid.create_field(space, init=default)

    def _aligned_explicit(self, name: str,
                          field: ScalarField) -> ScalarField:
        """
        Align a caller-supplied field to the space's factor order.

        Description
        -----------
        The supplied field may carry its factors in any order and
        may omit meshes it is constant along; alignment permutes the
        data axes into the requested space's mesh order and fills
        the omitted meshes with ``ConstantSpace`` factors, so the
        downstream ``diff``/``.to`` pipeline sees one canonical
        layout for defaults and supplied fields alike.

        Parameters
        ----------
        name : str
            The parameter name (declared coordinate dependence).
        field : ScalarField
            The caller-supplied parameter field.

        Returns
        -------
        ScalarField
            The aligned field on the space's mesh tuple.
        """
        if field.grid is not self._grid:
            raise ValueError(
                f"the supplied field for parameter {name!r} lives "
                "on a different grid than this mapping")
        coords = self._mapping._param_coords[name]  # noqa: SLF001
        by_mesh: dict[object, tuple[int, FunctionSpace]] = {}
        for axis, factor in enumerate(field.function_space.factors):
            if isinstance(factor, ConstantSpace):
                continue
            extra = tuple(n for n in factor.names
                          if n not in coords)
            if extra:
                raise ValueError(
                    f"parameter {name!r} is declared on coordinates "
                    f"{coords}; the supplied field varies along "
                    f"{extra}")
            by_mesh[factor.mesh] = (axis, factor)
        meshes = tuple(f.mesh for f in self._space.factors)
        order = [by_mesh[mesh][0] for mesh in meshes
                 if mesh in by_mesh]
        data = field.data
        rest = tuple(i for i in range(data.ndim)
                     if i not in order)
        data = jnp.transpose(data, (*order, *rest))
        shape = tuple(
            by_mesh[mesh][1].shape[0] if mesh in by_mesh else 1
            for mesh in meshes)
        factors = [
            by_mesh[mesh][1] if mesh in by_mesh else mesh.constant
            for mesh in meshes]
        space = _product(factors, self._space.layout)
        stored = store(self._grid.decomposition, space,
                       data.reshape(shape))
        return ScalarField(self._grid, space, stored,
                           FieldMetadata.create(name=name))

    def _at_space(self, field: ScalarField) -> ScalarField:
        """Interpolate an aligned field onto the requested space."""
        target = _product(
            tuple(
                base if isinstance(base, ConstantSpace)
                else mine
                for base, mine in zip(
                    field.function_space.bare.factors,
                    (f.bare for f in self._space.factors),
                    strict=True)),
            self._space.layout)
        return field.to(target)


def _product(factors: list[FunctionSpace] | tuple[FunctionSpace, ...],
             layout: object) -> SpaceLike:
    """Assemble a (lone-factor collapsing) laid-out product space."""
    space: SpaceLike = (
        factors[0] if len(factors) == 1
        else TensorProductSpace.of(*factors))
    return space.with_layout(layout)


# ================================================================
#  Evaluation helpers (map/chart tangents via jax.jvp)
# ================================================================
def _arguments(
    decl: _Declared, ctx: _Derivation,
) -> dict[str, jax.Array]:
    """Materialize a declared callable's arguments on the space."""
    values: dict[str, jax.Array] = {}
    for name in decl.order:
        if name in decl.params:
            values[name] = ctx.param_value(name)
        else:
            values[name] = ctx.coord(name)
    return values


def _tangents(
    decl: _Declared,
    ctx: _Derivation,
    values: dict[str, jax.Array],
    wrt: str,
) -> tuple[jax.Array, ...]:
    """
    Tangent vectors of the arguments along coordinate ``wrt``.

    Description
    -----------
    The single chain-rule assembly shared by maps and charts: the
    ``wrt`` coordinate argument gets a unit tangent, every parameter
    argument its discrete ``d<param>/d<wrt>`` (zero when it is a
    structural zero), everything else zero — one ``jax.jvp`` then
    yields the full derivative including the parameter chain terms.

    Parameters
    ----------
    decl : _Declared
        The declared callable.
    ctx : _Derivation
        The query context (parameter tangent supplier).
    values : dict[str, jax.Array]
        The materialized arguments (from ``_arguments``).
    wrt : str
        The coordinate to differentiate along.

    Returns
    -------
    tuple[jax.Array, ...]
        One tangent per argument, in signature order.
    """
    tangents: list[jax.Array] = []
    for name in decl.order:
        value = values[name]
        if name in decl.params:
            tangent = ctx.param_tangent(name, wrt)
            tangents.append(jnp.zeros_like(value)
                            if tangent is None else tangent)
        elif name == wrt:
            tangents.append(jnp.ones_like(value))
        else:
            tangents.append(jnp.zeros_like(value))
    return tuple(tangents)


def _jvp(decl: _Declared, ctx: _Derivation,
         wrt: str) -> tuple[object, object]:
    """Autodiff a declared callable along coordinate ``wrt``."""
    values = _arguments(decl, ctx)
    primals = tuple(values[name] for name in decl.order)
    tangents = _tangents(decl, ctx, values, wrt)

    def positional(*args: jax.Array) -> object:
        return decl.fn(**dict(zip(decl.order, args, strict=True)))

    return jax.jvp(positional, primals, tangents)


def _chart_tangents(
    decl: _Declared, ctx: _Derivation,
) -> list[tuple[jax.Array, ...]]:
    """Ambient tangent vectors, one per chart coordinate."""
    tangents: list[tuple[jax.Array, ...]] = []
    for wrt in decl.coords:
        primal, tangent = _jvp(decl, ctx, wrt)
        if not isinstance(primal, tuple | list) or not primal:
            raise TypeError(
                "chart callables must return a non-empty tuple of "
                f"ambient components, got {primal!r}")
        tangents.append(tuple(jnp.asarray(t) for t in tangent))
    return tangents


def _chart_matrix(
    decl: _Declared, ctx: _Derivation,
) -> jax.Array:
    """Assemble the induced metric as a ``(..., k, k)`` matrix."""
    tangents = _chart_tangents(decl, ctx)
    entries = [
        [sum(a * b for a, b in zip(ti, tj, strict=True))
         for tj in tangents]
        for ti in tangents]
    shape = jnp.broadcast_shapes(
        *(e.shape for row in entries for e in row))
    return jnp.stack(
        [jnp.stack([jnp.broadcast_to(e, shape) for e in row],
                   axis=-1)
         for row in entries], axis=-2)


# ================================================================
#  Derivation recipes (one per metric name)
# ================================================================
class _MapDerivative:

    """``d<p>_d<wrt>``: map derivative with parameter chain rule."""

    def __init__(self, decl: _Declared, wrt: str,
                 deps: frozenset[str]) -> None:
        self.decl = decl
        self.wrt = wrt
        self.deps = deps

    def evaluate(self, ctx: _Derivation) -> jax.Array:
        """Autodiff the map along ``wrt`` at the space's nodes."""
        _, tangent = _jvp(self.decl, ctx, self.wrt)
        return jnp.asarray(tangent)


class _MapInverse:

    """``d<b>_d<p>``: reciprocal Jacobian of a single-base column."""

    def __init__(self, decl: _Declared, base: str,
                 deps: frozenset[str]) -> None:
        self.decl = decl
        self.base = base
        self.deps = deps

    def evaluate(self, ctx: _Derivation) -> jax.Array:
        """Return the reciprocal of the column Jacobian."""
        _, tangent = _jvp(self.decl, ctx, self.base)
        return 1.0 / jnp.asarray(tangent)


class _Supplied:

    """A user-supplied metric callable, sampled at the nodes."""

    def __init__(self, decl: _Declared,
                 deps: frozenset[str]) -> None:
        self.decl = decl
        self.deps = deps

    def evaluate(self, ctx: _Derivation) -> jax.Array:
        """Evaluate the callable on coordinates and parameters."""
        return jnp.asarray(self.decl.fn(**_arguments(self.decl,
                                                     ctx)))


class _ChartEntry:

    """One induced-metric entry (``g``/``inv_g``/``sqrt_g``)."""

    def __init__(self, decl: _Declared, kind: str, i: int, j: int,
                 deps: frozenset[str]) -> None:
        self.decl = decl
        self.kind = kind
        self.i = i
        self.j = j
        self.deps = deps

    def evaluate(self, ctx: _Derivation) -> jax.Array:
        """Derive the entry from the chart's tangent vectors."""
        if self.kind == "g":
            tangents = _chart_tangents(self.decl, ctx)
            return sum(a * b for a, b in zip(
                tangents[self.i], tangents[self.j], strict=True))
        matrix = _chart_matrix(self.decl, ctx)
        if self.kind == "sqrt_g":
            return jnp.sqrt(jnp.linalg.det(matrix))
        return jnp.linalg.inv(matrix)[..., self.i, self.j]


# ================================================================
#  The public declaration class
# ================================================================
class CoordinateMapping:

    """
    Static coordinate-transform declaration; metrics on demand.

    Description
    -----------
    See the module docstring for the three declaration forms and
    the metric-name vocabulary. The descriptor is attached at grid
    build (``Grid(..., mapping=...)``); it is fully static — map
    callables and static parameter defaults only, identity-hashed —
    and every :meth:`metric` call materializes to the requested
    staggered space at trace time. Time-dependent geometry is
    module-owned state threaded through the ``params=`` overload;
    derived metrics are recomputed from the passed values at every
    query and never cached (rules 2.3, 3.8).

    Parameters
    ----------
    maps : Mapping[str, Callable] | None, optional
        Analytic maps: physical coordinate name -> callable of base
        coordinates and parameters by keyword (default: None).
    chart : Mapping[str, Callable] | None, optional
        Embedding charts (CS-D1): chart name -> callable of base
        coordinates (and parameters) returning the tuple of ambient
        components (default: None).
    metrics : Mapping[str, Callable] | None, optional
        Supplied metrics: metric name -> callable of coordinates
        and parameters by keyword (default: None).
    params : Mapping[str, Callable] | None, optional
        Static parameter defaults as callables of physical
        coordinates, materialized on demand — nothing is stored
        (default: None).
    """

    def __init__(
        self,
        maps: Mapping[str, Callable[..., jax.Array]] | None = None,
        *,
        chart: Mapping[str, Callable[..., object]] | None = None,
        metrics: (
            Mapping[str, Callable[..., jax.Array]] | None) = None,
        params: (
            Mapping[str, Callable[..., jax.Array]] | None) = None,
    ) -> None:
        """Declare the transform; see the class docstring."""
        maps = dict(maps or {})
        chart = dict(chart or {})
        metrics = dict(metrics or {})
        params = dict(params or {})
        if not (maps or chart or metrics):
            raise ValueError(
                "a CoordinateMapping declares at least one of "
                "maps=, chart=, or metrics=")
        for label, table in (("maps", maps), ("chart", chart),
                             ("metrics", metrics),
                             ("params", params)):
            for key, fn in table.items():
                if not isinstance(key, str) or not callable(fn):
                    raise TypeError(
                        f"{label}= maps names to callables, got "
                        f"{key!r}: {fn!r}")
        self._params: dict[str, Callable[..., jax.Array]] = params
        self._param_coords: dict[str, tuple[str, ...]] = {
            name: tuple(inspect.signature(fn).parameters)
            for name, fn in params.items()}
        param_names = tuple(params)
        self._maps: dict[str, _Declared] = {}
        self._charts: dict[str, _Declared] = {}
        self._recipes: dict[str, object] = {}
        for name, fn in maps.items():
            decl = _Declared(fn, param_names)
            if not decl.coords:
                raise ValueError(
                    f"map {name!r} must depend on at least one "
                    "base coordinate")
            self._maps[name] = decl
            self._declare_map(name, decl)
        for name, fn in chart.items():
            decl = _Declared(fn, param_names)
            if not decl.coords:
                raise ValueError(
                    f"chart {name!r} must depend on at least one "
                    "base coordinate")
            self._charts[name] = decl
            self._declare_chart(name, decl)
        for name, fn in metrics.items():
            decl = _Declared(fn, param_names)
            self._add(name, _Supplied(decl, self._deps(decl)))
        self._grid: Grid | None = None

    # ================================================================
    #  Identity (static aux discipline, matching Grid)
    # ================================================================
    def __eq__(self, other: object) -> bool:
        """Identity comparison: ``self is other`` (static aux)."""
        return self is other

    def __hash__(self) -> int:
        """Identity hash, consistent with ``__eq__``."""
        return id(self)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def param_names(self) -> tuple[str, ...]:
        """The named parameters of the map (H, eta, ...)."""
        return tuple(self._params)

    @property
    def metric_names(self) -> tuple[str, ...]:
        """The metric names this mapping can supply."""
        return tuple(self._recipes)

    # ================================================================
    #  Grid binding (called by the grid at attachment)
    # ================================================================
    def _bind(self, grid: Grid) -> None:
        """Bind to the grid; validate the coordinate vocabulary."""
        if self._grid is not None:
            raise ValueError(
                "this CoordinateMapping is already attached to a "
                "grid; descriptors are grid-bound — build a new "
                "one per grid")
        names = set(grid.names)
        for mapped in self._maps:
            if mapped in names:
                raise ValueError(
                    f"map {mapped!r} collides with a grid "
                    "coordinate name; analytic maps declare the "
                    "*physical* coordinate a base coordinate maps "
                    "to")
        for recipe in self._recipes.values():
            decl = recipe.decl
            unknown = tuple(n for n in decl.coords
                            if n not in names)
            if unknown:
                raise ValueError(
                    f"unknown coordinates {unknown} in a declared "
                    f"callable; grid coordinates are {grid.names} "
                    "— parameter fields must be declared via "
                    "params=")
        for name, coords in self._param_coords.items():
            unknown = tuple(n for n in coords if n not in names)
            if unknown:
                raise ValueError(
                    f"the default of parameter {name!r} names "
                    f"unknown coordinates {unknown}; grid "
                    f"coordinates are {grid.names}")
        self._grid = grid

    def _bound_grid(self) -> Grid:
        """Return the bound grid, raising when unattached."""
        if self._grid is None:
            raise RuntimeError(
                "this CoordinateMapping is not attached to a grid; "
                "pass it as Grid(..., mapping=...)")
        return self._grid

    # ================================================================
    #  Metric derivation (delegation target of grid.metric)
    # ================================================================
    def metric(
        self,
        space: SpaceLike,
        name: str,
        *,
        params: Mapping[str, ScalarField] | None = None,
    ) -> ScalarField:
        """
        Derive the named metric on the requested staggered space.

        Description
        -----------
        Materializes the metric at the space's evaluation nodes:
        coordinate derivatives by autodiff of the declared callable,
        parameter-field derivatives by the registry ``diff``
        operators plus ``.to`` interpolation (module docstring). The
        result is tagged with the querying space, every factor the
        metric does not involve replaced by its ``ConstantSpace``,
        so it broadcasts exactly under the strict algebra (mirroring
        ``grid.measure``). With ``params=`` given, the supplied
        fields **replace** the static defaults (CS-D4): they enter
        as dynamic arrays and trace through jit without recompiles
        across values.

        Parameters
        ----------
        space : SpaceLike
            The querying space; it must resolve every coordinate
            the metric involves through non-constant nodal/average
            factors.
        name : str
            The metric name (one of ``metric_names``).
        params : Mapping[str, ScalarField] | None, optional
            Caller-supplied parameter fields overriding the static
            defaults (default: None).

        Returns
        -------
        ScalarField
            The metric field, tagged with the querying space.
        """
        grid = self._bound_grid()
        laid = grid._laid_out(space)  # noqa: SLF001 — grid seam
        recipe = self._recipes.get(name)
        if recipe is None:
            raise ValueError(
                f"unknown metric {name!r}; this mapping supplies "
                f"{self.metric_names}")
        explicit = dict(params or {})
        unknown = tuple(key for key in explicit
                        if key not in self._params)
        if unknown:
            raise ValueError(
                f"unknown parameters {unknown} in params=; this "
                f"mapping declares {self.param_names}")
        deps = recipe.deps
        for coord in sorted(deps):
            try:
                factor = laid.factor(coord)
            except KeyError:
                raise ValueError(
                    f"metric {name!r} involves coordinate "
                    f"{coord!r}, which the requested space does "
                    "not resolve") from None
            if isinstance(factor, ConstantSpace):
                # a value error (bad space choice), not a type error
                raise ValueError(  # noqa: TRY004
                    f"metric {name!r} involves coordinate "
                    f"{coord!r}, but the requested space is "
                    "constant along it")
            if isinstance(factor, CoefficientSpace):
                # a value error (bad space choice), not a type error
                raise ValueError(  # noqa: TRY004
                    f"metric fields have no coefficient-space "
                    f"representation; factor along {coord!r} is "
                    f"{factor!r}")
        ctx = _Derivation(self, grid, laid, explicit)
        value = jnp.asarray(recipe.evaluate(ctx))
        factors = tuple(
            factor if any(n in deps for n in factor.names)
            else factor.mesh.constant
            for factor in laid.factors)
        result = _product(list(factors), laid.layout)
        data = jnp.broadcast_to(value, result.shape).astype(
            dtype_real())
        stored = store(grid.decomposition, result, data)
        return ScalarField(grid, result, stored,
                           FieldMetadata.create(name=name))

    # ================================================================
    #  Internal declaration helpers
    # ================================================================
    def _deps(self, decl: _Declared) -> frozenset[str]:
        """Structural coordinate dependence of one declaration."""
        coords = set(decl.coords)
        for param in decl.params:
            coords.update(self._param_coords.get(param, ()))
        return frozenset(coords)

    def _add(self, name: str, recipe: object) -> None:
        """Register one recipe; reject duplicate metric names."""
        if name in self._recipes:
            raise ValueError(
                f"duplicate metric name {name!r} across the "
                "declarations")
        self._recipes[name] = recipe

    def _declare_map(self, mapped: str, decl: _Declared) -> None:
        """Register the derivative recipes of one analytic map."""
        deps = self._deps(decl)
        wrts = dict.fromkeys(decl.coords)
        for param in decl.params:
            wrts.update(
                dict.fromkeys(self._param_coords.get(param, ())))
        for wrt in wrts:
            self._add(f"d{mapped}_d{wrt}",
                      _MapDerivative(decl, wrt, deps))
        if len(decl.coords) == 1:
            base = decl.coords[0]
            self._add(f"d{base}_d{mapped}",
                      _MapInverse(decl, base, deps))

    def _declare_chart(
        self,
        chart: str,  # noqa: ARG002 — names error messages later
        decl: _Declared,
    ) -> None:
        """Register the induced-metric recipes of one chart."""
        deps = self._deps(decl)
        coords = decl.coords
        for i, u in enumerate(coords):
            for j, v in enumerate(coords):
                self._add(f"g_{u}{v}",
                          _ChartEntry(decl, "g", i, j, deps))
                self._add(f"inv_g_{u}{v}",
                          _ChartEntry(decl, "inv_g", i, j, deps))
        self._add("sqrt_g",
                  _ChartEntry(decl, "sqrt_g", 0, 0, deps))

    def _corrections(self) -> dict[str, tuple[str, str]]:
        """
        Collect the couplings the derivative kinds are seeded on.

        Description
        -----------
        The grid-seam behind the ``"physical_diff"`` dispatch row:
        for every analytic map with exactly one base coordinate (a
        mapped column, ``z = Z(sigma, params)``), each coordinate
        the map couples — the base itself and every parameter
        coordinate — maps to ``(mapped, base)``. Two maps coupling
        one coordinate raise (nested mappings are out of scope for
        stage C1). Multi-base analytic maps supply Jacobian metrics
        but seed no derivative kind (no unambiguous column).

        Returns
        -------
        dict[str, tuple[str, str]]
            Coordinate name -> (mapped physical name, base name).
        """
        table: dict[str, tuple[str, str]] = {}
        for mapped, decl in self._maps.items():
            if len(decl.coords) != 1:
                continue
            base = decl.coords[0]
            coupled = dict.fromkeys((base,))
            for param in decl.params:
                coupled.update(dict.fromkeys(
                    self._param_coords.get(param, ())))
            for coord in coupled:
                if coord in table:
                    raise ValueError(
                        f"coordinate {coord!r} is coupled by two "
                        "analytic maps; nested mappings are out of "
                        "scope (coordinate-systems plan, stage C1)")
                table[coord] = (mapped, base)
        return table

    def _chart_coords(self) -> tuple[str, ...] | None:
        """
        Return the chart-coupled base coordinates, or None.

        Description
        -----------
        The grid-seam behind the metric-aware vector-calculus rows
        (coordinate-systems plan, stage C2): a mapping carrying an
        embedding chart (CS-D1) couples exactly the chart's base
        coordinates through the induced metric, and the grid seeds
        the ``"grad"``/``"div"``/``"curl"``/``"laplacian"`` and
        ``"raise_index"``/``"lower_index"`` kinds on them. At most
        one chart can be declared per mapping (the induced-metric
        names collide otherwise), so the coupling is a single
        coordinate tuple.

        Returns
        -------
        tuple[str, ...] | None
            The chart's base coordinates in signature order, or
            None when no chart is declared.
        """
        for decl in self._charts.values():
            return decl.coords
        return None
