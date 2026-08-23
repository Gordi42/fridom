r"""The z\* vertical coordinate (Adcroft & Campin 2004).

Description
-----------
Flow-following-coordinates plan, stage Z
(``design/plans/active/flow_following_coordinates_plan.md`` §2/§3).
z\* is the terrain-following sigma column stretched by the **free
surface**: on the base column :math:`z \in [-1, 0]` the physical
height is

.. math::

    z_p = \eta + (H + \eta)\, z ,

so :math:`z = 0` sits at the moving surface :math:`\eta(x, y, t)` and
:math:`z = -1` at the bottom :math:`-H(x, y)`. The layer thickness
breathes uniformly with the surface instead of the top cell absorbing
the whole excursion (the z-coordinate "vanishing top cell" problem).
The metrics follow from the map by the ordinary
:class:`~fridom.spatial.coordinate_mapping.CoordinateMapping`
derivation: the column Jacobian :math:`J = \partial z_p/\partial z =
H + \eta`, its reciprocal, the coordinate-surface slopes
:math:`\partial z_p/\partial x_i = (1 + z)\,\partial_i\eta +
z\,\partial_i H`, and the parameter sensitivity
:math:`\partial z_p/\partial\eta = 1 + z` — the mesh-velocity
ingredient :math:`\dot z_p = (1 + z)\,\dot\eta` that
:class:`~fridom.model.modules.MeshVelocityCorrection` contracts.

Two objects live here: :func:`zstar_mapping` builds the mapping, and
:class:`ZStarGeometry` is the **state-driven geometry module** that
owns :math:`\eta` and :math:`\dot\eta` as model state. The pair is
installed through the preset's ``modules_extra``::

    grid = fr.spatial.Grid(meshes, mapping=hy.zstar_mapping(depth))
    model = hy.Model(
        grid=grid, core=hy.Core(gravity=9.81),
        free_surface=hy.ExplicitFreeSurface(),
        time_stepper=fr.model.time_steppers.AdamBashforth(dt, order=3),
        modules_extra=(hy.ZStarGeometry(),
                       fr.model.modules.MeshVelocityCorrection()))

The geometric conservation law
------------------------------
The survey's one hard rule (plan §2): the mesh-induced thickness
change must be the **same discrete operator** that moves the geometry
parameter — an independently evaluated :math:`\partial_t J` breaks
constancy and tracer conservation (Campin et al. 2004; MITgcm
``dEtaHdt``). Here the map is **linear** in :math:`\eta`, so the
realized thickness change is exactly

.. math::

    J^{n+1} - J^{n}
      = \frac{\partial J}{\partial\eta}\,(\eta^{n+1} - \eta^{n}) ,

and the free-surface family evolves the surface pressure by
:math:`\partial_t p_s = -g\,T^*` with the J-weighted transport
divergence :math:`T^* = \int[\partial_x(Ju) + \partial_y(Jv)]\,
\mathrm{d}z`. This module writes :math:`\eta = p_s/g` and
:math:`\dot\eta = -T^*` by the **same formula at the same substages**,
so with the **explicit** free surface
(:class:`~fridom.hydrostatic.modules.free_surface.ExplicitFreeSurface`)
the discrete GCL holds to round-off: constancy is exact and the
tracer content :math:`\int J b` is conserved semi-discretely.

With the **implicit** or **split-explicit** variants
:math:`p_s^{n+1}` comes from a solve / a subcycle, so the
:math:`\dot\eta` the ALE term consumed differs from the realized
:math:`\Delta\eta/\Delta t` by :math:`O(\Delta t)`. Constancy stays
exact (the mesh-velocity bracket cancels on a uniform field by
construction), but conservation of :math:`\int J b` drifts at that
order — the MITgcm/ROMS "``h`` and :math:`\eta` must not evolve
independently" caveat, a documented owner call (plan §5.1). Use the
explicit variant where conservation is the point.

Validity
--------
The column is non-degenerate only while :math:`J = H + \eta > 0`,
i.e. :math:`\eta > -H`: the surface must stay above the bed. Nothing
clips or guards this (a clip would poison the reverse-mode
derivative); a run that dries a column produces a non-positive
Jacobian and the usual NaN seam catches it downstream.

Scope of iteration 1
--------------------
The dimensional free surface only (:math:`\eta = p_s/g` needs the
physical ``hydrostatic.gravity``; the nondimensional
``froude_number=`` variant is a taught error), and no immersed
(cut-cell) domain — the wet-fraction weighting of :math:`T^*` is not
modelled yet, and a silently unweighted transport divergence would
break the GCL rather than merely lose accuracy.
"""
# Flow-following-coordinates plan, stage Z: the z* coordinate
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import fridom as fr
from fridom.hydrostatic.modules.terrain import (
    discover_column,
    jacobian_name,
)
from fridom.hydrostatic.params import GRAVITY
from fridom.model.modules.moving_geometry import mapping_params
from fridom.spatial.coordinate_mapping import CoordinateMapping
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.fields.metadata import UNKNOWN_UNITS
from fridom.spatial.operators.integrate import Integral

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable

    import jax

    from fridom.spatial.fields.scalar_field import ScalarField

#: the mapping-parameter name of the static column depth — the same
#: spelling the terrain (sigma) mapping ``zp = z * H(x, y)`` uses, so a
#: z* grid reads as "the terrain grid plus a dynamic ``eta``"
DEPTH: str = "H"

#: the horizontal (zonal, meridional) coordinate-name default, shared
#: by the mapping builder and the module
_HORIZONTAL: tuple[str, str] = ("x", "y")


# ================================================================
#  Shared validation
# ================================================================
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


def _named(fn: Callable[..., jax.Array],
           names: tuple[str, ...]) -> Callable[..., jax.Array]:
    """Stamp a keyword-only callable with a named signature.

    Description
    -----------
    ``CoordinateMapping`` reads the *argument names* of every declared
    callable (they select base coordinates and parameters), so a
    builder whose names are configurable must supply them explicitly:
    the body takes ``**coords`` and the advertised
    ``inspect.Signature`` names the arguments. Every call site in the
    mapping passes arguments by keyword, so the stamped signature and
    the body agree.

    Parameters
    ----------
    fn : Callable
        The ``**coords`` body.
    names : tuple[str, ...]
        The advertised argument names, in order.

    Returns
    -------
    Callable
        ``fn`` with ``__signature__`` stamped.
    """
    fn.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
        [inspect.Parameter(name,
                           inspect.Parameter.POSITIONAL_OR_KEYWORD)
         for name in names])
    return fn


def _constant(value: float, names: tuple[str, ...]) -> Callable:
    """Return a constant callable declared on ``names``."""
    def default(**coords: jax.Array) -> jax.Array:
        return value + 0.0 * sum(coords.values())

    return _named(default, names)


def _zstar_map(base: str, eta: str) -> Callable:
    r"""Return the map callable ``zp = eta + (H + eta) * base``."""
    def zstar(**coords: jax.Array) -> jax.Array:
        surface = coords[eta]
        return surface + (coords[DEPTH] + surface) * coords[base]

    return _named(zstar, (base, DEPTH, eta))


# ================================================================
#  The mapping builder
# ================================================================
def zstar_mapping(
    depth: float | Callable[..., jax.Array],
    *,
    mapped: str = "zp",
    base: str = "z",
    eta: str = "eta",
    horizontal: tuple[str, str] = _HORIZONTAL,
) -> CoordinateMapping:
    r"""Build the z\* coordinate mapping ``zp = eta + (H + eta) z``.

    Description
    -----------
    The module docstring's map, declared as an analytic single-base
    column with the **static** depth parameter ``H`` and the
    **dynamic** free-surface parameter ``eta`` (static default
    ``0``, rewritten every substage by :class:`ZStarGeometry`). The
    base column is expected to run from ``-1`` (the bed) to ``0``
    (the surface): with any other extent the map still evaluates,
    but ``H`` stops being the column depth and ``eta`` the surface
    elevation.

    A **stretched** base mesh (``MappedIntervalMesh``) composes: the
    stretching rides ``grid.measure`` and the chart rides
    ``grid.metric``, so a graded base column becomes a graded z\*
    column whose grading breathes with the surface, exactly like the
    uniform one.

    Parameters
    ----------
    depth : float | Callable
        The static column depth ``H``: a positive number (uniform
        depth) or a callable of the horizontal coordinates,
        ``H(x, y)`` — the same declaration a terrain (sigma) mapping
        takes. A callable's own signature declares the coordinates
        ``H`` varies along, and must not exceed `horizontal`.
    mapped : str, optional
        The mapped (physical height) coordinate name
        (default: ``"zp"``).
    base : str, optional
        The base (computational) column coordinate name
        (default: ``"z"``).
    eta : str, optional
        The dynamic free-surface parameter name; :class:`ZStarGeometry`
        names its state fields after it (default: ``"eta"``).
    horizontal : tuple[str, str], optional
        The (zonal, meridional) coordinate names ``eta`` is declared
        on — and ``H`` too when `depth` is a number (default:
        ``("x", "y")``).

    Returns
    -------
    CoordinateMapping
        The z\* mapping, ready for ``Grid(..., mapping=...)``.

    Raises
    ------
    TypeError
        If `horizontal` is not two distinct strings.
    ValueError
        If the coordinate / parameter names collide, or a callable
        `depth` varies along coordinates outside `horizontal`.
    """
    horizontal = _validate_horizontal(horizontal)
    names = (mapped, base, eta, DEPTH)
    if len(set(names)) != len(names):
        raise ValueError(
            f"the z* names must be distinct, got mapped={mapped!r}, "
            f"base={base!r}, eta={eta!r} against the depth parameter "
            f"{DEPTH!r}")
    if callable(depth):
        extra = tuple(name
                      for name in inspect.signature(depth).parameters
                      if name not in horizontal)
        if extra:
            raise ValueError(
                f"the depth callable varies along {extra}, but the "
                f"z* mapping declares the horizontal coordinates "
                f"{horizontal}; a depth that varies along the column "
                "is not a column depth")
        depth_default: Callable = depth
    else:
        depth_default = _constant(float(depth), horizontal)
    return CoordinateMapping(
        maps={mapped: _zstar_map(base, eta)},
        params={DEPTH: depth_default,
                eta: _constant(0.0, horizontal)})


# ================================================================
#  ZStarGeometry: the state-driven free-surface geometry
# ================================================================
class ZStarGeometry(fr.model.Module):

    r"""Drive the z\* mapping's ``eta`` from the free surface.

    Description
    -----------
    The state-driven twin of
    :class:`~fridom.model.modules.MovingGeometry` (whose parameters
    follow a prescribed *schedule*): this module owns the mapping
    parameter ``eta`` and its time derivative ``eta_dot`` as
    AUXILIARY, ``time_dependent`` fields on the barotropic
    ``Profile(zonal, meridional)`` — the same space the free
    surface's ``ps`` lives on — and rewrites both every substage from
    the CURRENT state in one SELF_UPDATE stage (S1, before every
    consumer):

    .. math::

        \eta = \frac{p_s}{g} , \qquad
        \dot\eta = -\int\bigl[\partial_x(Ju) + \partial_y(Jv)\bigr]
                   \,\mathrm{d}z ,

    the kinematic free-surface condition in the J-weighted flux form
    (the free-surface family's ``T^*``), evaluated with the **new**
    :math:`\eta` so the Jacobian the mesh velocity is built from is
    the one the substage's physics sees. Reading other modules' state
    from a SELF_UPDATE stage is the sanctioned idiom (``reads=``; the
    split-explicit free surface's depth-mean snapshot is the
    precedent).

    Geometry consumers pick both fields up through the ordinary
    stage-C4 discovery convention — a state field named exactly after
    the mapping parameter
    (:func:`~fridom.model.modules.moving_geometry.mapping_params`) —
    so nothing here is z\*-specific downstream:
    :class:`~fridom.model.modules.MeshVelocityCorrection` contracts
    ``dzp_deta * eta_dot`` into the ALE term with no changes, and the
    hydrostatic core's metric reads see the current surface.

    See the module docstring for the GCL argument, the
    explicit-vs-implicit consistency caveat and the
    :math:`\eta > -H` validity bound.

    Defaults: both fields allocate to **zeros**, which is the
    consistent initial geometry exactly when ``ps`` starts at zero
    (the ordinary IC). It is also self-correcting: the SELF_UPDATE
    runs first in every substage, so the very first consumer already
    sees ``eta = ps/g`` — the zero default is never read by the step
    path, only by host-side inspection of a freshly assembled model.

    Parameters
    ----------
    eta : str, optional
        The mapping-parameter name; the two state fields are ``eta``
        and ``<eta>_dot`` (default: ``"eta"``).
    vertical : str, optional
        The base column coordinate the transport divergence reduces
        over (default: ``"z"``).
    horizontal : tuple[str, str], optional
        The (zonal, meridional) coordinate names of the horizontal
        transport divergence (default: ``("x", "y")``).
    """

    def __init__(
        self,
        *,
        eta: str = "eta",
        vertical: str = "z",
        horizontal: tuple[str, str] = _HORIZONTAL,
    ) -> None:
        """Store the names; see the class docstring."""
        if not isinstance(eta, str) or not eta:
            raise TypeError(
                "eta names the z* mapping's free-surface parameter "
                f"(a non-empty string), got {eta!r}")
        self._eta: str = eta
        self._dot: str = f"{eta}_dot"
        self._vertical: str = vertical
        self._horizontal: tuple[str, str] = _validate_horizontal(
            horizontal)
        #: the (mapped, base) column, resolved at bind
        self._column: tuple[str, str] | None = None
        #: the grid coordinate names, resolved at bind (extra_halo)
        self._coords: tuple[str, ...] = ()

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def param_names(self) -> tuple[str, ...]:
        """The driven mapping-parameter names (``MovingGeometry`` twin)."""
        return (self._eta,)

    # ================================================================
    #  Declarations and references
    # ================================================================
    @property
    def field_declarations(
        self,
    ) -> tuple[fr.model.FieldDeclaration, ...]:
        """The parameter field ``eta`` and its time derivative.

        Both live on the barotropic ``Profile(zonal, meridional)`` —
        the ``ps`` space, and the base layout the mapping's metric
        derivation aligns supplied parameter fields to, so at
        ``eta == 0`` the dynamic pipeline reproduces the static
        default pipeline bitwise (the frozen-surface gate). Units
        follow the ``MovingGeometry`` convention: a mapping parameter
        carries the ``FieldDeclaration`` sentinel, never a claimed
        unit the scaling would have to render.
        """
        zonal, meridional = self._horizontal
        space = fr.spatial.Profile(zonal, meridional)
        return (
            fr.model.FieldDeclaration(
                self._eta, space=space,
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                long_name=f"Mapping parameter {self._eta} "
                          "(free-surface elevation)",
                units=UNKNOWN_UNITS, time_dependent=True),
            fr.model.FieldDeclaration(
                self._dot, space=space,
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                long_name=f"Mapping parameter {self._eta} time "
                          "derivative",
                units=UNKNOWN_UNITS, time_dependent=True),
        )

    @property
    def field_references(self) -> tuple[fr.model.FieldReference, ...]:
        """``ps`` (the surface) and the horizontal velocities."""
        return (
            fr.model.FieldReference(
                "ps", hint="the z* surface elevation is eta = ps/g; "
                           "the surface pressure is declared by a "
                           "free-surface module, e.g. "
                           "free_surface=hy.ExplicitFreeSurface()"),
            fr.model.FieldReference(
                "u", hint="the kinematic free-surface tendency reads "
                          "the horizontal velocity (hy.Core declares "
                          "u/v)"),
            fr.model.FieldReference(
                "v", hint="the kinematic free-surface tendency reads "
                          "the horizontal velocity (hy.Core)"),
        )

    @property
    def parameter_references(
        self,
    ) -> tuple[fr.model.ParameterReference, ...]:
        """The dimensional gravity converting ``ps`` to ``eta``."""
        return (fr.model.ParameterReference(
            GRAVITY,
            hint="ZStarGeometry converts the surface pressure to the "
                 "free-surface elevation, eta = ps/g, so it needs the "
                 "physical gravitational acceleration: assemble the "
                 "dimensional model, core=hy.Core(gravity=...). The "
                 "nondimensional free-surface variant "
                 "(froude_number=) is not modelled in iteration 1 of "
                 "the z* coordinate"),)

    # ================================================================
    #  Bind-time validation (taught errors)
    # ================================================================
    def bind(self, table) -> None:  # noqa: ANN001
        """Validate the grid's mapping against the driven parameter.

        Raises
        ------
        ValueError
            If the grid carries no ``CoordinateMapping``, the mapping
            declares no such parameter, the parameter is not declared
            on exactly the horizontal coordinates, or ``ps`` is not
            PROGNOSTIC (the rigid lid has no free surface).
        NotImplementedError
            If the grid carries an immersed (cut-cell) domain, or the
            vertical axis is not the base of a single-base analytic
            column (through ``discover_column``).
        """
        grid = table.grid
        mapping = getattr(grid, "mapping", None)
        if mapping is None:
            raise ValueError(
                "ZStarGeometry drives the z* mapping's free-surface "
                "parameter, but the grid carries no coordinate "
                "mapping; build the grid with "
                "Grid(..., mapping=hy.zstar_mapping(depth))")
        declared = mapping.param_coords
        if self._eta not in declared:
            raise ValueError(
                f"ZStarGeometry drives the mapping parameter "
                f"{self._eta!r}, which the grid's mapping does not "
                f"declare (it declares {tuple(declared)}); build the "
                "grid with Grid(..., mapping=hy.zstar_mapping(depth))"
                " — or pass ZStarGeometry(eta=...) naming the "
                "mapping's own parameter")
        coords = declared[self._eta]
        if set(coords) != set(self._horizontal):
            raise ValueError(
                f"the mapping declares {self._eta!r} on coordinates "
                f"{coords}, but the z* free surface lives on the "
                f"horizontal {self._horizontal}: eta must vary on "
                "exactly the horizontal coordinates (a column-varying "
                "or degenerate eta is not a free surface). Align the "
                "mapping's params= default with "
                "ZStarGeometry(horizontal=...)")
        if getattr(grid, "immersed", None) is not None:
            raise NotImplementedError(
                "the z* coordinate does not model an immersed "
                "(cut-cell) domain yet (flow-following-coordinates "
                "plan, stage Z): the kinematic surface tendency would "
                "need the wet-fraction-weighted transport divergence, "
                "and an unweighted one breaks the geometric "
                "conservation law rather than merely losing accuracy. "
                "Assemble on a z* grid without an immersed mask, or "
                "use the static terrain (sigma) column")
        if table["ps"].lifecycle is not fr.model.Lifecycle.PROGNOSTIC:
            raise ValueError(
                "the z* coordinate is the free surface's own vertical "
                "coordinate, but this model's 'ps' is "
                f"{table['ps'].lifecycle.name}: the rigid lid "
                "(hy.ImplicitFreeSurface(epsilon=0)) carries no "
                "surface elevation at all — its 'ps' is the Lagrange "
                "multiplier enforcing a non-divergent depth mean, not "
                "g*eta, so eta = ps/g would be meaningless. Assemble "
                "with a genuine free surface (epsilon > 0, or "
                "hy.ExplicitFreeSurface / hy.SplitExplicitFreeSurface)"
                ", or drop ZStarGeometry and run the static terrain "
                "(sigma) column")
        self._column = discover_column(grid, self._vertical)
        self._coords = tuple(grid.names)

    # ================================================================
    #  The SELF_UPDATE stage (S1, per substage)
    # ================================================================
    @property
    def extra_halo(self) -> HaloSpec:
        """Two halo cells per coordinate (interpolate + difference).

        The surface tendency multiplies the column Jacobian onto the
        velocity faces (a ``grid.metric`` derivation the halo tracer's
        ``_TracerGrid`` cannot materialize, V-N2) and differences the
        product, so the reach is the metric's own staggering hop plus
        the C-grid difference — depth two, the
        ``MeshVelocityCorrection`` declaration pattern.
        """
        return HaloSpec(dict.fromkeys(self._coords, 2))

    @property
    def stages(self) -> tuple[fr.model.Stage, ...]:
        """The per-substage surface update (SELF_UPDATE, S1)."""
        return (
            fr.model.Stage(
                kind=fr.model.StageKind.SELF_UPDATE,
                fn="_update_geometry", name="zstar_geometry",
                reads=("ps", "u", "v", self._eta, self._dot),
                writes=(self._eta, self._dot)),
        )

    def _update_geometry(self, state, ctx) -> dict:  # noqa: ANN001
        r"""Rewrite ``eta = ps/g`` and ``eta_dot = -T^*``.

        SELF_UPDATE runs first in every substage (S1), so every
        geometry consumer of the substage — the DIAGNOSE stages, the
        tendency terms, the ALE correction — sees the surface and the
        mesh velocity at the stage time, consistent with
        ``eval_params``. The transport divergence is evaluated with
        the **new** ``eta`` (the substage's own geometry), which is
        what makes the realized ``Delta J`` and the ALE term's
        ``D_b(zp_dot)`` the same discrete operator.
        """
        gravity = ctx.params[GRAVITY]
        eta = state[self._eta]
        eta_new = eta.with_data(state["ps"].data / gravity)
        params = dict(mapping_params(state, eta.grid) or {})
        params[self._eta] = eta_new
        transport_div = self._transport_div(state, params)
        return {
            self._eta: eta_new,
            self._dot: state[self._dot].with_data(
                -transport_div.data),
        }

    def _transport_div(self, state, params: dict) -> ScalarField:  # noqa: ANN001
        r"""Return the J-weighted transport divergence ``T^*``.

        Description
        -----------
        The flux-form horizontal transport divergence
        ``T^* = \int[\partial_x(Ju) + \partial_y(Jv)]\,\mathrm{d}z``
        on the ``ps`` cell — the spelling of the free-surface family's
        ``_terrain_transport_div``, with the column Jacobian read from
        the CURRENT parameter fields through the ``params=`` seam
        (that thread is what makes the surface tendency and the
        barotropic ``\partial_t p_s = -g\,T^*`` the same discrete
        quantity). Duplicated rather than imported: the free-surface
        family owns its own copy and the two evolve independently.

        Parameters
        ----------
        state : object
            The current state (reads ``u`` and ``v``).
        params : dict
            The dynamic mapping-parameter fields (carrying the new
            ``eta``).

        Returns
        -------
        ScalarField
            ``T^*`` on the barotropic ``Profile`` cell.
        """
        zonal, meridional = self._horizontal
        u, v = state["u"], state["v"]
        jname = jacobian_name(self._column)
        ju = u * u.grid.metric(u.function_space.bare, jname,
                               params=params)
        jv = v * v.grid.metric(v.function_space.bare, jname,
                               params=params)
        div_h = ju.diff(zonal) + jv.diff(meridional)
        return Integral()[self._vertical](div_h)
