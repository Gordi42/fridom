r"""
The shallow-water core module (``sw.Core``): dual scaling variants.

Description
-----------
``Core`` declares the state vocabulary (``u``, ``v``, ``p``), owns the
squared phase speed field ``csqr`` and the DIAGNOSE-stage ``thickness``
field, and contributes the single **linear** wave (pressure-gradient /
geopotential-divergence) term. It is the shallow-water owner of the
``gravity_wave`` scaling mechanism (``fr.scaling``): the two
mutually-exclusive constructor kwarg sets fix the **variant** at
construction —

- **dimensional** (``gravity=`` + ``depth=``): physical parameters,
  the ``csqr`` field is :math:`c^2 = g\,D`, and the traced step
  carries **zero** scaling operations:

  .. math::
      \partial_t \boldsymbol{u} = - \nabla p , \qquad
      \partial_t p = -\nabla\cdot\left(c^2 \boldsymbol{u}\right)

- **nondimensional** (``froude_number=`` + optional ``depth=``
  :math:`\tilde D` profile): the ``csqr`` field is the depth ratio
  :math:`\tilde D` and the geopotential divergence carries the live
  mechanism ratio :math:`(\varepsilon/\mathrm{Fr})^2` **outside** the
  flux (read from ``ctx.params`` at stage time; under the matching
  ``fr.scaling.GravityWave()`` the ratio self-normalizes to an exact
  ``1.0``):

  .. math::
      \partial_t \boldsymbol{u} = - \nabla p , \qquad
      \partial_t p = -\left(\frac{\varepsilon}{\mathrm{Fr}}\right)^2
          \nabla\cdot\left(\tilde D\,\boldsymbol{u}\right)

**Thickness (DIAGNOSE)**: the full geopotential thickness is a
DIAGNOSTIC-lifecycle field ``thickness`` on ``p``'s centre space (the
hydrostatic ``w``/``p_hyd`` precedent), recomputed every substage
before any term runs, so every consumer (the Sadourny advection, the
conserving Coriolis routes, diagnostics) reads one stage-fresh field:

.. math::
    h = c^2 + p \quad\text{(dimensional)}, \qquad
    h = \tilde D + \varepsilon
        \left(\frac{\mathrm{Fr}}{\varepsilon}\right)^2 p
    \quad\text{(nondimensional)}

The nondimensional surface-displacement coefficient is spelled
:math:`\varepsilon\,(\mathrm{Fr}/\varepsilon)^2` VERBATIM — do not
simplify to :math:`\mathrm{Fr}^2/\varepsilon`: under the matching
scaling the ratio is an exact ``1.0`` (the alias row binds
:math:`\varepsilon` and :math:`\mathrm{Fr}` to ONE leaf), which is
what makes the today-parity mapping bitwise.

**Variable depth**: ``depth`` accepts a callable ``D(y)`` (static
profile) or a ``fr.model.ProfileFunction`` ``D(y,t)`` (TDF-D7) in both
variants; the ``csqr`` field is then declared on a meridional
``fr.spatial.Profile`` and the constant ``shallowwater.depth`` scalar
is **not** provided (provides-implies-constancy). A time-dependent
``gravity``/``depth`` (an ``fr.Ramp``, or a law) marks ``csqr``
``time_dependent`` and a SELF_UPDATE stage rewrites it each substage
with the stage-time :math:`g(t)\,D(y,t)` (or :math:`\tilde D(y,t)`).
Pair a spatially varying depth with a Coriolis module carrying
``metric_weight="csqr"`` (the thickness-weighted rotation); the
``sw.Model`` preset checks this via :attr:`Core.variable_depth`.

The rotation is **not** a core term: it is carried by the shared
Coriolis module family, opt-in. The nonlinear Sadourny advection is a
separate module.

Chart grids (coordinate-systems plan, stage C2)
-----------------------------------------------
The same module assembles on a chart-coupled grid (a
``CoordinateMapping`` embedding chart, e.g. the lat-lon sphere):
declare the coordinate names via ``coords=`` and the terms select
the metric-aware path (a static grid property, never a traced
value), resolving the seeded ``"grad"`` / ``"div"`` /
``"raise_index"`` kinds through the grid dispatch — the module never
hand-builds metric compositions.

**Velocity convention (physical components, ruling (c)):** on every
grid the prognostic ``u`` / ``v`` are the **physical** (m/s) velocity
components (``physical_state_components.md`` ruling (c)). The chart
wave term is written for the **contravariant** coordinate
velocities :math:`u^\lambda = \dot\lambda`, :math:`u^\varphi =
\dot\varphi`, so it converts at the seams (``chart.py``, D1): the
geopotential flux consumes :math:`u^i = U_i/\sqrt{g_{ii}}` (sealed
divide at entry), and the raised momentum tendency is rescaled
:math:`\mathrm{d}U_i = \sqrt{g_{ii}}\,\mathrm{d}u^i` at exit — the
metric derived per call via ``grid.metric`` on each component's own
staggered space. The chart-native coordinate velocities are exposed
read-only via ``state.chart`` (ruling (d)). On flat grids the
convention degenerates to the usual physical velocities and the flat
code path is taken verbatim (bitwise; the hard results-neutrality
gate).

The chart wave term is

.. math::
    \partial_t u^i = -\,g^{ij}\,\partial_j p , \qquad
    \partial_t p = -\frac{s}{\sqrt{g}}\,
        \partial_i\left(\sqrt{g}\, c^2 u^i\right)

with :math:`s = 1` (dimensional) or
:math:`s = (\varepsilon/\mathrm{Fr})^2` (nondimensional), via
``grad`` -> ``raise_index`` on the pressure and the flux-form metric
``div`` on the tagged geopotential flux (both in contravariant
components between the seam conversions).
"""
from __future__ import annotations

import inspect
import numbers
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.model.halo_demand import derive_extra_halo
from fridom.model.scheduled_field import ProfileFunction, profile_coords
from fridom.model.stages import Stage, StageKind
from fridom.model.time_dependent import TimeDependent, resolve_at
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.chart import (
    to_contravariant,
    to_physical_tendency,
)
from fridom.shallowwater2.diagnostics import DIAGNOSTICS
from fridom.shallowwater2.modules.immersed_weighting import (
    mask_field,
    scale_divergence,
    weight_flux,
)
from fridom.shallowwater2.state import State
from fridom.shallowwater2.units import (
    COMPONENT_FACTORS,
    coordinate_factors,
)
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.scalars import Variance

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


def _check_coords(coords: object) -> tuple[str, str]:
    """Validate the (zonal, meridional) coordinate names."""
    coords = tuple(coords)
    if (len(coords) != 2  # noqa: PLR2004 — zonal + meridional
            or not all(isinstance(c, str) for c in coords)
            or coords[0] == coords[1]):
        raise TypeError(
            "coords names the (zonal, meridional) coordinates: "
            f"two distinct strings, got {coords!r}")
    return coords


def _check_nonzero(name: str, value: object) -> None:
    r"""Refuse an exactly-zero float parameter (taught, cheap).

    The live scaling ratios divide by these leaves
    (:math:`\varepsilon/\mathrm{Fr}`), and the analytic consumers
    invert :math:`c^2`, so an exact zero would poison the run (and
    its VJP) far from the construction site. Ramp-valued leaves are
    not probed (their endpoints are live leaves).
    """
    if isinstance(value, numbers.Number) and float(value) == 0.0:
        raise TypeError(
            f"sw.Core {name}=0 is refused: the scaling ratios and "
            "the analytic consumers divide by it, so an exact zero "
            "poisons the run far from here; pass a nonzero value")


@partial(jaxify, dynamic=("gravity", "depth", "froude_number",
                          "_depth_law"))
class Core(fr.model.Module):

    r"""
    Shallow-water core: ``u``/``v``/``p``, wave term, thickness.

    Description
    -----------
    See the module docstring. The variant is fixed at construction by
    the kwarg set — dimensional ``gravity=`` + ``depth=`` XOR
    nondimensional ``froude_number=`` (+ optional ``depth=``
    :math:`\tilde D` profile) — and reported through
    :attr:`scaling_variant` for the assembly's ``fr.scaling``
    validation.

    Parameters
    ----------
    gravity : float | fr.model.Ramp | None, optional
        The gravitational acceleration :math:`g` [m/s^2]
        (dimensional variant only); published as
        ``shallowwater.gravity``. A Ramp drives a per-substage
        SELF_UPDATE of the ``csqr`` field (default: None).
    depth : float | Callable | ProfileFunction | None, optional
        The water depth :math:`D` [m] (dimensional; REQUIRED there)
        or the depth ratio :math:`\tilde D` (nondimensional;
        default 1.0). A float publishes ``shallowwater.depth``; a
        callable ``D(y)`` is the static variable depth; a
        ``ProfileFunction`` ``D(y,t)`` (TDF-D7) or an ``fr.Ramp``
        marks ``csqr`` time-dependent (default: None).
    froude_number : float | fr.model.Ramp | None, optional
        The Froude number :math:`\mathrm{Fr}` (nondimensional
        variant only); published as ``shallowwater.froude`` and — as
        the ``gravity_wave`` mechanism owner — aliased by the
        assembly onto ``scaling.nonlinearity`` under
        ``fr.scaling.GravityWave()`` (default: None).
    coords : tuple[str, str], optional
        The (zonal, meridional) coordinate names, in the grid's
        factor order — ``("lon", "lat")`` on the standard sphere
        chart (default: ``("x", "y")``).
    meridional : str | None, optional
        The meridional coordinate name a callable ``depth`` varies
        along; None uses ``coords[1]`` (default: None).
    """

    #: The vocabulary class this core supplies (D1.3 commitment 4).
    state_type = State

    #: Bound parameterful diagnostics (the D1.3 commitment-4 channel).
    diagnostics = DIAGNOSTICS

    #: fr.scaling traits: this family owns the gravity-wave mechanism
    scaling_mechanism = "gravity_wave"
    nonlinearity_attr = "froude_number"

    def __init__(
        self,
        *,
        gravity: float | None = None,
        depth: float | Callable | None = None,
        froude_number: float | None = None,
        coords: tuple[str, str] = ("x", "y"),
        meridional: str | None = None,
    ) -> None:
        """Store the leaves; fix the variant from the kwarg set.

        Raises
        ------
        TypeError
            On invalid ``coords``, a mixed/missing kwarg set, a
            callable ``gravity`` (spatial variation belongs in
            ``depth``), or a zero ``gravity``/``depth``/
            ``froude_number`` (the live ratios divide by them).
        """
        coords = _check_coords(coords)
        if (gravity is None) == (froude_number is None):
            raise TypeError(
                "sw.Core takes exactly one kwarg set: DIMENSIONAL "
                "gravity= + depth= (physical parameters, zero "
                "scaling ops in the trace) XOR NONDIMENSIONAL "
                "froude_number= (+ optional depth= as the depth "
                "ratio profile, default 1.0) under a nondimensional "
                "fr.scaling policy; got "
                f"gravity={gravity!r}, froude_number="
                f"{froude_number!r}")
        if gravity is not None:
            if callable(gravity) and not isinstance(
                    gravity, TimeDependent):
                raise TypeError(
                    "gravity= is the constant gravitational "
                    "acceleration (a float, or an fr.Ramp); spatial "
                    "variation belongs in depth= (a callable D(y) "
                    f"or a ProfileFunction D(y,t)); got {gravity!r}")
            if depth is None:
                raise TypeError(
                    "the dimensional sw.Core needs BOTH gravity= "
                    "and depth= (csqr = g * D); pass depth= (a "
                    "float, a callable D(y), or a ProfileFunction "
                    "D(y,t))")
        elif depth is None:
            depth = 1.0  # nondim: the flat depth ratio
        _check_nonzero("gravity", gravity)
        _check_nonzero("froude_number", froude_number)
        _check_nonzero("depth", depth)
        self.gravity = (None if gravity is None
                        else fr.model.leaf(gravity))
        self.froude_number = (None if froude_number is None
                              else fr.model.leaf(froude_number))
        # depth forms: law | scalar leaf (float / Ramp) | callable
        if isinstance(depth, ProfileFunction):
            self._depth_law: ProfileFunction | None = depth
            self._depth_fn = None
            self.depth = None
        elif isinstance(depth, TimeDependent) or not callable(depth):
            self._depth_law = None
            self._depth_fn = None
            self.depth = fr.model.leaf(depth)
        else:
            self._depth_law = None
            self._depth_fn = depth
            self.depth = None
        self._coords: tuple[str, str] = coords
        self._meridional = (coords[1] if meridional is None
                            else meridional)
        #: the constructor-fixed variant flag (host-side static)
        self._nondim: bool = froude_number is not None
        # whether the bound grid is chart-coupled; set by bind()
        self._charted: bool = False
        # whether the bound grid carries an immersed domain; bind()
        self._immersed: bool = False
        # the chart / immersed wave term's derived halo substitute
        # (V-N2); None on a plain flat grid (the traceable path)
        self._extra_halo: HaloSpec | None = None

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def coords(self) -> tuple[str, str]:
        """The (zonal, meridional) coordinate names."""
        return self._coords

    @property
    def scaling_variant(self) -> str:
        """The constructor-fixed variant (``fr.scaling`` seam)."""
        return "nondimensional" if self._nondim else "dimensional"

    @property
    def variable_depth(self) -> bool:
        """Whether the depth varies in space (profile or law)."""
        return (self._depth_fn is not None
                or self._depth_law is not None)

    @property
    def unit_factors(self) -> dict[str, fr.model.UnitFactor]:
        """Dimensional-factor rows (``model.units``, §D).

        The shallow-water amplitude table
        (:mod:`fridom.shallowwater2.units`) plus the two ``L``-valued
        coordinate rows — an instance property because ``coords=``
        renames the coordinate keys.
        """
        return {**coordinate_factors(*self._coords),
                **COMPONENT_FACTORS}

    # The chart-path wave term resolves the metric-aware kinds,
    # whose multi-row block application the halo tracer cannot
    # follow, so the module declares its stencil width and is
    # halo-trace exempt (V-N2, the Sadourny precedent). On an
    # immersed grid the flux-form continuity multiplies concrete
    # fraction fields (IP-D4) — also exempt. The width is DERIVED at
    # bind from the ``diff`` rows the term applies (width 1). None on
    # a plain flat grid: the unimmersed path stays fully halo-traced
    # (the parity guard).
    @property
    def extra_halo(self) -> HaloSpec | None:
        """Derived staggered ghost width on chart / immersed grids."""
        return self._extra_halo

    @property
    def _csqr_time_dependent(self) -> bool:
        """Whether the ``csqr`` field needs a per-substage rewrite."""
        return (self._depth_law is not None
                or isinstance(self.depth, TimeDependent)
                or isinstance(self.gravity, TimeDependent))

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_declarations(self) -> tuple[fr.model.FieldDeclaration, ...]:
        """U/v/p, the ``csqr`` AUXILIARY, the ``thickness`` DIAGNOSTIC."""
        time_dependent = self._csqr_time_dependent
        if self.depth is not None:
            csqr_decl = fr.model.FieldDeclaration(
                "csqr", space=fr.spatial.Profile(),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._csqr_default,
                long_name="Squared phase speed", units="m^2/s^2",
                time_dependent=time_dependent)
        elif self._depth_law is not None:
            csqr_decl = fr.model.FieldDeclaration(
                "csqr", space=fr.spatial.Profile(self._meridional),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._csqr_law_default,
                long_name="Squared phase speed", units="m^2/s^2",
                time_dependent=True)
        else:
            csqr_decl = fr.model.FieldDeclaration(
                "csqr", space=fr.spatial.Profile(self._meridional),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._csqr_profile_default,
                long_name="Squared phase speed", units="m^2/s^2",
                time_dependent=time_dependent)
        zonal, meridional = self._coords
        return (
            fr.model.FieldDeclaration.velocity(
                "u", zonal, space=fr.spatial.Staggered(zonal),
                long_name=f"Velocity ({zonal})", units="m/s"),
            fr.model.FieldDeclaration.velocity(
                "v", meridional,
                space=fr.spatial.Staggered(meridional),
                long_name=f"Velocity ({meridional})", units="m/s"),
            fr.model.FieldDeclaration(
                "p", space=fr.spatial.Collocated(),
                long_name="Pressure (g*eta)", units="m^2/s^2"),
            csqr_decl,
            fr.model.FieldDeclaration(
                "thickness", space=fr.spatial.Collocated(),
                lifecycle=fr.model.Lifecycle.DIAGNOSTIC,
                long_name="Full geopotential thickness",
                units="m^2/s^2"),
        )

    @property
    def parameter_declarations(
        self,
    ) -> tuple[fr.model.ParameterDeclaration, ...]:
        """The variant's provides (provides-implies-constancy).

        Dimensional: ``shallowwater.gravity`` always, plus
        ``shallowwater.depth`` when the depth is a scalar leaf (a
        spatially varying depth provides no constant).
        Nondimensional: ``shallowwater.froude`` always, plus
        ``shallowwater.depth`` (the depth RATIO) when constant. The
        retired ``shallowwater.csqr`` is provided by neither —
        analytic consumers read the primitives.
        """
        decls: tuple[fr.model.ParameterDeclaration, ...]
        if self._nondim:
            decls = (fr.model.ParameterDeclaration(
                sw_params.FROUDE, attr="froude_number", units="1",
                doc="Froude number (the gravity-wave mechanism)"),)
        else:
            decls = (fr.model.ParameterDeclaration(
                sw_params.GRAVITY, attr="gravity", units="m/s^2",
                doc="gravitational acceleration"),)
        if self.depth is not None:
            decls += (fr.model.ParameterDeclaration(
                sw_params.DEPTH, attr="depth", units="m",
                doc=("water depth" if not self._nondim
                     else "depth ratio (nondimensional)")),)
        return decls

    # ================================================================
    #  csqr materialization (the one shared g*D / D-tilde recipe)
    # ================================================================
    def _csqr_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: fill the one-DOF profile.

        ``g * D`` (dimensional) or the depth ratio (nondimensional),
        with time-dependent leaves materialized at ``t = 0`` (the
        SELF_UPDATE stage rewrites them per substage). No pre-syncing
        (GAP-B).
        """
        value = resolve_at(self.depth, 0.0)
        if not self._nondim:
            value = resolve_at(self.gravity, 0.0) * value
        return grid.create_field(
            space, data=jnp.full(space.shape, value), name="csqr")

    def _csqr_profile_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: materialize the static profile.

        ``g(0) * D(y)`` (dimensional) or the ratio profile; the
        signature is stamped dynamically to match
        ``self._meridional`` (the ``BetaPlaneCoriolis._f_default``
        precedent). No pre-syncing (GAP-B).
        """
        fn, mer = self._depth_fn, self._meridional
        gravity = (None if self._nondim
                   else resolve_at(self.gravity, 0.0))

        def init(**coords: object) -> object:
            depth = fn(coords[mer])
            return depth if gravity is None else gravity * depth

        init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(
                mer, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        return grid.create_field(space, init=init, name="csqr")

    def _csqr_law_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: sample the depth law at ``t = 0``.

        The AUXILIARY field is materialized as the ``t = 0`` snapshot
        so it keeps a valid static treedef; the SELF_UPDATE stage
        rewrites it with the stage-time value each substage (the
        frozen snapshot is never read at run time). No pre-syncing
        (GAP-B).
        """
        coords = profile_coords(grid, space, (self._meridional,))
        data = self._depth_law.sample(coords, 0.0, space.shape)
        if not self._nondim:
            data = resolve_at(self.gravity, 0.0) * data
        return grid.create_field(space, data=data, name="csqr")

    # ================================================================
    #  Stages: SELF_UPDATE (time-dependent csqr) + DIAGNOSE thickness
    # ================================================================
    @property
    def stages(self) -> tuple[Stage, ...]:
        """The csqr rewrite (when time-dependent) + the thickness."""
        stages: tuple[Stage, ...] = ()
        if self._csqr_time_dependent:
            stages += (Stage(
                kind=StageKind.SELF_UPDATE, fn="_update_csqr",
                name="csqr", reads=("csqr",), writes=("csqr",)),)
        stages += (Stage(
            kind=StageKind.DIAGNOSE, fn="_update_thickness",
            name="thickness"),)
        return stages

    def _update_csqr(self, state, ctx) -> dict:  # noqa: ANN001
        """Re-evaluate ``csqr`` at the substage clock (TDF-D7).

        SELF_UPDATE runs first in every substage (S1), so every
        ``csqr`` consumer — the wave term, the thickness DIAGNOSE,
        the Sadourny advection, the thickness-weighted rotation, the
        energy metric, diagnostics — reads the stage-time
        ``g(t) D(y,t)`` (or the ratio profile), consistent with
        ``eval_params``. A static-callable depth under a ramped
        ``gravity`` is re-evaluated on the profile coordinates here
        (the callable must be jax-traceable on that path).
        """
        time = getattr(ctx.clock, "time", ctx.clock)
        field = state["csqr"]
        space = field.function_space
        if self._depth_law is not None:
            coords = profile_coords(field.grid, space,
                                    (self._meridional,))
            value = self._depth_law.sample(coords, time, space.shape)
        elif self._depth_fn is not None:
            coords = profile_coords(field.grid, space,
                                    (self._meridional,))
            value = jnp.broadcast_to(
                jnp.asarray(
                    self._depth_fn(coords[self._meridional])),
                space.shape)
        else:
            value = jnp.full(space.shape,
                             resolve_at(self.depth, time))
        if not self._nondim:
            value = resolve_at(self.gravity, time) * value
        return {"csqr": field.with_data(value)}

    def _update_thickness(self, state, ctx) -> dict:  # noqa: ANN001
        r"""Diagnose the full geopotential thickness (S1').

        Dimensional: :math:`h = c^2 + p`. Nondimensional:
        :math:`h = \tilde D + \varepsilon\,(\mathrm{Fr}/
        \varepsilon)^2\,p` — the ``x/x`` spelling is VERBATIM (module
        docstring): under the matching ``GravityWave`` scaling the
        ratio is an exact ``1.0`` because :math:`\varepsilon` and
        :math:`\mathrm{Fr}` alias one leaf, so the coefficient
        collapses to :math:`\varepsilon` bitwise (today-parity).
        Pointwise on the centre space — halo-safe (at most one extra
        scalar exchange per substage multi-device).
        """
        p = state["p"]
        base = state["csqr"].to(p)
        if not self._nondim:
            return {"thickness": base + p}
        eps = ctx.params[fr.model.params.SCALING_NONLINEARITY]
        froude = ctx.params[sw_params.FROUDE]
        ratio = froude / eps
        return {"thickness": base + eps * (ratio * ratio) * p}

    # ================================================================
    #  Bind-time validation (taught errors)
    # ================================================================
    def bind(self, table) -> None:  # noqa: ANN001
        """On chart grids, require ``coords`` to match the chart.

        Raises
        ------
        ValueError
            If the grid carries an embedding chart whose coordinate
            family does not match ``coords`` in the grid's factor
            order (the metric-aware kinds match vector components
            to axes positionally, so the order is load-bearing).
        NotImplementedError
            If the grid carries **both** an embedding chart and an
            immersed domain: the chart wave/continuity path is
            unmasked, so it would silently ignore the immersed mask
            (silent wrong physics) — the same deferral the Sadourny
            advection guard names. This fires regardless of the
            ``advection`` setting, so even a linear model is refused.
        """
        grid = table.grid
        self._immersed = getattr(grid, "immersed", None) is not None
        chart = grid.chart_coords
        self._charted = chart is not None
        if chart is not None and self._immersed:
            raise NotImplementedError(
                "sw.Core does not support a grid carrying "
                "BOTH an embedding chart and an immersed (cut-cell) "
                "domain: the metric-aware chart wave/continuity "
                "path is unmasked, so it would silently ignore the "
                "immersed mask and let the geopotential flux cross "
                "the wet-region boundary (silent wrong physics — the "
                "fraction-weighted immersed path is flat-only). This "
                "is refused for any sw2 model on such a grid, linear "
                "or not. sw2 mapped+immersed is a recorded follow-up "
                "of the mapped+immersed composition plan; until it "
                "lands, drop the immersed domain or run on an "
                "unmapped (flat) grid.")
        if chart is not None:
            expected = tuple(
                name for name in grid.names if name in set(chart))
            if self._coords != expected:
                raise ValueError(
                    f"sw.Core coords={self._coords!r} do not "
                    f"match the grid's chart coordinates {expected!r} "
                    "(in factor order); pass coords=(zonal, meridional) "
                    "matching the grid, e.g. coords=('lon', 'lat') on "
                    "the standard sphere chart")
        self._extra_halo = self._derive_extra_halo(table)

    def _derive_extra_halo(self, table) -> HaloSpec | None:  # noqa: ANN001
        r"""Derive the chart / immersed wave term's ghost width (V-N2).

        Description
        -----------
        On a plain flat grid the wave term is a traceable staggered
        difference, so no substitute is declared (``None``) —
        **unless** a time-dependent ``csqr`` drives a SELF_UPDATE
        rewrite from raw data (halo-trace exempt, V-N2), in which
        case the module declares the term's reach itself (one ghost
        per axis, merged with any chart/immersed derivation). On a
        **chart** or **immersed** grid it resolves metric-aware /
        fraction-weighted rows the halo trace cannot follow, so the
        module declares its own width — derived from the order-2
        staggered ``diff`` rows the term applies, not a literal
        (reach 1 per coordinate; the cross-interp hop telescopes
        two-sidedly, ``storage_halo_width.md`` §1,
        ``pressure_solver_halo.md``). A registry override of the
        differences moves the value.
        """
        grid = table.grid
        derived: HaloSpec | None = None
        if self._charted or self._immersed:
            registry = grid.dispatch
            p = table["p"].space
            vel = (table["u"].space, table["v"].space)
            grad_leg: dict[str, list[tuple[str, object]]] = {}
            div_leg: dict[str, list[tuple[str, object]]] = {}
            for axis in self._coords:
                centre = p.factor(axis)
                grad_leg[axis] = [("diff", centre)]
                face = next(vs.factor(axis) for vs in vel
                            if vs.factor(axis) is not centre)
                div_leg[axis] = [("diff", face)]
            derived = derive_extra_halo(
                registry, self._coords, [div_leg, grad_leg])
        # a time-dependent csqr SELF_UPDATE rewrites the field from
        # raw sampled data (halo-trace exempt, V-N2), so the module
        # declares the wave term's reach itself: one ghost per axis
        # covers the staggered diff / interp hops (reach 1), merged
        # with any chart/immersed derivation above.
        if self._csqr_time_dependent:
            profile = HaloSpec(dict.fromkeys(grid.names, 1))
            derived = (profile if derived is None
                       else derived.merge_max(profile))
        return derived

    # ================================================================
    #  The wave term (linear)
    # ================================================================
    def _wave_factor(self, ctx) -> object:  # noqa: ANN001
        r"""Return the live ratio :math:`(\varepsilon/\mathrm{Fr})^2`.

        Nondimensional variant only (the dimensional trace never
        calls this). Under the matching ``GravityWave`` scaling the
        alias row binds :math:`\varepsilon` and :math:`\mathrm{Fr}`
        to ONE leaf, so the ratio is an exact ``1.0`` and the
        multiply is bitwise-neutral (today-parity).
        """
        eps = ctx.params[fr.model.params.SCALING_NONLINEARITY]
        froude = ctx.params[sw_params.FROUDE]
        ratio = eps / froude
        return ratio * ratio

    @fr.model.term(advances=("u", "v", "p"), linear=True,
                   linear_fields=("csqr",),
                   linear_params=(
                       fr.model.params.SCALING_NONLINEARITY,
                       sw_params.FROUDE),
                   name="gravity")
    def gravity_term(self, state, ctx) -> dict:  # noqa: ANN001
        r"""Pressure gradient and geopotential divergence.

        .. math::
            \partial_t \boldsymbol{u} = - \nabla p , \qquad
            \partial_t p = -\,s\,
                \nabla\cdot\left(c^2 \boldsymbol{u}\right)

        with :math:`s = 1` (dimensional — the branch carries ZERO
        scaling operations, verbatim the pre-scaling flux form) or
        :math:`s = (\varepsilon/\mathrm{Fr})^2` (nondimensional; the
        factor multiplies OUTSIDE the flux divergence on all three
        paths — flat, immersed, chart — read from ``ctx.params`` at
        stage time). Pure field arithmetic: ``c^2`` sits INSIDE the
        divergence (``(c.to(u) * u).diff("x")``, the flux form) so
        the discrete stencil matches ``diff(c^2 u)``. The ``csqr``
        field lifts from its one-DOF ``fr.spatial.Profile()`` onto
        each velocity face via the ConstantSpace broadcast in
        ``.to``.

        The pressure-gradient entries retag onto their velocities:
        nodal stencil outputs are BC-free, but on a walled grid each
        wall-normal velocity carries the derived Dirichlet wall tag
        on its own axis, so the entry adopts it (the nonhydro
        projection precedent) — identity on periodic grids. The flux
        entries need no retag: ``csqr.to(u)`` adopts the velocity's
        tag (BC-sibling adoption) and the divergence lands BC-free,
        which is ``p``'s space.

        On a chart grid (module docstring) the physical components
        are converted to the contravariant coordinate velocities at
        entry (``chart.py``) and the same physics resolves the
        seeded metric-aware kinds: ``grad`` -> ``raise_index`` turns
        the covariant pressure gradient into the contravariant
        tendency (``-g^{ij} d_j p``), and the flux-form ``div``
        carries the ``sqrt_g``-weighted geopotential flux; the
        raised momentum tendencies are rescaled back to physical at
        exit and the retags restore the velocities' wall tags
        (``dp`` needs no conversion).
        """
        u, v, p = state["u"], state["v"], state["p"]
        csqr = state["csqr"]
        zonal, meridional = self._coords
        if u.grid.chart_coords is None:
            if getattr(u.grid, "immersed", None) is not None:
                return self._gravity_immersed(u, v, p, csqr, ctx)
            dp = (-(csqr.to(u) * u).diff(zonal)
                  - (csqr.to(v) * v).diff(meridional))
            if self._nondim:
                dp = self._wave_factor(ctx) * dp
            return {
                "u": (-p.diff(zonal)).retag(u),
                "v": (-p.diff(meridional)).retag(v),
                "p": dp,
            }
        # entry seam: physical U -> contravariant u^i for the flux
        # (the raised pressure gradient does not read u/v); chart.py
        u = to_contravariant(u, zonal)
        v = to_contravariant(v, meridional)
        dispatch = u.grid.dispatch
        con = Variance.CONTRAVARIANT
        grad = dispatch.resolve("grad", p.function_space.bare)
        gp = grad(p)
        raise_index = dispatch.resolve(
            "raise_index", gp[zonal].function_space.bare)
        raised = raise_index(gp)
        flux = VectorField({
            zonal: (csqr.to(u) * u).with_variance(con),
            meridional: (csqr.to(v) * v).with_variance(con)})
        div = dispatch.resolve(
            "div", flux[zonal].function_space.bare)
        dp = -div(flux)
        if self._nondim:
            dp = self._wave_factor(ctx) * dp
        # exit seam: rescale the contravariant momentum tendencies to
        # physical (dp is a scalar rate — no conversion)
        return {
            "u": to_physical_tendency((-raised[zonal]).retag(u), zonal),
            "v": to_physical_tendency(
                (-raised[meridional]).retag(v), meridional),
            "p": dp,
        }

    def _gravity_immersed(self, u, v, p, csqr, ctx) -> dict:  # noqa: ANN001
        r"""Masked pressure gradient and fraction-weighted continuity.

        Description
        -----------
        The cut-cell weighting of the linear physics (IP-D10):

        .. math::
            \partial_t \boldsymbol{u} = -\,m \odot \nabla p , \qquad
            \partial_t p = -\frac{s}{\theta}\,
                \nabla\cdot\left(\alpha \odot c^2 \boldsymbol{u}\right)

        The geopotential flux :math:`c^2 u` is weighted by the
        open-area face fraction :math:`\alpha_f` before the
        divergence, whose sum is then divided by the wet plan-area
        fraction :math:`\theta_c` (guarded: a dry cell stays exactly
        ``0``) — the ``theta V``-weighted mass ``sum_c theta_c V_c
        p_c`` is conserved to machine zero (the open-area flux
        differences telescope, an :math:`\alpha = 0` face carrying
        none). The pressure gradient is the plain two-point
        difference masked by the boolean per-space face mask
        :math:`m` (``theta > 0``), so no tendency drives a closed
        face. The nondimensional factor :math:`s` multiplies OUTSIDE
        the whole scaled divergence. This path runs **only** on an
        immersed grid; the term is halo-trace exempt here
        (``extra_halo``) because the fraction multiplies drop to
        concrete fields.
        """
        immersed = u.grid.immersed
        zonal, meridional = self._coords
        flux_u = weight_flux(immersed, csqr.to(u) * u)
        flux_v = weight_flux(immersed, csqr.to(v) * v)
        div = -(flux_u.diff(zonal) + flux_v.diff(meridional))
        dp = scale_divergence(immersed, div)
        if self._nondim:
            dp = self._wave_factor(ctx) * dp
        return {
            "u": mask_field(immersed, (-p.diff(zonal)).retag(u)),
            "v": mask_field(immersed, (-p.diff(meridional)).retag(v)),
            "p": dp,
        }
