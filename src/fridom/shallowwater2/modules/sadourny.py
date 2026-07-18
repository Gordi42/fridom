r"""
The Sadourny [1975] energy/enstrophy-conserving advection scheme.

Description
-----------
The nonlinear shallow-water advection, written in the vector-invariant
form that conserves the discrete total energy (and, in the enstrophy
variant, potential enstrophy):

.. math::
    \partial_t \boldsymbol{u} =
        - \underset{\neg}{\boldsymbol{f_u}}\,q
        - \nabla\, E_\mathrm{kin} , \qquad
    \partial_t p = - \nabla\cdot\left(\boldsymbol{u}\,p\right)

with the mass flux :math:`\boldsymbol{f_u} = p_\mathrm{full}
\boldsymbol{u}`, the potential vorticity
:math:`q = \zeta / p_\mathrm{full}`, and the full geopotential
thickness :math:`p_\mathrm{full} = c^2 + \mathrm{Ro}\,p`. Every term
is scaled by the Rossby number ``scaling.rossby`` (read from
``ctx.params``); the module owns no numeric leaves.

**Signed delta vs the old model (§8.8):** the old scheme read the
*scalar* ``csqr`` in ``p_full`` (an outright bug on variable depth).
This port reads the ``csqr`` **field** — ``state["csqr"]`` — so
``p_full`` is spatially correct. Bitwise-identical to the old scheme
only for constant depth.

Walled grids (free-slip)
------------------------
On bounded mesh factors (topology-driven walls) the scheme realizes
**impermeable free-slip** walls; the treatment is fully explicit —
no boundary values are patched numerically (the old model's
``nan_to_num(q)`` corresponds to the free-slip corner tag below).

- *Impermeability* is structural: the wall-normal velocity lives on
  interior faces only (``Inner``, Dirichlet), the wall face is a
  boundary condition, not a DOF. Thickness and momentum fluxes
  through the wall vanish through the Dirichlet ghost fill of the
  flux divergences, so **mass is conserved exactly**.
- *Free slip* enters in exactly one place: the vorticity
  :math:`\zeta = \partial_x v - \partial_y u` is assembled from
  BC-free stencil outputs and **retagged onto the Dirichlet corner
  space** (wall-normal factor of each velocity). Declaring
  :math:`\partial_y u` odd across a ``y`` wall asserts zero wall
  shear, i.e. :math:`\zeta = 0` and hence :math:`q = 0` **at** the
  wall — the free-slip condition. Wall corners carry no DOFs; the
  choice acts only through the one-layer ghost fills below.
- The corner chain (:math:`q`, the corner mass fluxes) inherits the
  Dirichlet tag. A tag is a *wall-value claim* consumed by the
  staggered fills, and every wall value the scheme actually reads is
  an exact zero: :math:`(f_v q)`, :math:`(f_u q)` and :math:`v^2` /
  :math:`u^2` all contain a factor that vanishes on the wall (the
  wall-normal mass flux by impermeability, :math:`q` by free slip).
  The corner-interpolated thickness is also Dirichlet-tagged as a
  formal device; its (false) fill is never consumed.

**Discrete budgets on the walled grid** (verified to machine
precision, channel and doubly-walled):

- The vorticity-flux exchange — the ``u`` equation's
  :math:`+\overline{f_v q}` against the ``v`` equation's
  :math:`-\overline{f_u q}` — is an antisymmetric coupling under the
  thickness-weighted energy sum for **any** finite corner ``q``:
  transposing the wall-zero average lands exactly on the interior
  corners, so energy conservation does *not* depend on the
  :math:`q = 0` choice, only on the zero *wall value of the flux*
  (exact, since the wall-normal mass flux vanishes).
- The total energy
  :math:`E = \sum \tfrac12 \bar{h}^x u^2 + \tfrac12 \bar{h}^y v^2
  + \tfrac12 p^2` (with :math:`h = p_\mathrm{full}`) is conserved
  **exactly** (semi-discretely) by this term *together with* the
  core's linear gravity term — the mass flux :math:`u h` is split
  across the two terms, so the invariant belongs to the pair. The
  kinetic-energy gradient / mass-flux pair telescopes because the
  scheme's :math:`E_\mathrm{kin}` uses the same wall-zero average of
  :math:`v^2` (exact: :math:`v = 0` on the wall).
- :math:`q = 0` at the wall governs the **circulation/enstrophy**
  budget instead: the advective vorticity flux through the wall
  corners is identically zero (no boundary vorticity/enstrophy
  source), and the total circulation changes only through the
  along-wall tangential acceleration.
- The split Coriolis module (``f`` outside ``q``) conserves the
  quadratic M-energy exactly but commits an :math:`O(\mathrm{Ro})`
  commutator error in the thickness-weighted energy — as in v1.
  ``sw.modules.CoriolisEnergyCorrection`` (and the conserving
  ``sw.modules.NonlinearFPlaneCoriolis`` family) restores the
  ``f``-part of the PV flux — the *same* corner averages this scheme
  uses, with :math:`\zeta` replaced by :math:`f` — and closes the
  invariant exactly, rotation included (``sw.modules.coriolis``).

Prescribed background flow (``background=``)
--------------------------------------------
With a prescribed, stationary, divergence-free background
:math:`\boldsymbol{u}_b` the scheme contributes a **second, linear**
term (V-S3: a separate ``linear=True`` term, so ``fr.model.linearize``
keeps it while dropping the nonlinear self-advection):

.. math::
    \partial_t \boldsymbol{u} \mathrel{-}=
        \nabla\cdot\left(\boldsymbol{u}_b \otimes \boldsymbol{u}
        \right), \qquad
    \partial_t p \mathrel{-}=
        \nabla\cdot\left(\boldsymbol{u}_b\, p\right)

in the old scheme's flux form: the momentum fluxes are the centred
products ``mean(u_b) * mean(u)`` at the cell centres /
:math:`\zeta`-corners, the pressure flux is ``u_b * mean(p)`` on the
velocity faces. Because the pressure flux is bilinear, the
background term plus the nonlinear term reproduce the single
full-velocity flux :math:`-\nabla\cdot((\boldsymbol{u}' +
\boldsymbol{u}_b)\,p)` exactly (to rounding); the vorticity-flux /
kinetic-energy machinery of the nonlinear term stays pure
perturbation — exactly the old module's term split.

**Scaling (signed delta vs the old model):** the old stack set
``tendencies.advection.scaling = rossby_number`` and that factor
multiplied the background transport too, so old users passed
*pre-scaled* backgrounds (the effective advecting velocity was
``Ro * u_b``). Here the background term carries **no** Rossby
factor: ``u_b`` is an O(1) velocity, like every other linear
module's coefficient. Migrating an old setup, pass the physical
(unscaled) background.

The background samples at each velocity component's own staggered
nodes (AUXILIARY fields ``u_background`` / ``v_background``,
materialized at assembly — the profile-sampling precedent) and is
validated at bind: the **discrete** divergence of the sampled
background must vanish (the term split and the exact pressure/tracer
identities assume :math:`\nabla\cdot\boldsymbol{u}_b = 0`
*discretely* — derive the background from a streamfunction), and on
bounded axes the wall-normal component must vanish at the wall
(impermeability; structural for the sampled field, checked on the
user's input). The wall flux stencils reuse the free-slip retag /
Dirichlet-fill machinery of the nonlinear term: every wall value
the background term reads contains the wall-normal background
factor, an exact zero.

**Energy:** the background term alone conserves the plain quadratic
energy :math:`\sum \tfrac12(u^2 + v^2 + p^2)` exactly
(semi-discretely, periodic and walled): flux-form transport by a
discretely divergence-free field telescopes. The thickness-weighted
invariant of the gravity + self-advection pair is *not* conserved
in the presence of a background (the background exchanges energy
with the perturbation through the :math:`h`-weighting) — expect an
:math:`O(\mathrm{Ro})` drift of that functional, controlled by the
time-integration error.

Chart grids (coordinate-systems plan, stage C2)
-----------------------------------------------
On a chart-coupled grid (``grid.chart_coords`` is not None) the
scheme takes the metric-aware path with the prognostic velocities
as **contravariant** components (the convention recorded in
``core.py``); coordinate names come from ``coords=``. The
vector-invariant form generalizes with the mass fluxes
:math:`F^i = \sqrt{g}\,h\,u^i`:

.. math::
    \partial_t u_\lambda = +\,q\,\overline{F^\varphi}
        - \partial_\lambda E_\mathrm{kin} , \qquad
    \partial_t u_\varphi = -\,q\,\overline{F^\lambda}
        - \partial_\varphi E_\mathrm{kin} , \qquad
    \partial_t p = -\frac{\mathrm{Ro}}{\sqrt{g}}\,
        \partial_i F^i ,

with :math:`q = \zeta / h`, the relative vorticity the metric curl
of the **lowered** components
:math:`\zeta = (\partial_\lambda u_\varphi^{cov} -
\partial_\varphi u_\lambda^{cov})/\sqrt{g}`, and the covariant
momentum tendencies **raised** (``raise_index``) onto the
prognostic contravariant components. The thickness divergence
resolves the seeded flux-form ``"div"`` kind, vorticity the
``"curl"`` kind, index moves the ``"raise_index"`` /
``"lower_index"`` kinds — module overrides propagate; only the
scheme-specific corner flux/PV averaging multiplies
``grid.metric`` fields directly (derived per application, never
cached).

**Metric-forced discrete choices (recorded):**

- the corner mass fluxes are the *same* :math:`\sqrt{g}`-weighted
  fluxes entering the thickness divergence, interpolated to the
  corner (``sqrt_g`` derived on each velocity's own space —
  staggered-consistent with the ``div`` kind's internal
  derivation); this is what makes the vorticity-flux exchange
  antisymmetric under the thickness-weighted energy
  :math:`E = \sum \sqrt{g}\,\bar h\,g_{ii}(u^i)^2/2 + \dots` —
  the transpose of the corner average lands exactly on
  :math:`F^i`.
- the kinetic energy at centres interpolates the
  :math:`\sqrt{g}`-weighted quadratic and divides by the centre
  :math:`\sqrt{g}`:
  :math:`E_\mathrm{kin} = (\overline{\sqrt{g}\,g_{\lambda\lambda}
  (u^\lambda)^2} + \overline{\sqrt{g}\,g_{\varphi\varphi}
  (u^\varphi)^2}) / (2\sqrt{g})`, the placement that makes the
  KE-gradient / mass-flux pair telescope against the
  :math:`\bar h`-tendency of the same energy functional (the flat
  proof carries over line by line, every step weighted by the
  pointwise :math:`\sqrt{g}`); exactness holds to the rounding of
  the :math:`g_{ii}\,g^{ii}` round-trip (~1 ulp per node).

On the identity chart (X = (x, y)) every metric factor is an exact
1.0 and the chart path reproduces the flat scheme **bitwise** (the
flat-limit gate in ``tests/validation/test_spherical_shallowwater``).
The wall machinery is unchanged: the polar caps of a bounded-lat
sphere are the same Dirichlet/free-slip closures as the Cartesian
channel walls (impermeability structural, :math:`\zeta = 0` claimed
at the wall by the corner retag, every consumed wall value an exact
zero). The prescribed ``background=`` flow is **not** generalized to
chart grids (taught error at bind).
"""
from __future__ import annotations

import inspect
import numbers
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

import fridom as fr
from fridom.shallowwater2.modules.immersed_weighting import (
    mask_field,
    scale_divergence,
    weight_flux,
)
from fridom.shallowwater2.state import _wall_free
from fridom.spatial.bc import BC
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.fields.scalar_field import ScalarField
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.scalars import Variance
from fridom.spatial.spaces.constant import ConstantSpace

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable, Mapping

    from fridom.spatial.grid import Grid

#: wall-normal background components must vanish at the wall to this
#: relative tolerance (impermeability)
_WALL_TOL = 1e-12

#: the sampled background's discrete divergence must cancel to this
#: fraction of its individual flux-difference terms
_DIV_TOL = 1e-11

_BG_COMPONENTS = ("u", "v")


def _potential_vorticity(
    zeta: ScalarField, thickness: ScalarField,
) -> ScalarField:
    r"""Return the potential vorticity :math:`q = \zeta / h`.

    Description
    -----------
    The full thickness ``h`` (``p_full`` interpolated to the vorticity
    corner) is an exact zero in the unsealed ghost/padding cells — the
    outermost ring is never valid at this point of the step, and there
    ``zeta`` vanishes too, so the bare quotient is a masked ``0/0``. The
    forward pass discards those cells (sealed/stripped before any
    output), but **reverse-mode autodiff does not**: the quotient VJP
    (:math:`-\zeta/h^2` with :math:`h = 0`) turns the zero cotangent of
    a sealed cell into ``0 * inf = NaN`` and poisons every gradient with
    a data path. Replacing the exact-zero denominators by 1 keeps ``q``
    finite there; valid cells (``h != 0``) divide by the true thickness
    and are bitwise unchanged (same ghost-validity claim), forward and
    reverse. The guard is a no-op on the interior of any gravity grid
    (``h`` is bounded below by :math:`c^2 > 0` there).
    """
    guarded = jnp.where(thickness.storage == 0.0, 1.0, thickness.storage)
    safe = ScalarField(
        thickness.grid, thickness.function_space, guarded,
        thickness.metadata, halo_valid=thickness.halo_valid)
    return zeta / safe


def _sealed_metric_divide(
    num: ScalarField, den: ScalarField,
) -> ScalarField:
    r"""Return ``num / den`` with the metric denominator VJP-sealed.

    Description
    -----------
    The chart kinetic energy divides the lowered velocity quadratics by
    the centre metric :math:`\sqrt g` (``sqg_p``). On a walled chart
    (the lat-lon sphere's polar caps) that weight is an **exact zero**
    in the never-valid storage/halo padding, where the numerator
    vanishes too, so the bare quotient is a masked ``0/0``. The forward
    pass strips those cells (sealed/stripped before any output), but
    reverse-mode autodiff does not: the quotient VJP
    (:math:`-\mathrm{num}/\mathrm{den}^2` with ``den == 0``) turns the
    zero cotangent of a sealed cell into ``0 * inf = NaN`` and poisons
    every gradient with a data path through the Sadourny kinetic energy
    — the same masked singularity ``_potential_vorticity`` cures for the
    PV divide, and ``coriolis._safe_metric_divide`` for the rotation
    weights. Replacing the exact-zero denominators by 1 keeps the ratio
    finite there; valid cells (``den != 0``) divide by the true metric
    and are bitwise unchanged, forward and reverse. Like
    ``_potential_vorticity`` (its sibling in this method), this runs only
    on real storage — the chart advection tendency is never halo-traced
    with storage-less operands — so no storage-less escape hatch is
    needed.
    """
    guarded = jnp.where(den.storage == 0.0, 1.0, den.storage)
    safe = ScalarField(
        den.grid, den.function_space, guarded,
        den.metadata, halo_valid=den.halo_valid)
    return num / safe


class SadournyAdvection(fr.model.Module):

    r"""
    Energy/enstrophy-conserving advection for ``u``, ``v``, ``p``.

    Description
    -----------
    Reads ``u``, ``v``, ``p`` and the ``csqr`` field (all owned by
    the core, declared here as references) and the ``scaling.rossby``
    parameter. Contributes one nonlinear term transporting all three
    prognostic components by name (the D1.4 name-coupling; no role
    selection) and, when a background flow is prescribed, one
    additional **linear** background-advection term (module
    docstring). Works on periodic, channel, and doubly-walled grids;
    on bounded axes the scheme realizes impermeable free-slip walls
    (module docstring).

    Parameters
    ----------
    background : Mapping[str, float | Callable] | None, optional
        Prescribed stationary background flow: maps the velocity
        component names ``"u"`` / ``"v"`` to a constant or to a
        coordinate callable (e.g. ``{"u": lambda y: -psi_y(y)}``),
        sampled at each component's own staggered nodes; a missing
        component is zero. Must be discretely divergence-free and
        impermeable on walled axes (validated at bind, taught
        errors). Carries **no** Rossby factor — pass the physical,
        unscaled velocity (the old stack scaled the background by
        ``Ro``; see the module docstring). ``None`` disables the
        background term entirely (default: None).
    """

    field_references = (
        fr.model.FieldReference("u", hint="a shallow-water core"),
        fr.model.FieldReference("v", hint="a shallow-water core"),
        fr.model.FieldReference("p", hint="a shallow-water core"),
        fr.model.FieldReference("csqr", hint="a shallow-water core"))

    parameter_references = (
        fr.model.Param(fr.model.params.SCALING_ROSSBY, default=1.0),)

    # The Rossby scaling multiplies a traced ``ctx.params`` scalar
    # into the tendency (a raw-data op the halo tracer cannot follow),
    # so the module declares its stencil width and is halo-trace
    # exempt (V-N2): the vector-invariant scheme reaches two cells in
    # each direction (nested interpolations to the vorticity corner);
    # the background fluxes and the chart path (whose worst chain is
    # one difference plus one interpolation hop) stay within the
    # same reach.
    @property
    def extra_halo(self) -> HaloSpec:
        """Two halo cells per coordinate (the corner chain)."""
        return HaloSpec(dict.fromkeys(self._coords, 2))

    def __init__(
        self,
        background: Mapping[str, float | Callable] | None = None,
        *,
        coords: tuple[str, str] = ("x", "y"),
    ) -> None:
        """Normalize and locally validate the background slot."""
        coords = tuple(coords)
        if (len(coords) != 2  # noqa: PLR2004 — zonal + meridional
                or not all(isinstance(c, str) for c in coords)
                or coords[0] == coords[1]):
            raise TypeError(
                "coords names the (zonal, meridional) coordinates: "
                f"two distinct strings, got {coords!r}")
        self._coords: tuple[str, str] = coords
        self._bg_axes = {"u": coords[0], "v": coords[1]}
        if background is None:
            self._background: dict[str, float | Callable] | None = (
                None)
            return
        items = dict(background)
        unknown = sorted(set(items) - set(_BG_COMPONENTS))
        if unknown:
            raise TypeError(
                f"background= keys are the velocity component names "
                f"{_BG_COMPONENTS}, got the unknown {unknown}")
        if not items:
            raise ValueError(
                "background= needs at least one of the components "
                f"{_BG_COMPONENTS}; None disables the background "
                "term")
        for name, value in items.items():
            if not (isinstance(value, numbers.Number)
                    or callable(value)):
                raise TypeError(
                    f"background[{name!r}] must be a constant or a "
                    f"coordinate callable, got {value!r}")
        self._background = {
            name: items.get(name, 0.0) for name in _BG_COMPONENTS}

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def background(self) -> dict[str, float | Callable] | None:
        """The prescribed background flow (or None)."""
        return (None if self._background is None
                else dict(self._background))

    @property
    def coords(self) -> tuple[str, str]:
        """The (zonal, meridional) coordinate names."""
        return self._coords

    # ================================================================
    #  Background declarations (AUXILIARY, materialized at assembly)
    # ================================================================
    @property
    def field_declarations(self) -> tuple[fr.model.FieldDeclaration, ...]:
        """The sampled background velocities (when prescribed)."""
        if self._background is None:
            return ()
        zonal, meridional = self._coords
        return (
            fr.model.FieldDeclaration(
                "u_background",
                space=fr.spatial.Staggered(
                    zonal, wall_bc={zonal: BC.DIRICHLET}),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._u_background_default,
                long_name=f"Background velocity ({zonal})",
                units="m/s"),
            fr.model.FieldDeclaration(
                "v_background",
                space=fr.spatial.Staggered(
                    meridional, wall_bc={meridional: BC.DIRICHLET}),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._v_background_default,
                long_name=f"Background velocity ({meridional})",
                units="m/s"),
        )

    def _u_background_default(
        self, grid, space,  # noqa: ANN001
    ) -> ScalarField:
        """Owner-method default: sample ``u_b`` at the u nodes."""
        return self._sample(grid, space, "u", "u_background")

    def _v_background_default(
        self, grid, space,  # noqa: ANN001
    ) -> ScalarField:
        """Owner-method default: sample ``v_b`` at the v nodes."""
        return self._sample(grid, space, "v", "v_background")

    def _sample(
        self, grid: Grid, space, component: str,  # noqa: ANN001
        name: str,
    ) -> ScalarField:
        """Materialize one background component on its own nodes.

        A constant fills the space; a callable is sampled at the
        component's staggered nodes, with the callable free to name
        any subset of the coordinates (the wrapper stamps the full
        signature ``create_field`` expects — the profile-sampling
        precedent).
        """
        value = self._background[component]
        if not callable(value):
            data = jnp.full(space.shape, value)
            return grid.create_field(space, data=data, name=name)
        fn = value
        wanted = tuple(inspect.signature(fn).parameters)
        names = tuple(
            factor_name
            for factor in space.factors
            if not isinstance(factor, ConstantSpace)
            for factor_name in factor.names)

        def init(**coords: object) -> object:
            return fn(**{key: coords[key] for key in wanted})

        init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(
                factor_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD)
             for factor_name in names])
        return grid.create_field(space, init=init, name=name)

    # ================================================================
    #  Bind-time validation of the background (taught errors)
    # ================================================================
    def bind(self, table) -> None:  # noqa: ANN001
        """Validate the prescribed background against the grid.

        Description
        -----------
        Sampling and both checks are host-side and transient (the
        real AUXILIARY fields materialize at assembly step 8 through
        the declared defaults, from the same initializers).

        Raises
        ------
        ValueError
            If the grid carries an embedding chart whose coordinate
            family does not match ``coords`` (in factor order), if
            a background callable names a coordinate the grid does
            not have, if a wall-normal background component does
            not vanish on its wall, or if the sampled background is
            not discretely divergence-free.
        NotImplementedError
            If a background flow is prescribed on a chart grid (the
            background term's flux stencils are not generalized to
            the metric path).
        """
        grid = table.grid
        if (self._background is not None
                and getattr(grid, "immersed", None) is not None):
            raise NotImplementedError(
                "background= is not supported on immersed (cut-cell) "
                "grids: the background-transport flux stencils are "
                "unmasked, so a prescribed background would silently "
                "advect across the wet-region boundary (immersed-"
                "partial-cells plan, IP-D8 — masked background "
                "transport is designed-for). Run the immersed model "
                "without a prescribed background flow.")
        chart = grid.chart_coords
        if chart is not None:
            expected = tuple(
                name for name in grid.names if name in set(chart))
            if self._coords != expected:
                raise ValueError(
                    f"SadournyAdvection coords={self._coords!r} do "
                    "not match the grid's chart coordinates "
                    f"{expected!r} (in factor order); pass "
                    "coords=(zonal, meridional) matching the grid")
            if self._background is not None:
                raise NotImplementedError(
                    "background= is not supported on chart grids: "
                    "the background-transport flux stencils are "
                    "Cartesian (metric generalization pending); "
                    "run the sphere without a prescribed "
                    "background flow")
        if self._background is None:
            return
        self._check_coordinate_names(grid)
        fields = {
            component: self._sample(
                grid, table[f"{component}_background"].space,
                component, f"{component}_background")
            for component in _BG_COMPONENTS}
        self._check_wall_normal(grid, fields)
        self._check_divergence_free(fields)

    def _check_coordinate_names(self, grid: Grid) -> None:
        """Raise if a background callable names a bad coordinate."""
        for component in _BG_COMPONENTS:
            value = self._background[component]
            if not callable(value):
                continue
            wanted = tuple(inspect.signature(value).parameters)
            unknown = sorted(set(wanted) - set(grid.names))
            if unknown:
                raise ValueError(
                    f"the background[{component!r}] callable names "
                    f"the coordinate(s) {unknown}, which the grid "
                    f"does not have (coordinates: {grid.names}); "
                    "background callables vary along the "
                    "coordinates their parameters name")

    def _check_wall_normal(
        self, grid: Grid, fields: dict[str, ScalarField],
    ) -> None:
        """Impermeability: wall-normal background zero at the wall.

        The sampled field is structurally impermeable (the wall face
        is not a DOF), so the check runs on the *user's input*: a
        constant must be zero, a callable must evaluate to zero at
        the wall positions (over the tangential nodes it names).
        """
        for component in _BG_COMPONENTS:
            axis = self._bg_axes[component]
            mesh = next(
                (factor for factor in grid.factors
                 if axis in factor.names), None)
            if mesh is None or getattr(mesh, "periodic", True):
                continue
            value = self._background[component]
            scale = max(
                1.0,
                float(np.max(np.abs(np.asarray(
                    fields[component].data)))))
            if not callable(value):
                worst = abs(float(value))
            else:
                worst = self._wall_values(
                    grid, fields[component], value, axis,
                    mesh.extent)
            if worst > _WALL_TOL * scale:
                raise ValueError(
                    f"background[{component!r}] does not vanish at "
                    f"the {axis!r} wall (max wall value {worst:.3e})"
                    ": the walls are impermeable, so the wall-normal"
                    " background component must be zero on the wall "
                    "— derive the background from a streamfunction "
                    "that is constant along each wall")

    @staticmethod
    def _wall_values(
        grid: Grid, field: ScalarField, fn: Callable, axis: str,
        extent: tuple[float, float],
    ) -> float:
        """Return the largest |fn| over both walls of ``axis``."""
        wanted = tuple(inspect.signature(fn).parameters)
        worst = 0.0
        for wall in extent:
            coords: dict[str, object] = {}
            for name in wanted:
                if name == axis:
                    coords[name] = wall
                else:
                    coords[name] = np.asarray(grid.evaluation_nodes(
                        field.function_space, name).data).ravel()
            worst = max(worst, float(np.max(np.abs(
                np.asarray(fn(**coords))))))
        return worst

    def _check_divergence_free(
        self, fields: dict[str, ScalarField],
    ) -> None:
        """Require a discretely solenoidal sampled background.

        The scheme's term split (and the exact pressure/tracer flux
        identities) assume the *discrete* divergence of the sampled
        background vanishes; the tolerance is relative to the
        individual flux-difference terms (pure cancellation error
        passes, an analytically-but-not-discretely solenoidal field
        does not).
        """
        parts = (fields["u"].diff(self._coords[0]),
                 fields["v"].diff(self._coords[1]))
        div = parts[0] + parts[1]
        max_div = float(np.max(np.abs(np.asarray(div.data))))
        scale = max(
            float(np.max(np.abs(np.asarray(part.data))))
            for part in parts)
        if max_div > _DIV_TOL * scale:
            raise ValueError(
                "the sampled background flow is not discretely "
                f"divergence-free (max |div u_b| = {max_div:.3e}, "
                f"flux-difference scale {scale:.3e}): the Sadourny "
                "background term assumes div(u_b) = 0 on the grid — "
                "a non-solenoidal background silently breaks the "
                "pressure/tracer transport identities. Provide a "
                "divergence-free background, e.g. u_b = -d(psi)/dy, "
                "v_b = +d(psi)/dx from a streamfunction psi sampled "
                "at the vorticity corners")

    @fr.model.term(advances=("u", "v", "p"),
             transports=("u", "v", "p"), linear=False)
    def advect(self, state, ctx) -> dict:  # noqa: ANN001
        """Return the Sadourny vector-invariant tendency (Ro-scaled).

        Description
        -----------
        On walled grids the only wall-specific step is the free-slip
        retag of the vorticity onto the Dirichlet corner space (and
        the matching retags of the kinetic-energy gradients onto the
        velocities, the nonhydro-projection precedent); every retag
        is the identity on periodic axes, so the periodic scheme is
        reproduced bit for bit. On chart grids the metric-aware path
        is taken (module docstring); on the identity chart it
        reproduces the flat scheme bitwise.
        """
        rossby = ctx.params[fr.model.params.SCALING_ROSSBY]
        u, v, p = state["u"], state["v"], state["p"]
        c = state["csqr"]
        zonal, meridional = self._coords

        # full geopotential thickness (centre) — the csqr-FIELD fix
        # (c is the centre csqr field, never the scalar; old bug)
        p_full = c.to(p) + rossby * p

        if u.grid.chart_coords is not None:
            return self._advect_chart(u, v, p, p_full, rossby)

        if getattr(u.grid, "immersed", None) is not None:
            return self._advect_immersed(u, v, p, p_full, rossby)

        # --- thickness tendency  dp = -Ro div(u p_e, v p_n) --------
        # (wall-normal flux lives on interior faces; the Dirichlet
        # fill closes the divergence with a zero wall flux)
        flux_u = u * p.to(u)                       # u face (east)
        flux_v = v * p.to(v)                       # v face (north)
        dp = rossby * -(flux_u.diff(zonal)
                        + flux_v.diff(meridional))

        # --- momentum: vorticity flux + kinetic-energy gradient ----
        # The NE-corner space adopts each velocity's wall tag on the
        # OTHER velocity's axis (Dirichlet on bounded factors);
        # retagging the BC-free stencil outputs onto it is the
        # explicit free-slip claim zeta = 0 at the wall (docstring).
        corner = u.function_space.bare.replace(**{
            meridional: v.function_space.bare.factor(meridional)})
        zeta = (v.diff(zonal).retag(corner)
                - u.diff(meridional).retag(corner))
        q = _potential_vorticity(zeta, p_full.to(zeta))
        fu = (u * p_full.to(u)).to(zeta)           # mass flux, NE
        fv = (v * p_full.to(v)).to(zeta)
        ekin = 0.5 * ((u * u).to(p) + (v * v).to(p))  # centre
        du = rossby * ((fv * q).to(u)
                       - ekin.diff(zonal).retag(u))
        dv = rossby * (-(fu * q).to(v)
                       - ekin.diff(meridional).retag(v))
        return {"u": du, "v": dv, "p": dp}

    def _advect_immersed(
        self,
        u: ScalarField,
        v: ScalarField,
        p: ScalarField,
        p_full: ScalarField,
        rossby: object,
    ) -> dict:
        r"""Return the Sadourny tendency under immersed boolean masks.

        Description
        -----------
        The cut-cell path (IP-D10) on a flat immersed grid. **Mass** is
        conserved exactly: the thickness transport is fraction-weighted
        exactly like the linear core continuity — every face flux
        carries the open-area fraction :math:`\alpha_f` and the
        divergence divides by the wet plan-area :math:`\theta_c`
        (guarded), so ``sum_c theta_c V_c p_c`` is machine-zero
        conserved for any fractions (the flux differences telescope, an
        :math:`\alpha = 0` face carrying none).

        **Momentum** runs under **boolean masks** (genuine-fraction
        Sadourny weighting is designed-for): the corner potential
        vorticity is zeroed wherever its corner touches a dry cell
        (``q <- q * mask(corner)``) — no vorticity is computed from dry
        velocities, the immersed free-slip closure (the wall analog of
        the ``zeta = 0`` corner retag) — and the momentum tendency is
        masked onto its own face (``du <- du * mask(u)``) so neither the
        vorticity flux nor the kinetic-energy gradient drives a closed
        face.

        **Conservation scope.** Away from the mask (a fully wet region:
        :math:`\theta = 1`, every mask ``True``) the scheme reduces to
        the flat scheme term for term, so the discrete energy/enstrophy
        telescoping is **interior-exact**. At the wet-region boundary
        the boolean-masked corner fluxes make energy/enstrophy
        conservation **approximate** (mass stays exact) — the price of
        the boolean simplification, documented here per IP-D10.
        """
        immersed = u.grid.immersed
        zonal, meridional = self._coords

        # --- thickness: mass-conserving fraction-weighted transport --
        # dp = -(Ro/theta) div(alpha u p) — the core-continuity idiom,
        # so mass conserves to machine zero on any (partial) fractions
        flux_u = weight_flux(immersed, u * p.to(u))
        flux_v = weight_flux(immersed, v * p.to(v))
        div = -(flux_u.diff(zonal) + flux_v.diff(meridional))
        dp = rossby * scale_divergence(immersed, div)

        # --- momentum: boolean-masked vorticity flux + KE gradient ---
        corner = u.function_space.bare.replace(**{
            meridional: v.function_space.bare.factor(meridional)})
        zeta = (v.diff(zonal).retag(corner)
                - u.diff(meridional).retag(corner))
        # zero the corner PV where it touches a dry cell (immersed
        # free-slip: no vorticity from dry velocities). The division is
        # routed through the guarded helper *before* the mask: masking
        # after the bare quotient leaves the immersed interior dry cells
        # (p_full == 0 there, not only the never-valid padding) as live
        # 0/0 candidates whose reverse-mode VJP (-zeta/h^2, h=0) is NaN
        # — the same masked-singularity that poisons the flat/chart PV
        # divisions (see _potential_vorticity). Guarding first keeps the
        # forward mask bitwise identical while the gradient stays finite.
        q = mask_field(
            immersed, _potential_vorticity(zeta, p_full.to(zeta)))
        fu = (u * p_full.to(u)).to(zeta)           # mass flux, NE
        fv = (v * p_full.to(v)).to(zeta)
        ekin = 0.5 * ((u * u).to(p) + (v * v).to(p))  # centre
        du = rossby * ((fv * q).to(u) - ekin.diff(zonal).retag(u))
        dv = rossby * (-(fu * q).to(v)
                       - ekin.diff(meridional).retag(v))
        # no momentum tendency into a closed face
        return {
            "u": mask_field(immersed, du),
            "v": mask_field(immersed, dv),
            "p": dp,
        }

    def _advect_chart(
        self,
        u: ScalarField,
        v: ScalarField,
        p: ScalarField,
        p_full: ScalarField,
        rossby: object,
    ) -> dict:
        r"""Return the metric-aware vector-invariant tendency.

        Description
        -----------
        The module docstring's generalization: flux-form thickness
        transport through the seeded ``"div"`` kind, vorticity as
        the metric ``"curl"`` of the lowered components (free-slip
        retagged onto the Dirichlet corner), corner PV fluxes as the
        :math:`\sqrt{g}`-weighted mass fluxes, kinetic energy from
        the lowered quadratics, and the covariant momentum
        tendencies raised back onto the prognostic contravariant
        components. Every metric coefficient is derived per
        application via ``grid.metric``.
        """
        grid = u.grid
        dispatch = grid.dispatch
        zonal, meridional = self._coords
        con = Variance.CONTRAVARIANT

        # --- thickness: dp = -(Ro/sqrt_g) d_i(sqrt_g u^i p) --------
        flux = VectorField({
            zonal: (u * p.to(u)).with_variance(con),
            meridional: (v * p.to(v)).with_variance(con)})
        div = dispatch.resolve(
            "div", flux[zonal].function_space.bare)
        dp = rossby * -(div(flux))

        # --- vorticity: metric curl of the lowered components ------
        # (tags stripped before the curl so the stencil outputs
        # share one corner space; the free-slip claim re-asserted by
        # the corner retag — the flat scheme's placement)
        corner = u.function_space.bare.replace(**{
            meridional: v.function_space.bare.factor(meridional)})
        lower = dispatch.resolve(
            "lower_index", u.function_space.bare)
        covariant = lower(VectorField({
            zonal: _wall_free(u, zonal).with_variance(con),
            meridional: _wall_free(v, meridional).with_variance(
                con)}))
        curl = dispatch.resolve(
            "curl", covariant[zonal].function_space.bare)
        zeta = curl(covariant).retag(corner)
        q = _potential_vorticity(zeta, p_full.to(zeta))

        # --- sqrt_g-weighted corner mass fluxes F^i ----------------
        # (the same fluxes the thickness divergence carries, so the
        # vorticity-flux exchange stays antisymmetric — docstring)
        sqg_u = grid.metric(u.function_space.bare, "sqrt_g")
        sqg_v = grid.metric(v.function_space.bare, "sqrt_g")
        fu = (sqg_u * (u * p_full.to(u))).to(zeta)
        fv = (sqg_v * (v * p_full.to(v))).to(zeta)

        # --- kinetic energy at centres -----------------------------
        # K = (mean(sqrt_g g_ii (u^i)^2)) / (2 sqrt_g) — the
        # placement pairing with the h-tendency (docstring)
        sqg_p = grid.metric(p.function_space.bare, "sqrt_g")
        g_uu = grid.metric(u.function_space.bare,
                           f"g_{zonal}{zonal}")
        g_vv = grid.metric(v.function_space.bare,
                           f"g_{meridional}{meridional}")
        # the / sqg_p divide is VJP-sealed: sqg_p is an exact zero in
        # the walled chart's never-valid padding, so the bare quotient's
        # reverse mode poisons the gradient with a masked 0/0 (see
        # _sealed_metric_divide; the same cure as the PV divide above)
        ekin_num = 0.5 * (((sqg_u * g_uu) * (u * u)).to(p)
                          + ((sqg_v * g_vv) * (v * v)).to(p))
        ekin = _sealed_metric_divide(ekin_num, sqg_p)

        # --- covariant momentum tendencies, raised -----------------
        cov = Variance.COVARIANT
        tu = ((fv * q).to(u)
              - ekin.diff(zonal).retag(u)).with_variance(cov)
        tv = (-(fu * q).to(v)
              - ekin.diff(meridional).retag(v)).with_variance(cov)
        raise_index = dispatch.resolve(
            "raise_index", tu.function_space.bare)
        raised = raise_index(VectorField({
            zonal: tu, meridional: tv}))
        du = rossby * raised[zonal].retag(u)
        dv = rossby * raised[meridional].retag(v)
        return {"u": du, "v": dv, "p": dp}

    @fr.model.term(name="background_advection",
             advances=("u", "v", "p"), linear=True)
    def background_advection(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""Flux-form transport by the prescribed background flow.

        Description
        -----------
        ``-div(u_b (x) u)`` on the velocities (centre / NE-corner
        flux products, the old module's stencils) and
        ``-div(u_b p)`` on the pressure — together with the
        nonlinear term's ``-div(u' p)`` this is exactly the
        bilinear full-velocity pressure flux. **Unscaled**: no
        Rossby factor (module docstring). The wall handling reuses
        the nonlinear term's machinery: the corner space adopts the
        Dirichlet wall tags, so every wall value consumed by the
        flux divergences carries the wall-normal background factor
        — an exact zero (impermeability); the retags onto the
        velocities are the identity on periodic axes.
        """
        u, v, p = state["u"], state["v"], state["p"]
        ub = state["u_background"]
        vb = state["v_background"]
        zonal, meridional = self._coords
        corner = u.function_space.bare.replace(**{
            meridional: v.function_space.bare.factor(meridional)})

        # --- thickness: dp = -div(u_b p) ---------------------------
        dp = -((ub * p.to(ub)).diff(zonal)
               + (vb * p.to(vb)).diff(meridional))

        # --- momentum: -div(u_b (x) u) -----------------------------
        # (fluxes at the centre along the own axis, at the NE corner
        # along the other axis — the old module's positions)
        fx_u = ub.to(p) * u.to(p)                  # centre
        fy_u = vb.to(corner) * u.to(corner)        # NE corner
        du = -(fx_u.diff(zonal).retag(u)
               + fy_u.diff(meridional).retag(u))
        fx_v = ub.to(corner) * v.to(corner)        # NE corner
        fy_v = vb.to(p) * v.to(p)                  # centre
        dv = -(fx_v.diff(zonal).retag(v)
               + fy_v.diff(meridional).retag(v))
        return {"u": du, "v": dv, "p": dp}

    def tendency_terms(self) -> tuple[fr.model.TendencyTerm, ...]:
        """Collect the terms; drop the unused background term."""
        terms = super().tendency_terms()
        if self._background is None:
            terms = tuple(term for term in terms
                          if term.name != "background_advection")
        return terms
