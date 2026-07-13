r"""
The shallow-water dynamical core module.

Description
-----------
``DynamicalCore`` declares the state vocabulary (``u``, ``v``,
``p``), owns the squared phase speed :math:`c^2` (the AUXILIARY
``csqr`` field and, when constant, the ``shallowwater.csqr`` scalar)
and the Rossby scaling (``scaling.rossby``), and contributes the
single **linear** pressure-gradient / geopotential-divergence term:

.. math::
    \partial_t \boldsymbol{u} = - \nabla p , \qquad
    \partial_t p = -\nabla\cdot\left(c^2 \boldsymbol{u}\right)

**Variable depth**: ``csqr`` accepts a callable :math:`c^2(y)` (the
coriolis two-type precedent, folded into one core because the core
also owns the whole state vocabulary): the ``csqr`` field is then
declared on a meridional ``fr.spatial.Profile("y")`` and the constant
``shallowwater.csqr`` scalar is **not** provided
(provides-implies-constancy, 02_rules) — analytic consumers keyed on
the provide reject the model, the dense-column channel engine serves
it. The tendency terms are untouched either way: they read the
``csqr`` *field* (:math:`c^2` sits inside the divergence — the flux
form), which is exactly the sampling the variable-depth energy
metric ``diag(c^2, c^2, 1)`` pairs with. Pair a varying ``csqr``
with a Coriolis module carrying ``metric_weight="csqr"`` (the
thickness-weighted rotation) so the rotation stays energy-conserving
under that metric; the ``sw.Model`` preset wires this automatically.

The rotation :math:`f\,\underset{\neg}{\boldsymbol{u}}` is **not** a
core term: it is carried by the shared Coriolis module
(``fr.model.modules.FPlaneCoriolis`` / ``BetaPlaneCoriolis``, or
``RotationCoriolis`` on a chart grid), which declares the
``f_coriolis`` field and the ``+f v`` / ``-f u`` coupling — and is
opt-in: a model assembled without one simply does not rotate. The
gravity term here is unscaled (the Rossby number multiplies only the
advection, D2.2). The nonlinear Sadourny advection is a separate
module.

Chart grids (coordinate-systems plan, stage C2)
-----------------------------------------------
The same module assembles on a chart-coupled grid (a
``CoordinateMapping`` embedding chart, e.g. the lat-lon sphere):
declare the coordinate names via ``coords=`` and the terms select
the metric-aware path (a static grid property, never a traced
value), resolving the seeded ``"grad"`` / ``"div"`` /
``"raise_index"`` kinds through the grid dispatch — the module never
hand-builds metric compositions.

**Velocity convention (recorded per the C2 task):** on chart grids
the prognostic ``u`` / ``v`` are the **contravariant** components
:math:`u^\lambda = \dot\lambda`, :math:`u^\varphi = \dot\varphi`
(units 1/s on the sphere), stored untagged in the state; terms tag
them ``CONTRAVARIANT`` at the seams. The conversion points to
physical (m/s) components are ``State.u_physical`` /
``State.v_physical`` and the metric-aware ``ekin`` diagnostic —
:math:`u_{\rm east} = \sqrt{g_{\lambda\lambda}}\,u^\lambda`,
:math:`v_{\rm north} = \sqrt{g_{\varphi\varphi}}\,u^\varphi`,
derived per call via ``grid.metric``. On flat grids the convention
degenerates to the usual physical velocities and the flat code path
is taken verbatim (bitwise; the hard results-neutrality gate).

The chart gravity term is

.. math::
    \partial_t u^i = -\,g^{ij}\,\partial_j p , \qquad
    \partial_t p = -\frac{1}{\sqrt{g}}\,
        \partial_i\left(\sqrt{g}\, c^2 u^i\right)

via ``grad`` -> ``raise_index`` on the pressure and the flux-form
metric ``div`` on the tagged geopotential flux. A spherical model is
assembled through the same preset (see ``sw.Model``'s ``coords=``
docs for the grid recipe).
"""
from __future__ import annotations

import inspect
from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom as fr
from fridom.framework.utils import jaxify
from fridom.shallowwater2 import params as sw_params
from fridom.shallowwater2.diagnostics import DIAGNOSTICS
from fridom.shallowwater2.state import State
from fridom.spatial.decomposition.halo import HaloSpec
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.scalars import Variance

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Callable


@partial(jaxify, dynamic=("csqr", "rossby_number"))
class DynamicalCore(fr.model.Module):

    r"""
    Shallow-water core: declares ``u``, ``v``, ``p``; linear physics.

    Parameters
    ----------
    csqr : float | Callable, optional
        The squared gravity-wave phase speed :math:`c^2`. A float is
        the constant depth: published as ``shallowwater.csqr`` and
        materialized into the one-DOF ``csqr`` field. A callable
        ``csqr(y)`` (evaluated on the meridional coordinate) is the
        variable depth: materialized into a ``csqr`` field on
        ``fr.spatial.Profile("y")``, with **no** ``shallowwater.csqr``
        provide (provides-implies-constancy) (default: 1.0).
    rossby_number : float | fr.model.Ramp, optional
        The Rossby number scaling the (separate) advection term;
        published as ``scaling.rossby`` (default: 1.0); may be a
        ``fr.model.Ramp`` for a spun-up nonlinearity.
    coords : tuple[str, str], optional
        The (zonal, meridional) coordinate names, in the grid's
        factor order — ``("lon", "lat")`` on the standard sphere
        chart (default: ``("x", "y")``).
    meridional : str | None, optional
        The meridional coordinate name a callable ``csqr`` varies
        along; None uses ``coords[1]`` (default: None).
    """

    #: The vocabulary class this core supplies (D1.3 commitment 4).
    state_type = State

    #: Bound parameterful diagnostics (the D1.3 commitment-4 channel).
    diagnostics = DIAGNOSTICS

    # The chart-path gravity term resolves the metric-aware kinds,
    # whose multi-row block application the halo tracer cannot
    # follow (it collects traced operands into VectorFields), so the
    # module declares its stencil width and is halo-trace exempt
    # (V-N2, the Sadourny precedent): one staggered difference plus
    # at most one cross-term interpolation hop per axis (the
    # non-diagonal raise_index worst case).
    @property
    def extra_halo(self) -> HaloSpec:
        """Two halo cells per coordinate (chart worst case)."""
        return HaloSpec(dict.fromkeys(self._coords, 2))

    def __init__(
        self,
        csqr: float | Callable = 1.0,
        rossby_number: float | fr.model.Ramp = 1.0,
        *,
        coords: tuple[str, str] = ("x", "y"),
        meridional: str | None = None,
    ) -> None:
        """Store the leaves; a callable ``csqr`` stays static."""
        coords = tuple(coords)
        if (len(coords) != 2  # noqa: PLR2004 — zonal + meridional
                or not all(isinstance(c, str) for c in coords)
                or coords[0] == coords[1]):
            raise TypeError(
                "coords names the (zonal, meridional) coordinates: "
                f"two distinct strings, got {coords!r}")
        self._csqr_fn = csqr if callable(csqr) else None
        self.csqr = None if callable(csqr) else fr.model.leaf(csqr)
        self.rossby_number = fr.model.leaf(rossby_number)
        self._coords: tuple[str, str] = coords
        self._meridional = (coords[1] if meridional is None
                            else meridional)

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def coords(self) -> tuple[str, str]:
        """The (zonal, meridional) coordinate names."""
        return self._coords

    # ================================================================
    #  Declarations
    # ================================================================
    @property
    def field_declarations(self) -> tuple[fr.model.FieldDeclaration, ...]:
        """U (east face), v (north face), p (centre), csqr (AUX)."""
        if self._csqr_fn is None:
            csqr_decl = fr.model.FieldDeclaration(
                "csqr", space=fr.spatial.Profile(),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._csqr_default,
                long_name="Squared phase speed", units="m^2/s^2")
        else:
            csqr_decl = fr.model.FieldDeclaration(
                "csqr", space=fr.spatial.Profile(self._meridional),
                lifecycle=fr.model.Lifecycle.AUXILIARY,
                default=self._csqr_profile_default,
                long_name="Squared phase speed", units="m^2/s^2")
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
        )

    @property
    def parameter_declarations(
        self,
    ) -> tuple[fr.model.ParameterDeclaration, ...]:
        """Rossby always; ``shallowwater.csqr`` only when constant."""
        decls = (
            fr.model.ParameterDeclaration(
                fr.model.params.SCALING_ROSSBY, attr="rossby_number"),
        )
        if self._csqr_fn is None:
            decls += (
                fr.model.ParameterDeclaration(
                    sw_params.CSQR, attr="csqr", units="m^2/s^2"),
            )
        return decls

    def _csqr_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: fill the one-DOF profile with ``csqr``.

        The field is declared on ``fr.spatial.Profile()`` (constant depth is a
        single degree of freedom); the GAP-A ConstantSpace/Profile
        broadcast lifts it to the nodal join wherever a term multiplies
        it (``c.to(u) * u``). No ``grid.sync``
        pre-syncing: the GAP-B fix keeps carry-resident AUXILIARY
        fields scan-treedef-stable without pre-flooding their halos.
        """
        return grid.create_field(
            space, data=jnp.full(space.shape, self.csqr),
            name="csqr")

    def _csqr_profile_default(
        self, grid, space,  # noqa: ANN001
    ) -> fr.spatial.ScalarField:
        """Owner-method default: materialize the ``csqr(y)`` profile.

        The meridional profile carries a single non-constant
        coordinate, so ``init`` names exactly that coordinate; the
        signature is stamped dynamically to match ``self._meridional``
        (the ``BetaPlaneCoriolis._f_default`` precedent). No
        pre-syncing (GAP-B).
        """
        fn, mer = self._csqr_fn, self._meridional

        def init(**coords: object) -> object:
            return fn(coords[mer])

        init.__signature__ = inspect.Signature(  # type: ignore[attr-defined]
            [inspect.Parameter(
                mer, inspect.Parameter.POSITIONAL_OR_KEYWORD)])
        return grid.create_field(space, init=init, name="csqr")

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
        """
        grid = table.grid
        chart = grid.chart_coords
        if chart is None:
            return
        expected = tuple(
            name for name in grid.names if name in set(chart))
        if self._coords != expected:
            raise ValueError(
                f"DynamicalCore coords={self._coords!r} do not "
                f"match the grid's chart coordinates {expected!r} "
                "(in factor order); pass coords=(zonal, meridional) "
                "matching the grid, e.g. coords=('lon', 'lat') on "
                "the standard sphere chart")

    # ================================================================
    #  Tendency terms (linear)
    # ================================================================
    @fr.model.term(advances=("u", "v", "p"), linear=True)
    def gravity(self, state, ctx) -> dict:  # noqa: ANN001, ARG002
        r"""Pressure gradient and geopotential divergence.

        .. math::
            \partial_t \boldsymbol{u} = - \nabla p , \qquad
            \partial_t p = -\nabla\cdot\left(c^2 \boldsymbol{u}\right)

        Pure field arithmetic: ``c^2`` sits INSIDE the divergence
        (``(c.to(u) * u).diff("x")``, the flux form) so the discrete
        stencil matches ``diff(c^2 u)``. The ``csqr`` field lifts from
        its one-DOF ``fr.spatial.Profile()`` onto each velocity face via the
        ConstantSpace broadcast in ``.to``.

        The pressure-gradient entries retag onto their velocities:
        nodal stencil outputs are BC-free, but on a walled grid each
        wall-normal velocity carries the derived Dirichlet wall tag
        on its own axis, so the entry adopts it (the nonhydro
        projection precedent) — identity on periodic grids. The flux
        entries need no retag: ``csqr.to(u)`` adopts the velocity's
        tag (BC-sibling adoption) and the divergence lands BC-free,
        which is ``p``'s space.

        On a chart grid (module docstring) the same physics resolves
        the seeded metric-aware kinds: ``grad`` -> ``raise_index``
        turns the covariant pressure gradient into the contravariant
        tendency (``-g^{ij} d_j p``), and the flux-form ``div``
        carries the ``sqrt_g``-weighted geopotential flux; the final
        retags strip the variance claim and restore the velocities'
        wall tags. On the identity chart every metric factor is an
        exact 1.0, reproducing the flat path bitwise.
        """
        u, v, p = state["u"], state["v"], state["p"]
        csqr = state["csqr"]
        zonal, meridional = self._coords
        if u.grid.chart_coords is None:
            return {
                "u": (-p.diff(zonal)).retag(u),
                "v": (-p.diff(meridional)).retag(v),
                "p": (-(csqr.to(u) * u).diff(zonal)
                      - (csqr.to(v) * v).diff(meridional)),
            }
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
        return {
            "u": (-raised[zonal]).retag(u),
            "v": (-raised[meridional]).retag(v),
            "p": -div(flux),
        }
