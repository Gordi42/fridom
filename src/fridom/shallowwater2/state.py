r"""
The shallow-water State vocabulary class.

Description
-----------
``State`` is the shallow-water *vocabulary* subclass of
``fr.spatial.VectorField`` (model.md section 4): curated ``u`` / ``v`` / ``p``
accessors plus parameter-free diagnostics for the shallow-water
family. It is **not** a structural entity — it declares no fields and
owns no assembly logic (the ``DynamicalCore`` module does), and it
inherits ``VectorField``'s constructor and pytree registration
unchanged. A model assembled without a core produces a plain
``VectorField`` and loses only the sugar (CS-14: no transform may
``isinstance`` on this class).

Component vocabulary (D1.3):

- ``u``: zonal velocity, on ``fr.spatial.Staggered(zonal)`` (the
  C-grid east face);
- ``v``: meridional velocity, on ``fr.spatial.Staggered(meridional)``
  (the north face);
- ``p``: the pressure / geopotential perturbation :math:`p = g\eta`,
  on ``fr.spatial.Collocated()`` (cell centre).

The coordinate names are read off the component spaces (grid factor
order, zonal first — the core's ``coords`` convention), so the same
vocabulary serves the Cartesian ``(x, y)`` grids and the spherical
``(lon, lat)`` chart. On every grid ``u`` / ``v`` hold the
**physical** (m/s) velocity components (``physical_state_components.md``
ruling (c)); the chart-native contravariant coordinate velocities
:math:`\dot\lambda`, :math:`\dot\varphi` live behind the read-only
:attr:`~State.chart` namespace (ruling (d), ``chart.py``). The
metric-aware ``rel_vort`` / ``divergence`` convert at their chart-path
entry and resolve the metric-aware kinds.

Only ``rel_vort`` and ``divergence`` are parameter-free and live
here; ``ekin`` / ``epot`` carry :math:`c^2` and the
Rossby number and are bound diagnostics on the model (D2.3), not
State properties.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.shallowwater2.chart import chart_component, to_contravariant
from fridom.spatial.errors import MissingComponentError
from fridom.spatial.fields.chart_view import ChartView
from fridom.spatial.fields.vector_field import VectorField
from fridom.spatial.scalars import Variance

if TYPE_CHECKING:  # pragma: no cover
    import fridom as fr
    from fridom.spatial.fields.scalar_field import ScalarField

__all__ = ["MissingComponentError", "State"]


def _wall_free(field: ScalarField, axis: str) -> ScalarField:
    """Return the field retagged onto the BC-free ``axis`` sibling.

    The chart path's tag-strip (shared with the Sadourny module):
    the metric ``curl`` entry needs its two stencil outputs on one
    shared corner space, so a wall-normal Dirichlet claim is
    stripped before the curl (data-neutral) and re-asserted on the
    output by the corner retag. Identity on periodic axes.
    """
    factor = field.function_space.bare.factor(axis)
    return field.retag(factor.mesh.nodal(factor.node_set))


class State(VectorField):

    r"""
    Shallow-water state vector: ``u``, ``v``, ``p`` (+ diagnostics).

    Description
    -----------
    A curated read surface over a ``VectorField`` whose components
    are named ``u`` / ``v`` / ``p``. The accessors raise a hinted
    :class:`MissingComponentError` when the component is absent (a
    model assembled without the core), never a bare ``KeyError``.
    """

    # ================================================================
    #  Curated component accessors (hinted; no setters)
    # ================================================================
    @property
    def u(self) -> fr.spatial.ScalarField:
        """Zonal velocity (declared by a shallow-water core)."""
        return self.require(
            "u", hint="declared by a shallow-water core module, "
                      "e.g. sw.Core")

    @property
    def v(self) -> fr.spatial.ScalarField:
        """Meridional velocity (declared by a shallow-water core)."""
        return self.require(
            "v", hint="declared by a shallow-water core module, "
                      "e.g. sw.Core")

    @property
    def p(self) -> fr.spatial.ScalarField:
        r"""Pressure / geopotential perturbation :math:`p = g\eta`."""
        return self.require(
            "p", hint="declared by a shallow-water core module, "
                      "e.g. sw.Core")

    def _axes(self) -> tuple[str, str]:
        """Read the (zonal, meridional) names off the ``u`` space."""
        names = self.u.function_space.names
        return names[0], names[1]

    # ================================================================
    #  Chart-native view (read-only expert surface)
    # ================================================================
    @property
    def chart(self) -> ChartView:
        r"""The read-only chart-native view of ``u`` / ``v`` (ruling (d)).

        Description
        -----------
        ``state.chart["u"]`` / ``state.chart.u`` is the chart-native
        **coordinate velocity** :math:`\dot\lambda =
        U/\sqrt{g_{\lambda\lambda}}` on a chart grid (the sealed entry
        conversion, ``chart.py``), the identity on a flat grid;
        ``chart["v"]`` is its meridional twin. ``u, v =
        state.chart.velocities`` destructures the pair in grid axis
        order. The view is read-only — write the physical component on
        the state instead. The prognostic ``u`` / ``v`` themselves are
        the **physical** (m/s) velocities on every grid
        (``physical_state_components.md``).
        """
        return ChartView(self, chart_component, ("u", "v"))

    # ================================================================
    #  Parameter-free diagnostics (field algebra; D2.3)
    # ================================================================
    @property
    def rel_vort(self) -> fr.spatial.ScalarField:
        r"""
        Relative vorticity :math:`\zeta = \partial_x v - \partial_y u`.

        Description
        -----------
        On the C-grid ``v`` staggers in y and ``u`` in x, so both
        derivatives land on the north-east vorticity corner and the
        difference is well defined without an interpolation. On a
        walled grid the BC-free stencil outputs are retagged onto
        the Dirichlet corner space (each velocity's wall tag on the
        other velocity's axis) — the free-slip claim
        :math:`\zeta = 0` at the wall, matching the Sadourny
        advection module; identity on periodic axes. On chart grids
        the physical components are converted to the contravariant
        coordinate velocities at entry (``chart.py``) and this is the
        metric curl of the lowered components,
        :math:`\zeta = (\partial_\lambda v_{cov} - \partial_\varphi
        u_{cov})/\sqrt{g}` (the physical scalar vorticity), through
        the seeded ``"lower_index"`` / ``"curl"`` kinds.
        """
        u, v = self.u, self.v
        zonal, meridional = self._axes()
        corner = u.function_space.bare.replace(**{
            meridional: v.function_space.bare.factor(meridional)})
        if u.grid.chart_coords is None:
            zeta = (v.diff(zonal).retag(corner)
                    - u.diff(meridional).retag(corner))
        else:
            # entry seam: physical U -> contravariant u^i (chart.py);
            # the metric curl flow below runs verbatim
            u = to_contravariant(u, zonal)
            v = to_contravariant(v, meridional)
            dispatch = u.grid.dispatch
            con = Variance.CONTRAVARIANT
            lower = dispatch.resolve(
                "lower_index", u.function_space.bare)
            covariant = lower(VectorField({
                zonal: _wall_free(u, zonal).with_variance(con),
                meridional: _wall_free(v, meridional)
                .with_variance(con)}))
            curl = dispatch.resolve(
                "curl", covariant[zonal].function_space.bare)
            zeta = curl(covariant).retag(corner)
        return zeta.with_metadata(
            name="rel_vort", long_name="Relative vorticity",
            units="1/s")

    @property
    def divergence(self) -> fr.spatial.ScalarField:
        r"""
        Horizontal divergence :math:`\nabla\cdot\boldsymbol{u}`.

        Description
        -----------
        ``u`` staggers in x and ``v`` in y, so ``u.diff("x")`` and
        ``v.diff("y")`` both land at the cell centre. On chart
        grids the physical components are converted to the
        contravariant coordinate velocities at entry (``chart.py``)
        and this is the flux-form metric divergence
        :math:`\partial_i(\sqrt{g}\,u^i)/\sqrt{g}`, through the
        seeded ``"div"`` kind.
        """
        u, v = self.u, self.v
        zonal, meridional = self._axes()
        if u.grid.chart_coords is None:
            div = u.diff(zonal) + v.diff(meridional)
        else:
            # entry seam: physical U -> contravariant u^i (chart.py);
            # the flux-form metric divergence below runs verbatim
            u = to_contravariant(u, zonal)
            v = to_contravariant(v, meridional)
            con = Variance.CONTRAVARIANT
            vec = VectorField({
                zonal: u.with_variance(con),
                meridional: v.with_variance(con)})
            div = u.grid.dispatch.resolve(
                "div", u.function_space.bare)(vec)
        return div.with_metadata(
            name="divergence", long_name="Horizontal divergence",
            units="1/s")
