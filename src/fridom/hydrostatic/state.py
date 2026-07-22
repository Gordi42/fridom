"""The hydrostatic ``State`` vocabulary class.

Description
-----------
A model-package vocabulary subclass of ``fr.grid.VectorField``
(model.md section 4): curated component accessors carrying hinted
errors, plus **parameter-free** diagnostics written in the field
algebra. Parameterful diagnostics (``ekin`` carries ``n2``,
``pot_vort`` carries ``f0``/``n2``) live in ``hy.diagnostics`` (D2.3),
not here.

The class is supplied by ``hy.Core`` through
``Module.state_type`` and constructed by assembly with the base
``Mapping[str, ScalarField]`` constructor, so it adds no ``__init__``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.hydrostatic.modules.terrain import chart_component
from fridom.spatial.fields.chart_view import ChartView
from fridom.spatial.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField


class State(VectorField):

    r"""Hydrostatic state vocabulary: u, v, w, b, ps + diagnostics.

    Description
    -----------
    ``state["u"]`` stays the primary, module-facing spelling; the
    named properties are notebook sugar that raise a hinted
    ``MissingComponentError`` (a ``KeyError`` subclass) when a
    component is absent — the curated-hint contract of D1.5, delegated
    to :meth:`~fridom.spatial.VectorField.require`. ``w`` and ``ps``
    are diagnosed / barotropic; ``u``, ``v``, ``b`` are prognostic.

    The velocity components ``u``, ``v``, ``w`` are the **physical**
    velocities (m/s) on every grid — flat, stretched, terrain-following
    (``physical_state_components.md``). The chart-native quantities (the
    contravariant volume flux ``J\omega`` on a mapped column) live behind
    the read-only :attr:`chart` namespace, derived on demand.
    """

    # ================================================================
    #  Curated component accessors
    # ================================================================
    @property
    def u(self) -> ScalarField:
        """Zonal velocity (declared by a hydrostatic core)."""
        return self.require(
            "u", hint="declared by a hydrostatic core, "
                      "e.g. hy.Core")

    @property
    def v(self) -> ScalarField:
        """Meridional velocity (declared by a hydrostatic core)."""
        return self.require(
            "v", hint="declared by a hydrostatic core, "
                      "e.g. hy.Core")

    @property
    def w(self) -> ScalarField:
        r"""Diagnosed **physical** vertical velocity (m/s).

        Description
        -----------
        The physical vertical velocity on every grid (hydrostatic
        continuity). On a terrain-following sigma column it is
        ``w = J\omega + u\,Z_x + v\,Z_y`` — the contravariant volume
        flux plus the slope advection — so it is nonzero at the bed over
        a slope (the *flux* vanishes at the terrain, not physical ``w``).
        The chart-native flux ``J\omega`` is ``state.chart["w"]``.

        .. note::

            On a future time-dependent map (a ``MovingGeometry`` whose
            terrain moves, ``\partial_t Z \neq 0``) the physical ``w``
            gains the mesh-velocity term ``\partial_t Z`` on top of the
            slope advection. Not implemented — the current maps are
            static; this is a breadcrumb for the moving-terrain case.
        """
        return self.require(
            "w", hint="diagnosed by a hydrostatic core, "
                      "e.g. hy.Core")

    @property
    def b(self) -> ScalarField:
        """Buoyancy (present when a stratification module is used)."""
        return self.require(
            "b", hint="add a stratification module, e.g. "
                      "hy.ConstantStratification")

    @property
    def ps(self) -> ScalarField:
        """Surface pressure ``g*eta`` (a free-surface module)."""
        return self.require(
            "ps", hint="declared by a free-surface module, e.g. "
                       "hy.ExplicitFreeSurface")

    # ================================================================
    #  Chart-native view (read-only expert surface)
    # ================================================================
    @property
    def chart(self) -> ChartView:
        r"""The read-only chart-native view of the velocity trio (ruling (d)).

        Description
        -----------
        ``state.chart["w"]`` / ``state.chart.w`` is the chart-native
        vertical quantity: the **contravariant volume flux** ``J\omega``
        on a terrain (sigma) column, the identity on an unmapped grid;
        ``chart["u"]`` / ``chart["v"]`` are the identity (the horizontal
        components are uncoupled). ``u, v, w = state.chart.velocities``
        destructures the trio in grid axis order (vertical last). The
        view is read-only — write the physical component on the state
        instead.
        """
        return ChartView(self, chart_component, ("u", "v", "w"))

    # ================================================================
    #  Parameter-free diagnostics (field algebra only)
    # ================================================================
    @property
    def rel_vort_z(self) -> ScalarField:
        """Vertical relative vorticity ``d_x v - d_y u`` (cell edge).

        Description
        -----------
        Parameter-free (the ``f0``/``n2`` weightings live on the
        parameterful ``pot_vort`` diagnostic). ``dvdx`` sets the
        target position; ``dudy`` is interpolated onto it.
        """
        dvdx = self.v.diff("x")
        dudy = self.u.diff("y").to(dvdx)
        return (dvdx - dudy).with_metadata(
            name="rel_vort_z", long_name="Vertical relative vorticity",
            units="1/s")

    @property
    def hor_divergence(self) -> ScalarField:
        """Horizontal divergence ``d_x u + d_y v`` (cell center).

        Description
        -----------
        The continuity source of the diagnosed ``w``
        (``d_z w = -hor_divergence``); parameter-free kinematics on
        the C-grid, both derivatives landing at the pressure cell.
        """
        div = self.u.diff("x") + self.v.diff("y")
        return div.with_metadata(
            name="hor_divergence", long_name="Horizontal divergence",
            units="1/s")
