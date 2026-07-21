"""The nonhydrostatic ``State`` vocabulary class.

Description
-----------
A model-package vocabulary subclass of ``fr.grid.VectorField``
(model.md section 4): curated component accessors carrying hinted
errors, plus **parameter-free** diagnostics written in the field
algebra. Parameterful diagnostics (``ekin`` carries ``dsqr``,
``pot_vort`` carries ``f0``/``n2``/``rossby``/``dsqr``) are NOT here
— they are ``model.diagnostics`` functions (D2.3).

The class is supplied by ``nh.Core`` through
``Module.state_type`` and constructed by assembly with the base
``Mapping[str, ScalarField]`` constructor, so it adds no ``__init__``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.nonhydro2.chart import chart_component
from fridom.spatial.fields.chart_view import ChartView
from fridom.spatial.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField


class State(VectorField):

    r"""Nonhydrostatic state vocabulary: u, v, w, b + diagnostics.

    Description
    -----------
    ``state["u"]`` stays the primary, module-facing spelling; the
    named properties are notebook sugar that raise a hinted
    ``MissingComponentError`` (a ``KeyError`` subclass) when a
    component is absent — the curated-hint contract of D1.5, delegated
    to :meth:`~fridom.spatial.VectorField.require`.

    The velocity components ``u``, ``v``, ``w`` are the **physical**
    velocities (m/s) on every grid — mapped columns included, where
    ``w`` is the prognostic physical vertical velocity
    (``physical_state_components.md``). The chart-native contravariant
    flux ``J\omega`` on a mapped column lives behind the read-only
    :attr:`chart` namespace, derived on demand.
    """

    # ================================================================
    #  Curated component accessors
    # ================================================================
    @property
    def u(self) -> ScalarField:
        """Zonal velocity (declared by a dynamical-core module)."""
        return self.require(
            "u", hint="declared by a dynamical-core module, "
                      "e.g. nh.Core")

    @property
    def v(self) -> ScalarField:
        """Meridional velocity (declared by a dynamical-core module)."""
        return self.require(
            "v", hint="declared by a dynamical-core module, "
                      "e.g. nh.Core")

    @property
    def w(self) -> ScalarField:
        r"""Vertical velocity — the **physical** prognostic ``w`` (m/s).

        Description
        -----------
        The physical vertical velocity on every grid, mapped columns
        included (the prognostic *is* physical ``w``, so a physical ``w``
        IC works directly). The chart-native contravariant flux
        ``J\omega = w - \sum_i Z_i I(u_i)`` is ``state.chart["w"]``.

        .. note::

            On a future time-dependent map (a ``MovingGeometry`` whose
            terrain moves, ``\partial_t Z \neq 0``) the physical ``w``
            gains the mesh-velocity term ``\partial_t Z``. Not
            implemented — a breadcrumb for the moving-terrain case.
        """
        return self.require(
            "w", hint="declared by a dynamical-core module, "
                      "e.g. nh.Core")

    @property
    def b(self) -> ScalarField:
        """Buoyancy (present when a stratification module is used)."""
        return self.require(
            "b", hint="add a stratification module, e.g. "
                      "nh.ConstantStratification")

    # ================================================================
    #  Chart-native view (read-only expert surface)
    # ================================================================
    @property
    def chart(self) -> ChartView:
        r"""The read-only chart-native view of the velocity trio (ruling (d)).

        Description
        -----------
        ``state.chart["w"]`` / ``state.chart.w`` is the chart-native
        vertical quantity: the **contravariant volume flux**
        ``J\omega = w - \sum_i Z_i I(u_i)`` on a terrain (sigma) column
        (equal to the mapped pressure solver's divergence right-hand
        side quantity), the identity on an unmapped grid; ``chart["u"]``
        / ``chart["v"]`` are the identity. ``u, v, w =
        state.chart.velocities`` destructures the trio in grid axis order
        (vertical last). The view is read-only.
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
        Parameter-free (the ``dsqr``/``Ro`` weightings live on the
        parameterful ``pot_vort`` diagnostic). ``dvdx`` sets the
        target position; ``dudy`` is interpolated onto it.
        """
        dvdx = self.v.diff("x")
        dudy = self.u.diff("y").to(dvdx)
        return dvdx - dudy
