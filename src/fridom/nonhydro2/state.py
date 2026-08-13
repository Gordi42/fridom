"""The nonhydrostatic ``State`` vocabulary class.

Description
-----------
A model-package vocabulary subclass of ``fr.spatial.VectorField``
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
    from fridom.spatial.spaces.tensor_product import SpaceLike


def vorticity_corner(
    u: ScalarField, v: ScalarField,
) -> SpaceLike:
    r"""Return the vertical vorticity edge space of a ``u``/``v`` pair.

    Description
    -----------
    The C-grid edge :math:`\partial_x v` and :math:`\partial_y u`
    share: ``u``'s x factor tensor ``v``'s y factor (the rest of
    ``u``'s product space, which ``u`` and ``v`` agree on). Each
    velocity contributes its **own** wall tag, which the other
    velocity's difference lands on BC-free — a staggered first
    difference cannot claim its wall value (boundary_plan.md R1), so
    the tag has to be re-asserted here.

    On a walled axis that tag is the wall-normal velocity's
    homogeneous Dirichlet claim, and adopting it for the vorticity is
    the free-slip claim :math:`\zeta = 0` at the wall — the same
    claim :class:`~fridom.nonhydro2.SmagorinskyLilly`'s shear strain
    and ``sw.State.rel_vort`` already make. It is what grounds a
    later conversion to cell centres: on the BC-free sibling the wall
    ghost is undefined and the reconstruction would read it.
    Identity on periodic axes.

    Parameters
    ----------
    u : ScalarField
        The zonal velocity (sets the x factor).
    v : ScalarField
        The meridional velocity (sets the y factor).

    Returns
    -------
    SpaceLike
        The bare (unlaid-out) vorticity edge space.
    """
    return u.function_space.bare.replace(
        y=v.function_space.bare.factor("y"))


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
        """Buoyancy (present when a buoyancy module is used)."""
        return self.require(
            "b", hint="add a buoyancy module, e.g. "
                      "nh.ConstantStratification or nh.BuoyancyTracer")

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
        r"""Vertical relative vorticity ``d_x v - d_y u`` (cell edge).

        Description
        -----------
        Parameter-free (the ``dsqr``/``Ro`` weightings live on the
        parameterful ``pot_vort`` diagnostic). On the C-grid ``v``
        staggers in x and ``u`` in y, so both differences already land
        on the vertical vorticity edge and no interpolation is needed;
        the two BC-free stencil outputs are retagged onto the shared
        :func:`vorticity_corner` (the free-slip wall claim
        :math:`\zeta = 0`, identity on periodic axes).

        Retagging is what makes the result usable at a wall. A
        difference emits a BC-free bounded factor, whose exterior is
        undefined (boundary_plan.md R1); converting that to cell
        centres reads the unrepaired wall ghost and reports a
        spurious wall column of order :math:`v/\Delta x`, which grows
        without bound under refinement. The tag hands the
        reconstruction the wall value instead.

        The declaration carries ``v``'s ``nondimensional`` flag: the
        difference is a value-computing op, so it resets the whole
        record, and re-declaring only the physical unit would claim
        ``1/s`` on a nondimensional model.
        """
        corner = vorticity_corner(self.u, self.v)
        dvdx = self.v.diff("x").retag(corner)
        dudy = self.u.diff("y").retag(corner)
        return (dvdx - dudy).with_metadata(
            name="rel_vort_z", long_name="Vertical relative vorticity",
            units="1/s")
