"""The hydrostatic ``State`` vocabulary class.

Description
-----------
A model-package vocabulary subclass of ``fr.grid.VectorField``
(model.md section 4): curated component accessors carrying hinted
errors, plus **parameter-free** diagnostics written in the field
algebra. Parameterful diagnostics (``ekin`` carries ``n2``,
``pot_vort`` carries ``f0``/``n2``) live in ``hy.diagnostics`` (D2.3),
not here.

The class is supplied by ``hy.HydrostaticCore`` through
``Module.state_type`` and constructed by assembly with the base
``Mapping[str, ScalarField]`` constructor, so it adds no ``__init__``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    from fridom.spatial.fields.scalar_field import ScalarField


class State(VectorField):

    """Hydrostatic state vocabulary: u, v, w, b, ps + diagnostics.

    Description
    -----------
    ``state["u"]`` stays the primary, module-facing spelling; the
    named properties are notebook sugar that raise a hinted
    ``MissingComponentError`` (a ``KeyError`` subclass) when a
    component is absent — the curated-hint contract of D1.5, delegated
    to :meth:`~fridom.spatial.VectorField.require`. ``w`` and ``ps``
    are diagnosed / barotropic; ``u``, ``v``, ``b`` are prognostic.
    """

    # ================================================================
    #  Curated component accessors
    # ================================================================
    @property
    def u(self) -> ScalarField:
        """Zonal velocity (declared by a hydrostatic core)."""
        return self.require(
            "u", hint="declared by a hydrostatic core, "
                      "e.g. hy.HydrostaticCore")

    @property
    def v(self) -> ScalarField:
        """Meridional velocity (declared by a hydrostatic core)."""
        return self.require(
            "v", hint="declared by a hydrostatic core, "
                      "e.g. hy.HydrostaticCore")

    @property
    def w(self) -> ScalarField:
        """Diagnosed vertical velocity (hydrostatic continuity)."""
        return self.require(
            "w", hint="diagnosed by a hydrostatic core, "
                      "e.g. hy.HydrostaticCore")

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
