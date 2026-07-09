"""The nonhydrostatic ``State`` vocabulary class.

Description
-----------
A model-package vocabulary subclass of ``fr.grid.VectorField``
(model.md section 4): curated component accessors carrying hinted
errors, plus **parameter-free** diagnostics written in the field
algebra. Parameterful diagnostics (``ekin`` carries ``dsqr``,
``pot_vort`` carries ``f0``/``n2``/``rossby``/``dsqr``) are NOT here
— they are ``model.diagnostics`` functions (D2.3).

The class is supplied by ``nh.DynamicalCore`` through
``Module.state_type`` and constructed by assembly with the base
``Mapping[str, ScalarField]`` constructor, so it adds no ``__init__``.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework2.grid.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    from fridom.framework2.grid.fields.scalar_field import ScalarField


class State(VectorField):

    """Nonhydrostatic state vocabulary: u, v, w, b + diagnostics.

    Description
    -----------
    ``state["u"]`` stays the primary, module-facing spelling; the
    named properties are notebook sugar that raise a hinted
    ``MissingComponentError`` (a ``KeyError`` subclass) when a
    component is absent — the curated-hint contract of D1.5, delegated
    to :meth:`~fridom.framework2.grid.VectorField.require`.
    """

    # ================================================================
    #  Curated component accessors
    # ================================================================
    @property
    def u(self) -> ScalarField:
        """Zonal velocity (declared by a dynamical-core module)."""
        return self.require(
            "u", hint="declared by a dynamical-core module, "
                      "e.g. nh.DynamicalCore")

    @property
    def v(self) -> ScalarField:
        """Meridional velocity (declared by a dynamical-core module)."""
        return self.require(
            "v", hint="declared by a dynamical-core module, "
                      "e.g. nh.DynamicalCore")

    @property
    def w(self) -> ScalarField:
        """Vertical velocity (declared by a dynamical-core module)."""
        return self.require(
            "w", hint="declared by a dynamical-core module, "
                      "e.g. nh.DynamicalCore")

    @property
    def b(self) -> ScalarField:
        """Buoyancy (present when a stratification module is used)."""
        return self.require(
            "b", hint="add a stratification module, e.g. "
                      "nh.ConstantStratification")

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
        dudy = self.u.diff("y").to(dvdx.function_space)
        return dvdx - dudy
