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
    ``KeyError`` when a component is absent (the curated-hint contract
    of D1.5; the landed grid layer raises ``KeyError`` rather than the
    designed ``MissingComponentError`` — deviation noted in the port
    report).
    """

    # ================================================================
    #  Curated component accessors
    # ================================================================
    def _component(self, name: str, *, hint: str) -> ScalarField:
        """Return the named component or raise a hinted ``KeyError``."""
        if name not in self:
            raise KeyError(
                f"no component {name!r}; {hint}. Components present: "
                f"{', '.join(self.component_names)}.")
        return self[name]

    @property
    def u(self) -> ScalarField:
        """Zonal velocity (declared by a dynamical-core module)."""
        return self._component(
            "u", hint="declared by a dynamical-core module, "
                      "e.g. nh.DynamicalCore")

    @property
    def v(self) -> ScalarField:
        """Meridional velocity (declared by a dynamical-core module)."""
        return self._component(
            "v", hint="declared by a dynamical-core module, "
                      "e.g. nh.DynamicalCore")

    @property
    def w(self) -> ScalarField:
        """Vertical velocity (declared by a dynamical-core module)."""
        return self._component(
            "w", hint="declared by a dynamical-core module, "
                      "e.g. nh.DynamicalCore")

    @property
    def b(self) -> ScalarField:
        """Buoyancy (present when a stratification module is used)."""
        return self._component(
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
