r"""
The shallow-water State vocabulary class (framework2 port).

Description
-----------
``State`` is the shallow-water *vocabulary* subclass of
``fr.VectorField`` (model.md section 4): curated ``u`` / ``v`` / ``h``
accessors plus parameter-free diagnostics for the shallow-water
family. It is **not** a structural entity — it declares no fields and
owns no assembly logic (the ``ShallowWaterCore`` module does), and it
inherits ``VectorField``'s constructor and pytree registration
unchanged. A model assembled without a core produces a plain
``VectorField`` and loses only the sugar (CS-14: no transform may
``isinstance`` on this class).

Component vocabulary (D1.3, note the delta from the pre-Phase-2 code
which spelled the pressure ``p``):

- ``u``: velocity in x, on ``fr.Staggered("x")`` (the C-grid east
  face);
- ``v``: velocity in y, on ``fr.Staggered("y")`` (the north face);
- ``h``: the geopotential / pressure perturbation :math:`h = g\\eta`
  (the old ``p``), on ``fr.Collocated()`` (cell centre).

Only ``rel_vort`` and ``divergence`` are parameter-free and live
here; ``ekin`` / ``epot`` / ``pot_vort`` carry :math:`c^2` and the
Rossby number and are bound diagnostics on the model (D2.3), not
State properties.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.framework2.grid.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    import fridom.framework2 as fr


class MissingComponentError(KeyError):

    """A vocabulary accessor named a component the state lacks."""


class State(VectorField):

    r"""
    Shallow-water state vector: ``u``, ``v``, ``h`` (+ diagnostics).

    Description
    -----------
    A curated read surface over a ``VectorField`` whose components
    are named ``u`` / ``v`` / ``h``. The accessors raise a hinted
    :class:`MissingComponentError` when the component is absent (a
    model assembled without the core), never a bare ``KeyError``.
    """

    # ================================================================
    #  Curated component accessors (hinted; no setters)
    # ================================================================
    @property
    def u(self) -> fr.ScalarField:
        """Velocity in x (declared by a shallow-water core)."""
        return self._component(
            "u", hint="declared by a shallow-water core module, "
                      "e.g. sw.modules.ShallowWaterCore")

    @property
    def v(self) -> fr.ScalarField:
        """Velocity in y (declared by a shallow-water core)."""
        return self._component(
            "v", hint="declared by a shallow-water core module, "
                      "e.g. sw.modules.ShallowWaterCore")

    @property
    def h(self) -> fr.ScalarField:
        r"""Geopotential / pressure perturbation :math:`h = g\eta`."""
        return self._component(
            "h", hint="declared by a shallow-water core module, "
                      "e.g. sw.modules.ShallowWaterCore")

    def _component(self, name: str, *, hint: str) -> fr.ScalarField:
        """Return the named component or raise a hinted error."""
        if name not in self:
            raise MissingComponentError(
                f"no component {name!r}: {hint}. Components present: "
                f"{', '.join(self.component_names) or '(none)'}.")
        return self[name]

    # ================================================================
    #  Parameter-free diagnostics (field algebra; D2.3)
    # ================================================================
    @property
    def rel_vort(self) -> fr.ScalarField:
        r"""
        Relative vorticity :math:`\zeta = \partial_x v - \partial_y u`.

        Description
        -----------
        On the C-grid ``v`` staggers in y and ``u`` in x, so both
        derivatives land on the north-east vorticity corner and the
        difference is well defined without an interpolation.
        """
        return (self.v.diff("x") - self.u.diff("y")).with_metadata(
            name="rel_vort", long_name="Relative vorticity",
            units="1/s")

    @property
    def divergence(self) -> fr.ScalarField:
        r"""
        Horizontal divergence :math:`\nabla\cdot\boldsymbol{u}`.

        Description
        -----------
        ``u`` staggers in x and ``v`` in y, so ``u.diff("x")`` and
        ``v.diff("y")`` both land at the cell centre.
        """
        return (self.u.diff("x") + self.v.diff("y")).with_metadata(
            name="divergence", long_name="Horizontal divergence",
            units="1/s")
