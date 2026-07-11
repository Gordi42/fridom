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

- ``u``: velocity in x, on ``fr.spatial.Staggered("x")`` (the C-grid east
  face);
- ``v``: velocity in y, on ``fr.spatial.Staggered("y")`` (the north face);
- ``p``: the pressure / geopotential perturbation :math:`p = g\\eta`,
  on ``fr.spatial.Collocated()`` (cell centre).

Only ``rel_vort`` and ``divergence`` are parameter-free and live
here; ``ekin`` / ``epot`` / ``pot_vort`` carry :math:`c^2` and the
Rossby number and are bound diagnostics on the model (D2.3), not
State properties.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fridom.spatial.errors import MissingComponentError
from fridom.spatial.fields.vector_field import VectorField

if TYPE_CHECKING:  # pragma: no cover
    import fridom as fr

__all__ = ["MissingComponentError", "State"]


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
        """Velocity in x (declared by a shallow-water core)."""
        return self.require(
            "u", hint="declared by a shallow-water core module, "
                      "e.g. sw.modules.DynamicalCore")

    @property
    def v(self) -> fr.spatial.ScalarField:
        """Velocity in y (declared by a shallow-water core)."""
        return self.require(
            "v", hint="declared by a shallow-water core module, "
                      "e.g. sw.modules.DynamicalCore")

    @property
    def p(self) -> fr.spatial.ScalarField:
        r"""Pressure / geopotential perturbation :math:`p = g\eta`."""
        return self.require(
            "p", hint="declared by a shallow-water core module, "
                      "e.g. sw.modules.DynamicalCore")

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
        advection module; identity on periodic axes.
        """
        u, v = self.u, self.v
        corner = u.function_space.bare.replace(
            y=v.function_space.bare.factor("y"))
        return (v.diff("x").retag(corner)
                - u.diff("y").retag(corner)).with_metadata(
            name="rel_vort", long_name="Relative vorticity",
            units="1/s")

    @property
    def divergence(self) -> fr.spatial.ScalarField:
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
