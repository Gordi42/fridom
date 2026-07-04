"""State vectors for the hydrostatic model."""
from __future__ import annotations

from collections import OrderedDict

import fridom.framework as fr
import fridom.hydrostatic as hs

NEUMANN = fr.grid.BCType.NEUMANN
DIRICHLET = fr.grid.BCType.DIRICHLET

@fr.utils.jaxify
class State(hs.VectorField):

    """
    State vector for the hydrostatic model.

    Description
    -----------
    The state vector for the hydrostatic model consists of the following
    scalar fields:

    - u: Velocity in the x-direction (zonal wind).
    - v: Velocity in the y-direction (meridional wind).
    - b: Buoyancy.

    """

    def __init__(self, mset: hs.ModelSettings, **kwargs: any) -> None:
        super().__init__(mset, **kwargs)
        # we set the class to State, so that child classes will always be
        # of type State
        self.__class__ = State

    @staticmethod
    def _create_default_fields(mset: hs.ModelSettings,
                               vector_dim: int | None,  # noqa: ARG004
                               **kwargs: any,
                               ) -> OrderedDict[str, hs.ScalarField]:
        cell_center = mset.grid.cell_center

        u = fr.ScalarField(
            mset,
            name="u",
            long_name="u - velocity",
            units="m/s",
            position=cell_center.shift(axis=0),
            bc_types=(DIRICHLET, NEUMANN, NEUMANN),
            flags={"ENABLE_FRICTION": True},
            **kwargs)

        v = fr.ScalarField(
            mset,
            name="v",
            long_name="v - velocity",
            units="m/s",
            position=cell_center.shift(axis=1),
            bc_types=(NEUMANN, DIRICHLET, NEUMANN),
            flags={"ENABLE_FRICTION": True},
            **kwargs)

        b = fr.ScalarField(
            mset,
            name="b",
            long_name="Buoyancy",
            units="m/s²",
            position=cell_center.shift(axis=2),
            bc_types=(NEUMANN, NEUMANN, DIRICHLET),
            flags={"ENABLE_MIXING": True},
            **kwargs)

        fields = OrderedDict([("u", u), ("v", v), ("b", b)])
        return State._add_custom_fields(
            mset, fields, mset.custom_state_fields, **kwargs)

    # ----------------------------------------------------------------
    #  State Variables
    # ----------------------------------------------------------------

    @property
    def u(self) -> fr.ScalarField:
        """Velocity in the x-direction."""
        return self.fields["u"]

    @u.setter
    def u(self, value: fr.ScalarField) -> None:
        self.fields["u"] = value

    @property
    def v(self) -> fr.ScalarField:
        """Velocity in the y-direction."""
        return self.fields["v"]

    @v.setter
    def v(self, value: fr.ScalarField) -> None:
        """Velocity in the y-direction."""
        self.fields["v"] = value

    @property
    def b(self) -> fr.ScalarField:
        """Buoyancy."""
        return self.fields["b"]

    @b.setter
    def b(self, value: fr.ScalarField) -> None:
        self.fields["b"] = value

    @property
    def velocity(self) -> fr.VectorField:
        """The horizontal velocity vector field."""
        return self[:2]

    @property
    def tracers(self) -> fr.VectorField:
        """The tracer fields."""
        return self[2:]

    # ----------------------------------------------------------------
    #  Derived Variables
    # ----------------------------------------------------------------
    @property
    def w(self) -> fr.ScalarField:
        r"""
        Vertical velocity derived from horizontal divergence.

        The vertical velocity is derived from the horizontal divergence of the
        horizontal velocity field:

        .. math::

            w = - \int_{0}^{z} (\partial_x u + \partial_y v) \, dz

        """
        divergence = self.u.diff(axis=0) + self.v.diff(axis=1)
        return - divergence.cumulative_integral(axis=2, direction="forward")

class DiagnosticState(hs.VectorField):

    """Diagnostic state vector for the hydrostatic model."""
