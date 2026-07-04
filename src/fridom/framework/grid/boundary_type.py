"""Enum class for the type of boundary conditions for scalar fields."""
from enum import StrEnum


class BCType(StrEnum):

    r"""
    Enum class for the type of boundary conditions for scalar fields.

    DIRICHLET: Dirichlet boundary conditions (:math:`u = 0` at the boundary).
    NEUMANN: Neumann boundary conditions (:math:`\partial_n u = 0` at the
    boundary).

    """

    DIRICHLET = "dirichlet"
    NEUMANN = "neumann"
