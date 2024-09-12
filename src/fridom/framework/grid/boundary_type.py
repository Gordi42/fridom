from enum import Enum, auto

class BCType(Enum):
    r"""
    Enum class for the type of boundary conditions for field variables.

    DIRICHLET: Dirichlet boundary conditions (:math:`u = 0` at the boundary).
    NEUMANN: Neumann boundary conditions (:math:`\partial_n u = 0` at the boundary).
    """
    DIRICHLET = auto()
    NEUMANN = auto()