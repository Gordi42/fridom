from fridom.Framework.BoundaryConditions import *
from fridom.NonHydrostatic.ModelSettings import ModelSettings

class UBoundary(BoundaryConditions):
    """
    Boundary conditions for the u velocity.
    """
    def __init__(self, mset:ModelSettings) -> None:
        x_bounds = Dirichlet(mset, axis=0, btype=1)
        y_bounds = Dirichlet(mset, axis=1, btype=2)
        z_bounds = Neumann(mset, axis=2, btype=2)
        super().__init__([x_bounds, y_bounds, z_bounds])

class VBoundary(BoundaryConditions):
    """
    Boundary conditions for the v velocity.
    """
    def __init__(self, mset: ModelSettings) -> None:
        x_bounds = Dirichlet(mset, axis=0, btype=2)
        y_bounds = Dirichlet(mset, axis=1, btype=1)
        z_bounds = Neumann(mset, axis=2, btype=2)
        super().__init__([x_bounds, y_bounds, z_bounds])

class WBoundary(BoundaryConditions):
    """
    Boundary conditions for the w velocity.
    """
    def __init__(self, mset: ModelSettings) -> None:
        x_bounds = Neumann(mset, axis=0, btype=2)
        y_bounds = Neumann(mset, axis=1, btype=2)
        z_bounds = Dirichlet(mset, axis=2, btype=1)
        super().__init__([x_bounds, y_bounds, z_bounds])

class BBoundary(BoundaryConditions):
    """
    Boundary conditions for the buoyancy.
    """
    def __init__(self, mset: ModelSettings) -> None:
        x_bounds = Neumann(mset, axis=0, btype=2)
        y_bounds = Neumann(mset, axis=1, btype=2)
        z_bounds = Dirichlet(mset, axis=2, btype=2)
        super().__init__([x_bounds, y_bounds, z_bounds])

class PBoundary(BoundaryConditions):
    """
    Boundary conditions for the pressure.
    """
    def __init__(self, mset: ModelSettings) -> None:
        x_bounds = Neumann(mset, axis=0, btype=2)
        y_bounds = Neumann(mset, axis=1, btype=2)
        z_bounds = Neumann(mset, axis=2, btype=2)
        super().__init__([x_bounds, y_bounds, z_bounds])

class TriplePeriodic(BoundaryConditions):
    """
    Boundary conditions for triple periodic domain.
    """
    def __init__(self, mset: ModelSettings) -> None:
        x_bounds = Periodic(mset, axis=0)
        y_bounds = Periodic(mset, axis=1)
        z_bounds = Periodic(mset, axis=2)
        super().__init__([x_bounds, y_bounds, z_bounds])