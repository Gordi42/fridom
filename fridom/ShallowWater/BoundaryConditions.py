from fridom.Framework.BoundaryConditions import BoundaryConditions
from fridom.ShallowWater.ModelSettings import ModelSettings

class UBoundary(BoundaryConditions):
    """
    Boundary conditions for the u velocity.
    """
    def __init__(self, mset:ModelSettings) -> None:
        from fridom.Framework.BoundaryConditions import Dirichlet
        x_bounds = Dirichlet(mset, axis=0, btype=1)
        y_bounds = Dirichlet(mset, axis=1, btype=2)
        super().__init__([x_bounds, y_bounds])

class VBoundary(BoundaryConditions):
    """
    Boundary conditions for the v velocity.
    """
    def __init__(self, mset: ModelSettings) -> None:
        from fridom.Framework.BoundaryConditions import Dirichlet
        x_bounds = Dirichlet(mset, axis=0, btype=2)
        y_bounds = Dirichlet(mset, axis=1, btype=1)
        super().__init__([x_bounds, y_bounds])

class HBoundary(BoundaryConditions):
    """
    Boundary conditions for the Layer Thickness h.
    """
    def __init__(self, mset: ModelSettings) -> None:
        from fridom.Framework.BoundaryConditions import Neumann
        x_bounds = Neumann(mset, axis=0, btype=2)
        y_bounds = Neumann(mset, axis=1, btype=2)
        super().__init__([x_bounds, y_bounds])

# remove symbols from namespace
del BoundaryConditions, ModelSettings