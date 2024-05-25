from abc import ABC, abstractmethod

from fridom.framework.grid_base import GridBase
from fridom.framework.state_base import StateBase

class Projection:
    """
    Base class for projections. All projections should inherit from this class.

    Methods:
        project : Project a state to a subspace (e.g. geostrophic subspace).
    """
    def __init__(self, grid:GridBase) -> None:
        self.mset = grid.mset
        self.grid = grid

    @abstractmethod
    def __call__(self, z: StateBase) -> StateBase:
        """
        Abstract method for projecting a state to a subspace. All subclasses must implement this method.

        Arguments:
            z      (State) : The state to project.

        Returns:
            z_proj (State) : The projected state.
        """
        pass

# remove symbols from namespace
del ABC, abstractmethod, GridBase, StateBase