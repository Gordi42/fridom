import numpy as np
from .position_base import PositionBase

class InterpolationBase:
    def interpolate(self, 
                    arr: np.ndarray, 
                    origin: PositionBase, 
                    destination: PositionBase) -> np.ndarray:
        """
        Interpolate an array from one position to another.
        """
        raise NotImplementedError