# Import external modules
from typing import TYPE_CHECKING
# Import internal modules
from fridom.framework.modules import Module
# Import type information
if TYPE_CHECKING:
    from numpy import ndarray
    from .position_base import PositionBase
    from fridom.framework.model_settings_base import ModelSettingsBase

class InterpolationBase(Module):
    """
    The base class for interpolation methods.
    
    Description
    -----------
    All interpolation methods should inherit from this class. An interpolation
    method should have a `setup` method that takes a `ModelSettingsBase` object
    as input and sets up the interpolation method. The interpolation method should
    also have an `interpolate` method that takes an array, an origin position, and
    a destination position as input and returns the interpolated array.
    
    Methods
    -------
    `setup(mset: ModelSettingsBase) -> None`
        Set up the interpolation method.
    `interpolate(arr: ndarray, origin: PositionBase, destination: PositionBase) -> ndarray`
        Interpolate an array from one position to another.
    """
    def __init__(self, name="Interpolation"):
        super().__init__(name=name)
        return
    
    def setup(self, mset: 'ModelSettingsBase') -> None:
        return

    def interpolate(self, 
                    arr: 'ndarray', 
                    origin: 'PositionBase', 
                    destination: 'PositionBase') -> 'ndarray':
        """
        Interpolate an array from one position to another.
        
        Description
        -----------
        This method should be implemented by child classes to interpolate an array
        from one position to another. For example, from a edge to a center.
        Each grid type may have different interpolation methods and different
        position types.
        
        Parameters
        ----------
        `arr` : `ndarray`
            The array to interpolate.
        `origin` : `PositionBase`
            The position of the array.
        `destination` : `PositionBase`
            The position to interpolate to.
        
        Returns
        -------
        `ndarray`
            The interpolated array.
        """
        return arr