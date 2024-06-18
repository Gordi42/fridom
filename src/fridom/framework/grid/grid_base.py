from fridom.framework.modelsettings import ModelSettingsBase
from fridom.framework.field_variable import FieldVariable

class GridBase:
    """
    This is the base class for all grids in the framework. A grid is a container
    with the meshgrid of the domain, and all differential operators.
    Physical meshgrids should be stored in the X attribute, while spectral 
    meshgrids should be stored in the K attribute.
    """
    def __init__(self) -> None:
        pass

    def setup(self) -> None:
        pass

    def cpu(self) -> "GridBase":
        pass

    def fft(self, f:FieldVariable) -> FieldVariable:
        pass

    def ifft(self, f:FieldVariable) -> FieldVariable:
        pass

    def interpolate(self, f:FieldVariable, axis:int) -> FieldVariable:
        """
        
        """
        pass

    def derivative(self, f:FieldVariable, axis:int, offset: int) -> FieldVariable:
        """
        Compute the derivative of a field variable in a given direction and offset.
        For example for a (x,y,z) grid, axis=1 and offset=0 would compute a forward
        derivative in the y direction, offset=1 would compute a backward derivative.
        """
        pass