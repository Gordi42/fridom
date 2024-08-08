import fridom.framework as fr
from abc import abstractmethod


@fr.utils.jaxify
class InterpolationModule(fr.modules.Module):
    """
    The base class for interpolation methods.
    
    Description
    -----------
    An interpolation module is a class that interpolates a field from one position
    to another. For example, from the cell face to the cell center.
    """
    name = "Interpolation Module"
    _is_mod_submodule = True
    
    @abstractmethod
    def interpolate(self, 
                    f: fr.FieldVariable,
                    destination: fr.grid.Position) -> fr.FieldVariable:
        """
        Interpolate the field to the destination position.
        
        Parameters
        ----------
        `f` : `fr.FieldVariable`
            The field to interpolate.
        `destination` : `fr.grid.Position`
            The position to interpolate to.
        
        Returns
        -------
        `fr.FieldVariable`
            The interpolated field.
        """
        raise NotImplementedError