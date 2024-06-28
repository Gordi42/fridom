from fridom.framework.field_variable import FieldVariable
from fridom.framework.modules.module import Module

class InterpolationModule(Module):
    """
    Base class for interpolation modules. This class defines the interface for
    interpolating fields forwards and backwards in x, y, and z directions.

    Lets consider the grid:
    [b]-------------[b]-------------[b]        
    j-1              j              j+1
            [x]-------------[x]-------------[x]
            j-1              j              j+1
                    [f]-------------[f]-------------[f]
                    j-1              j              j+1

    - the grid cells are marked with [x]
    - the backward interpolation is marked with [b]
    - the forward interpolation is marked with [f]

    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # ==============================
    #   SYMMETRIC INTERPOLATION
    # ==============================

    def sym_xf(self, f: FieldVariable):
        """
        Forward interpolation in x-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            f_int (Array)  : Interpolated field
        """
        raise NotImplementedError

    def sym_xb(self, f: FieldVariable):
        """
        Backward interpolation in x-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            f_int (Array)  : Interpolated field.
        """
        raise NotImplementedError
    
    def sym_yf(self, f: FieldVariable):
        """
        Forward interpolation in y-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            f_int (Array)  : Interpolated field.
        """
        raise NotImplementedError

    def sym_yb(self, f: FieldVariable):
        """
        Backward interpolation in y-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            f_int (Array)  : Interpolated field.
        """
        raise NotImplementedError
    
    def sym_zf(self, f: FieldVariable):
        """
        Forward interpolation in z-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            f_int (Array)  : Interpolated field.
        """
        raise NotImplementedError
    
    def sym_zb(self, f: FieldVariable):
        """
        Backward interpolation in z-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            f_int (Array)  : Interpolated field.
        """
        raise NotImplementedError

    # ==============================
    #   LEFT-BIASED INTERPOLATION
    # ==============================

    def lb_xf(self, f: FieldVariable):
        """
        Left-biased forward interpolation in x-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            Array  : Interpolated field
        """
        raise NotImplementedError
    
    def lb_xb(self, f: FieldVariable):
        """
        Left-biased backward interpolation in x-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            Array  : Interpolated field.
        """
        raise NotImplementedError
    
    def lb_yf(self, f: FieldVariable):
        """
        Left-biased forward interpolation in y-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            Array  : Interpolated field.
        """
        raise NotImplementedError
    
    def lb_yb(self, f: FieldVariable):
        """
        Left-biased backward interpolation in y-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            Array  : Interpolated field.
        """
        raise NotImplementedError
    
    def lb_zf(self, f: FieldVariable):
        """
        Left-biased forward interpolation in z-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            Array  : Interpolated field.
        """
        raise NotImplementedError
    
    def lb_zb(self, f: FieldVariable):
        """
        Left-biased backward interpolation in z-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            Array  : Interpolated field.
        """
        raise NotImplementedError


# remove symbols from the namespace
del FieldVariable, Module