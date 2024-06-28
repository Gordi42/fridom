from fridom.framework.field_variable import FieldVariable
from fridom.framework.modules.module import start_module
from .interpolation_module import InterpolationModule

class LinearInterpolation(InterpolationModule):
    """
    This class provides the linear interpolation of fields forwards and backwards in x, y, and z directions.
    """
    def __init__(self):
        super().__init__(name="linear interpolation")

    @start_module
    def start(self):
        self.cp = self.grid.cp
        # prepare the boundary conditions
        mset = self.grid.mset
        self.half = mset.dtype(0.5)
        self.bcx = "wrap" if mset.periodic_bounds[0] else "constant"
        self.bcy = "wrap" if mset.periodic_bounds[1] else "constant"
        self.bcz = "wrap" if mset.periodic_bounds[2] else "constant"

    # ==============================
    #   SYMMETRIC INTERPOLATION
    # ==============================

    def sym_xf(self, f: FieldVariable):
        """
        Forward interpolation in x-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field
        """
        # append one more cell to the right
        f_ext = self.cp.pad(f, ((0, 1), (0, 0), (0, 0)), mode=self.bcx)
        return (f_ext[:-1] + f_ext[1:]) * self.half
    
    def sym_xb(self, f: FieldVariable):
        """
        Backward interpolation in x-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field.
        """
        # append one more cell to the left
        f_ext = self.cp.pad(f, ((1, 0), (0, 0), (0, 0)), mode=self.bcx)
        return (f_ext[:-1] + f_ext[1:]) * self.half
    
    def sym_yf(self, f: FieldVariable):
        """
        Forward interpolation in y-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field.
        """
        # append one more cell to the top
        f_ext = self.cp.pad(f, ((0, 0), (0, 1), (0, 0)), mode=self.bcy)
        return (f_ext[:, :-1] + f_ext[:, 1:]) * self.half
    
    def sym_yb(self, f: FieldVariable):
        """
        Backward interpolation in y-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field.
        """
        # append one more cell to the bottom
        f_ext = self.cp.pad(f, ((0, 0), (1, 0), (0, 0)), mode=self.bcy)
        return (f_ext[:, :-1] + f_ext[:, 1:]) * self.half
    
    def sym_zf(self, f: FieldVariable):
        """
        Forward interpolation in z-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field.
        """
        # append one more cell to the front
        f_ext = self.cp.pad(f, ((0, 0), (0, 0), (0, 1)), mode=self.bcz)
        return (f_ext[:, :, :-1] + f_ext[:, :, 1:]) * self.half
    
    def sym_zb(self, f: FieldVariable):
        """
        Backward interpolation in z-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field.
        """
        # append one more cell to the back
        f_ext = self.cp.pad(f, ((0, 0), (0, 0), (1, 0)), mode=self.bcz)
        return (f_ext[:, :, :-1] + f_ext[:, :, 1:]) * self.half

    def __repr__(self) -> str:
        return self.name
