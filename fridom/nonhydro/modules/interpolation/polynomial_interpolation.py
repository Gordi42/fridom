from fridom.nonhydro.grid import Grid
from fridom.framework.field_variable import FieldVariable

from fridom.nonhydro.modules.interpolation.interpolation_module import InterpolationModule, InterpolationConstructor

class PolynomialInterpolation(InterpolationModule):
    """
    This class provides the polynomial interpolation of fields forwards and backwards in x, y, and z directions.
    """
    def __init__(self, grid: Grid, order: int):
        super().__init__(grid)
        # order must be an odd number
        assert order % 2 == 1

        self.order = order
        self.padding = order // 2
        self.cp = grid.cp
        mset = grid.mset
        self.half = mset.dtype(0.5)
        self.bcx = "wrap" if mset.periodic_bounds[0] else "constant"
        self.bcy = "wrap" if mset.periodic_bounds[1] else "constant"
        self.bcz = "wrap" if mset.periodic_bounds[2] else "constant"

        # coefficients for the polynomial interpolation

        # Let n be the order of the polynomial interpolation.
        # We consider the grid points x_i = (i - n/2) * dx, i = 0, 1, ..., n.
        # The polynomial interpolation is given by:
        # f(x) = \sum_{i=0}^{n} (
        #   \prod_{j=0, j!=i}^{n} (
        #       (x - x_j) / (x_i - x_j) * f(x_i)
        #   )
        # )
        # The coefficients for f(x_i) at x = 0 are given by:
        # c_i = \prod_{j=0, j!=i}^{n} (x_j / (x_j - x_i))
        #     = \prod_{j=0, j!=i}^{n} (j - n/2) / (j - i)
        
        coeffs = []
        for i in range(order+1):
            c = mset.dtype(1)
            for j in range(order+1):
                if j != i:
                    c *= (j - order/2) / (j - i)
            coeffs.append(c)
        self.coeffs = self.cp.asarray(coeffs)

        # create the slices for the interpolation
        slices = [slice(i, -order + i) for i in range(order)]
        slices.append(slice(order, None))

        self.xslices = [(s, slice(None), slice(None)) for s in slices]
        self.yslices = [(slice(None), s, slice(None)) for s in slices]
        self.zslices = [(slice(None), slice(None), s) for s in slices]

    # ==============================
    #   SYMMETRIC INTERPOLATION
    # ==============================

    def interpolate(self, 
                    f: FieldVariable, 
                    slices: list, 
                    padding: tuple,
                    bc: str):
        """
        Interpolate the field f using the slices and padding.
        """
        # append cells
        f_ext = self.cp.pad(f, padding, mode=bc)
        f_res = self.cp.zeros_like(f)
        for s, c in zip(slices, self.coeffs):
            f_res += f_ext[s] * c
        return f_res

    def sym_xf(self, f: FieldVariable):
        """
        Forward interpolation in x-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field
        """
        return self.interpolate(
            f, 
            self.xslices, 
            ((self.padding, self.padding + 1), (0, 0), (0, 0)), 
            self.bcx)

    def sym_xb(self, f: FieldVariable):
        """
        Backward interpolation in x-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field.
        """
        return self.interpolate(
            f, 
            self.xslices, 
            ((self.padding + 1, self.padding), (0, 0), (0, 0)), 
            self.bcx)
    
    def sym_yf(self, f: FieldVariable):
        """
        Forward interpolation in y-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field.
        """
        return self.interpolate(
            f, 
            self.yslices, 
            ((0, 0), (self.padding, self.padding + 1), (0, 0)), 
            self.bcy)
    
    def sym_yb(self, f: FieldVariable):
        """
        Backward interpolation in y-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field.
        """
        return self.interpolate(
            f, 
            self.yslices, 
            ((0, 0), (self.padding + 1, self.padding), (0, 0)), 
            self.bcy)
    
    def sym_zf(self, f: FieldVariable):
        """
        Forward interpolation in z-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field.
        """
        return self.interpolate(
            f, 
            self.zslices, 
            ((0, 0), (0, 0), (self.padding, self.padding + 1)), 
            self.bcz)
    
    def sym_zb(self, f: FieldVariable):
        """
        Backward interpolation in z-direction.
        
        Args:
            f (FieldVariable)  : Field to be interpolated.
        
        Returns:
            (Array)  : Interpolated field.
        """
        return self.interpolate(
            f, 
            self.zslices, 
            ((0, 0), (0, 0), (self.padding + 1, self.padding)), 
            self.bcz)

class PolynomialInterpolationConstructor(InterpolationConstructor):
    def __init__(self, order: int = 1):
        self.order = order

    def __call__(self, grid: Grid):
        return PolynomialInterpolation(grid, self.order)

    def __repr__(self) -> str:
        return "polynomial interpolation of order {}".format(self.order)

# remove symbols from the namespace
del Grid, FieldVariable, \
    InterpolationModule, InterpolationConstructor