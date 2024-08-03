# Import external modules
from typing import TYPE_CHECKING
from functools import partial
# Import internal modules
import fridom.framework as fr
from fridom.framework import config, utils
from fridom.framework.modules import setup_module, module_method
from fridom.framework.grid.interpolation_base import InterpolationBase
# Import type information
if TYPE_CHECKING:
    from numpy import ndarray


class PolynomialInterpolation(InterpolationBase):
    _dynamic_attributes = []
    def __init__(self, order: int = 1):
        super().__init__(name="Polynomial Interpolation")
        # order must be an odd number
        assert order % 2 == 1

        self.required_halo = order // 2 + 1
        self.order = order
        self._coeffs = None
        self._slices = None
        self._nexts = None
        self._prevs = None
        return

    @setup_module
    def setup(self) -> None:
        self.ndim = ndim = self.mset.grid.n_dims
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
        order = self.order
        coeffs = []
        for i in range(order+1):
            c = config.dtype_real(1)
            for j in range(order+1):
                if j != i:
                    c *= (j - order/2) / (j - i)
            coeffs.append(c)
        self._coeffs = coeffs

        slices = [slice(i, -order + i) for i in range(order)]
        slices.append(slice(order, None))

        all_slices = []
        for axis in range(ndim):
            sl = []
            for sli in slices:
                s = [slice(None)] * ndim
                s[axis] = sli
                sl.append(tuple(s))
            all_slices.append(sl)
        self._slices = all_slices

        self._nexts = tuple(self._get_slices(axis)[0] for axis in range(ndim))
        self._prevs = tuple(self._get_slices(axis)[1] for axis in range(ndim))
        return

    @partial(utils.jaxjit, static_argnames=('origin', 'destination'))
    def interpolate(self, 
                    arr: 'ndarray', 
                    origin: fr.grid.Position, 
                    destination: fr.grid.Position) -> 'ndarray':
        for axis in range(arr.ndim):
            arr = self.interpolate_axis(
                arr, 
                axis, 
                origin.positions[axis], 
                destination.positions[axis])
        return arr
    
    @partial(utils.jaxjit, static_argnames=('axis', 'origin', 'destination'))
    @module_method
    def interpolate_axis(self, 
                         arr: 'ndarray', 
                         axis: int,
                         origin: fr.grid.AxisPosition, 
                         destination: fr.grid.AxisPosition) -> 'ndarray':

        if arr.shape[axis] == 1:
            # no interpolation when the axis has only one cell
            return arr
        
        match origin, destination:
            case fr.grid.AxisPosition.LEFT, fr.grid.AxisPosition.LEFT:
                return arr
            case fr.grid.AxisPosition.LEFT, fr.grid.AxisPosition.CENTER:
                return self._half_forward(arr, axis)
            case fr.grid.AxisPosition.LEFT, fr.grid.AxisPosition.RIGHT:
                return self._full_forward(arr, axis)
            case fr.grid.AxisPosition.CENTER, fr.grid.AxisPosition.LEFT:
                return self._half_backward(arr, axis)
            case fr.grid.AxisPosition.CENTER, fr.grid.AxisPosition.CENTER:
                return arr
            case fr.grid.AxisPosition.CENTER, fr.grid.AxisPosition.RIGHT:
                return self._half_forward(arr, axis)
            case fr.grid.AxisPosition.RIGHT, fr.grid.AxisPosition.LEFT:
                return self._full_backward(arr, axis)
            case fr.grid.AxisPosition.RIGHT, fr.grid.AxisPosition.CENTER:
                return self._half_backward(arr, axis)
            case fr.grid.AxisPosition.RIGHT, fr.grid.AxisPosition.RIGHT:
                return arr

    def _raw_interpolate(self, arr: 'ndarray', axis: int) -> 'ndarray':
        return sum(arr[s] * self._coeffs[i] 
                   for i, s in enumerate(self._slices[axis]))

    def _half_forward(self, arr: 'ndarray', axis: int) -> 'ndarray':
        return utils.modify_array(
            arr, self._prevs[axis], self._raw_interpolate(arr, axis))
    
    def _half_backward(self, arr: 'ndarray', axis: int) -> 'ndarray':
        return utils.modify_array(
            arr, self._nexts[axis], self._raw_interpolate(arr, axis))
    
    def _full_forward(self, arr: 'ndarray', axis: int) -> 'ndarray':
        return utils.modify_array(
            arr, self._prevs[axis], self._raw_interpolate(arr, axis))
    
    def _full_backward(self, arr: 'ndarray', axis: int) -> 'ndarray':
        return utils.modify_array(
            arr, self._nexts[axis], self._raw_interpolate(arr, axis))
    
    def _get_slices(self, axis):
        n = self.order // 2
        if n == 0:
            end = None
        else:
            end = -n
        next = tuple(slice(n+1, end) if i == axis else slice(None) 
                     for i in range(self.ndim))
        prev = tuple(slice(n, -1-n) if i == axis else slice(None) 
                     for i in range(self.ndim))
        return next, prev

utils.jaxify_class(PolynomialInterpolation)