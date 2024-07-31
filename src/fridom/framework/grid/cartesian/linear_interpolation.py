# Import external modules
from typing import TYPE_CHECKING
from functools import partial
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.modules import setup_module, module_method
from fridom.framework.grid.interpolation_base import InterpolationBase
from .position import Position, AxisOffset
# Import type information
if TYPE_CHECKING:
    from numpy import ndarray


class LinearInterpolation(InterpolationBase):
    """
    Interpolation of the cartesian grid using linear interpolation.
    """
    _dynamic_attributes = []
    def __init__(self) -> None:
        super().__init__(name="Linear Interpolation")
        self.ndim: int = None
        self._nexts: tuple[slice] = None
        self._prevs: tuple[slice] = None
        return

    @setup_module
    def setup(self) -> None:
        self.ndim = ndim = self.mset.grid.n_dims
        self._nexts = tuple(self._get_slices(axis)[0] for axis in range(ndim))
        self._prevs = tuple(self._get_slices(axis)[1] for axis in range(ndim))
        return

    @partial(utils.jaxjit, static_argnames=('origin', 'destination'))
    def interpolate(self, 
                    arr: 'ndarray', 
                    origin: Position, 
                    destination: Position) -> 'ndarray':
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
                         origin: AxisOffset, 
                         destination: AxisOffset) -> 'ndarray':
        match origin, destination:
            case AxisOffset.LEFT, AxisOffset.LEFT:
                return arr
            case AxisOffset.LEFT, AxisOffset.CENTER:
                return self._half_forward(arr, axis)
            case AxisOffset.LEFT, AxisOffset.RIGHT:
                return self._full_forward(arr, axis)
            case AxisOffset.CENTER, AxisOffset.LEFT:
                return self._half_backward(arr, axis)
            case AxisOffset.CENTER, AxisOffset.CENTER:
                return arr
            case AxisOffset.CENTER, AxisOffset.RIGHT:
                return self._half_forward(arr, axis)
            case AxisOffset.RIGHT, AxisOffset.LEFT:
                return self._full_backward(arr, axis)
            case AxisOffset.RIGHT, AxisOffset.CENTER:
                return self._half_backward(arr, axis)
            case AxisOffset.RIGHT, AxisOffset.RIGHT:
                return arr

    @partial(utils.jaxjit, static_argnames=('axis'))
    def _half_forward(self, arr: 'ndarray', axis: int) -> 'ndarray':
        """
        Interpolates half a grid cell forward along an axis.
        """
        res = config.ncp.empty_like(arr)
        next = self._nexts[axis]; prev = self._prevs[axis]
        return utils.modify_array(res, prev, 0.5 * (arr[next] + arr[prev]))

    @partial(utils.jaxjit, static_argnames=('axis'))
    def _half_backward(self, arr: 'ndarray', axis: int) -> 'ndarray':
        """
        Interpolates half a grid cell backward along an axis.
        """
        res = config.ncp.empty_like(arr)
        next = self._nexts[axis]; prev = self._prevs[axis]
        return utils.modify_array(res, next, 0.5 * (arr[next] + arr[prev]))
    
    @partial(utils.jaxjit, static_argnames=('axis'))
    def _full_forward(self, arr: 'ndarray', axis: int) -> 'ndarray':
        """
        Interpolates a full grid cell forward along an axis.
        """
        res = config.ncp.empty_like(arr)
        next = self._nexts[axis]; prev = self._prevs[axis]
        return utils.modify_array(res, prev, arr[next])
    
    @partial(utils.jaxjit, static_argnames=('axis'))
    def _full_backward(self, arr: 'ndarray', axis: int) -> 'ndarray':
        """
        Interpolates a full grid cell backward along an axis.
        """
        res = config.ncp.empty_like(arr)
        next = self._nexts[axis]; prev = self._prevs[axis]
        return utils.modify_array(res, next, arr[prev])

    def _get_slices(self, axis):
        next = tuple(slice(1, None) if i == axis else slice(None) 
                     for i in range(self.ndim))
        prev = tuple(slice(None, -1) if i == axis else slice(None) 
                     for i in range(self.ndim))
        return next, prev

utils.jaxify_class(LinearInterpolation)