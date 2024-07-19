# Import external modules
from typing import TYPE_CHECKING
import numpy as np
from functools import partial
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.grid.diff_base import DiffBase
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase


class FiniteDifferences(DiffBase):
    _dynamic_attributes = ['_dx1']
    def __init__(self) -> None:
        super().__init__()
        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self._dx1 = None
        return

    def setup(self, mset: 'ModelSettingsBase') -> None:
        from .grid import Grid
        if not isinstance(mset.grid, Grid):
            raise ValueError("Finite differences only work with Cartesian grids.")
        
        # ----------------------------------------------------------------
        #  Calculate the operator coefficients
        # ----------------------------------------------------------------
        self._dx1 = 1 / config.ncp.array(mset.grid.dx, dtype=config.dtype_real)
        return

    @partial(utils.jaxjit, static_argnames=('axis', 'type'))
    def diff(self, 
             arr: np.ndarray, 
             axis: int, 
             type: str,
             **kwargs) -> np.ndarray:
        match type:
            case 'forward':
                return self._diff_forward(arr, axis)
            case 'backward':
                return self._diff_backward(arr, axis)
            case 'centered':
                return self._diff_centered(arr, axis)
            case None:
                return self._diff_centered(arr, axis)

    @partial(utils.jaxjit, static_argnames=('axes',))
    def div(self,
            arrs: list[np.ndarray],
            axes: list[int] | None = None,
            **kwargs) -> np.ndarray:
        if axes is None:
            axes = list(range(len(arrs)))
        
        res = config.ncp.zeros_like(arrs[axes[0]])
        for axis, arr in zip(axes, arrs):
            res += self._diff_backward(arr, axis)
        return res

    @partial(utils.jaxjit, static_argnames=('axes',))
    def grad(self,
             arr: np.ndarray,
             axes: list[int] | None = None,
             **kwargs) -> list[np.ndarray]:
            if axes is None:
                axes = list(range(arr.ndim))
            
            return [self._diff_forward(arr, i) if i in axes else None
                    for i in range(arr.ndim)]
    
    @partial(utils.jaxjit, static_argnames=('axes',))
    def laplacian(self,
                  arr: np.ndarray,
                  axes: list[int] | None = None,
                  **kwargs) -> np.ndarray:
            if axes is None:
                axes = list(range(arr.ndim))
            
            res = config.ncp.zeros_like(arr)
            for axis in range(axes):
                res += self._diff_forward(self._diff_backward(arr, axis), axis)
            return res

    @partial(utils.jaxjit, static_argnames=('axes',))
    def curl(self,
             arrs: list[np.ndarray],
             axes: list[int] | None = None,
             **kwargs) -> np.ndarray:
        if axes is None:
            axes = list(range(len(arrs)))
        if arrs[axes[0]].ndim == 2:
            return (  self._diff_forward(arrs[1], axis=0) 
                    - self._diff_forward(arrs[0], axis=1)  )
        if arrs[axes[0]].ndim == 3:
            ax1 = [1, 2, 0]
            ax2 = [2, 0, 1]
            res = []
            for i in range(3):
                if i not in axes:
                    res.append(None)
                    continue
                res.append(  self._diff_forward(arrs[(i+2)%3], axis=ax1[(i+1)%3])
                           - self._diff_forward(arrs[(i+1)%3], axis=ax2[(i+2)%3])  )
            return res
        else:
            raise ValueError("Curl is not implemented for this dimension.")

    @partial(utils.jaxjit, static_argnames=('axis',))
    def _diff_forward(self, arr: np.ndarray, axis: int) -> np.ndarray:
        res = config.ncp.empty_like(arr)

        next = tuple(slice(1, None) if i == axis else slice(None) 
                     for i in range(arr.ndim))
        prev = tuple(slice(None, -1) if i == axis else slice(None) 
                     for i in range(arr.ndim))

        diff = (arr[next] - arr[prev]) * self._dx1[axis]
        res = utils.modify_array(res, prev, diff)

        return res

    @partial(utils.jaxjit, static_argnames=('axis',))
    def _diff_backward(self, arr: np.ndarray, axis: int) -> np.ndarray:
        res = config.ncp.empty_like(arr)

        next = tuple(slice(1, None) if i == axis else slice(None) 
                     for i in range(arr.ndim))
        prev = tuple(slice(None, -1) if i == axis else slice(None) 
                     for i in range(arr.ndim))

        diff = (arr[next] - arr[prev]) * self._dx1[axis]
        res = utils.modify_array(res, next, diff)

        return res

    @partial(utils.jaxjit, static_argnames=('axis',))
    def _diff_centered(self, arr: np.ndarray, axis: int) -> np.ndarray:
        res = config.ncp.empty_like(arr)

        next = tuple(slice(2, None) if i == axis else slice(None) 
                     for i in range(arr.ndim))
        prev = tuple(slice(None, -2) if i == axis else slice(None) 
                     for i in range(arr.ndim))
        cent = tuple(slice(1, -1) if i == axis else slice(None)
                     for i in range(arr.ndim))
        
        diff = (arr[next] - arr[prev]) * 0.5 * self._dx1[axis]
        res = utils.modify_array(res, cent, diff)
        return res

utils.jaxify_class(FiniteDifferences)