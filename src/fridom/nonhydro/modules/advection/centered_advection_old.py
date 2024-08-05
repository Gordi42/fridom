# Import external modules
from typing import TYPE_CHECKING, Union
import numpy as np
# Import internal modules
from fridom.framework.modules import Module
# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase


class DiffBase(Module):
    def __init__(self, name="Differentiation"):
        super().__init__(name=name)
        return
    
    def setup(self, mset: 'ModelSettingsBase') -> None:
        raise NotImplementedError

    def diff(self, 
             arr: np.ndarray, 
             axis: int, 
             **kwargs) -> np.ndarray:
        """
        Differentiate an array along an axis.
        
        Description
        -----------
        This method should be implemented by child classes to compute the
        derivative of an array along a specified axis.
        
        Parameters
        ----------
        `arr` : `np.ndarray`
            The array to differentiate.
        `axis` : `int`
            The axis to differentiate along.
        `**kwargs`
            Additional keyword arguments. For example, the type of derivative
            (forward, backward, centered) for finite differences.
        
        Returns
        -------
        `np.ndarray`
            The derivative of the array along the specified axis. 
            (same shape as `arr`)
        """
        raise NotImplementedError

    def div(self, 
            arrs: list[np.ndarray], 
            axes: list[int] | None = None, 
            **kwargs) -> np.ndarray:
        """
        Calculate the divergence of a vector field.
        
        Description
        -----------
        This method should be implemented by child classes to compute the
        divergence of a vector field. The vector field is represented by a list
        of arrays, where each array represents a component of the vector field.
        Note that the number of arrays in the list should be equal to the number
        of dimensions of the grid. An optional argument `axes` should be provided
        to specify the axes along which to compute the divergence.
        
        Parameters
        ----------
        `arrs` : `list[np.ndarray]`
            The list of arrays representing the vector field.
        `axes` : `list[int]` or `None` (default: `None`)
            The axes along which to compute the divergence. If `None`, the
            divergence is computed along all axes.
        `**kwargs`
            Additional keyword arguments.
        
        Returns
        -------
        `np.ndarray`
            The divergence of the vector field. (same shape as the arrays in `arrs`)
        
        Examples
        --------
        .. code-block:: python

            # Create diff module (Let mset be a ModelSettingsBase object)
            diff = DiffBase(...)
            diff.setup(mset)
            # let u, v, w be the components of the vector field
            # Calculate 3D divergence
            div = diff.div([u, v, w])
            # Calculate 2D horizontal divergence
            div = diff.div([u, v], axes=[0, 1])
        """
        raise NotImplementedError
    
    def grad(self, 
             arr: np.ndarray, 
             axes: list[int] | None = None,
             **kwargs) -> list[Union[np.ndarray,None]]:
        """
        Calculate the gradient of a scalar field.
        
        Description
        -----------
        This method should be implemented by child classes to compute the
        gradient of a scalar field. The gradient is computed along the specified
        axes. 
        
        Parameters
        ----------
        `arr` : `np.ndarray`
            The array representing the scalar field.
        `axes` : `list[int]` or `None` (default: `None`)
            The axes along which to compute the gradient. If `None`, the
            gradient is computed along all axes.
        `**kwargs`
            Additional keyword arguments.
        
        Returns
        -------
        `list[np.ndarray | None]`
            The gradient of the scalar field. The list contains the gradient
            components along each axis. Axis which are not included in `axes`
            will have a value of `None`. 
            E.g. for a 3D grid, `diff.grad(arr, axes=[0, 2])` will return
            `[du/dx, None, dv/dz]`.
        """
        raise NotImplementedError

    def laplacian(self,
                  arr: np.ndarray,
                  axes: list[int] | None = None,
                  **kwargs) -> np.ndarray:
        """
        Calculate the Laplacian of a scalar field.

        Description
        -----------
        This method should be implemented by child classes to compute the
        Laplacian of a scalar field. The Laplacian is computed along the
        specified axes.

        Parameters
        ----------
        `arr` : `np.ndarray`
            The array representing the scalar field.
        `axes` : `list[int]` or `None` (default: `None`)
            The axes along which to compute the Laplacian. If `None`, the
            Laplacian is computed along all axes.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        `np.ndarray`
            The Laplacian of the scalar field.
        """
        raise NotImplementedError

    def curl(self,
            arrs: list[np.ndarray],
            axes: list[int] | None = None,
            **kwargs) -> list[np.ndarray]:
        """
        Calculate the curl of a vector field. (\\nab \\times \\mathbf{u})

        Description
        -----------
        This method should be implemented by child classes to compute the
        curl of a vector field. The vector field is represented by a list
        of arrays, where each array represents a component of the vector field.
        Note that the number of arrays in the list should be equal to the number
        of dimensions of the grid. An optional argument `axes` should be provided
        to specify the axes along which to compute the curl.

        Parameters
        ----------
        `arrs` : `list[np.ndarray]`
            The list of arrays representing the vector field.
        `axes` : `list[int]` or `None` (default: `None`)
            The axes along which to compute the curl. If `None`, the
            curl is computed along all axes.
        `**kwargs`
            Additional keyword arguments.

        Returns
        -------
        `list[np.ndarray]`
            The curl of the vector field. (same shape as the arrays in `arrs`)

        Examples
        --------
        .. code-block:: python

            # Create diff module (Let mset be a ModelSettingsBase object)
            diff = DiffBase(...)
            diff.setup(mset)
            # let u, v, w be the components of the vector field
            # Calculate 3D curl
            curl = diff.rot([u, v, w])
            # yield [dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy]
            # Calculate horizontal curl
            curl = diff.rot([u, v, None], axes=[2])
            # yield [None, None, dv/dx - du/dy]
        """
        raise NotImplementedError


# Import external modules
from typing import TYPE_CHECKING
import numpy as np
from functools import partial
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.modules import setup_module, module_method

# Import type information
if TYPE_CHECKING:
    from fridom.framework.model_settings_base import ModelSettingsBase


class FiniteDifferences(DiffBase):
    _dynamic_attributes = ['_dx1']
    def __init__(self) -> None:
        super().__init__(name="Finite Differences")
        # ----------------------------------------------------------------
        #  Set attributes
        # ----------------------------------------------------------------
        self._dx1 = None
        return

    @setup_module
    def setup(self) -> None:
        
        # ----------------------------------------------------------------
        #  Calculate the operator coefficients
        # ----------------------------------------------------------------
        self._dx1 = 1 / config.ncp.array(self.mset.grid.dx, dtype=config.dtype_real)
        return

    @partial(utils.jaxjit, static_argnames=('axis', 'type'))
    @module_method
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
    @module_method
    def div(self,
            arrs: list[np.ndarray],
            axes: list[int] | None = None,
            **kwargs) -> np.ndarray:
        if axes is None:
            axes = list(range(len(arrs)))
        
        res = config.ncp.zeros(arrs[axes[0]].shape)
        for axis, arr in zip(axes, arrs):
            res += self._diff_backward(arr, axis)
        return res

    @partial(utils.jaxjit, static_argnames=('axes',))
    @module_method
    def grad(self,
             arr: np.ndarray,
             axes: list[int] | None = None,
             **kwargs) -> list[np.ndarray]:
            if axes is None:
                axes = list(range(arr.ndim))
            
            return [self._diff_forward(arr, i) if i in axes else None
                    for i in range(arr.ndim)]
    
    @partial(utils.jaxjit, static_argnames=('axes',))
    @module_method
    def laplacian(self,
                  arr: np.ndarray,
                  axes: list[int] | None = None,
                  **kwargs) -> np.ndarray:
            if axes is None:
                axes = list(range(arr.ndim))
            
            res = config.ncp.zeros(arr.shape)
            for axis in range(axes):
                res += self._diff_forward(self._diff_backward(arr, axis), axis)
            return res

    @partial(utils.jaxjit, static_argnames=('axes',))
    @module_method
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
        res = config.ncp.empty(arr.shape)

        next = tuple(slice(1, None) if i == axis else slice(None) 
                     for i in range(arr.ndim))
        prev = tuple(slice(None, -1) if i == axis else slice(None) 
                     for i in range(arr.ndim))

        diff = (arr[next] - arr[prev]) * self._dx1[axis]
        res = utils.modify_array(res, prev, diff)

        return res

    @partial(utils.jaxjit, static_argnames=('axis',))
    def _diff_backward(self, arr: np.ndarray, axis: int) -> np.ndarray:
        res = config.ncp.empty(arr.shape)

        next = tuple(slice(1, None) if i == axis else slice(None) 
                     for i in range(arr.ndim))
        prev = tuple(slice(None, -1) if i == axis else slice(None) 
                     for i in range(arr.ndim))

        diff = (arr[next] - arr[prev]) * self._dx1[axis]
        res = utils.modify_array(res, next, diff)

        return res

    @partial(utils.jaxjit, static_argnames=('axis',))
    def _diff_centered(self, arr: np.ndarray, axis: int) -> np.ndarray:
        res = config.ncp.empty(arr.shape)

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


import fridom.framework as fr
import fridom.nonhydro as nh
# Import external modules
from typing import TYPE_CHECKING
from functools import partial
# Import internal modules
from fridom.framework import config, utils
from fridom.framework.modules.module import Module, setup_module, module_method
# Import type information
if TYPE_CHECKING:
    from numpy import ndarray
    from fridom.framework.model_state import ModelState
    from fridom.framework.grid import InterpolationBase, DiffBase
    from fridom.framework.field_variable import FieldVariable

class CenteredAdvection(Module):
    """
    Centered advection scheme.

    Description
    -----------
    Let :math:`\\mathbf{v}` be the velocity field and :math:`q` be the quantity
    to be advected. For a divergence-free velocity field, the advection term
    can be written as:

    .. math::
        \\mathbf{v} \\cdot \\nabla q = \\nabla \\cdot (\\mathbf{v} q)

    Lets consider the :math:`x`-component of the flux 
    :math:`\\mathbf{F}=\\mathbf{v} q`. The flux divergence 
    :math:`\\partial_x F_x` is calculated using forward or backward differences.
    For that the flux is interpolated to the cell faces of the quantity :math:`q`.

    Parameters
    ----------
    `diff` : `DiffBase | None`, (default=None)
        Differentiation module to use.
        If None, the differentiation module of the grid is used.
    `interpolation` : `InterpolationBase | None`, (default=None)
        The interpolation module to use.
        If None, the interpolation module of the grid is used.
    """
    _dynamic_attributes = set(["mset"])
    def __init__(self, 
                 diff: 'DiffBase | None' = None,
                 interpolation: 'InterpolationBase | None' = None):
        super().__init__(name="Centered Advection")
        self._diff = FiniteDifferences()
        self._interpolation = interpolation
        return

    @setup_module
    def setup(self):
        # setup the differentiation modules
        if self.diff is None:
            self.diff = self.mset.grid._diff_mod
        else:
            self.diff.setup(mset=self.mset)

        # setup the interpolation modules
        if self.interpolation is None:
            self.interpolation = self.mset.grid._interp_mod
        else:
            self.interpolation.setup(mset=self.mset)
        return

    @utils.jaxjit
    def flux_divergence(self, 
                        velocity: 'tuple[FieldVariable]',
                        quantity: 'FieldVariable') -> 'ndarray':
        # shorthand notation
        inter = self.interpolation.interpolate
        Ro = self.mset.Ro
        q_pos = quantity.position

        flux_divergence = config.ncp.zeros_like(quantity.arr)
        for axis, v in enumerate(velocity):
            match q_pos[axis]:
                case fr.grid.AxisPosition.CENTER:
                    flux_pos = q_pos.shift(axis)
                    diff_type = "backward"
                case fr.grid.AxisPosition.FACE:
                    flux_pos = q_pos.shift(axis)
                    diff_type = "forward"

            # interpolate the velocity and quantity to the flux position
            flux = (  inter(v, flux_pos)
                    * inter(quantity, flux_pos)  ) 
            # calculate the flux divergence
            flux_divergence += self.diff.diff(flux, axis, type=diff_type)
        return Ro * flux_divergence

    @utils.jaxjit
    def advect_state(self, 
                     z: nh.State, 
                     dz: nh.State, 
                     velocity: tuple[fr.FieldVariable]) -> nh.State:
        # calculate the advection term
        for name, quantity in z.fields.items():
            if quantity.flags["NO_ADV"]:
                continue
            dz.fields[name].arr -= self.flux_divergence(velocity, quantity)
        return dz


    @module_method
    def update(self, mz: 'ModelState') -> None:
        """
        Compute the advection term of the state vector z.
        """
        # calculate the full velocity field
        # in future, this enables the use of a background velocity field
        zf = mz.z
        velocity = (zf.u, zf.v, zf.w)

        mz.dz = self.advect_state(zf, mz.dz, velocity)

        return mz

    # ================================================================
    #  Properties
    # ================================================================
    @property
    def info(self) -> dict:
        res = super().info
        res["diff"] = self.diff
        res["interpolation"] = self.interpolation
        return res

    @property
    def diff(self) -> 'DiffBase':
        """The differentiation module."""
        return self._diff
    
    @diff.setter
    def diff(self, value: 'DiffBase'):
        self._diff = value
        return

    @property
    def interpolation(self) -> 'InterpolationBase':
        """The interpolation module."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: 'InterpolationBase'):
        self._interpolation = value
        return

utils.jaxify_class(CenteredAdvection)