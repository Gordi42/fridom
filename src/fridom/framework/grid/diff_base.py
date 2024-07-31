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