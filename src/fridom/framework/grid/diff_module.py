from copy import deepcopy
import fridom.framework as fr
from abc import abstractmethod
from functools import partial


@fr.utils.jaxify
class DiffModule(fr.modules.Module):
    """
    Base class for differentiation modules.

    Description
    -----------
    A differentiation module is a class that computes derivatives of a field,
    for example the partial derivative in a specific direction, or the gradient
    of a field, or divergence of a vector etc.
    """
    name = "Diff. Module"
    _is_mod_submodule = True

    @abstractmethod
    def diff(self,
             f: fr.ScalarField,
             axis: int,
             ) -> fr.ScalarField:
        r"""
        Compute the partial derivative of a field along an axis.

        .. math::
            \partial_i^n f

        with axis :math:`i`.

        Parameters
        ----------
        `f` : `fr.ScalarField`
            The field to differentiate.
        `axis` : `int`
            The axis along which to differentiate.

        Returns
        -------
        `fr.ScalarField`
            The derivative of the field along the specified axis.
        """
        raise NotImplementedError

    @fr.modules.module_method
    def grad(self,
             f: fr.ScalarField,
             axes: list[int] | None = None
             ) -> tuple[fr.ScalarField | None]:
        r"""
        Compute the gradient of a field.

        .. math::
            \nabla f = 
            \begin{pmatrix} \partial_1 f \\ \dots \\ \partial_n f \end{pmatrix}

        Parameters
        ----------
        `f` : `fr.ScalarField`
            The field to differentiate.
        `axes` : `list[int] | None` (default is None)
            The axes along which to compute the gradient. If `None`, the
            gradient is computed along all axes.

        Returns
        -------
        `tuple[fr.ScalarField | None]`
            The gradient of the field along the specified axes. The list contains 
            the gradient components along each axis. Axis which are not included 
            in `axes` will have a value of `None`. 
            E.g. for a 3D grid, `diff.grad(f, axes=[0, 2])` will return
            `[df/dx, None, df/dz]`.
        """
        if axes is None:
            axes = list(range(f.arr.ndim))
            
        return [self.diff(f, i) if i in axes else None
                for i in range(f.arr.ndim)]

    @fr.modules.module_method
    def div(self,
            vec: tuple[fr.ScalarField | None]
            ) -> fr.ScalarField:
        r"""
        Compute the divergence of a vector field.
        
        .. math::
            \nable \cdot \boldsymbol{v} = \sum_{i=1}^n \partial_i v_i

        Parameters
        ----------
        `vec` : `tuple[fr.ScalarField | None]`
            The vector field to compute the divergence of. Tuple entries that
            are `None` are ignored (for example to calculate 2D divergence
            in a 3D system).

        Returns
        -------
        `fr.ScalarField`
            The divergence of the field.

        Examples
        --------
        .. code-block:: python

            # Create diff module (Let mset be a ModelSettingsBase object)
            diff = DiffModule(...)
            diff.setup(mset)
            # let u, v, w be the components of the vector field
            # Calculate 3D divergence
            div = diff.div((u, v, w))
            # Calculate 2D horizontal divergence
            div = diff.div((u, v, None))
        """
        div = sum(self.diff(f, axis) 
                  for axis, f in enumerate(vec) if f is not None)
        return div

    @fr.modules.module_method
    def laplacian(self,
                  f: fr.ScalarField,
                  axes: tuple[int] | None = None
                  ) -> fr.ScalarField:
        r"""
        Compute the Laplacian of a scalar field.

        .. math::
            \nabla^2 f = \sum_{i=1}^n \partial_i^2 f

        Parameters
        ----------
        `f` : `fr.ScalarField`
            The field to differentiate.
        `axes` : `tuple[int] | None` (default is None)
            The axes along which to compute the Laplacian. If `None`, the
            Laplacian is computed along all axes.

        Returns
        -------
        `fr.ScalarField`
            The Laplacian of the field.
        """
        if axes is None:
            axes = list(range(f.arr.ndim))

        laplace = fr.ScalarField(mset=f.mset, mdata=deepcopy(f.mdata))
        for axis in axes:
            laplace += self.diff(self.diff(f, axis), axis)
        return laplace
