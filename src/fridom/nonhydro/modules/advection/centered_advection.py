# Import external modules
from typing import TYPE_CHECKING
from functools import partial
# Import internal modules
import fridom.framework as fr
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
        self._diff = diff
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
                    flux_pos = q_pos.shift(axis, "forward")
                    diff_type = "backward"
                case fr.grid.AxisPosition.RIGHT:
                    flux_pos = q_pos.shift(axis, "backward")
                    diff_type = "forward"
                case fr.grid.AxisPosition.LEFT: # should not happen
                    flux_pos = q_pos.shift(axis, "forward")
                    diff_type = "backward"

            # interpolate the velocity and quantity to the flux position
            flux = (  inter(v.arr, v.position, flux_pos)
                    * inter(quantity.arr, q_pos, flux_pos)  ) 
            # calculate the flux divergence
            flux_divergence += self.diff.diff(flux, axis, type=diff_type)
        return Ro * flux_divergence

    @module_method
    def update(self, mz: 'ModelState') -> None:
        """
        Compute the advection term of the state vector z.
        """
        # calculate the full velocity field
        # in future, this enables the use of a background velocity field
        zf = mz.z
        velocity = (zf.u, zf.v, zf.w)

        # calculate the advection term
        for name, quantity in zf.fields.items():
            if quantity.no_adv:
                continue
            mz.dz.fields[name].arr -= self.flux_divergence(velocity, quantity)

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