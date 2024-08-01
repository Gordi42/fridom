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
    :math:`\\partial_x F_x` is calculated using forward or backward differences,
    for that the flux is interpolated to the cell faces of the quantity :math:`q`.

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

    @partial(utils.jaxjit, static_argnames=('directions',))
    def flux_divergence(self, 
                        velocity: 'tuple[FieldVariable]',
                        quantity: 'FieldVariable',
                        directions: 'tuple[str]') -> 'ndarray':
        # shorthand notation
        inter = self.interpolation.interpolate
        q_pos = quantity.position

        flux_divergence = config.ncp.zeros_like(quantity.arr)
        for axis, (v, direction) in enumerate(zip(velocity, directions)):
            # find the position of cell face where the flux is calculated
            flux_pos = q_pos.shift(axis, direction)
            # interpolate the velocity and quantity to the flux position
            flux = (  inter(v.arr, v.position, flux_pos)
                    * inter(quantity.arr, q_pos, flux_pos)  ) 
            # calculate the flux divergence
            diff_type = "forward" if direction == "backward" else "backward"
            flux_divergence += self.diff.diff(flux, axis, type=diff_type)
        return flux_divergence

    @module_method
    def update(self, mz: 'ModelState') -> None:
        """
        Compute the advection term of the state vector z.
        """
        # calculate the full velocity field
        # in future, this enables the use of a background velocity field
        zf = mz.z

        # shorthand notation
        Ro = self.mset.Ro
        velocity = (zf.u, zf.v, zf.w)

        mz.dz.u.arr -= Ro * self.flux_divergence(
            velocity, mz.z.u, ("backward", "forward", "forward"))
        mz.dz.v.arr -= Ro * self.flux_divergence(
            velocity, mz.z.v, ("forward", "backward", "forward"))
        mz.dz.w.arr -= Ro * self.flux_divergence(
            velocity, mz.z.w, ("forward", "forward", "backward"))
        mz.dz.b.arr -= Ro * self.flux_divergence(
            velocity, mz.z.b, ("forward", "forward", "forward"))

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