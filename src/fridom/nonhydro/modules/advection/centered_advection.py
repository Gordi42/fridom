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
    from fridom.framework.grid import InterpolationModule, DiffBase
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
                 interpolation: 'InterpolationModule | None' = None):
        super().__init__(name="Centered Advection")
        self._diff = diff
        self._interpolation = interpolation
        self.required_halo = 2
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
                        quantity: 'FieldVariable') -> 'FieldVariable':
        # shorthand notation
        inter = self.interpolation.interpolate
        Ro = self.mset.Ro
        q_pos = quantity.position

        res = fr.FieldVariable(**quantity.get_kw())

        for axis, v in enumerate(velocity):
            # the flux position should be shifted from the quantity position
            flux_pos = q_pos.shift(axis)
            flux = inter(v, flux_pos) * inter(quantity, flux_pos)
            res += flux.diff(axis, order=1)
        return Ro * res

    @utils.jaxjit
    def advect_state(self, 
                     z: nh.State, 
                     dz: nh.State, 
                     velocity: tuple[fr.FieldVariable]) -> nh.State:
        # calculate the advection term
        for name, quantity in z.fields.items():
            if quantity.flags["NO_ADV"]:
                continue
            dz.fields[name] -= self.flux_divergence(velocity, quantity)
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
    def interpolation(self) -> 'InterpolationModule':
        """The interpolation module."""
        return self._interpolation

    @interpolation.setter
    def interpolation(self, value: 'InterpolationModule'):
        self._interpolation = value
        return

utils.jaxify_class(CenteredAdvection)