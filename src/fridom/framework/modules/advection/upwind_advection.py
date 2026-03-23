"""Upwind advection scheme."""
from __future__ import annotations

from copy import deepcopy

import fridom.framework as fr


@fr.utils.jaxify
class UpwindAdvection(fr.modules.advection.AdvectionBase):

    r"""
    Upwind advection scheme.

    Description
    -----------
    For the upwind advection scheme, we assume that the velocity field is
    divergence-free. The advection term can then be written as:

    .. math::
        \mathcal{A}(\boldsymbol{v}, q) = -\boldsymbol{v} \cdot \nabla q =
            - \nabla \cdot (\boldsymbol{v} q)

    where :math:`q` is the quantity to be advected and :math:`\boldsymbol{v}`
    is the velocity field. The flux divergence :math:`\nabla \cdot (\boldsymbol{v} q)`
    is calculated using forward or backward differences. For that the flux is
    interpolated to the cell faces of the quantity :math:`q`:

    ::

                    Position of the quantity q
                                ↓
        |       x       |       x       |       x       |
                        ↑
            Position of the flux Fx

    The quantities at the cell faces are interpolated using an upwind interpolation
    scheme. The velocity field is interpolated to the cell faces of q
    using a centered interpolation scheme.

    """

    name = "Upwind Advection"

    def __init__(self,
                 order: int = 3,
                 symmetric_inter: fr.grid.InterpolationModule = None,
                 biased_inter: fr.grid.UpwindInterpolation = None,
                 ) -> None:

        super().__init__()

        cart = fr.grid.cartesian

        self.order = order
        self.interp_module = symmetric_inter or cart.PolynomialInterpolation(
            order=order - 2)
        self.biased_inter = biased_inter or cart.UpwindInterpolation(order=order-1)

    def _on_setup(self) -> None:
        self.biased_inter.setup(self.mset)

    def advection(self,  # noqa: D102
                  velocity: fr.VectorField,
                  quantity: fr.ScalarField) -> fr.ScalarField:

        # Get the interpolation functions
        symmetric_interpolate = self.interp_module.interpolate
        biased_interpolate = self.biased_inter.interpolate
        diff = self.diff_module.diff

        res = fr.ScalarField(mset=quantity.mset, mdata=deepcopy(quantity.mdata))

        for axis, v in enumerate(velocity):
            # the flux position should be shifted from the quantity position
            flux_pos = quantity.position.shift(axis)

            # interpolate the velocity to the flux position using
            # a symmetric interpolation
            v_at_flux = symmetric_interpolate(v, flux_pos)

            # interpolate the quantity to the flux position using WENO
            q_at_flux = biased_interpolate(quantity, v_at_flux, flux_pos)

            # calculate the flux
            flux = v_at_flux * q_at_flux

            res -= diff(flux, axis)
        return res

    @property
    def required_halo(self) -> int:
        """The required halo size based on the interpolation modules."""
        return max(self.interp_module.required_halo, self.biased_inter.required_halo)
