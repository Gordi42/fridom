"""A WENO advection scheme following S. Mishra et al. (2021)."""
from __future__ import annotations

from copy import deepcopy

import fridom.framework as fr
import fridom.nonhydro as nh


@fr.utils.jaxify
class WENO(nh.modules.advection.AdvectionBase):

    r"""
    Weighted Essentially Non-Oscillatory (WENO) advection scheme.

    Description
    -----------
    This class implements the WENO advection scheme following S. Mishra et al. (2021).
    It is designed to handle discontinuities and sharp gradients in the solution
    while maintaining high accuracy.

    References
    ----------
    .. [1] S. Mishra, C. Pares-Pulido, and K. G. Pressel, "Arbitrarily high-order
    (weighted) essentially non-oscillatory finite difference schemes for anelastic
    flows on staggered meshes" *Communications in Computational Physics*, 2021.

    """

    name = "WENO Advection"

    def __init__(self,
                 order: int = 5,
                 symmetric_inter: fr.grid.InterpolationModule = None,
                 biased_inter: fr.grid.cartesian.InterWENO = None,
                 ) -> None:

        super().__init__()

        # check if the order is valid (only odd orders are allowed)
        if order % 2 == 0 or order < 1:
            msg = f"Invalid order {order}. Only odd orders >= 1 are allowed."
            raise ValueError(msg)

        cart = fr.grid.cartesian

        self.order = order
        self.symmetric_inter = symmetric_inter or cart.PolynomialInterpolation(
            order=order - 1)
        self.biased_inter = biased_inter or cart.InterWENO(order=order)

    def _on_setup(self) -> None:
        self.symmetric_inter.setup(self.mset)
        self.biased_inter.setup(self.mset)

    def advection(self,
                  velocity: fr.VectorField,
                  quantity: fr.ScalarField) -> fr.ScalarField:
        # Get the interpolation functions
        symmetric_interpolate = self.symmetric_inter.interpolate
        biased_interpolate = self.biased_inter.interpolate
        diff = self.diff_module

        res = fr.ScalarField(mset=quantity.mset, mdata=deepcopy(quantity.mdata))

        for axis, v in enumerate(velocity):
            # the flux position should be shifted from the quantity position
            flux_pos = quantity.position.shift(axis)

            # interpolate the velocity to the flux position using 
            # a symmetric interpolation
            v_at_flux = symmetric_interpolate(v, flux_pos)

            # interpolate the quantity to the flux position using WENO
            q_at_flux = biased_interpolate(quantity, flux_pos, velocity=v_at_flux)

            # calculate the flux
            flux = v_at_flux * q_at_flux

            res -= diff(flux, axis)
        return res

    @property
    def required_halo(self) -> int:
        """The required halo size based on the interpolation modules."""
        return self.inter.required_halo + self.weno.required_halo
