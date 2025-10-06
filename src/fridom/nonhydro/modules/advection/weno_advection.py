"""A WENO advection scheme following S. Mishra et al. (2021)."""
from __future__ import annotations

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
                 inter: fr.grid.InterpolationModule = None,
                 weno: fr.grid.cartesian.InterWENO = None,
                 flux_function: fr.modules.flux_functions.FluxFunctionBase = None,
                 ) -> None:

        super().__init__()

        # check if the order is valid (only odd orders are allowed)
        if order % 2 == 0 or order < 1:
            msg = f"Invalid order {order}. Only odd orders >= 1 are allowed."
            raise ValueError(msg)

        self.order = order
        self.inter = inter or fr.grid.cartesian.PolynomialInterpolation(order=order)
        self.weno = weno or fr.grid.cartesian.InterWENO(order=order)
        self.flux_function = flux_function or fr.modules.flux_functions.Upwind()

    def _on_setup(self) -> None:
        self.inter.setup(self.mset)
        self.weno.setup(self.mset)
        self.flux_function.setup(self.mset)

    @fr.utils.jaxjit
    def advect_state(self, z: nh.State, dz: nh.State) -> nh.State:  # noqa: D102
        # Get the interpolation functions
        ip = self.inter.interpolate
        weno = self.weno.reconstruct
        flux_fun = self.flux_function.compute
        diff = self.diff_module

        # ----------------------------------------------------------------
        #  Momentum advection
        # ----------------------------------------------------------------
        for v1 in z.velocity:
            for axis, v2 in enumerate(z.velocity):
                # Interpolate v2 to the position of v1
                v2_at_v1 = ip(v2, v1.position)
                # Interpolate v1*v2 to the face of v1 using WENO
                flux_left, flux_right = weno(v1*v2_at_v1,
                                                v1.position.shift(axis=axis))
                # Take the flux based on the advecting velocity
                # flux = flux_fun(flux_left=flux_left,
                #                 flux_right=flux_right,
                #                 velocity=v2_at_v1)
                flux = 0.5 * (flux_left + flux_right)# - 0.1 * (flux_right - flux_left)
                dz[v1.name] -= self.scaling * diff.diff(flux, axis=axis)

        # ----------------------------------------------------------------
        #  Tracer advection
        # ----------------------------------------------------------------
        # interpolate the velocity to the cell centers
        vel_at_center = (ip(v, self.grid.cell_center) for v in z.velocity)
        for field in z.tracers:
            if field.flags["NO_ADV"]:
                continue
            for axis, v in enumerate(vel_at_center):
                # Interpolate the tracer to the face using WENO
                tracer_left, tracer_right = weno(field,
                                                    field.position.shift(axis=axis))
                # Take the tracer from the face based on the advecting velocity
                # tracer = flux_fun(flux_left=tracer_left,
                #                   flux_right=tracer_right,
                #                   velocity=v)
                tracer = 0.5 * (tracer_left + tracer_right)
                # Add the gradient to the field
                dz[field.name] -= self.scaling * diff.diff(tracer, axis=axis) * v

        return dz

    @property
    def required_halo(self) -> int:
        """The required halo size based on the interpolation modules."""
        return self.inter.required_halo + self.weno.required_halo
