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
                 eno_ip: fr.grid.InterpolationModule = None,
                 weno_ip: fr.grid.InterpolationModule = None,
                 flux_function: fr.modules.flux_functions.FluxFunctionBase = None,
                 ) -> None:

        super().__init__()

        # check if the order is valid (only odd orders are allowed)
        if order % 2 == 0 or order < 1:
            msg = f"Invalid order {order}. Only odd orders >= 1 are allowed."
            raise ValueError(msg)

        self.order = order
        self.eno_ip = eno_ip  # TODO(Silvano): Set a default
        self.weno_ip = weno_ip  # TODO(Silvano): Set a default
        self.flux_function = flux_function  # TODO(Silvano): Set a default

    def _on_setup(self) -> None:
        self.eno_ip.setup(self.mset)
        self.weno_ip.setup(self.mset)
        self.flux_function.setup(self.mset)

    @fr.utils.jaxjit
    def advect_state(self, z: nh.State, dz: nh.State) -> nh.State:  # noqa: D102
        # Get the interpolation functions
        eno_ip = self.eno_ip.interpolate
        weno_ip = self.weno_ip.interpolate
        flux_fun = self.flux_function.compute
        diff = self.diff_module

        # ----------------------------------------------------------------
        #  Momentum advection
        # ----------------------------------------------------------------
        for v1 in z.velocity:
            flux_components = []
            for axis, v2 in enumerate(z.velocity):
                # Interpolate v2 to the position of v1 using ENO
                v2_at_v1 = eno_ip(v2, v1.position)
                # Interpolate v1*v2 to the face of v1 using WENO
                flux = weno_ip(v1*v2_at_v1, v1.position.shift(axis=axis))
                # Take the flux based on the advecting velocity
                flux = flux_fun(flux=flux, velocity=v2_at_v1, axis=axis)
                flux_components.append(flux)

            # create the flux vector field
            flux_vector = nh.VectorField(self.mset, flux_components)
            dz[v1.name] -= self.scaling * diff.div(flux_vector)

        # ----------------------------------------------------------------
        #  Tracer advection
        # ----------------------------------------------------------------
        # interpolate the velocity to the cell centers
        vel_at_center = (eno_ip(v, self.grid.cell_center) for v in z.velocity)
        for field in z.tracers:
            if field.flags["NO_ADV"]:
                continue
            for axis, v in enumerate(vel_at_center):
                # Interpolate the tracer to the face using WENO
                tracer = weno_ip(field, field.position.shift(axis=axis))
                # Take the tracer from the face based on the advecting velocity
                tracer = flux_fun(flux=tracer, velocity=v, axis=axis)
                # Add the gradient to the field
                dz[field.name] -= self.scaling * diff.diff(tracer, axis=axis) * v

        return dz

    @property
    def required_halo(self) -> int:
        """The required halo size based on the interpolation modules."""
        return max(self.eno_ip.required_halo,
                   self.weno_ip.required_halo,
                   self.flux_function.required_halo)
