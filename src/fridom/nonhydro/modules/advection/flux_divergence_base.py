"""Base class for flux divergence advection schemes."""
from __future__ import annotations

from abc import abstractmethod

import fridom.framework as fr
import fridom.nonhydro as nh


class FluxDivergenceBase(fr.modules.advection.AdvectionBase):

    r"""
    Base class for flux divergence advection schemes.

    Description
    -----------
    Assuming that the velocity field is divergence-free, then we can write
    the advection of the momentum flux tensor as:

    .. math::
        (\mathbf{u} \cdot \nabla) \mathbf{u} = \nabla \cdot \mathbf{F}

    where :math:`\mathbf{F}` is the momentum flux tensor:

    .. math::
        \mathbf{F} = \mathbf{u}^T \mathbf{u} = \begin{bmatrix}
            u^2 & uv & uw \\
            uv & v^2 & vw \\
            uw & vw & w^2
        \end{bmatrix}

    Child classes must implement the compute_momentum_flux_tensor method.

    """

    @abstractmethod
    def compute_momentum_flux_tensor(self, velocity: nh.VectorField,
                                     ) -> tuple[nh.VectorField]:
        """
        Compute the momentum flux tensor.

        Parameters
        ----------
        velocity : nh.VectorField
            Velocity field (u, v, w).

        Returns
        -------
        tuple[nh.VectorField]
            Momentum flux tensor (vel*u, vel*v, vel*w).

        """
        msg = "Subclasses must implement the compute_momentum_flux_tensor method."
        raise NotImplementedError(msg)

    @abstractmethod
    def compute_tracer_advection(self,
                                 velocity: nh.VectorField,
                                 field: nh.ScalarField) -> nh.ScalarField:
        """
        Compute the advection of a tracer field.

        Parameters
        ----------
        velocity : nh.VectorField
            Velocity field (u, v, w).
        field : nh.ScalarField
            Tracer field to be advected.

        Returns
        -------
        nh.ScalarField
            Advection term for the tracer field.

        """
        msg = "Subclasses must implement the compute_tracer_advection method."
        raise NotImplementedError(msg)

    def advect_state(self, z: nh.State, dz: nh.State) -> nh.State:
        """Advect the state vector."""
        # get the differential operator
        diff = self.diff_module
        # calculate the momentum flux tensor
        fu, fv, fw = self.compute_momentum_flux_tensor(z.velocity)
        # calculate the flux divergence
        dz.u += self.scaling * diff.div(fu)
        dz.v += self.scaling * diff.div(fv)
        dz.w += self.scaling * diff.div(fw)
        # advection of tracer fields
        for field in z.tracers:
            if field.flags["NO_ADV"]:
                continue
            dz[field.name] += self.scaling * self.compute_tracer_advection(
                                                            z.velocity, field)
        return dz
