"""Upwind flux function."""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import TYPE_CHECKING

import fridom.framework as fr

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np


@fr.utils.jaxify
class Upwind(fr.modules.flux_functions.FluxFunctionBase):

    r"""
    Upwind flux function.

    Description
    -----------
    Let's assume we have a flux from which we want to compute the divergence

    .. math::
        \partial_x F
        \quad
        \text{where } F = u q

    with the advection velocity :math:`u` and the advected quantity :math:`q`.
    In a finite-volume or finite-difference discretization, the upwind flux
    is computed by selecting the value of :math:`q` from the upstream direction,
    depending on the sign of the velocity :math:`u`. This function performs
    upwinding based on the staggered grid layout.

    Let :math:`F_i^f` be the flux at the face and :math:`F_i^c` be the flux
    at the center of the cell. The index :math:`i` refers to the cell index.
    To compute the upwinded flux, we first need to determine the left and right
    fluxes of the corresponding cell. This behavior depends on the position of
    the flux field:

    - If the flux is defined at **cell faces**, the divergence is computed
    on the cell centers which are located in front of the faces. Thus the flux at
    the right face is given by :math:`F_R = F_i^f` and the flux at the left face
    is given by :math:`F_L = F_{i-1}^f`.

    - If the flux is defined at **cell centers**, the divergence is computed
    on the cell faces which are located in front of the centers. Thus the flux at
    the right of the cell face is given by :math:`F_R = F_{i+1}^c` and the flux
    at the left of the cell face is given by :math:`F_L = F_{i}^c`.

    Finally, the velocity :math:`u` is interpolated to the flux position and the
    upwind value is chosen between left and right fluxes:

    .. math::
        F_i = \begin{cases}
            F_L & \text{if } u \ge 0 \\
            F_R & \text{if } u < 0
        \end{cases}

    """

    name = "Upwind"

    @partial(fr.utils.jaxjit, static_argnames=["axis"])
    def compute(self,  # noqa: D102
                flux: fr.ScalarField,
                velocity: fr.ScalarField,
                axis: int) -> fr.ScalarField:
        # interpolate velocity to the flux position
        u = self.interp_module.interpolate(velocity, flux.position)

        # Compute flux based on the flux position
        match flux.position[axis]:
            case fr.grid.AxisPosition.CENTER:
                return self._flux_at_center(flux, u, axis)
            case fr.grid.AxisPosition.FACE:
                return self._flux_at_face(flux, u, axis)

    def _flux_at_face(self,
                      flux: fr.ScalarField,
                      velocity: fr.ScalarField,
                      axis: int) -> fr.ScalarField:
        res = fr.ScalarField(mset=flux.mset, mdata=deepcopy(flux.mdata))

        @self.grid.domain_decomp.shard_map
        def _flux(right_flux: np.ndarray, u: np.ndarray) -> np.ndarray:
            # roll the array one to the left
            left_flux = fr.config.ncp.roll(right_flux, 1, axis=axis)
            pos_velocity = u > 0
            return fr.config.ncp.where(pos_velocity, left_flux, right_flux)

        res.arr = _flux(flux.arr, velocity.arr)

        return res

    def _flux_at_center(self,
                      flux: fr.ScalarField,
                      velocity: fr.ScalarField,
                      axis: int) -> fr.ScalarField:
        res = fr.ScalarField(mset=flux.mset, mdata=deepcopy(flux.mdata))

        @self.grid.domain_decomp.shard_map
        def _flux(left_flux: np.ndarray, u: np.ndarray) -> np.ndarray:
            # roll the array one to the right
            right_flux = fr.config.ncp.roll(left_flux, -1, axis=axis)
            pos_velocity = u >= 0
            return fr.config.ncp.where(pos_velocity, left_flux, right_flux)

        res.arr = _flux(flux.arr, velocity.arr)

        return res
