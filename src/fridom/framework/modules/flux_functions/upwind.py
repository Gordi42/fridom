"""Upwind flux function."""
from __future__ import annotations

from copy import deepcopy
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
    Let's assume we have two estimations of the flux :math:`F_L` and :math:`F_R`
    where :math:`F_L` is a flux which is biased to the left (e.g. for its 
    computation was more influenced by values on the left side of the cell) and
    :math:`F_R` is a flux which is biased to the right.
    The upwind flux is computed by selecting the flux based on the sign of the
    velocity :math:`u`:

    .. math::
        F = \begin{cases}
            F_L & \text{if } u \ge 0 \\
            F_R & \text{if } u < 0
        \end{cases}

    """

    name = "Upwind"

    @fr.utils.jaxjit
    def compute(self,  # noqa: D102
                flux_left: fr.ScalarField,
                flux_right: fr.ScalarField,
                velocity: fr.ScalarField) -> fr.ScalarField:
        # We assume that the flux_left and flux_right are at the same position
        # interpolate velocity to the flux position
        u = self.interp_module.interpolate(velocity, flux_left.position)

        flux = fr.ScalarField(mset=flux_left.mset, mdata=deepcopy(flux_left.mdata))

        @self.grid.domain_decomp.shard_map
        def _flux(right_flux: np.ndarray,
                  left_flux: np.ndarray,
                  u: np.ndarray) -> np.ndarray:
            return fr.config.ncp.where(u>=0, left_flux, right_flux)

        flux.arr = _flux(flux_right.arr, flux_left.arr, u.arr)
        return flux
