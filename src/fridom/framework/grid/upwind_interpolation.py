"""Upwind interpolation methods."""
from __future__ import annotations

import fridom.framework as fr

ncp = fr.config.ncp

@fr.utils.jaxify
class UpwindInterpolation(fr.grid.BiasedInterpolationModule):

    """Upwind interpolation."""

    name = "Upwind Interpolation"

    def _interpolate_axis(self,
                          x: ncp.ndarray,
                          bias: ncp.ndarray,
                          axis: int,
                          destination: fr.grid.AxisPosition) -> ncp.ndarray:

        shift = 0 if destination == fr.grid.AxisPosition.FACE else 1

        @self.grid.domain_decomp.shard_map
        def _interpolate(arr: ncp.ndarray, axis: int, v: ncp.ndarray) -> ncp.ndarray:
            left = ncp.roll(arr, shift, axis=axis)
            right = ncp.roll(arr, shift - 1, axis=axis)
            return ncp.where(v > 0, left, right)

        return _interpolate(x, axis=axis, v=bias)
