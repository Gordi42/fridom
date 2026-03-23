"""Linear interpolation for cartesian grids."""
from __future__ import annotations

import fridom.framework as fr

ncp = fr.config.ncp

@fr.utils.jaxify
class LinearInterpolation(fr.grid.InterpolationModule):

    r"""
    Simple linear interpolation for cartesian grids.

    .. math::
        f(x + 0.5 \Delta x) = \frac{1}{2} (f(x) + f(x + \Delta x))
    """

    name = "Linear Interpolation"
    required_halo = 1

    def _interpolate_axis(self,
                          x: ncp.ndarray,
                          axis: int,
                          destination: fr.grid.AxisPosition) -> ncp.ndarray:

        shift = -1 if destination == fr.grid.AxisPosition.FACE else 1

        @self.grid.domain_decomp.shard_map
        def _interpolate(arr: ncp.ndarray) -> ncp.ndarray:
            rolled = ncp.roll(arr, shift=shift, axis=axis)
            return 0.5 * (arr + rolled)

        return _interpolate(x)
