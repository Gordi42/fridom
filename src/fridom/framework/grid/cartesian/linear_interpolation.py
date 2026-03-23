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

        stencil = fr.grid.Stencil(
            grid=self.grid, size=2, offset=0, destination=destination)
        return sum(stencil.view(x, axis=axis)) * 0.5
