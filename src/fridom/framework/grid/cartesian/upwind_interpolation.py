"""Upwind interpolation methods."""
from __future__ import annotations

from typing import Literal

import fridom.framework as fr

ncp = fr.config.ncp

@fr.utils.jaxify
class UpwindInterpolation(fr.grid.BiasedInterpolationModule):

    """Upwind interpolation."""

    name = "Upwind Interpolation"

    def __init__(
            self,
            order: int = 2,
            method: Literal["pointwise", "cell_average"] = "cell_average",
    ) -> None:

        super().__init__()

        # check if the order is valid (only odd orders are allowed)
        if order % 2 != 0 or order < 0:
            msg = f"Invalid order {order}. Only even orders >= 0 are allowed."
            raise ValueError(msg)

        self.required_halo = order // 2
        self.order = order
        self.stencil_size = order + 1
        self.method = method

        if method == "pointwise":
            coeffs = fr.grid.cartesian.compute_polynomial_coefficients_pointwise(
                stencil_size=self.stencil_size)
        elif method == "cell_average":
            coeffs = fr.grid.cartesian.compute_polynomial_coefficients_cell_average(
                stencil_size=self.stencil_size)
        else:
            msg = (f"Invalid method {method}",
                   "Only 'pointwise' and 'cell_average' are supported.")
            raise ValueError(msg)


        self.left_coeffs = coeffs[order//2 + 1]
        self.right_coeffs = coeffs[order//2]


    def _interpolate_axis(self,
                          x: ncp.ndarray,
                          bias: ncp.ndarray,
                          axis: int,
                          destination: fr.grid.AxisPosition) -> ncp.ndarray:

        offset = 1 if destination == fr.grid.AxisPosition.CENTER else 0
        start = self.order // 2 + offset

        cl = self.left_coeffs
        cr = self.right_coeffs

        @self.grid.domain_decomp.shard_map
        def _interpolate(arr: ncp.ndarray, axis: int, v: ncp.ndarray) -> ncp.ndarray:
            left = sum(ncp.roll(arr, shift=start - i, axis=axis) * cl[i]
                        for i in range(self.stencil_size))
            right = sum(ncp.roll(arr, shift=start - i - 1, axis=axis) * cr[i]
                        for i in range(self.stencil_size))

            return ncp.where(v > 0, left, right)

        return _interpolate(x, axis=axis, v=bias)
