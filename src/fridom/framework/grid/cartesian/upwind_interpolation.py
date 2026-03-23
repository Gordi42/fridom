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

        coeffs = fr.grid.cartesian.compute_polynomial_coefficients(
            stencil_size=self.stencil_size, method=method)
        self.left_coeffs = coeffs[order//2 + 1]
        self.right_coeffs = coeffs[order//2]

    def _interpolate_axis(self,
                          x: ncp.ndarray,
                          bias: ncp.ndarray,
                          axis: int,
                          destination: fr.grid.AxisPosition) -> ncp.ndarray:

        size = self.stencil_size
        start = self.order // 2

        left_view = fr.grid.Stencil(
            grid=self.grid, size=size, offset=start, destination=destination,
            ).view(x, axis=axis)

        right_view = fr.grid.Stencil(
            grid=self.grid, size=size, offset=start-1, destination=destination,
            ).view(x, axis=axis)

        left = sum(c * v for c, v in zip(self.left_coeffs, left_view))
        right = sum(c * v for c, v in zip(self.right_coeffs, right_view))
        return ncp.where(bias > 0, left, right)
