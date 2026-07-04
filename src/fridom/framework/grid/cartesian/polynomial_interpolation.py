"""Centered polynomial interpolation for cartesian grids."""
from __future__ import annotations

from typing import Literal

import fridom.framework as fr

ncp = fr.config.ncp

@fr.utils.jaxify
class PolynomialInterpolation(fr.grid.InterpolationModule):

    """Polynomial interpolation for cartesian grids."""

    name = "Polynomial Interpolation"

    def __init__(
            self,
            order: int = 1,
            method: Literal["pointwise", "cell_average"] = "cell_average",
) -> None:
        super().__init__()

        # check if the order is valid (only odd orders are allowed)
        if order % 2 == 0 or order < 1:
            msg = f"Invalid order {order}. Only odd orders >= 1 are allowed."
            raise ValueError(msg)

        self.required_halo = order // 2 + 1
        self.order = order
        self.method = method

        # this is a symmetric interpolation, so we only need the coefficients
        # for interpolation to the center of the stencil
        coeffs = fr.grid.cartesian.compute_polynomial_coefficients(
            stencil_size=order+1, method=method)
        self._coeffs = coeffs[(self.order + 1) // 2]

    def _interpolate_axis(self,
                          x: ncp.ndarray,
                          axis: int,
                          destination: fr.grid.AxisPosition) -> ncp.ndarray:

        stencil_size = self.order + 1
        offset = self.order // 2
        stencil_view = fr.grid.Stencil(
            grid=self.grid, size=stencil_size, offset=offset,
            destination=destination,
            ).view(x, axis=axis)

        return sum(c * v for c, v
                   in zip(self._coeffs, stencil_view, strict=False))
