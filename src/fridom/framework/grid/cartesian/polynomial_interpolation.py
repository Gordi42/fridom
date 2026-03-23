"""Centered polynomial interpolation for cartesian grids."""
from __future__ import annotations

import fridom.framework as fr

ncp = fr.config.ncp

@fr.utils.jaxify
class PolynomialInterpolation(fr.grid.InterpolationModule):

    r"""
    Polynomial interpolation for cartesian grids.

    Description
    -----------
    Consider the following grid points:

    .. math::
        x_i = (i - n/2) \Delta x, \quad i = 0, 1, \ldots, n

    where :math:`n` is the (odd) order of the polynomial interpolation. For
    example for :math:`n = 3` we have the following grid points:

    ::

            We want to interpolate the field to this point (x=0)
                                    ↓
                |   x_0   |   x_1   |   x_2   |   x_3   |
        x/dx =     -3/2      -1/2       1/2       3/2

    Let :math:`f_i` be the field values at :math:`x_i`. We define the
    continuous extension of the field as:

    .. math::
        f(x) = \sum_{i=0}^{n} \left(
            \prod_{j=0, j \neq i}^{n} \left(
                \frac{x - x_j}{x_i - x_j} f_i
            \right)
        \right)

    By definition, :math:`f(x_i) = f_i` holds. Finally, to interpolate the
    field to the point :math:`x=0`, we insert :math:`x=0` into the above
    expression. Note that the grid spacing :math:`\Delta x` cancels out.

    .. math::
        f(0) = \sum_{i=0}^{n} c_i f_i

    with the coefficients :math:`c_i` given by:

    .. math::
        c_i = \prod_{j=0, j \neq i}^{n} \frac{j-n/2}{j - i}

    """

    name = "Polynomial Interpolation"
    def __init__(self, order: int = 1) -> None:
        super().__init__()

        # check if the order is valid (only odd orders are allowed)
        if order % 2 == 0 or order < 1:
            msg = f"Invalid order {order}. Only odd orders >= 1 are allowed."
            raise ValueError(msg)

        self.required_halo = order // 2 + 1
        self.order = order

        coeffs = []
        for i in range(order+1):
            c = fr.config.dtype_real(1)
            for j in range(order+1):
                if j != i:
                    c *= (j - order/2) / (j - i)
            coeffs.append(c)

        self._coeffs = ncp.asarray(coeffs, dtype=fr.config.dtype_real)

    def _interpolate_axis(self,
                          x: ncp.ndarray,
                          axis: int,
                          destination: fr.grid.AxisPosition) -> ncp.ndarray:

        shift = 0 if destination == fr.grid.AxisPosition.FACE else 1
        start_shift = self.order // 2 + shift

        @self.grid.domain_decomp.shard_map
        def _interpolate(arr: ncp.ndarray) -> ncp.ndarray:
            return sum(ncp.roll(arr, shift=start_shift - i, axis=axis) * self._coeffs[i]
                       for i in range(self.order + 1))

        return _interpolate(x)
