"""WENO interpolation module."""
from __future__ import annotations

from typing import Literal

import jax.numpy as jnp

import fridom.framework as fr

COEFFS_D = {
    1: [1.0],
    2: [1.0/3.0, 2.0/3.0],
    3: [1.0/10.0, 6.0/10.0, 3.0/10.0],
    4: [1.0/35.0, 12.0/35.0, 18.0/35.0, 4.0/35.0],
    5: [1.0/126.0, 20.0/126.0, 60.0/126.0, 40.0/126.0, 5.0/126.0],
}

COEFFS_BETA1 = {
    2: [[[-1, 1]],
        [[-1, 1]]],
    3: [[[1, -2, 1], [1, -4, 3]],
        [[1, -2, 1], [1, 0, -1]],
        [[1, -2, 1], [3, -4, 1]]],
}

COEFFS_BETA2 = {
    2: [1.0],
    3: [13.0/12.0, 1.0/4.0],
}

# only stencil sizes for which all coefficient tables are available
# are supported
MIN_STENCIL_SIZE = max(
    min(COEFFS_D), min(COEFFS_BETA1), min(COEFFS_BETA2))
MAX_STENCIL_SIZE = min(
    max(COEFFS_D), max(COEFFS_BETA1), max(COEFFS_BETA2))


@fr.utils.jaxify
class InterWENO(fr.grid.BiasedInterpolationModule):

    """
    WENO interpolation module.

    Parameters
    ----------
    order : int
        Order of the WENO interpolation (must be odd), default is 5.
    eps : float
        Small value to avoid division by zero, default is 1e-10.

    """

    name = "WENO Interpolation"

    def __init__(
            self,
            order: int = 5,
            eps: float = 1e-10,
            method: Literal["pointwise", "cell_average"] = "cell_average",
    ) -> None:

        super().__init__()

        # check if the order is valid (only odd orders are allowed)
        if order % 2 == 0:
            msg = (f"Order {order} is not odd. "
                   "Please use an odd order for WENO.")
            raise ValueError(msg)

        self.order = order
        self.stencil_size = (order + 1) // 2
        self.required_halo = self.stencil_size
        self.eps = eps

        if not MIN_STENCIL_SIZE <= self.stencil_size <= MAX_STENCIL_SIZE:
            msg = f"Order {order} is not supported. "
            msg += (f"Please use an order between "
                    f"{2 * MIN_STENCIL_SIZE - 1} and "
                    f"{2 * MAX_STENCIL_SIZE - 1}.")
            raise ValueError(msg)

        self.pol_coeffs = fr.grid.cartesian.compute_polynomial_coefficients(
            stencil_size=self.stencil_size, method=method)
        self.coeff_d = jnp.asarray(COEFFS_D[self.stencil_size])
        self.coeff_beta1 = jnp.asarray(COEFFS_BETA1[self.stencil_size])
        self.coeff_beta2 = jnp.asarray(COEFFS_BETA2[self.stencil_size])

    def _interpolate_axis(self,
                          x: jnp.ndarray,
                          bias: jnp.ndarray,
                          axis: int,
                          destination: fr.grid.AxisPosition) -> jnp.ndarray:

        size = self.stencil_size

        all_stencils = [fr.grid.Stencil(
            grid=self.grid, size=size, offset=size - n - 1,
            destination=destination,
            ).view(x, axis=axis) for n in range(size + 1)]

        left_weights = self._compute_weights(all_stencils[:-1], mode="left")
        right_weights = self._compute_weights(all_stencils[1:], mode="right")

        # compute the interpolations candidates for all stencils
        candidates = [sum(v*c for c, v in zip(
                          self.pol_coeffs[size - n], stencil, strict=False))
                      for n, stencil in enumerate(all_stencils)]

        left = sum(w*c for w, c
                   in zip(left_weights, candidates[:-1], strict=False))
        right = sum(w*c for w, c
                    in zip(right_weights, candidates[1:], strict=False))

        return jnp.where(bias > 0, left, right)


    def _compute_weights(
            self,
            stencils: list[fr.grid.Stencil],
            mode: Literal["left", "right"],
    ) -> list[jnp.ndarray]:

        def sort(x: list) -> list:
            return x if mode == "left" else list(x)[::-1]

        beta1 = self.coeff_beta1
        beta2 = self.coeff_beta2
        derivative_orders = self.stencil_size - 1

        # compute the smoothness indicators for each stencil
        smoothness_indicators = [
            sum(beta2[derivative_order] * (
                sum(beta1[stencil_number, derivative_order, stencil_index]
                        * value
                        for stencil_index, value in enumerate(sort(stencil))
                    ) ** 2
                ) for derivative_order in range(derivative_orders)
            ) for stencil_number, stencil in enumerate(sort(stencils))]

        # compute the weights for each stencil
        weights = [
            self.coeff_d[stencil_number] / ( (beta + self.eps) ** 2 )
            for stencil_number, beta in enumerate(smoothness_indicators)]

        # normalize the weights
        weights_sum = sum(weights)

        return sort([w / weights_sum for w in weights])
