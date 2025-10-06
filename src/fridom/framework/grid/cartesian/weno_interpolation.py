"""WENO interpolation module."""
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import fridom.framework as fr

if TYPE_CHECKING:  # pragma: no cover
    import numpy as np

@fr.utils.jaxify
class InterWENO(fr.grid.cartesian.InterENO):

    """
    WENO interpolation module.

    Description
    -----------
    TODO

    Parameters
    ----------
    order : int
        Order of the WENO interpolation (must be odd), default is 5.
    eps : float
        Small value to avoid division by zero, default is 1e-10.

    """

    name = "WENO Interpolation"
    max_implementation_order = 5

    def __init__(self, order: int = 5, eps: float = 1e-10) -> None:
        super().__init__()

        if order % 2 == 0:
            msg = f"Order {order} is not odd. Please use an odd order for WENO."
            raise ValueError(msg)

        if order > self.max_implementation_order:
            msg = f"Order {order} is too high. "
            msg += f"Please use an order <= {self.max_implementation_order}."
            raise ValueError(msg)

        self.order = order
        self.stencil_size = (order + 1) // 2
        self.required_halo = self.stencil_size
        self.eps = eps

        self.pol_coeffs = None  # Coefficients for polynomial reconstruction
        self.stencil_slices = None
        self.d_coeffs = None
        self.beta_inner = None
        self.beta_outer = None

    def _on_setup(self) -> None:
        # check that the grid is periodic in all dimensions
        if not all(self.grid.periodic_bounds):
            msg = "WENO only works on periodic grids"
            raise ValueError(msg)

        # Initialize polynomial coefficients
        self.pol_coeffs = self.compute_polynomial_coefficients_cell_average(
            self.stencil_size)

        # Precompute the axis slices for the stencil
        self.stencil_slices = self._compute_stencil_slices(
            self.stencil_size, self.grid.n_dims)

        # Precompute the d coefficients for the WENO reconstruction
        self.d_coeffs = self.compute_d_coeffs(self.stencil_size)

        # Precompute the beta coefficients and slices
        self.beta_coeffs = self.compute_beta_coeffs(
            self.stencil_size)
        self.beta_slices = self._compute_b_slices(
            self.stencil_size, self.grid.n_dims)

    def _compute_stencil_slices(self, stencil_size: int, ndim: int,
                                ) -> list[list[tuple[slice]]]:
        """
        Precompute the slices for each stencil.

        The first index of the output list corresponds to the axis, and the second
        index corresponds to the stencil offset.
        """
        k = stencil_size

        # Precompute the axis slices for the stencil
        axis_slices = [
            slice(i if i != 0 else None, -k + i + 1 if i != k - 1 else None)
            for i in range(k)
        ]

        # Build full slice tuples per stencil offset and per axis
        return [[
            tuple(slice(None) if ax != axis else axis_slices[i]
                  for ax in range(ndim))
            for i in range(k)
        ] for axis in range(ndim)]

    def _compute_b_slices(self, stencil_size: int, ndim: int,
                                ) -> list[list[tuple[slice]]]:
        """
        Precompute the slices for each stencil for the b coefficients.

        The first index of the output list corresponds to the axis, and the second
        index corresponds to the stencil offset.
        """
        k = stencil_size

        # Precompute the axis slices for the stencil
        axis_slices = [
            slice(i if i != 0 else None, -2*(k-1)+i if i != 2*(k-1) else None)
            for i in range(2*k-1)
        ]

        # Build full slice tuples per stencil offset and per axis
        return [[
            tuple(slice(None) if ax != axis else axis_slices[i]
                  for ax in range(ndim))
            for i in range(2*k-1)
        ] for axis in range(ndim)]

    def compute_d_coeffs(self, stencil_size: int) -> np.ndarray:
        """Compute the coefficients d_i for the WENO reconstruction."""
        tabulated_coeffs = {
            1: [1.0],
            2: [1.0/3.0, 2.0/3.0],
            3: [1.0/10.0, 6.0/10.0, 3.0/10.0],
            4: [1.0/35.0, 12.0/35.0, 18.0/35.0, 4.0/35.0],
            5: [1.0/126.0, 20.0/126.0, 60.0/126.0, 40.0/126.0, 5.0/126.0],
        }[stencil_size]
        return fr.config.ncp.asarray(tabulated_coeffs, dtype=fr.config.dtype_real)

    def compute_beta_coeffs(self, stencil_size: int) -> tuple[np.ndarray]:
        """Coefficients that are used to compute the smoothness indicators."""
        beta_order3 = [[[-1, 1]],
                       [[-1, 1]]]
        beta_order5 = [[[1, -2, 1], [1, -4, 3]],
                       [[1, -2, 1], [1, 0, -1]],
                       [[1, -2, 1], [3, -4, 1]]]
        beta_inner = {2: beta_order3, 3: beta_order5}[stencil_size]
        beta_outer = {
            2: [1.0],
            3: [13.0/12.0, 1.0/4.0],
        }[stencil_size]

        return (fr.config.ncp.asarray(beta_inner, dtype=fr.config.dtype_real),
                fr.config.ncp.asarray(beta_outer, dtype=fr.config.dtype_real) )

    @fr.utils.jaxjit
    def reconstruct(self,
                    f: fr.ScalarField,
                    destination: fr.grid.Position) -> tuple[fr.ScalarField]:
        """
        Reconstruct the field at the destination points using WENO interpolation.

        Parameters
        ----------
        f : fr.ScalarField
            The field to be reconstructed.
        destination : fr.grid.Position
            The position where the field is to be reconstructed. (Can only 
            reconstuct in one dimension at a time)

        Returns
        -------
        tuple[fr.ScalarField]
            Left and right biased reconstructed fields. (Should be combined
            with a flux limiter to get the final reconstructed field)

        """
        # First, we need to find out the axis in which we are reconstructing
        axis = self._get_reconstruction_axis(f, destination)

        # Get the interpolation from the individual stencils
        interpolations = self._interpolate_stencils(f, axis)

        # Compute the smoothness indicators
        beta = self.compute_smoothness_indicators(f, axis)

        # Compute the left and right biased weights
        left_weights = self._compute_weights(self.d_coeffs, beta)
        right_weights = self._compute_weights(self.d_coeffs[::-1], beta)
        # left_weights = self.d_coeffs
        # right_weights = self.d_coeffs[::-1]
        # Compute the left and right biased reconstructed fields
        slices = self.stencil_slices[axis]
        left_arr = sum(left_weights[i] * interpolations[i][slices[i]]
                       for i in range(self.stencil_size))
        right_arr = sum(right_weights[i] * interpolations[i+1][slices[i]]
                        for i in range(self.stencil_size))

        # Create the reconstructed scalar fields
        left = fr.ScalarField(f.mset, mdata=deepcopy(f.mdata))
        right = fr.ScalarField(f.mset, mdata=deepcopy(f.mdata))

        # If the destination is the face, we interpolate one to the right
        offset = 0
        if destination.positions[axis] == fr.grid.AxisPosition.FACE:
            offset = 1

        # Set the reconstructed values
        right.arr = fr.utils.modify_array(right.arr,
                              self.beta_slices[axis][self.stencil_size-1-offset],
                              right_arr)
        left.arr = fr.utils.modify_array(left.arr,
                              self.beta_slices[axis][self.stencil_size-offset],
                              left_arr)

        # Set the positions of the reconstructed fields
        left.position = destination
        right.position = destination

        return left, right

    def _get_reconstruction_axis(self,
                                 f: fr.ScalarField,
                                 destination: fr.grid.Position) -> int:
        """
        Get the axis in which we are reconstructing.

        Parameters
        ----------
        f : fr.ScalarField
            The field to be reconstructed.
        destination : fr.grid.Position
            The position where the field is to be reconstructed. (Can only 
            reconstuct in one dimension at a time)

        Returns
        -------
        int
            The axis in which we are reconstructing.

        Raises
        ------
        ValueError
            1. If the reconstruction is not done in one dimension.
            2. If the destination position is the same as the field position.

        """
        # First, we need to find out the axis in which we are reconstructing
        diff_axes = [i for i, (pos, dest) in enumerate(
            zip(f.position.positions, destination.positions)) if pos != dest]

        if len(diff_axes) > 1:
            msg = "Reconstruction can only be done in one dimension at a time"
            raise ValueError(msg)
        if len(diff_axes) == 0:
            msg = "Destination position is the same as the field position"
            raise ValueError(msg)

        return diff_axes[0]

    def _interpolate_stencils(self,
                                f: fr.ScalarField,
                                axis: int) -> list[np.ndarray]:
            """
            Interpolate to each point in the stencil.

            Parameters
            ----------
            f : fr.ScalarField
                The field to be reconstructed.
            axis : int
                The axis in which we are reconstructing.

            Returns
            -------
            list[np.ndarray]
                The interpolated values at each point in the stencil.

            """
            k = self.stencil_size
            slices = self.stencil_slices[axis]
            coeffs = self.pol_coeffs

            return [sum(coeffs[n,i] * f.arr[slices[i]] for i in range(k))
                    for n in range(k+1)][::-1]  # reverse the order of the stencils

    def compute_smoothness_indicators(self,
                                      f: fr.ScalarField,
                                      axis: int) -> list[np.ndarray]:
        """Compute the smoothness indicators for each stencil."""
        stencil_size = self.stencil_size
        bi, co = self.beta_coeffs
        slices = self.beta_slices[axis]
        return [
            sum(
                co[j] * sum(
                    bi[n,j,i] * f.arr[slices[i+n]]
                    for i in range(stencil_size) # loop over each point in stencil
                ) ** 2  # square the derivative
                for j in range(stencil_size - 1))  # derivative loop
            for n in range(stencil_size)  # loop over stencils
        ]

    def _compute_weights(self,
                         d: np.ndarray,
                         beta: list[np.ndarray]) -> tuple[np.ndarray]:
        # compute the weights
        weights = [d[i] / ( (self.eps + beta[i])**2 )
                        for i in range(self.stencil_size)]
        # normalize the weights
        weights_sum = sum(weights)
        return [w / weights_sum for w in weights]
