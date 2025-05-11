"""ENO interpolation module."""
from __future__ import annotations

from copy import deepcopy
from functools import partial
from typing import Literal

import numpy as np

import fridom.framework as fr


@fr.utils.jaxify
class InterENO(fr.grid.InterpolationModule):

    """
    ENO interpolation module.

    Description
    -----------
    Let's consider a field :math: `f(x)` which cell averages are known at the
    grid points :math: `x_{i+1/2}` and we want to interpolate the field at the
    grid points :math: `x_{i+1}`:

    ::

                                            Interpolate here
                                                  v
        | x_{i-1-1/2} |  x_{i-1/2}  |  x_{i+1/2}  | x_{i+1+1/2} | x_{i+2+1/2} |
        ^             ^             ^             ^             ^             ^
      x_{i-2}      x_{i-1}        x_{i}        x_{i+1}      x_{i+2}         x_{i+3}

    For this, we first select a stencil of size :math: `n+1` that contains the
    grid points :math: `x_{i-1/2}` and :math: `x_{i+1/2}` based on the smoothness
    of the field. For this we need to store the store the start grid point of the
    stencil, for more details see the `compute_start_of_stencil` method.

    Once the stencil is selected, we interpolate the field either pointwise
    (for finite difference methods) or cell average (for finite volume methods)
    using polynomial interpolation.

    Parameters
    ----------
    order : int, optional
        The order of the ENO interpolation. Default is 5.
    method : str, optional
        The method of interpolation. Can be either "pointwise" or
        "cell_average". Default is "pointwise".
        "pointwise" for finite difference methods. It will treat the function
        values as pointwise values.
        "cell_average" for finite volume methods. It will treat the function
        values as cell averages and construct a polynomial which cell averages
        are equal to the cell averages of the field at the grid points.

    References
    ----------
    [1] Shu, Chi-Wang. „Essentially Non-Oscillatory and Weighted Essentially
    Non-Oscillatory Schemes for Hyperbolic Conservation Laws“. In Advanced 
    Numerical Approximation of Nonlinear Hyperbolic Equations: Lectures given at
    the 2nd Session of the Centro Internazionale Matematico Estivo (C.I.M.E.)
    Held in Cetraro, Italy, June 23–28, 1997

    """

    name = "ENO Interpolation"

    def __init__(self,
                 order: int = 5,
                 method: Literal["pointwise", "cell_average"] = "pointwise",
                 ) -> None:
        super().__init__()
        self.order = order
        self.required_halo = order
        self.method = method
        self._coeffs = None

    def _on_setup(self) -> None:
        # check that the grid is periodic in all dimensions
        if not all(self.grid.periodic_bounds):
            msg = "ENO interpolation only works on periodic grids"
            raise ValueError(msg)

        if self.method == "pointwise":
            self.compute_polynomial_coefficients_pointwise()
        elif self.method == "cell_average":
            self.compute_polynomial_coefficients_cell_average()
        else:
            msg = f"Unknown method {self.method}, must be 'pointwise' or 'cell_average'"
            raise ValueError(msg)

    def compute_polynomial_coefficients_pointwise(self) -> None:
        r"""
        Polynomial coefficients for the ENO interpolation for pointwise values.

        Description
        -----------
        We consider a stencil of size :math:`n+1` with grid positions
        :math:`x_{i+1/2}`. For a function :math:`f(x)` we have the function values
        at the grid points :math:`x_{i+1/2}`:

        ::

            |    x_{1/2}    |   x_{1+1/2}   |   x_{2+1/2}   |
           x_0             x_1             x_2             x_3
                 f_{1/2}        f_{1+1/2}       f_{2+1/2}

        We search for a polynomial :math:`p(x)` of order :math:`n` such that
        the function value is equal to :math:`f_{i+1/2}` at the grid points:

        .. math::
            p(x_{i+1/2}) = f_{i+1/2}

        Such a polynomial can be constructed using the lagrange interpolation
        polynomial :math:`\ell_i(x)`:

        .. math::
            p(x) = \sum_{i=0}^{n} f_{i+1/2} \ell_i(x)
            \quad \text{with} \quad
            \ell_i(x) = \prod_{j=0, j \neq i}^{n}
            \frac{x - x_{j+1/2}}{x_{i+1/2} - x_{j+1/2}}

        Hence, at the grid points :math:`x_k`, we have:

        .. math::
            p(x_k) = \sum_{i=0}^{n} c_{ki} f_{i+1/2}
            \quad \text{with} \quad
            c_{ki} = \prod_{j=0, j \neq i}^{n}
            \frac{x_k - x_{j+1/2}}{x_{i+1/2} - x_{j+1/2}}

        for a uniform grid with :math:`x_i = i \Delta x`, the coefficients
        :math:`c_{ki}` can be computed as

        .. math::
            c_{ki} = \prod_{j=0, j \neq i}^{n}
            \frac{k - j - 1/2}{i - j}

        """
        stencil_size = self.order + 1

        # we zero out and then accumulate
        coeffs = np.zeros((stencil_size+1, stencil_size))

        def compute_coeff(k: int, i: int, n: int) -> float:
            """Compute the coefficients for the ENO interpolation."""
            total = 1.0
            for j in range(n+1):
                if j == i:
                    continue
                total *= (k - j - 0.5) / (i - j)
            return total

        # sum over each i and k of the coefficients
        for i in range(stencil_size):
            for k in range(stencil_size+1):
                coeffs[k, i] = compute_coeff(k, i, self.order)

        self._coeffs = fr.config.ncp.asarray(coeffs, dtype=fr.config.dtype_real)


    def compute_polynomial_coefficients_cell_average(self) -> None:
        r"""
        Polynomial coefficients for the ENO interpolation for cell averages.

        Description
        -----------
        We follow Shu (1998) to compute reconstruction coefficients. For this,
        we consider a stencil of size :math:`n+1` with grid positions
        :math:`x_i`. For a function :math:`f(x)` cell averages are known at the
        grid points :math:`x_{i+1/2}`:

        ::

            |    x_{1/2}    |   x_{1+1/2}   |   x_{2+1/2}   |
           x_0             x_1             x_2             x_3
                 f_{1/2}        f_{1+1/2}       f_{2+1/2}

        with the cell averages given by

        .. math::
            f_{i+1/2} = \frac{1}{x_{i+1} - x_i} \int_{x_i}^{x_{i+1}} f(x) dx

        We search for a polynomial :math:`p(x)` of order :math:`n-1` such that
        the cell averaged value is equal to :math:`f_{i+1/2}` at the grid points:

        .. math::
            f_{i+1/2} = \frac{1}{x_{i+1} - x_i} \int_{x_i}^{x_{i+1}} p(x) dx

        We now evaluate the integral of :math:`f(x)` at the positions :math:`x_i`:

        .. math::
            F_i = F(x_i) = \int_{x_0}^{x_i} f(x) dx
                = F_0 + \sum_{m=0}^{i-1} f_{m+1/2} (x_{m+1} - x_m)

        The integration constant :math:`F_0` does not matter and can be set to
        zero (:math:`F_0 = 0`). We now define the polynomial :math:`P(x)` using
        the lagrange interpolation polynomial :math:`\ell_i(x)`:

        .. math::
            P(x) = \sum_{i=0}^{n} F_i \ell_i(x)
            \quad \text{with} \quad
            \ell_i(x) = \prod_{j=0, j \neq i}^{n} \frac{x - x_j}{x_i - x_j}

        This can be rewritten as

        .. math::
            \begin{align*}
            P(x) &= \sum_{i=1}^{n} \sum_{m=0}^{i-1} f_{m+1/2} (x_{m+1} - x_m)
            \ell_i(x) \\
            &= \sum_{i=0}^{n-1} f_{i+1/2} \sum_{m=i+1}^{n} (x_{i+1} - x_i)
            \ell_m(x)
            \end{align*}

        By construction, the polynomial given by :math:`p(x) = P'(x)` satisfies

        .. math::
            \frac{1}{x_{i+1} - x_i} \int_{x_i}^{x_{i+1}} p(x) dx
            = \frac{1}{x_{i+1} - x_i} (P(x_{i+1}) - P(x_i))
            = \frac{1}{x_{i+1} - x_i} (F_{i+1} - F_i) = f_{i+1/2}

        which is the desired result. The reconstruction at the positions
        :math:`x_{k}` are then given by

        .. math::
            p(x_k) &= \sum_{i=0}^{n-1} c_{ki} f_{i+1/2}
            \quad \text{with} \quad
            c_{ki} = \sum_{m=i+1}^{n} (x_{i+1} - x_i)
            \left. \frac{d}{dx} \ell_m(x) \right|_{x=x_k}

        The derivative of the lagrange polynomial is given by

        .. math::
            \frac{d}{dx} \ell_m(x) = \sum_{j=0, j \neq m}^{n}
                \frac{1}{x_m - x_j}
                \prod_{r=0, r \neq m, j}^{n} \frac{x - x_r}{x_m - x_r}

        The coefficients :math:`c_{ki}` are then given by

        .. math::
            c_{ki} = \sum_{m=i+1}^{n} \sum_{j=0, j \neq m}^{n}
                \frac{x_{m+1} - x_m}{x_m - x_j}
                \prod_{r=0, r \neq m, j}^{n} \frac{x_k - x_r}{x_m - x_r}

        Or, on a uniform grid with :math:`x_i = i \Delta x`:

        .. math::
            c_{ki} = \sum_{m=i+1}^{n} \sum_{j=0, j \neq m}^{n}
                \frac{1}{m - j}
                \prod_{r=0, r \neq m, j}^{n} \frac{k - r}{m - r}

        """
        n = self.order + 1

        # we zero out and then accumulate
        coeffs = np.zeros((n+1, n))

        def compute_coeff(k: int, i: int, n: int) -> float:
            """Compute the coefficients for the ENO interpolation."""
            total = 0.0
            # sum over m and j
            for m in range(i+1, n+1):
                for j in range(n+1):
                    if j == m:
                        continue

                    # build the inner product for this (m,j)
                    term = 1.0
                    for r in range(n+1):
                        if r in (m, j):
                            continue
                        term *= (k - r) / (m - r)

                    # divide by (m-j) and add to the running total
                    total += term / (m - j)
            return total

        # sum over each i and k of the coefficients
        for i in range(n):
            for k in range(n+1):
                coeffs[k, i] = compute_coeff(k, i, n)

        self._coeffs = fr.config.ncp.asarray(coeffs, dtype=fr.config.dtype_real)

    @fr.utils.jaxjit
    def interpolate(self,  # noqa: D102
                    f: fr.ScalarField,
                    destination: fr.grid.Position) -> fr.ScalarField:
        for axis in range(f.arr.ndim):
            f = self._interpolate_axis(f, axis, destination.positions[axis])
        mask = self.grid.water_mask.get_mask(destination)
        f.arr = f.arr * mask
        return f

    def compute_start_of_stencil(self,
                                 f: fr.ScalarField,
                                 axis: int,
                                 destination: fr.grid.AxisPosition) -> None:
        r"""
        Compute the start of the stencil for the ENO interpolation.

        Description
        -----------
        We start with a stencil of size 2 that contains function values
        :math:`f_i` and :math:`f_{i+1}` at the grid points :math:`x_{i}` and
        :math:`x_{i+1}`, we want to interpolate to the grid point :math:`x_{i+1/2}`:

        ::

            |    x_i    |  x_{i+1}  |
            |    f_i    |  f_{i+1}  |
                        ^
                Interpolate here

        To do this, we want to increase the stencil size, so that it contains
        :math:`n+1` grid points, where :math:`n` is the aimed order of the
        interpolation. The selection of the stencil is based on the smoothness
        of the field, we will later explain how we decide whether to include
        the next or previous grid point. For now, we assume for example that
        our stencil also one grid point to the left and two grid points to
        the right, so that we have a stencil of size 5 (order 4):

        ::

            | x_{i-1} |   x_i   |  x_{i+1}  | x_{i+2}  | x_{i+3} |
                                ^
                        Interpolate here

        This function returns for every grid point :math:`x_{i}` a list of
        indices of the stencil for that grid point, e.g. for the above
        example, the stencil index list for :math:`x_{i}` would be
        [i-1, i, i+1, i+2, i+3]. Furthermore, we return the index of the
        coefficients for the polynomial interpolation, which is for the grid
        point :math:`x_{i}` in the above example 2 (0 would be to the left
        of :math:`x_{i-1}`, 1 would be at left of :math:`x_{i}` and 2 would
        be at the right of :math:`x_{i}`).

        Choosing the stencil
        ~~~~~~~~~~~~~~~~~~~~
        Let's assume we have the following stencil of size 3:

        ::

            Function f:          |  f_{i-1}  |    f_i    |  f_{i+1}  |
            First derivative:             f'_{i-1}     f'_{i}
            Second derivative:                 f''_{i-1}

        As we can see, from this stencil, we can compute the first and second
        derivative of the function. To compute the third derivative, we either
        need to include the previous grid point :math:`f_{i-2}` or the next
        grid point :math:`f_{i+2}`, from which we obtain the two derivatives
        :math:`f'''_{i-2}` and :math:`f'''_{i}`. We can now compare the
        absolute values of the two derivatives and choose the one with the
        smaller absolute value, and include the corresponding grid point in the
        stencil, so for example if :math:`|f'''_{i-2}| < |f'''_{i}|`, we include
        the grid point :math:`x_{i-2}` in the stencil and repeat the process
        until we have a stencil of size :math:`n+1`.

        We define the derivative of the function as:

        .. math::
            f^{(n)}_{i} = f^{(n-1)}_{i+1} - f^{(n-1)}_{i}

        As we work with uniform grids, we do not need to divide by the grid size
        :math:`\Delta x`, as it is the same for all grid points.

        Returns
        -------
        list[tuple[np.ndarray[int]]]
            A list of indices of the stencil for every grid point.
        np.ndarray[int]
            The index of the coefficients for the polynomial interpolation.

        """
        stencil_size = self.order + 1
        # Get the full field indices and remove halos
        full_indices = fr.config.ncp.indices(f.arr.shape)
        unpadded_indices = [self.grid.unpad(idx) for idx in full_indices]

        # Split into segments: before, along, and after the split axis
        indices_before = unpadded_indices[:axis]
        index_along_axis = unpadded_indices[axis]
        indices_after = unpadded_indices[axis + 1:]

        # Create an array to store the offset to the start of the stencil
        offsets = fr.config.ncp.zeros_like(index_along_axis)

        # if we map from face to centers, we have to reduce the offsets by one
        if destination == fr.grid.AxisPosition.CENTER:
            offsets -= 1

        # Prepare slices for the differences
        next_sl = [slice(None)] * f.arr.ndim
        next_sl[axis] = slice(1, None)
        prev_sl = [slice(None)] * f.arr.ndim
        prev_sl[axis] = slice(None, -1)
        next_sl = tuple(next_sl)
        prev_sl = tuple(prev_sl)

        # Compute the derivative of the field
        derivative = f.arr[next_sl] - f.arr[prev_sl]

        for _ in range(stencil_size - 2):
            # compute the next order derivative
            derivative = derivative[next_sl] - derivative[prev_sl]

            # get the indices of the next and previous cells
            next_indices = index_along_axis + offsets
            prev_indices = next_indices - 1

            # get the absolute value of the derivative
            abs_derivative = fr.config.ncp.abs(derivative)
            val_next = abs_derivative[(*indices_before, next_indices, *indices_after)]
            val_prev = abs_derivative[(*indices_before, prev_indices, *indices_after)]

            # if the previous derivative is smaller than the next one,
            # we reduce the offset by one
            offsets -= (val_prev < val_next)

        # TODO(Silvano): We need to check for the boundaries here
        # The stencil should not go out of bounds. We need to store arrays for
        # the maximum and minimum indices of the stencil and check if the
        # offsets are within the bounds of the grid.

        all_indices = [(*indices_before, index_along_axis+i+offsets, *indices_after)
                       for i in range(stencil_size)]

        # compute the index of coefficients
        match destination:
            case fr.grid.AxisPosition.CENTER:
                coeff_indices = -offsets
            case fr.grid.AxisPosition.FACE:
                coeff_indices = 1 - offsets

        return all_indices, coeff_indices

    @partial(fr.utils.jaxjit, static_argnames=("axis", "destination"))
    def _interpolate_axis(self,
                          f: fr.ScalarField,
                          axis: int,
                          destination: fr.grid.AxisPosition) -> fr.ScalarField:
        if not f.topo[axis]:
            # no interpolation when the field has no extend along the axis
            return f

        if f.position[axis] == destination:
            # no interpolation needed
            return f

        res = fr.ScalarField(mset=f.mset, mdata=deepcopy(f.mdata))

        all_indices, coeff_indices = self.compute_start_of_stencil(f, axis, destination)

        interp_arr = sum(self._coeffs[:,i][coeff_indices] * f.arr[s]
                         for i, s in enumerate(all_indices))

        res.arr = self.grid.pad(interp_arr)
        res.position = f.position.shift(axis)

        return res
