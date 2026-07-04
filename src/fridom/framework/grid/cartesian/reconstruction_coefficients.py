"""Various functions to compute reconstruction coefficients."""
from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
import numpy as np

import fridom.framework as fr


def compute_polynomial_coefficients_cell_average(
            stencil_size: int) -> jnp.ndarray:
    r"""
    Polynomial coefficients for cell averages.

    Parameters
    ----------
    stencil_size : int
        The size of the stencil.

    Returns
    -------
    np.ndarray
        The polynomial coefficients for the interpolation.
        The first dimension corresponds to the position where we want to
        reconstruct the field, and the second dimension corresponds to the
        position of the cell average in the stencil.

    Description
    -----------
    We consider a stencil of size :math:`n+1` with grid positions :math:`x_i`.
    ::

        |    x_{1/2}    |   x_{1+1/2}   |   x_{2+1/2}   |
        x_0             x_1             x_2             x_3
             f_{1/2}        f_{1+1/2}       f_{2+1/2}

    function values are given at the grid points :math:`x_{i+1/2}`. The
    reconstruction at the positions :math:`x_{k}` are then given by

    .. math::
        p(x_k) = \sum_{i=0}^{n-1} c_{ki} f_{i+1/2}


    We follow Shu (1998) to compute reconstruction coefficients. The
    function values correspond to cell, given by

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
        c_{ki} &= \sum_{m=i+1}^{n} (x_{i+1} - x_i)
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
    # we zero out and then accumulate
    coeffs = np.zeros((stencil_size+1, stencil_size))

    def compute_coeff(k: int, i: int, n: int) -> float:
        """Compute the coefficients for the interpolation."""
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
    for i in range(stencil_size):
        for k in range(stencil_size+1):
            coeffs[k, i] = compute_coeff(k, i, stencil_size)

    return jnp.asarray(coeffs, dtype=fr.utils.dtype_real())


def compute_polynomial_coefficients_pointwise(stencil_size: int) -> np.ndarray:
    r"""
    Polynomial coefficients for the interpolation for pointwise values.

    Parameters
    ----------
    stencil_size : int
        The size of the stencil.

    Returns
    -------
    np.ndarray
        The polynomial coefficients for the interpolation.

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
    # we zero out and then accumulate
    coeffs = np.zeros((stencil_size+1, stencil_size))

    def compute_coeff(k: int, i: int, n: int) -> float:
        """Compute the coefficients for the interpolation."""
        total = 1.0
        for j in range(n):
            if j == i:
                continue
            total *= (k - j - 0.5) / (i - j)
        return total

    # sum over each i and k of the coefficients
    for i in range(stencil_size):
        for k in range(stencil_size+1):
            coeffs[k, i] = compute_coeff(k, i, stencil_size)

    return jnp.asarray(coeffs, dtype=fr.utils.dtype_real())


def compute_polynomial_coefficients(
        stencil_size: int,
        method: Literal["pointwise", "cell_average"] = "cell_average",
        ) -> jnp.ndarray:
    """
    Compute the polynomial coefficients.

    Parameters
    ----------
    stencil_size : int
        The size of the stencil.
    method : Literal["pointwise", "cell_average"] (default is "cell_average")
        The method for computing the polynomial coefficients.
        "pointwise" computes coefficients for pointwise values.
        "cell_average" computes coefficients for cell averages.

    Returns
    -------
    np.ndarray
        The polynomial coefficients for the interpolation.

    """
    if method == "pointwise":
        coeffs = compute_polynomial_coefficients_pointwise(
            stencil_size=stencil_size)
    elif method == "cell_average":
        cell_average = (
            fr.grid.cartesian.compute_polynomial_coefficients_cell_average)
        coeffs = cell_average(stencil_size=stencil_size)
    else:
        msg = (f"Invalid method {method}. "
               "Only 'pointwise' and 'cell_average' are supported.")
        raise ValueError(msg)

    return coeffs
