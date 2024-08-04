r"""
Operators for spectral analysis of discrete Cartesian grids.

Averaging operators
-------------------
The linear interpolation operator on a field :math:`u` is defined as:

.. math::
    \overline{u}^{x\pm} = \frac{u(x \pm \Delta x) + u(x)}{2}

where :math:`\pm` denotes the forward (+) or backward (-) linear 
interpolation. A fourier transform yields the discrete spectral operator:

.. math::
    \overline{u}^{x\pm} \rightarrow \frac{e^{\pm ik_x \Delta x} + 1}{2}u =
        \hat{1}_x^\pm u
    
Hence, the discrete spectral operator `one_hat` is given by:

.. math::
    \hat{1}_x^\pm = \frac{e^{\pm ik_x \Delta x} + 1}{2}

In the continuous case, e.g. :math:`\Delta x \rightarrow 0`, the limit yields

.. math::
    \lim_{\Delta x \rightarrow 0} \hat{1}_x^\pm = 1

When applying a forward and consecutive backward linear interpolation, (
or vice versa), the discrete spectral operator `one_hat_squared` is given by:

.. math::
    \hat{1}_x^2 = \hat{1}_x^+ \hat{1}_x^- =
        \frac{1 + \cos(k_x \Delta x)}{2}

Differentiation operators
-------------------------
The finite difference operator is defined as:

.. math::
    \delta_x^\pm u = \pm \frac{u(x \pm \Delta x) - u(x)}{\Delta x}

where :math:`\pm` denotes the forward (+) or backward (-) finite difference.
A fourier transform yields the discrete spectral operator:

.. math::
    \delta_x^\pm u \rightarrow \pm \frac{e^{\pm ik_x \Delta x} - 1}{\Delta x}u =
        i \hat{k}_x^\pm u
    
Hence, the discrete spectral operator `k_hat` is given by:

.. math::
    \hat{k}_x^\pm = \mp i \frac{e^{\pm ik_x \Delta x} - 1}{\Delta x}

For the continuous case, e.g. :math:`\Delta x \rightarrow 0`, the limit yields

.. math::
    \lim_{\Delta x \rightarrow 0} \hat{k}_x^\pm = k_x

When applying a forward and consecutive backward finite difference, (or vice
versa), the discrete spectral operator `k_hat_squared` is given by:

.. math::
    \hat{k}_x^2 = \hat{k}_x^+ \hat{k}_x^- =
        2 \frac{1 - \cos(k_x \Delta x)}{\Delta x^2}
"""
import fridom.framework as fr
from numpy import ndarray

# ================================================================
#  Discrete spectral operators (one-hat-plus etc.)
# ================================================================
def one_hat(kx: ndarray, dx: float, sign: int, use_discrete: bool = True) -> ndarray:
    r"""
    Spectral operator for the forward linear interpolation.
    
    Description
    -----------
    Computes the spectral operator :math:`\hat{1}_x^\pm` that arises from the
    discrete forward (+1) or backward (-1) linear interpolation:

    .. math::
        \hat{1}_x^\pm = \frac{e^{\pm ik_x \Delta x} + 1}{2}
    
    Parameters
    ----------
    `kx` : `ndarray`
        The wavenumber
    `dx` : `float`
        The grid spacing
    `sign` : `int`
        The sign of the operator (+1 for forward, -1 for backward)
    `use_discrete` : `bool` (default: True)
        If True, the discrete operator is returned. Otherwise, the continuous
        operator is returned which is 1 for forward and backward interpolation.
    
    Returns
    -------
    `ndarray`
        The spectral operator.
    """
    if use_discrete:
        return (1 + fr.config.ncp.exp(sign * 1j * kx * dx)) / 2
    else:
        return 1

def one_hat_squared(kx: ndarray, dx: float, use_discrete: bool = True) -> ndarray:
    r"""
    Discrete spectral operator of forward - backward linear interpolation.
    
    Description
    -----------
    Computes the spectral operator :math:`\hat{1}_x^2` that arises from the
    discrete forward - backward linear interpolation:

    .. math::
        \hat{1}_x^2 = \hat{1}_x^+ \hat{1}_x^- =
            \frac{1 + \cos(k_x \Delta x)}{2}

    Parameters
    ----------
    `kx` : `ndarray`
        The wavenumber
    `dx` : `float`
        The grid spacing
    `use_discrete` : `bool`
        If True, the discrete operator is returned. Otherwise, the continuous
        operator is returned which is always 1.
    
    Returns
    -------
    `ndarray`
        The spectral operator.
    """
    if use_discrete:
        return (1 + fr.config.ncp.cos(kx*dx)) / 2
    else:
        return 1

def k_hat(kx: ndarray, dx: float, sign: int, use_discrete: bool = True) -> ndarray:
    r"""
    Spectral operator for the forward finite difference.
    
    Description
    -----------
    Computes the spectral operator :math:`\hat{k}_x^\pm` that arises from the
    discrete forward (+1) or backward (-1) finite difference.

    .. math::
        \hat{k}_x^\pm = \mp i \frac{e^{\pm ik_x \Delta x} - 1}{\Delta x}
    
    Parameters
    ----------
    `kx` : `ndarray`
        The wavenumber
    `dx` : `float`
        The grid spacing
    `sign` : `int`
        The sign of the operator (+1 for forward, -1 for backward)
    `use_discrete` : `bool`
        If True, the discrete operator is returned. Otherwise, the continuous
        operator is returned which is always kx.
    
    Returns
    -------
    `ndarray`
        The spectral operator.
    """
    if use_discrete:
        return sign * 1j * (1 - fr.config.ncp.exp(sign * 1j * kx * dx)) / dx
    else:
        return kx

def k_hat_squared(kx: ndarray, dx: float, use_discrete: bool = True) -> ndarray:
    r"""
    Spectral operator of forward - backward finite difference.
    
    Description
    -----------
    Computes the spectral operator :math:`\hat{k}_x^2` that arises from 
    forward - backward finite difference:

    .. math::
        \hat{k}_x^2 = \hat{k}_x^+ \hat{k}_x^- =
            2 \frac{1 - \cos(k_x \Delta x)}{\Delta x^2}

    Parameters
    ----------
    `kx` : `ndarray`
        The wavenumber
    `dx` : `float`
        The grid spacing
    `use_discrete` : `bool`
        If True, the discrete operator is returned. Otherwise, the continuous
        operator is returned which is always kx**2.
    
    Returns
    -------
    `ndarray`
        The spectral operator.
    """
    if use_discrete:
        return 2 * (1 - fr.config.ncp.cos(kx*dx)) / dx**2
    else:
        return kx**2

# ================================================================
#  Utility functions
# ================================================================

def set_nyquist_to_zero(z: fr.StateBase) -> fr.StateBase:
    r"""
    Set the nyquist frequency to zero in the spectral domain.
    
    Parameters
    ----------
    `z` : `State`
        The state which nyquist frequency should be set to zero.
    
    Returns
    -------
    `State`
        The state with the nyquist frequency set to zero.
    """
    ncp = fr.config.ncp
    grid = z.grid
    # Set nyquist frequency to zero
    for axis in range(grid.n_dims):
        # We only need to consider periodic axes since nonperiodic axes
        # work differently with cosine and sine transforms instead of
        # Fourier transforms
        if not grid.periodic_bounds[axis]:
            continue
        # We only need to consider axes with an even number of grid points
        nx = grid.N[axis]
        if nx % 2 != 0:
            continue
        # Find the position of the nyquist frequency in the local domain
        k_nyquist = grid.k_global[axis][nx//2]
        nyquist = (grid.K[axis] == k_nyquist)
        # Set the nyquist frequency to zero
        for field in z.fields.values():
            field.arr = ncp.where(nyquist, 0, field.arr)
    return z