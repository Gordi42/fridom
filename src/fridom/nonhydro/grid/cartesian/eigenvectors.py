r"""
Eigenvalues and eigenvectors of the system matrix of the nonhydrostatic model.

Continuous Case
===============

The System Matrix
-----------------
We start from the scaled linearized nonhydrostatic equations in spectral space:

.. math::
    \partial_t u = f v - i k_x p
.. math::
    \partial_t v = -f u - i k_y p
.. math::
    \partial_t w = \delta^{-2} b - \delta^{-2} i k_z p
.. math::
    \partial_t b = -wN^2

and the diagnostic pressure equation obtained by taking the divergence of the
momentum equations:


.. math::
    0 = if (k_x v - k_y u)
        + \delta^{-2} i k_z b + ( k_x^2 + k_y^2 + \delta^{-2} k_z^2) p

Solving the diagnostic pressure equation for the pressure and substituting it
back into the momentum equations, we obtain the following system of equations:

.. math::
    \partial_t \boldsymbol{z} = -i \mathbf{A} \cdot \boldsymbol{z}

with

.. math::
    \boldsymbol{z} =
    \begin{pmatrix}
        u \\ v \\ w \\ b
    \end{pmatrix}
    \quad, \quad
    \mathbf{A} = \frac{-1}{\delta^2 k^2}
    \begin{pmatrix}
     -if\delta^2k_x k_y & -if\left( \delta^2k_y^2 + k_z^2 \right)
        & 0 & i k_x k_z \\
     if \left( \delta^2 k_x^2 + k_z^2 \right) & if\delta^2k_x k_y
        & 0 & i k_y k_z \\
     -i f k_y k_z & i f k_x k_z & 0 & -i k_h^2 \\
     0 & 0 & iN^2\delta^2 k^2 & 0
    \end{pmatrix}

with the wavenumbers:

.. math::
    k^2 = k_h^2 + \delta^{-2} k_z^2
    \quad \text{and} \quad
    k_h^2 = k_x^2 + k_y^2


Eigenvalues
-----------
The system matrix has three eigenvalues. One eigenvalue is zero, corresponding
to the geostrophic mode:

.. math::
    \omega^0 = 0

The other two eigenvalues correspond to the inertial-gravity wave modes:

.. math::
    \omega^\pm = \pm \sqrt{\frac{f^2 k_z^2 + N^2 k_h^2}{\delta^2 k^2}}

Eigenvectors
------------
For the eigenvectors we have to separately consider the case of purely
vertical, e.g. :math:`k_x = k_y = 0`, and the general case of nonzero
horizontal wavenumbers.
For the purely vertical case, the eigenvectors are:

.. math::
    \boldsymbol{q^0} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}
    \quad, \quad
    \boldsymbol{q^\pm} = \begin{pmatrix} \mp i \\ 1 \\ 0 \\ 0 \end{pmatrix}

For the general case of nonzero horizontal wavenumbers, the eigenvectors are:

.. math::

    \boldsymbol{q^0} = \begin{pmatrix} -k_y \\ k_x \\ 0 \\ fk_z \end{pmatrix}
    \quad, \quad
    \boldsymbol{q^\pm} = \begin{pmatrix}
        k_z ( -i \omega^\pm k_x + f k_y) \\
        k_z ( -i \omega^\pm k_y - f k_x) \\
        i\omega^\pm k_h^2 \\
        N^2 k_h^2
    \end{pmatrix}

Projection Vectors
------------------
Projection vectors should satisfy

.. math::
    {\boldsymbol{p^s}}^* \cdot \boldsymbol{q^{s'}} = \delta_{s,s'}

where the star denotes the hermitian transposed. Add divergent vector
:math:`\boldsymbol{q^d} = \begin{pmatrix} k_x & k_y & k_z & 0 \end{pmatrix}`
such that
:math:`(\boldsymbol{q^0}, \boldsymbol{q^+}, \boldsymbol{q^-},
\boldsymbol{q^d})`
form a basis.

For the purely vertical case, the projection vectors are identical to the
eigenvectors: :math:`\boldsymbol{p^s} = \boldsymbol{q^s}`. For the general case
of nonzero horizontal wavenumbers, the projection vectors are:

.. math::
    \boldsymbol{p^0} =
    \begin{pmatrix} -N^2 k_y \\ N^2 k_x \\ 0 \\ f k_z \end{pmatrix}
    \quad, \quad
    \boldsymbol{p^\pm} = \begin{pmatrix}
        k_z (-i \omega^\pm k_x + f \gamma k_y) \\
        k_z (-i \omega^\pm k_y - f \gamma k_x) \\
        i\omega^\pm k_h^2 \\
        \gamma k_h^2
    \end{pmatrix}

with

.. math::
    \gamma = \frac{k_h^2 + k_z^2}{\delta^2 k^2}

The projection vectors are normalized such that
:math:`{\boldsymbol{p^s}}^* \cdot \boldsymbol{p^s} = 1`.


Discrete Case
=============

Lets define the forward and backward finite difference operators as:

.. math::
    \delta_x^+ u = \frac{u(x + \Delta x) - u(x)}{\Delta x}
    \quad, \quad
    \delta_x^- u = \frac{u(x) - u(x - \Delta x)}{\Delta x}

A fourier transform yields the discrete spectral operators:

.. math::
    \delta_x^+ u \rightarrow \frac{e^{ik_x \Delta x} - 1}{\Delta x} =
        i \hat{k}_x^+ u
    \quad, \quad
    \delta_x^- u \rightarrow \frac{1 - e^{-ik_x \Delta x}}{\Delta x} =
        i \hat{k}_x^- u

Similarly, we define the forward and backward linear interpolation
operators as:

.. math::
    \overline{u}^{x+} = \frac{u(x + \Delta x) + u(x)}{2}
    \quad, \quad
    \overline{u}^{x-} = \frac{u(x) + u(x - \Delta x)}{2}

A fourier transform yields the discrete spectral operators:

.. math::
    \overline{u}^{x+} \rightarrow \frac{e^{ik_x \Delta x} + 1}{2} =
        \hat{1}_x^+ u
    \quad, \quad
    \overline{u}^{x-} \rightarrow \frac{1 + e^{-ik_x \Delta x}}{2} =
        \hat{1}_x^- u

Using these discretization operators, the discrete linear system of equations
in spectral space can be written as:

.. math::
    \partial_t u = f \hat{1}_x^+ \hat{1}_y^- v - i \hat{k}_x^+ p
.. math::
    \partial_t v = -f \hat{1}_x^- \hat{1}_y^+ u - i \hat{k}_y^+ p
.. math::
    \partial_t w = \delta^{-2} \hat{1}_z^+ b - \delta^{-2} i \hat{k}_z^+ p
.. math::
    \partial_t b = - \hat{1}_z^- w N^2

and the diagnostic pressure equation:

.. math::
    0 = i (\hat{1}_x^+ \hat{1}_y^- \hat{k}_x^- v
        - \hat{1}_x^- \hat{1}_y^+ \hat{k}_y^- u)
        + \delta^{-2} i \hat{1}_z^+ \hat{k}_z^- b
        + (\hat{k}_x^2 + \hat{k}_y^2 + \delta^{-2} \hat{k}_z^2) p

System Matrix
-------------
Following the same procedure as in the continuous case, we obtain the discrete
system matrix:

.. math::
    \mathbf{A} = \frac{1}{\hat{k}^2} \begin{pmatrix}
        -i \delta^2 \hat{1}_x^- \hat{1}_y^+ \hat{k}_x^+ \hat{k}_y^- f &
        -i f \hat{1}_x^+ \hat{1}_y^-
            \left( \delta^2 \hat{k}_y^2 + \hat{k}_z^2 \right ) &
        0 &
        i \hat{1}_z^+ \hat{k}_x^+ \hat{k}_z^- \\
        i f \hat{1}_x^- \hat{1}_y^+
            \left( \delta^2 \hat{k}_x^2 + \hat{k}_z^2 \right) &
        i \delta^2 \hat{1}_x^+ \hat{1}_y^- \hat{k}_x^- \hat{k}_y^+ f &
        0 &
        i \hat{1}_z^+ \hat{k}_y^+ \hat{k}_z^- \\
        -i \hat{1}_x^- \hat{1}_y^+ \hat{k}_y^- \hat{k}_z^+ f &
        i \hat{1}_x^+ \hat{1}_y^- \hat{k}_x^- \hat{k}_z^+ &
        0 &
        -i \hat{1}_z^+ \hat{k}_h^2 \\
        0 &
        0 &
        i N^2 \hat{1}_x^- \hat{k}^2 &
        0
    \end{pmatrix}

with

.. math::
    \hat{k}^2 = \hat{k}_h^2 + \delta^{-2} \hat{k}_z^2
    \quad \text{and} \quad
    \hat{k}_h^2 = \hat{k}_x^2 + \hat{k}_y^2
    \quad \text{and} \quad
    \hat{k}_x^2 = \hat{k}_x^+ \hat{k}_x^-

Eigenvalues
-----------
.. math::
    \omega^0 = 0
    \quad \text{and} \quad
    \omega^\pm =
        \sqrt{\frac{\hat{1}_x^2 \hat{1}_y^2 f^2 \hat{k}_z^2 +
        \hat{1}_z^2 N^2 \hat{k}_h^2}{\hat{k}^2}}

with

.. math::
    \hat{1}_x^2 = \hat{1}_x^+ \hat{1}_x^-

Eigenvectors
------------
For the purely vertical case, the discrete eigenvectors are identical to the
continuous eigenvectors. For the general case of nonzero horizontal
wavenumbers, the discrete eigenvectors are:

.. math::
    \boldsymbol{q^0} = \begin{pmatrix}
        - \hat{1}_x^+ \hat{1}_y^- \hat{1}_z^+ \hat{k}_y^+ \\
        \hat{1}_x^- \hat{1}_y^+ \hat{1}_z^+ \hat{k}_x^+ \\
        0 \\
        \hat{1}_x^2 \hat{1}_y^2 f \hat{k}_z^+
    \end{pmatrix}
    \quad \text{and} \quad
    \boldsymbol{q}^\pm = \begin{pmatrix}
        \hat{k}_z^- ( - i \omega^\pm \hat{k}_x^+
                      + \hat{1}_x^+ \hat{1}_y^- f \hat{k}_y^+) \\
        \hat{k}_z^- ( - i \omega^\pm \hat{k}_y^+
                      - \hat{1}_x^- \hat{1}_y^+ f \hat{k}_x^+) \\
        i \omega^\pm \hat{k}_h^2 \\
        \hat{1}_z^- N^2 \hat{k}_h^2
    \end{pmatrix}

The divergent vector is given by
:math:`\boldsymbol{q^d} = (\hat{k}_x^+, \hat{k}_y^+, \hat{k}_z^+, 0)`.
All eigenvectors are orthogonal to the divergent vector. Hence no eigenvector
project onto divergent velocity fields.

Projection Vectors
------------------
Similar to the continuous case, the projection vectors of the purely vertical
case are identical to the eigenvectors. Hence, we only show the projection
vectors for the general case of nonzero horizontal wavenumbers:

.. math::
    \boldsymbol{p^0} =
    \begin{pmatrix}
        - \hat{1}_x^+ \hat{1}_y^- \hat{1}_z^+ N^2 \hat{k}_y^+ \\
        \hat{1}_x^- \hat{1}_y^+ \hat{1}_z^+ N^2 \hat{k}_x^+ \\
        0 \\
        \hat{1}_x^2 \hat{1}_y^2 f \hat{k}_z^+
    \end{pmatrix} ~~, \quad \text{and} \quad
    \boldsymbol{p^\pm} = \begin{pmatrix}
        \hat{k}_z^- ( -i \omega^\pm \hat{k}_x^+
                      + \hat{1}_x^+ \hat{1}_y^- f \hat{\gamma} \hat{k}_y^+) \\
        \hat{k}_z^- ( -i \omega^\pm \hat{k}_y^+
                      - \hat{1}_x^- \hat{1}_y^+ f \hat{\gamma} \hat{k}_x^+) \\
        i \omega^\pm \hat{k}_h^2 \\
        \hat{1}_z^- \hat{\gamma} \hat{k}_h^2
    \end{pmatrix}

with

.. math::
    \hat{\gamma} = \frac{\hat{k}_h^2 + \hat{k}_z^2}{\delta^2 \hat{k}^2}
"""
from __future__ import annotations

from functools import partial

import numpy as np
from numpy import ndarray

import fridom.framework as fr
import fridom.nonhydro as nh

ncp = fr.config.ncp
dso = fr.grid.cartesian.discrete_spectral_operators

def _check_if_spectral_analysis_is_possible(mset: nh.ModelSettings) -> None:
    r"""
    Check if the spectral analysis is possible.

    Parameters
    ----------
    mset : nh.ModelSettings
        The model settings.

    Raises
    ------
    ValueError
        If spectral analysis is not possible.

    """
    # check if the coriolis parameter is constant
    if not fr.utils.array_is_constant(mset.f_coriolis.arr):
        msg = "The coriolis frequency is varying."
        raise ValueError(msg)
    # TODO (Silvano): check if N^2 is constant
    # check if the coriolis frequency and the stratification are zero
    if mset.f0 == 0 and mset.stratification_n2 == 0:
        msg = "The coriolis frequency and the stratification are zero."
        raise ValueError(msg)

def _check_for_horizontal_periodic_boundaries(mset: nh.ModelSettings) -> None:
    r"""
    Check if the periodic boundary in horizontal direction is set.

    Parameters
    ----------
    mset : nh.ModelSettings
        The model settings.

    Raises
    ------
    ValueError
        If the grid is nonperiodic in horizontal direction.

    """
    bc = mset.grid.periodic_bounds
    horizontal_periodic = bc[0] and bc[1]
    if not horizontal_periodic:
        msg = "The grid is nonperiodic in horizontal direction."
        raise ValueError(msg)

def _check_use_discrete_argument(
        dx: tuple[float] | None,
        use_discrete: bool,
) -> None:
    r"""
    Check if the use_discrete argument is used correctly.

    Parameters
    ----------
    dx : tuple[float] | None
        The grid spacing. Required if `use_discrete` is True.
    use_discrete : bool
        If True, the discrete eigenvalues are returned. Otherwise, the
        continuous eigenvalues are returned.

    Raises
    ------
    ValueError
        If `use_discrete` is True and `dx` is None.

    """
    if use_discrete and dx is None:
        msg = "The grid spacing `dx` is required when `use_discrete` is True."
        raise ValueError(msg)

# ================================================================
#  The eigenvalues
# ================================================================
def omega(s: int,
          f0: float,
          stratification_n2: float,
          dsqr: float,
          k: tuple[float] | tuple[ndarray],
          dx: tuple[float] | None = None,
          use_discrete: bool = False,
          ) -> ndarray:
    r"""
    Return eigenvalues of the System matrix.

    Parameters
    ----------
    s : int
        The mode of the eigenvector. (0, 1, -1)
    k : tuple[float] | tuple[ndarray]
        The wavenumbers.
    f0 : float
        The constant Coriolis parameter.
    stratification_n2 : float
        The constant stratification.
    dsqr : float
        The square of the aspect ratio.
    dx : tuple[float], optional
        The grid spacing. Required if `use_discrete` is True.
    use_discrete : bool (default: False)
        If True, the discrete eigenvalues are returned. Otherwise, the
        continuous eigenvalues are returned.

    """
    _check_use_discrete_argument(dx, use_discrete)

    # shorthand notation
    f2 = f0**2
    n_squared = stratification_n2
    # cast k to ndarray
    kx, ky, kz = tuple(ncp.asarray(k) for k in k)
    dx, dy, dz = dx or (None, None, None)

    # eigenvector of geostrophic mode is zero
    if s == 0:
        return ncp.zeros_like(kx)
    if s == "d":
        msg = "Divergent mode does not have eigenvalues."
        raise ValueError(msg)

    # Define the spectral operators
    def ohpm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat_squared(k, d, use_discrete)
    def khpm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.k_hat_squared(k, d, use_discrete)

    kh2 = khpm(kx, dx) + khpm(ky, dy)
    with np.errstate(divide="ignore"):
        coriolis_part = ohpm(kx,dx) * ohpm(ky,dy) * f2 * khpm(kz,dz)
        buoyancy_part = ohpm(kz,dz) * n_squared * kh2
        denominator = dsqr * kh2 + khpm(kz,dz)

    om = ncp.sqrt((coriolis_part + buoyancy_part) / denominator)

    # set the result to zero where the denominator is zero
    nonzero = (kx**2 + ky**2 + kz**2 > 0)
    om = ncp.where(nonzero, om, 0)
    return s * om

# ================================================================
#  The q eigenvectors
# ================================================================

def _vec_q_geostrophic(
        f0: float,
        k: tuple[float] | tuple[ndarray],
        dx: tuple[float] | None = None,
        use_discrete: bool = True,
) -> tuple[ndarray]:
    """Return geostrophic q eigenvector."""
    _check_use_discrete_argument(dx, use_discrete)

    kx, ky, kz = tuple(ncp.asarray(k) for k in k)
    dx, dy, dz = dx or (None, None, None)

    # Define the spectral operators
    def ohp(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat(k, d, +1, use_discrete)
    def ohm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat(k, d, -1, use_discrete)
    def khp(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.k_hat(k, d, +1, use_discrete)
    def ohpm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat_squared(k, d, use_discrete)

    # We first consider the case of nonzero horizontal wavenumbers
    u = -ohp(kx,dx)*ohm(ky,dy)*ohp(kz,dz)*khp(ky,dy)
    v =  ohm(kx,dx)*ohp(ky,dy)*ohp(kz,dz)*khp(kx,dx)
    w = ncp.zeros_like(kx)
    b = ohpm(kx,dx)*ohpm(ky,dy)*f0*khp(kz,dz)

    # Now we consider the purely vertical case
    # (ov -> only vertical)
    u_ov = 0
    v_ov = 0
    w_ov = 0
    b_ov = 1

    # Mask to separate inertial modes from inertia-gravity modes
    nonzero_horizontal = (kx**2 + ky**2 != 0)  # nonzero horizontal wavenumbers

    u = ncp.where(nonzero_horizontal, u, u_ov)
    v = ncp.where(nonzero_horizontal, v, v_ov)
    w = ncp.where(nonzero_horizontal, w, w_ov)
    b = ncp.where(nonzero_horizontal, b, b_ov)

    return u, v, w, b

def _vec_q_divergent(
        k: tuple[float] | tuple[ndarray],
        dx: tuple[float] | None = None,
        use_discrete: bool = True,
    ) -> tuple[ndarray]:
    """Return divergent q eigenvector."""
    _check_use_discrete_argument(dx, use_discrete)

    kx, ky, kz = tuple(ncp.asarray(k) for k in k)
    dx, dy, dz = dx or (None, None, None)

    # Define the spectral operators
    def khp(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.k_hat(k, d, +1, use_discrete)

    # We first consider the case of nonzero horizontal wavenumbers
    u = khp(kx,dx)
    v = khp(ky,dy)
    w = khp(kz,dz)
    b = ncp.zeros_like(kx)

    # Now we consider the purely vertical case
    # (ov -> only vertical)
    u_ov = khp(kx,dx) # should be zero
    v_ov = khp(ky,dy) # should be zero
    w_ov = khp(kz,dz)
    b_ov = 0

    # Mask to separate inertial modes from inertia-gravity modes
    nonzero_horizontal = (kx**2 + ky**2 != 0)  # nonzero horizontal wavenumbers

    u = ncp.where(nonzero_horizontal, u, u_ov)
    v = ncp.where(nonzero_horizontal, v, v_ov)
    w = ncp.where(nonzero_horizontal, w, w_ov)
    b = ncp.where(nonzero_horizontal, b, b_ov)

    return u, v, w, b

def _vec_q_igw(
        s: int,
        f0: float,
        n_squared: float,
        dsqr: float,
        k: tuple[float] | tuple[ndarray],
        dx: tuple[float] | None = None,
        use_discrete: bool = True,
    ) -> tuple[ndarray]:
    """Return inertial-gravity wave q eigenvectors."""
    _check_use_discrete_argument(dx, use_discrete)

    kx, ky, kz = tuple(ncp.asarray(k) for k in k)
    dx, dy, dz = dx or (None, None, None)

    # Define the spectral operators
    def ohp(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat(k, d, +1, use_discrete)
    def ohm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat(k, d, -1, use_discrete)
    def khp(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.k_hat(k, d, +1, use_discrete)
    def khm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.k_hat(k, d, -1, use_discrete)
    def khpm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.k_hat_squared(k, d, use_discrete)

    # Horizontal wavenumber squared
    kh2 = khpm(kx, dx) + khpm(ky, dy)

    # calculate eigenvalue
    om = omega(s=s, f0=f0, stratification_n2=n_squared, dsqr=dsqr,
               k=(kx, ky, kz), dx=(dx, dy, dz), use_discrete=use_discrete)

    # We first consider the case of nonzero horizontal wavenumbers
    u = (khm(kz,dz) * (-1j*om*khp(kx,dx) +
            ohp(kx,dx)*ohm(ky,dy)*f0*khp(ky,dy)))
    v = (khm(kz,dz) * (-1j*om*khp(ky,dy) -
            ohm(kx,dx)*ohp(ky,dy)*f0*khp(kx,dx)))
    w = 1j * om * kh2
    b = ohm(kz,dz) * n_squared * kh2

    # Now we consider the purely vertical case
    # (ov -> only vertical)
    u_ov = -s * 1j
    v_ov = 1
    w_ov = 0
    b_ov = 0

    # Mask to separate inertial modes from inertia-gravity modes
    nonzero_horizontal = (kx**2 + ky**2 != 0)  # nonzero horizontal wavenumbers

    u = ncp.where(nonzero_horizontal, u, u_ov)
    v = ncp.where(nonzero_horizontal, v, v_ov)
    w = ncp.where(nonzero_horizontal, w, w_ov)
    b = ncp.where(nonzero_horizontal, b, b_ov)

    return u, v, w, b

@partial(fr.utils.jaxjit, static_argnames=["s", "use_discrete"])
def _vec_q(
        s: int | str,
        f0: float,
        n_squared: float,
        dsqr: float,
        k: tuple[float] | tuple[ndarray],
        dx: tuple[float] | None = None,
        use_discrete: bool = True,
) -> tuple[ndarray]:
    if s == 0:
        return _vec_q_geostrophic(f0, k, dx, use_discrete)
    if s == "d":
        return _vec_q_divergent(k, dx, use_discrete)
    if s in (1, -1):
        return _vec_q_igw(s, f0, n_squared, dsqr, k, dx, use_discrete)
    msg = "Invalid mode of the eigenvector."
    raise ValueError(msg)

def vec_q(mset: nh.ModelSettings,
          s: int | str,
          use_discrete: bool = True) -> None:
    r"""
    Return eigenvectors of the System matrix and the divergence vector.

    Parameters
    ----------
    mset : ModelSettings
        The model settings.
    s : int | str
        The mode of the eigenvector.
        0  => geostrophic,
        "d" => divergent,
        1  => positive inertial-gravity,
        -1 => negative inertial-gravity
    use_discrete : bool (default: True)
        If True, the discrete eigenvectors are returned. Otherwise, the
        continuous eigenvectors are returned.

    Returns
    -------
    `State`
        The eigenvectors of the System matrix.

    Description
    -----------
    For the continuous case, and :math:`k_x = k_y = 0`, the eigenvectors are:

    .. math::
        \boldsymbol{q^0} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}
        \quad, \quad
        \boldsymbol{q^\pm} = \begin{pmatrix} \mp i \\ 1 \\ 0 \\ 0 \end{pmatrix}
        \quad, \quad
        \boldsymbol{q^d} = \begin{pmatrix} k_x \\ k_y \\ k_z \\ 0 \end{pmatrix}

    For the general case of nonzero horizontal wavenumbers, the
    eigenvectors are:

    .. math::
        \boldsymbol{q^0} =
        \begin{pmatrix} -k_y \\ k_x \\ 0 \\ fk_z \end{pmatrix}
        \quad, \quad
        \boldsymbol{q^\pm} = \begin{pmatrix}
            k_z ( -i \omega^\pm k_x + f k_y) \\
            k_z ( -i \omega^\pm k_y - f k_x) \\
            i\omega^\pm k_h^2 \\
            N^2 k_h^2
        \end{pmatrix}
        \quad, \quad
        \boldsymbol{q^d} = \begin{pmatrix} k_x \\ k_y \\ k_z \\ 0 \end{pmatrix}

    The discrete projection vector is given in the docstring of the
    eigenvectors.

    """
    # We first check if the spectral analysis is possible
    _check_if_spectral_analysis_is_possible(mset)
    _check_for_horizontal_periodic_boundaries(mset)

    # Then we calculate the eigenvector arrays
    u, v, w, b = _vec_q(s, mset.f0, mset.stratification_n2, mset.dsqr,
                        mset.grid.k_mesh, mset.grid.dx, use_discrete)

    # Finally we construct the state and set the nyquist frequency to zero
    z = nh.State(mset, is_spectral=True)
    z.u.arr = u
    z.v.arr = v
    z.w.arr = w
    z.b.arr = b
    return dso.set_nyquist_to_zero(z)


# ================================================================
#  The projection vectors
# ================================================================
def _normalize_p_vec(
        p: tuple[ndarray],
        q: tuple[ndarray],
        f0: float,
        n_squared: float,
        k: tuple[float] | tuple[ndarray],
) -> tuple[ndarray]:
    """Normalize the projection vectors."""
    kx, ky, kz = tuple(ncp.asarray(k) for k in k)

    f0 = ncp.asarray(f0)
    n_squared = ncp.asarray(n_squared)

    horizontal_mask = (kx**2 + ky**2 != 0)
    vertical_mask = (kz**2 != 0)
    full_mask = (kx**2 + ky**2 + kz**2 != 0)

    # if N2 == 0 => use the vertical mask
    mask = ncp.where(n_squared == 0, vertical_mask, full_mask)
    # if f0 == 0 => use the horizontal mask
    mask = ncp.where(f0 == 0, horizontal_mask, mask)
    # if both f0 and N2 are zero => This case should never happen

    norm = ncp.abs(
        sum(p_i * q_i.conj() for p_i, q_i in zip(p, q, strict=False)))

    return tuple(ncp.where(mask, p_i/norm, 0) for p_i in p)

def _vec_p_geostrophic(
        f0: float,
        n_squared: float,
        k: tuple[float] | tuple[ndarray],
        dx: tuple[float] | None = None,
        use_discrete: bool = True,
) -> tuple[ndarray]:
    """Return geostrophic projection vectors."""
    _check_use_discrete_argument(dx, use_discrete)

    kx, ky, kz = tuple(ncp.asarray(k) for k in k)
    dx, dy, dz = dx or (None, None, None)

    # Define the spectral operators
    def ohp(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat(k, d, +1, use_discrete)
    def ohm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat(k, d, -1, use_discrete)
    def khp(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.k_hat(k, d, +1, use_discrete)
    def ohpm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat_squared(k, d, use_discrete)

    # Mask to separate inertial modes from inertia-gravity modes
    nonzero_horizontal = (kx**2 + ky**2 != 0)

    # Construct the eigenvector
    q = _vec_q_geostrophic(f0, k, (dx, dy, dz), use_discrete)

    # the zero horizontal wavenumber modes are the same as the eigenvector
    # So we only need to calculate the horizontal varying modes
    u = - ohp(kx,dx) * ohm(ky,dy) * ohp(kz,dz) * n_squared * khp(ky,dy)
    v =   ohm(kx,dx) * ohp(ky,dy) * ohp(kz,dz) * n_squared * khp(kx,dx)
    w = 0
    b = ohpm(kx,dx) * ohpm(ky,dy) * f0 * khp(kz,dz)

    p = tuple(ncp.where(nonzero_horizontal, arr, q_i)
              for arr, q_i in zip((u, v, w, b), q, strict=False))

    return _normalize_p_vec(p, q, f0, n_squared, k)

def _vec_p_divergent(
    k: tuple[float] | tuple[ndarray],
    dx: tuple[float] | None = None,
    use_discrete: bool = True,
) -> tuple[ndarray]:
    """Return divergent projection vectors."""
    _check_use_discrete_argument(dx, use_discrete)

    kx, ky, kz = tuple(ncp.asarray(k) for k in k)

    # Divergent modes are the same as the eigenvector
    p = _vec_q_divergent(k, dx, use_discrete)

    # normalize the vector
    norm = ncp.abs(sum(p_i * p_i.conj() for p_i in p))

    # avoid division by zero
    mask = (kx**2 + ky**2 + kz**2 != 0)

    return tuple(ncp.where(mask, p_i/norm, 0) for p_i in p)

def _vec_p_igw(
        s: int,
        f0: float,
        n_squared: float,
        dsqr: float,
        k: tuple[float] | tuple[ndarray],
        dx: tuple[float] | None = None,
        use_discrete: bool = True,
) -> tuple[ndarray]:
    _check_use_discrete_argument(dx, use_discrete)

    kx, ky, kz = tuple(ncp.asarray(k) for k in k)
    dx, dy, dz = dx or (None, None, None)

    # Define the spectral operators
    def ohp(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat(k, d, +1, use_discrete)
    def ohm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.one_hat(k, d, -1, use_discrete)
    def khp(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.k_hat(k, d, +1, use_discrete)
    def khm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.k_hat(k, d, -1, use_discrete)
    def khpm(k: tuple[float] | tuple[ndarray], d: float) -> ndarray:
        return dso.k_hat_squared(k, d, use_discrete)

    # Horizontal wavenumber squared
    kh2 = khpm(kx, dx) + khpm(ky, dy)

    # Mask to separate inertial modes from inertia-gravity modes
    nonzero_horizontal = (kx**2 + ky**2 != 0)

    # Construct the eigenvector
    q = _vec_q_igw(s, f0, n_squared, dsqr, k, (dx, dy, dz), use_discrete)

    # the zero horizontal wavenumber modes are the same as the eigenvector
    # So we only need to calculate the horizontal varying modes
    # Scaling factor
    gamma = (  (kh2 + khpm(kz,dz))
                / (dsqr * kh2 + khpm(kz,dz)) )

    # compute the eigenvalues (frequency)
    om = omega(s=s, f0=f0, stratification_n2=n_squared, dsqr=dsqr,
               k=(kx, ky, kz), dx=(dx, dy, dz), use_discrete=use_discrete)

    u = (khm(kz,dz) * (-1j*om*khp(kx,dx) +
            ohp(kx,dx)*ohm(ky,dy)*f0*gamma*khp(ky,dy)))
    v = (khm(kz,dz) * (-1j*om*khp(ky,dy) -
            ohm(kx,dx)*ohp(ky,dy)*f0*gamma*khp(kx,dx)))
    w = 1j * om * kh2
    b = ohm(kz,dz) * gamma * kh2

    p = tuple(ncp.where(nonzero_horizontal, arr, q_i)
              for arr, q_i in zip((u, v, w, b), q, strict=False))

    # normalize the vector
    return _normalize_p_vec(p, q, f0, n_squared, k)


@partial(fr.utils.jaxjit, static_argnames=["s", "use_discrete"])
def _vec_p(
        s: int | str,
        f0: float,
        n_squared: float,
        dsqr: float,
        k: tuple[float] | tuple[ndarray],
        dx: tuple[float] | None = None,
        use_discrete: bool = True,
) -> tuple[ndarray]:
    if s == 0:
        return _vec_p_geostrophic(f0, n_squared, k, dx, use_discrete)
    if s == "d":
        return _vec_p_divergent(k, dx, use_discrete)
    if s in (1, -1):
        return _vec_p_igw(s, f0, n_squared, dsqr, k, dx, use_discrete)
    msg = "Invalid mode of the eigenvector."
    raise ValueError(msg)

def vec_p(mset: nh.ModelSettings,
          s: int | str,
          use_discrete: bool = True) -> None:
    r"""
    Return the projection vectors of the System matrix.

    Parameters
    ----------
    mset : ModelSettings
        The model settings.
    s : int
        The mode of the eigenvector.
        0  => geostrophic,
        "d" => divergent,
        1  => positive inertial-gravity,
        -1 => negative inertial-gravity
    use_discrete : bool (default: True)
        If True, the discrete eigenvectors are returned. Otherwise, the
        continuous eigenvectors are returned.

    Description
    -----------
    For the continuous case, and :math:`k_x = k_y = 0`, the projection
    vectors are:

    .. math::
        \boldsymbol{p^0} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}
        \quad, \quad
        \boldsymbol{p^\pm} = \begin{pmatrix} \mp i \\ 1 \\ 0 \\ 0 \end{pmatrix}
        \quad, \quad
        \boldsymbol{p^d} = \begin{pmatrix} k_x \\ k_y \\ k_z \\ 0 \end{pmatrix}

    For the general case of nonzero horizontal wavenumbers, the
    projection vectors are:

    .. math::
        \boldsymbol{p^0} = \begin{pmatrix}
            -N^2 k_y \\ N^2 k_x \\ 0 \\ f k_z
        \end{pmatrix}
        \quad, \quad
        \boldsymbol{p^\pm} = \begin{pmatrix}
            k_z ( -i \omega^\pm k_x + f \gamma k_y) \\
            k_z ( -i \omega^\pm k_y - f \gamma k_x) \\
            i\omega^\pm k_h^2 \\
            \gamma k_h^2
        \end{pmatrix}
        \quad, \quad
        \boldsymbol{p^d} = \begin{pmatrix} k_x \\ k_y \\ k_z \\ 0 \end{pmatrix}

    The discrete projection vector is given in the docstring of the
    eigenvectors.

    """
    _check_if_spectral_analysis_is_possible(mset)
    _check_for_horizontal_periodic_boundaries(mset)
    (u, v, w, b) = _vec_p(s, mset.f0, mset.stratification_n2, mset.dsqr,
                          mset.grid.k_mesh, mset.grid.dx, use_discrete)
    z = nh.State(mset, is_spectral=True)
    z.u.arr = u
    z.v.arr = v
    z.w.arr = w
    z.b.arr = b
    return dso.set_nyquist_to_zero(z)
