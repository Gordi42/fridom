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
     -if\delta^2k_x k_y & -if\left( \delta^2k_y^2 + k_z^2 \right) & 0 & i k_x k_z \\
     if \left( \delta^2 k_x^2 + k_z^2 \right) & if\delta^2k_x k_y & 0 & i k_y k_z \\
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
For the eigenvectors we have to separately consider the case of purely vertical,
e.g. :math:`k_x = k_y = 0`, and the general case of nonzero horizontal wavenumbers.
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
such that :math:`(\boldsymbol{q^0}, \boldsymbol{q^+}, \boldsymbol{q^-}, \boldsymbol{q^d})`
form a basis.

For the purely vertical case, the projection vectors are identical to the
eigenvectors: :math:`\boldsymbol{p^s} = \boldsymbol{q^s}`. For the general case
of nonzero horizontal wavenumbers, the projection vectors are:

.. math::
    \boldsymbol{p^0} = \begin{pmatrix} -N^2 k_y \\ N^2 k_x \\ 0 \\ f k_z \end{pmatrix} 
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
     
Similarly, we define the forward and backward linear interpolation operators as:

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
        -i f \hat{1}_x^+ \hat{1}_y^- \left( \delta^2 \hat{k}_y^2 + \hat{k}_z^2 \right ) &
        0 &
        i \hat{1}_z^+ \hat{k}_x^+ \hat{k}_z^- \\
        i f \hat{1}_x^- \hat{1}_y^+ \left( \delta^2 \hat{k}_x^2 + \hat{k}_z^2 \right) &
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
continuous eigenvectors. For the general case of nonzero horizontal wavenumbers,
the discrete eigenvectors are:

.. math::
    \boldsymbol{q^0} = \begin{pmatrix}
        - \hat{1}_x^+ \hat{1}_y^- \hat{1}_z^+ \hat{k}_y^+ \\
        \hat{1}_x^- \hat{1}_y^+ \hat{1}_z^+ \hat{k}_x^+ \\
        0 \\
        \hat{1}_x^2 \hat{1}_y^2 f \hat{k}_z^+
    \end{pmatrix} 
    \quad \text{and} \quad
    \boldsymbol{q}^\pm = \begin{pmatrix}
        \hat{k}_z^- ( -i \omega^\pm \hat{k}_x^+ + \hat{1}_x^+ \hat{1}_y^- f \hat{k}_y^+) \\
        \hat{k}_z^- ( -i \omega^\pm \hat{k}_y^+ - \hat{1}_x^- \hat{1}_y^+ f \hat{k}_x^+) \\
        i \omega^\pm \hat{k}_h^2 \\
        \hat{1}_z^- N^2 \hat{k}_h^2
    \end{pmatrix}

The divergent vector is given by :math:`\boldsymbol{q^d} = 
(\hat{k}_x^+, \hat{k}_y^+, \hat{k}_z^+, 0)`. All eigenvectors are orthogonal 
to the divergent vector. Hence no eigenvector project onto divergent velocity fields.

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
        \hat{k}_z^- ( -i \omega^\pm \hat{k}_x^+ + \hat{1}_x^+ \hat{1}_y^- f \hat{\gamma} \hat{k}_y^+) \\
        \hat{k}_z^- ( -i \omega^\pm \hat{k}_y^+ - \hat{1}_x^- \hat{1}_y^+ f \hat{\gamma} \hat{k}_x^+) \\
        i \omega^\pm \hat{k}_h^2 \\
        \hat{1}_z^- \hat{\gamma} \hat{k}_h^2
    \end{pmatrix}

with

.. math::
    \hat{\gamma} = \frac{\hat{k}_h^2 + \hat{k}_z^2}{\delta^2 \hat{k}^2}
"""
import numpy as np
from numpy import ndarray
import fridom.framework as fr
import fridom.nonhydro as nh

# ================================================================
#  The eigenvalues
# ================================================================
def omega(mset: nh.ModelSettings,
          s: int,
          k: tuple[float] | tuple[ndarray],
          use_discrete: bool = False
          ) -> ndarray:
    # shorthand notation
    ncp = fr.config.ncp
    dsqr = mset.dsqr
    f2 = mset.f0**2
    N2 = mset.N2
    kx, ky, kz = k
    if s == 0:
        return ncp.zeros_like(kx)
    # cast k to ndarray
    kx = ncp.asarray(kx); ky = ncp.asarray(ky); kz = ncp.asarray(kz)
    dx, dy, dz = mset.grid.dx

    if not ncp.allclose(f2, mset.f_coriolis.arr**2):
        fr.log.warning("The corioliis frequency is varying.")
        fr.log.warning("The eigenvalues and eigenvectors may be wrong.")
    if not ncp.allclose(N2, mset.N2):
        fr.log.warning("N^2 is varying.")
        fr.log.warning("The eigenvalues and eigenvectors may be wrong.")

    from fridom.framework.grid.cartesian import discrete_spectral_operators as dso
    ohpm = lambda k, d: dso.one_hat_squared(k, d, use_discrete)
    khpm = lambda k, d: dso.k_hat_squared(k, d, use_discrete)

    kh2 = khpm(kx, dx) + khpm(ky, dy)
    with np.errstate(divide='ignore'):
        coriolis_part = ohpm(kx,dx) * ohpm(ky,dy) * f2 * khpm(kz,dz)
        buoyancy_part = ohpm(kz,dz) * N2 * kh2
        denominator = dsqr * kh2 + khpm(kz,dz)

    om = ncp.sqrt((coriolis_part + buoyancy_part) / denominator)

    # set the result to zero where the denominator is zero
    nonzero = (kx**2 + ky**2 + kz**2 > 0)
    om = ncp.where(nonzero, om, 0)
    return s * om


# ================================================================
#  The eigenvectors
# ================================================================

def vec_q(mset: nh.ModelSettings, 
          s: int, 
          use_discrete=True) -> None:
    r"""
    The eigenvectors of the System matrix and the divergence vector.

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `s` : `int`
        The mode of the eigenvector.
        0  => geostrophic,
        "d" => divergent,
        1  => positive inertial-gravity,
        -1 => negative inertial-gravity
    `use_discrete` : `bool` (default: True)
        If True, the discrete eigenvectors are returned. Otherwise, the continuous
        eigenvectors are returned.

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
        \quad, \quad
        \boldsymbol{q^d} = \begin{pmatrix} k_x \\ k_y \\ k_z \\ 0 \end{pmatrix}

    The discrete projection vector is given in the docstring of the eigenvectors.
    """

    # Shortcuts
    grid = mset.grid
    ncp = fr.config.ncp
    kx, ky, kz = grid.K
    dx, dy, dz = grid.dx
    f0 = mset.f0; N0 = mset.N2**(1/2); dsqr = mset.dsqr

    from fridom.framework.grid.cartesian import discrete_spectral_operators as dso
    ohp = lambda k, d: dso.one_hat(k, d, +1, use_discrete)
    ohm = lambda k, d: dso.one_hat(k, d, -1, use_discrete)
    khp = lambda k, d: dso.k_hat(k, d, +1, use_discrete)
    khm = lambda k, d: dso.k_hat(k, d, -1, use_discrete)
    khpm = lambda k, d: dso.k_hat_squared(k, d, use_discrete)
    ohpm = lambda k, d: dso.one_hat_squared(k, d, use_discrete)
        

    # Horizontal wavenumber squared
    kh2 = khpm(kx, dx) + khpm(ky, dy)

    # Check the mode of the eigenvector
    is_geostrophic = (s == 0)
    is_divergent   = (s == "d")

    # We first consider the case of nonzero horizontal wavenumbers
    if is_geostrophic:
        u = -ohp(kx,dx)*ohm(ky,dy)*ohp(kz,dz)*khp(ky,dy)
        v =  ohm(kx,dx)*ohp(ky,dy)*ohp(kz,dz)*khp(kx,dx)
        w = ncp.zeros_like(kx)
        b = ohpm(kx,dx)*ohpm(ky,dy)*f0*khp(kz,dz)
    elif is_divergent:
        u = khp(kx,dx)
        v = khp(ky,dy)
        w = khp(kz,dz)
        b = ncp.zeros_like(kx)
    else:
        # calculate eigenvalue
        om = omega(mset=mset, s=s, k=(kx, ky, kz), use_discrete=use_discrete)
        u = (khm(kz,dz) * (-1j*om*khp(kx,dx) + 
                ohp(kx,dx)*ohm(ky,dy)*f0*khp(ky,dy)))
        v = (khm(kz,dz) * (-1j*om*khp(ky,dy) -
                ohm(kx,dx)*ohp(ky,dy)*f0*khp(kx,dx)))
        w = 1j * om * kh2
        b = ohm(kz,dz) * N0**2 * kh2
        
    # Now we consider the purely vertical case
    # (ov -> only vertical)
    if is_geostrophic:
        u_ov = 0
        v_ov = 0
        w_ov = 0
        b_ov = 1
    elif is_divergent:
        u_ov = khp(kx,dx) # should be zero
        v_ov = khp(ky,dy) # should be zero
        w_ov = khp(kz,dz)
        b_ov = 0
    else:
        u_ov = -s * 1j
        v_ov = 1
        w_ov = 0
        b_ov = 0

    # Mask to separate inertial modes from inertia-gravity modes
    nonzero_horizontal = (kx**2 + ky**2 != 0)  # nonzero horizontal wavenumbers

    z = nh.State(mset, is_spectral=True)
    z.u.arr = ncp.where(nonzero_horizontal, u, u_ov)
    z.v.arr = ncp.where(nonzero_horizontal, v, v_ov)
    z.w.arr = ncp.where(nonzero_horizontal, w, w_ov)
    z.b.arr = ncp.where(nonzero_horizontal, b, b_ov)

    # Set the nyquist frequency to zero
    z = dso.set_nyquist_to_zero(z)
    return z

def vec_p(mset: nh.ModelSettings, 
          s: int,
          use_discrete: bool = True) -> None:
    r"""
    The projection vectors of the System matrix.

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `s` : `int`
        The mode of the eigenvector.
        0  => geostrophic,
        "d" => divergent,
        1  => positive inertial-gravity,
        -1 => negative inertial-gravity
    `use_discrete` : `bool` (default: True)
        If True, the discrete eigenvectors are returned. Otherwise, the continuous
        eigenvectors are returned.

    Description
    -----------
    For the continuous case, and :math:`k_x = k_y = 0`, the projection vectors are:

    .. math::
        \boldsymbol{p^0} = \begin{pmatrix} 0 \\ 0 \\ 0 \\ 1 \end{pmatrix}
        \quad, \quad
        \boldsymbol{p^\pm} = \begin{pmatrix} \mp i \\ 1 \\ 0 \\ 0 \end{pmatrix}
        \quad, \quad
        \boldsymbol{p^d} = \begin{pmatrix} k_x \\ k_y \\ k_z \\ 0 \end{pmatrix}

    For the general case of nonzero horizontal wavenumbers, the projection vectors are:

    .. math::
        \boldsymbol{p^0} = \begin{pmatrix} -N^2 k_y \\ N^2 k_x \\ 0 \\ f k_z \end{pmatrix}
        \quad, \quad
        \boldsymbol{p^\pm} = \begin{pmatrix}
            k_z ( -i \omega^\pm k_x + f \gamma k_y) \\
            k_z ( -i \omega^\pm k_y - f \gamma k_x) \\
            i\omega^\pm k_h^2 \\
            \gamma k_h^2
        \end{pmatrix}
        \quad, \quad
        \boldsymbol{p^d} = \begin{pmatrix} k_x \\ k_y \\ k_z \\ 0 \end{pmatrix}
    
    The discrete projection vector is given in the docstring of the eigenvectors.
    """
    # Shortcuts
    ncp = fr.config.ncp
    grid = mset.grid
    kx, ky, kz = grid.K
    dx, dy, dz = grid.dx
    f0 = mset.f0; N0 = mset.N2**(1/2); dsqr = mset.dsqr

    from fridom.framework.grid.cartesian import discrete_spectral_operators as dso
    ohp = lambda k, d: dso.one_hat(k, d, +1, use_discrete)
    ohm = lambda k, d: dso.one_hat(k, d, -1, use_discrete)
    khp = lambda k, d: dso.k_hat(k, d, +1, use_discrete)
    khm = lambda k, d: dso.k_hat(k, d, -1, use_discrete)
    khpm = lambda k, d: dso.k_hat_squared(k, d, use_discrete)
    ohpm = lambda k, d: dso.one_hat_squared(k, d, use_discrete)
        
    # Horizontal wavenumber squared
    kh2 = khpm(kx, dx) + khpm(ky, dy)

    # Check the mode of the eigenvector
    is_geostrophic = (s == 0)
    is_divergent = (s == "d")

    # Mask to separate inertial modes from inertia-gravity modes
    nonzero_horizontal = (kx**2 + ky**2 != 0)

    # Construct the eigenvector
    q = vec_q(mset, s, use_discrete=use_discrete)
    qu = q.u.arr; qv = q.v.arr; qw = q.w.arr; qb = q.b.arr

    # the zero horizontal wavenumber modes are the same as the eigenvector
    # So we only need to calculate the horizontal varying modes
    if is_geostrophic:
        u = - ohp(kx,dx) * ohm(ky,dy) * ohp(kz,dz) * N0**2 * khp(ky,dy)
        v =   ohm(kx,dx) * ohp(ky,dy) * ohp(kz,dz) * N0**2 * khp(kx,dx)
        w = 0
        b = ohpm(kx,dx) * ohpm(ky,dy) * f0 * khp(kz,dz)
    elif is_divergent:
        # Divergent modes are the same as the eigenvector
        u = qu
        v = qv
        w = qw
        b = qb
    else:
        # Scaling factor
        gamma = (  (kh2 + khpm(kz,dz))
                    / (dsqr * kh2 + khpm(kz,dz)) )

        # Eigenvalues (frequency)
        om = omega(mset, s, (kx, ky, kz), use_discrete=use_discrete)

        u = (khm(kz,dz) * (-1j*om*khp(kx,dx) +
                ohp(kx,dx)*ohm(ky,dy)*f0*gamma*khp(ky,dy)))
        v = (khm(kz,dz) * (-1j*om*khp(ky,dy) -
                ohm(kx,dx)*ohp(ky,dy)*f0*gamma*khp(kx,dx)))
        w = 1j * om * kh2
        b = ohm(kz,dz) * gamma * kh2

    z = nh.State(mset, is_spectral=True)
    z.u.arr = ncp.where(nonzero_horizontal, u, qu)
    z.v.arr = ncp.where(nonzero_horizontal, v, qv)
    z.w.arr = ncp.where(nonzero_horizontal, w, qw)
    z.b.arr = ncp.where(nonzero_horizontal, b, qb)

    # normalize the vector
    norm = ncp.abs((q.dot(z)).arr)
    # avoid division by zero
    mask = (norm > 1e-10)
    z.u.arr = ncp.where(mask, z.u.arr/norm, 0)
    z.v.arr = ncp.where(mask, z.v.arr/norm, 0)
    z.w.arr = ncp.where(mask, z.w.arr/norm, 0)
    z.b.arr = ncp.where(mask, z.b.arr/norm, 0)

    # Set the nyquist frequency to zero
    z = dso.set_nyquist_to_zero(z)
    return z
