r"""
Eigenvalues and eigenvectors of the system matrix of the shallow water model.

Continuous Case
===============

The System Matrix
-----------------
Linearizing the shallow water equations (i.e. taking the limit :math:`Ro \rightarrow 0`)
and performing a spatial Fourier transform, we obtain the following system:

.. math::
    \partial_t \boldsymbol{z} = -i \mathbf{A} \cdot \boldsymbol{z}

with

.. math::
    \boldsymbol{z} = 
    \begin{pmatrix}
        u \\ v \\ p 
    \end{pmatrix} 
    \quad, \quad
    \mathbf{A} = 
    \begin{pmatrix}
        0       & if      & k_x \\
        -if     & 0       & k_y \\
        c^2 k_x & c^2 k_y & 0
    \end{pmatrix} 

Eigenvalues
-----------
The system matrix has three eigenvalues. One eigenvalue is zero, corresponding
to the geostrophic mode:

.. math::
    \omega^0 = 0

The other two eigenvalues correspond to the inertial-gravity wave modes:

.. math::
    \omega^\pm = \pm \sqrt{f^2 + c^2 (k_x^2 + k_y^2)}

Eigenvectors
------------
For :math:`|\boldsymbol{k}| > 0` the eigenvectors that correspond to the 
mode :math:`s=0,+,-` are given by:

.. math::
    \boldsymbol{q^s} = \begin{pmatrix}
                            \omega^s k_x - i f k_y \\
                            \omega^s k_y + i f k_x \\
                            f^2 - (\omega^s)^2
                        \end{pmatrix}

For :math:`|\boldsymbol{k}| = 0` we obtain the eigenvectors for the inertial modes:

.. math::
    \boldsymbol{q^s} = \begin{pmatrix} -is \\ s^2 \\ 1 - s^2 \end{pmatrix}

Projection Vectors
------------------
Projection vectors should satisfy 

.. math::
    {\boldsymbol{p^s}}^* \cdot \boldsymbol{q^{s'}} = \delta_{s,s'}

where the star denotes the hermitian transposed. Solving this equation for
:math:`|\boldsymbol{k}| > 0` yields:

.. math::
    \boldsymbol{p^s} = \begin{pmatrix} q^s_x \\ q^s_y \\ c^{-2} q^s_z \end{pmatrix}

where :math:`q^s_i` denotes the :math:`i`-th component of the 
eigenvector :math:`\boldsymbol{q^s}`.

For the inertial modes (i.e. :math:`|\boldsymbol{k}| = 0`), the projection vectors
are equal to the eigenvectors. All projection vectors are normalized such that 
:math:`{\boldsymbol{p^s}}^* \cdot \boldsymbol{p^s} = 1` holds.


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

Using these discretization operators, the discrete linear system matrix is:

.. math::
    \mathbf{A} = \begin{pmatrix}
                    0 & i f \hat{1}_x^+ \hat{1}_y^-   & \hat{k}_x^+ \\
                    -i f \hat{1}_x^- \hat{1}_y^+ & 0  & \hat{k}_y^+ \\
                    c^2 \hat{k}_x^- & c^2 \hat{k}_y^- & 0
                 \end{pmatrix}

Eigenvalues
-----------
The eigenvalues of the discrete system matrix are:

.. math::
    \omega^0 = 0
    \quad \text{and} \quad
    \omega^\pm =
        \sqrt{\hat{1}_x^2 \hat{1}_y^2 f^2 + 
        c^2 (\hat{k}_x^2 + \hat{k}_y^2)}
    
with

.. math::
    \hat{1}_x^2 = \hat{1}_x^+ \hat{1}_x^-
    \quad \text{and} \quad
    \hat{k}_x^2 = \hat{k}_x^+ \hat{k}_x^-

Eigenvectors
-----------------------------------
The eigenvectors :math:`\boldsymbol{q^s}` for :math:`|\boldsymbol{k}| > 0` are:

.. math::
    \boldsymbol{q^s} = \begin{pmatrix}
        \omega^s \hat{k}_x^+ - i \hat{1}_x^+ \hat{1}_y^- f \hat{k}_y^+ \\
        \omega^s \hat{k}_y^+ + i \hat{1}_x^- \hat{1}_y^+ f \hat{k}_x^+ \\
        \hat{1}_x^2 \hat{1}_y^2 f^2 - (\omega^s)^2
    \end{pmatrix}

For :math:`|\boldsymbol{k}| = 0` the eigenvectors are identical to the 
continuous case. The projection vectors are the same as in the continuous case,
but by using the discrete eigenvectors.
"""
import numpy as np
from numpy import ndarray
import fridom.framework as fr
import fridom.shallowwater as sw

# ================================================================
#  The eigenvalues
# ================================================================
def omega(mset: sw.ModelSettings,
          s: int,
          k: tuple[float] | tuple[ndarray],
          use_discrete: bool = False
          ) -> ndarray:
    r"""
    The eigenvalues of the System matrix.

    Computes the continuous or discrete eigenvalues as described in
    :py:mod:`eigenvectors <fridom.shallowwater.grid.cartesian.eigenvectors>`.

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `s` : `int`
        The mode of the eigenvector.
        0  => geostrophic,
        1  => positive inertial-gravity,
        -1 => negative inertial-gravity
    `k` : `tuple[float] | tuple[ndarray]`
        The wavenumber tuple :math:`(k_x, k_y)`.
    `use_discrete` : `bool` (default: True)
        If True, the discrete eigenvectors are returned. Otherwise, the continuous
        eigenvectors are returned.

    Returns
    -------
    `ndarray`
        The eigenvalues of the System matrix.
    """
    # shorthand notation
    ncp = fr.config.ncp
    csqr = mset.csqr
    f2 = mset.f0**2
    kx, ky = k

    # ----------------------------------------------------------------
    #  The geostrophic mode
    # ----------------------------------------------------------------
    if s == 0:
        return ncp.zeros_like(kx)

    # ----------------------------------------------------------------
    #  The wave modes
    # ----------------------------------------------------------------

    # cast k to ndarray
    kx = ncp.asarray(kx); ky = ncp.asarray(ky)
    dx, dy = mset.grid.dx

    # print a warning if the coriolis frequency or c² is varying
    if not ncp.allclose(f2, mset.f_coriolis.arr**2):
        fr.log.warning("The corioliis frequency is varying.")
        fr.log.warning("The eigenvalues and eigenvectors may be wrong.")
    if not ncp.allclose(csqr, mset.csqr_field.arr):
        fr.log.warning("c² is varying.")
        fr.log.warning("The eigenvalues and eigenvectors may be wrong.")

    # get discrete spectral operators
    from fridom.framework.grid.cartesian import discrete_spectral_operators as dso
    ohpm = lambda k, d: dso.one_hat_squared(k, d, use_discrete)
    khpm = lambda k, d: dso.k_hat_squared(k, d, use_discrete)

    # compute each part of the eigenvalue
    kh2 = khpm(kx, dx) + khpm(ky, dy)
    coriolis_part = ohpm(kx,dx) * ohpm(ky,dy) * f2
    gravity_part = csqr * kh2

    # return the eigenvalue
    return s * ncp.sqrt(coriolis_part + gravity_part)


# ================================================================
#  The eigenvectors
# ================================================================

def vec_q(mset: sw.ModelSettings, 
          s: int, 
          use_discrete=True) -> sw.State:
    r"""
    The eigenvectors of the system matrix

    Computes the continuous or discrete eigenvectors as described in
    :py:mod:`eigenvectors <fridom.shallowwater.grid.cartesian.eigenvectors>`.

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `s` : `int`
        The mode of the eigenvector.
        0  => geostrophic,
        1  => positive inertial-gravity,
        -1 => negative inertial-gravity
    `use_discrete` : `bool` (default: True)
        If True, the discrete eigenvectors are returned. Otherwise, the continuous
        eigenvectors are returned.

    Returns
    -------
    `State`
        The eigenvectors of the system matrix.
    """
    # Shortcuts
    grid = mset.grid
    ncp = fr.config.ncp
    kx, ky = grid.K
    dx, dy = grid.dx
    f0 = mset.f0

    # import the discrete spectral operators
    from fridom.framework.grid.cartesian import discrete_spectral_operators as dso
    ohp = lambda k, d: dso.one_hat(k, d, +1, use_discrete)
    ohm = lambda k, d: dso.one_hat(k, d, -1, use_discrete)
    khp = lambda k, d: dso.k_hat(k, d, +1, use_discrete)
    ohpm = lambda k, d: dso.one_hat_squared(k, d, use_discrete)
        
    # compute the eigenvalue
    om = omega(mset, s, (kx, ky), use_discrete=use_discrete)

    # compute the eigenvector for k != 0
    u = om * khp(kx,dx) / f0 - 1j*ohp(kx,dx)*ohm(ky,dy)*khp(ky,dy)
    v = om * khp(ky,dy) / f0 + 1j*ohm(kx,dx)*ohp(ky,dy)*khp(kx,dx)
    p = f0 * ohpm(kx,dx) * ohpm(ky,dy) - om**2 / f0

    # Inertial mode (k = 0)
    u_in = -1j*s
    v_in = s**2
    p_in = 1 - s**2

    # Mask to separate (k != 0) from (k = 0)
    k_nonzero = (kx**2 + ky**2 != 0) 

    z = sw.State(mset, is_spectral=True)
    z.u.arr = ncp.where(k_nonzero, u, u_in)
    z.v.arr = ncp.where(k_nonzero, v, v_in)
    z.p.arr = ncp.where(k_nonzero, p, p_in)

    # Set the nyquist frequency to zero
    z = dso.set_nyquist_to_zero(z)
    return z

def vec_p(mset: sw.ModelSettings, 
          s: int,
          use_discrete: bool = True) -> sw.State:
    r"""
    The projection vectors of the system matrix.

    Computes the continuous or discrete projection vectors as described in
    :py:mod:`eigenvectors <fridom.shallowwater.grid.cartesian.eigenvectors>`.

    Parameters
    ----------
    `mset` : `ModelSettings`
        The model settings.
    `s` : `int`
        The mode of the eigenvector.
        0  => geostrophic,
        1  => positive inertial-gravity,
        -1 => negative inertial-gravity
    `use_discrete` : `bool` (default: True)
        If True, the discrete eigenvectors are returned. Otherwise, the continuous
        eigenvectors are returned.

    Returns
    -------
    `State`
        The projection vectors of the system matrix.
    """
    # Shortcuts
    ncp = fr.config.ncp
    kx, ky = mset.grid.K
    csqr = mset.csqr

    # Construct the eigenvector
    z = vec_q(mset, s, use_discrete=use_discrete)

    # Divide the pressure by c² for k != 0
    k_nonzero = (kx**2 + ky**2 != 0)
    z.p.arr = ncp.where(k_nonzero, z.p.arr/csqr, z.p.arr)

    # normalize the vector
    q = vec_q(mset, s, use_discrete=use_discrete)
    norm = ncp.abs((q.dot(z)).arr)
    # avoid division by zero
    mask = (norm > 1e-10)
    with np.errstate(divide='ignore', invalid='ignore'):
        z.u.arr = ncp.where(mask, z.u.arr/norm, 0)
        z.v.arr = ncp.where(mask, z.v.arr/norm, 0)
        z.p.arr = ncp.where(mask, z.p.arr/norm, 0)

    # Set the nyquist frequency to zero
    from fridom.framework.grid.cartesian import discrete_spectral_operators as dso
    z = dso.set_nyquist_to_zero(z)
    return z
