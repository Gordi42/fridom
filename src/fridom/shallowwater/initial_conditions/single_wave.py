"""Single wave initial condition for the shallow water model."""
from __future__ import annotations

import jax.numpy as jnp

import fridom.shallowwater as sw


class SingleWave(sw.State):

    r"""
    A single polarized wave with a given wavenumber and a given mode.

    Description
    -----------
    Creates a polarized wave from the eigenvectors of the linearized
    equations of motion. The wave is initizalized in spectral space as:

    .. math::
        z(\boldsymbol{k}) = \boldsymbol{q}_s(\boldsymbol{k})
                         \delta_{\boldsymbol{k}, \boldsymbol{k}_0} \exp(i\phi)

    where :math:`\boldsymbol{q}_s` is the eigenvector of the mode `s`
    (see :py:mod:`fridom.shallowwater.grid.cartesian.eigenvectors`), and
    :math:`\delta_{\boldsymbol{k}, \boldsymbol{k}_0}` is the Kronecker delta
    function:

    .. math::
        \delta_{\boldsymbol{k}, \boldsymbol{k}_0} = \begin{cases}
            1 & \text{if } \boldsymbol{k}
                = 2\pi\boldsymbol{k}_0/\boldsymbol{L} \\
            0 & \text{otherwise}
        \end{cases}

    with :math:`\boldsymbol{L}` the domain size in the x, y, and z directions
    and :math:`\boldsymbol{k}_0` the wavenumber that is passed as an argument.
    The phase :math:`\phi` is also passed as an argument. Finally, the state
    is fourier transformed to physical space and normalized so that its
    L2 norm is equal to 1.

    Parameters
    ----------
    mset : ModelSettings
        The model settings.
    k : tuple[int]
        The wavenumber in the x and y directions.
        A wavenumber of one means that the wave has a wavelength equal to the
        domain size in that direction.
    s : int
        The mode (0, 1, -1)
        0 => geostrophic mode
        1 => positive inertia-gravity mode
        -1 => negative inertia-gravity mode
    phase : float
        The phase of the wave. (default: 0)
    use_discrete : bool, optional
        Whether to use the discrete eigenvectors or the analytical ones
        (default: True).

    """

    def __init__(self,
                 mset: sw.ModelSettings,
                 k: tuple[int],
                 s: int = 1,
                 phase: float = 0,
                 use_discrete: bool = True) -> None:
        super().__init__(mset, is_spectral=False)

        # Shortcuts
        grid = mset.grid
        kx_mesh, ky_mesh = grid.k_mesh
        kx, ky = k
        lx, ly = grid.domain_size
        pi = jnp.pi

        # Find index of the wavenumber in the grid (nearest neighbor)
        kx = 2*pi*kx/lx
        ky = 2*pi*ky/ly
        is_kx = jnp.isclose(kx_mesh, kx)
        is_ky = jnp.isclose(ky_mesh, ky)
        k_loc = is_kx & is_ky

        # Construct the spectral field of the corresponding mode
        # all zeros except for the mode
        mask = jnp.where(k_loc, 1, 0)

        # Construct the eigenvector of the corresponding mode
        q = mset.grid.vec_q(s, use_discrete=use_discrete)

        # Construct the state
        z = (q * mask * jnp.exp(1j*phase)).ifft()

        # Normalize the state
        z /= z.norm_l2()

        # Set the state
        self.fields = z.fields

        # Calculate the frequency and period of the wave
        om = grid.omega((kx, ky), use_discrete=True)
        ts = mset.time_stepper
        self.omega = ts.time_discretization_effect(om)
        self.period = 2*pi/self.omega.real

        # save the wavenumbers
        self.kx = kx
        self.ky = ky
