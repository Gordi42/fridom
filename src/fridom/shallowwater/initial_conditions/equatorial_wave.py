"""Initial conditions for equatorial waves on the beta-plane."""
from __future__ import annotations

import numpy as np

import fridom.shallowwater as sw


class EquatorialWave(sw.State):

    r"""
    Equatorial wave on the beta-plane approximation.

    Parameters
    ----------
    mset : ModelSettings
        Model settings.
    longitudinal_mode : int
        Longitudinal mode of the wave (how many wavelengths in the x-direction).
    equatorial_mode : int
        Equatorial mode of the wave (Hermite polynomial order).
    wave_mode : int
        Mode of the wave:
            0 for the wave mode with negative frequency
            1 for the rossby mode (smallest absolute frequency)
            2 for the wave mode with positive frequency
    phase : float, optional
        Phase of the wave.

    Description
    -----------
    The equatorial wave solutions follow from solving the eigenvalue problem of
    the linearized shallow water equations on the equatorial beta-plane
    :math:`f = \beta y`. The frequencies of the m-th eigenmode can be obtained by
    solving

    .. math::

        \omega_m \left( \omega_m^2 - c^2
                    \left( k^2 + (2m + 1) \frac{\beta}{c} \right)
                 \right) = k \beta c^2

    for :math:`\omega_m`.

    The initial condition is then given by

    .. math::

        \mathbf z = \begin{pmatrix} u_m \\ v_m \\ p_m \end{pmatrix}
                    e^{- \frac{1}{2}\tilde{y}^2} e^{i(kx + \phi)}
        \quad \text{with} \quad
        \tilde{y} = \frac{y}{R_e}
        \quad \text{and} \quad
        R_e^2 = \frac{c}{\beta}

    with

    .. math::

        \begin{align}
            u_m &= \frac{i c}{2 R_e} \left( \frac{H_{m+1}(\tilde{y})}{\omega_m - c k}
                + \frac{2 m H_{m-1}(\tilde{y})}{\omega_m + c k} \right) \\
            v_m &= H_m(\tilde{y})  \\
            p_m &= \frac{i c^2}{2 R_e} \left( \frac{H_{m+1}(\tilde{y})}{\omega_m - c k}
                - \frac{2 m H_{m-1}(\tilde{y})}{\omega_m + c k} \right)
        \end{align}

    where :math:`H_m` is the m-th Hermite polynomial, given by the recursive relation

    .. math::

        H_m(x) = 2x H_{m-1}(x) - 2(m-1) H_{m-2}(x)

    with :math:`H_{-1}(x) = 0` and :math:`H_0(x) = 1`.

    """

    def __init__(self,
                 mset: sw.ModelSettings,
                 longitudinal_mode: int,
                 equatorial_mode: int,
                 wave_mode: int,
                 phase: float = 0,
                 ) -> None:
        super().__init__(mset, is_spectral=False)

        # Shortcuts
        ncp = sw.config.ncp
        grid = mset.grid
        pi = ncp.pi

        # Compute physical parameters
        csqr = mset.csqr
        beta = mset.beta
        phase_velocity = csqr ** 0.5
        rossby_radius = (phase_velocity / beta) ** 0.5
        eigenvalue = (2 * equatorial_mode + 1) / (rossby_radius ** 2)

        y0 = grid.L[1] / 2
        kx = 2 * pi / grid.L[0] * longitudinal_mode

        # Compute the frequencies of the equatorial wave
        coeffs = [1, 0, -csqr * (kx**2 + eigenvalue), -kx * beta * csqr]
        omega = np.sort(np.roots(coeffs))[wave_mode]
        period = np.abs(2 * np.pi / omega)

        # Helper functions for the hermite polynomials
        def hermite_polynomial(order: int,
                               x: float,
                               memory: dict | None = None) -> float:
            """Calculate the Hermite polynomial of given order at x."""
            if memory is None:
                memory = {-1: x * 0, 0: x * 0 + 1}
            if order in memory:
                return memory[order]
            memory[order] = ( 2 * x * hermite_polynomial(order-1, x, memory)
                            - 2 * (order-1) * hermite_polynomial(order-2, x, memory) )
            return memory[order]

        def hermite_gaussian(order: int, x: float) -> float:
            """Calculate the Hermite-Gaussian function of given order at x."""
            return hermite_polynomial(order, x) * np.exp(-x**2/2)

        state = sw.State(mset)
        u, v, p = state

        # v
        x, y = v.get_mesh()
        y_star = (y - y0) / rossby_radius
        v_structure = hermite_gaussian(equatorial_mode, y_star)
        v.arr = ( v_structure * ncp.exp(1j * (kx * x + phase)) ).real

        # u
        x, y = u.get_mesh()
        y_star = (y - y0) / rossby_radius
        psi_np1 = hermite_gaussian(equatorial_mode + 1, y_star)
        psi_nm1 = hermite_gaussian(equatorial_mode - 1, y_star)
        u_structure = 1j * phase_velocity / (2 * rossby_radius) * (
                        psi_np1 / (omega - kx * phase_velocity)
                        + 2 * equatorial_mode * psi_nm1 / (omega + kx * phase_velocity))
        u.arr = ( u_structure * ncp.exp(1j * (kx * x + phase)) ).real

        # p
        x, y = p.get_mesh()
        y_star = (y - y0) / rossby_radius
        psi_np1 = hermite_gaussian(equatorial_mode + 1, y_star)
        psi_nm1 = hermite_gaussian(equatorial_mode - 1, y_star)
        p_structure = 1j * csqr / (2 * rossby_radius) * (
            psi_np1 / (omega - kx * phase_velocity)
            - 2 * equatorial_mode * psi_nm1 / (omega + kx * phase_velocity))
        p.arr = ( p_structure * ncp.exp(1j * (kx * x + phase)) ).real

        # Normalize the state
        u_amp = (u**2 + v**2).max() ** 0.5
        state /= u_amp

        # Set the state
        self.fields = state.fields
        self.omega = omega
        self.period = period
