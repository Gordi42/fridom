"""Kelvin wave initial condition."""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

import fridom.nonhydro as nh

if TYPE_CHECKING:
    from numpy import ndarray


class KelvinWave(nh.State):

    r"""
    Initial condition with a kelvin wave at the boundary.

    TODO: Add some more details about the math, add proper gallery example.

    Description
    -----------
    Lets consider a Kelvin wave at the southern boundary of the domain with
    the horizontal wavenumber `kh` and the vertical wavenumber `kz`. Searching
    in the linearized nonhydrostatic equations for a solution of the form:

    .. math::
        (U, V, W, B) \\exp(- k_n y) \\exp(i(k_h x + k_z z - \\omega t))

    Yields
    ------

    .. math::
        \\omega = \\sqrt{\\frac{k_h^2 N^2}{k_h^2 + k_z^2}} \\quad
        k_n = \\frac{k_h f_0}{\\omega}

    .. math::
        U = - k_z \\quad
        V = 0 \\quad
        W = k_h \\quad
        B = -i N^2 k_h / \\omega

    Similar polarizations can be found for the other boundaries.

    Parameters
    ----------
    mset : ModelSettings
        The model settings object.
    side : str
        The side of the domain where the wave is located.
        Possible values are 'N': North, 'S': South, 'E': East, 'W': West.
    kz : int
        The vertical wavenumber. (Can be positive or negative)
    kh : int
        The horizontal wavenumber in the direction of the boundary.
        (Must be positive)

    Examples
    --------
    .. code-block:: python

        import fridom.nonhydro as nh
        import numpy as np
        grid = nh.grid.cartesian.Grid(
            shape=[128]*3, domain_size=[1]*3,
            periodic_bounds=(True, False, True))
        mset = nh.ModelSettings(grid=grid)
        mset.time_stepper.dt = np.timedelta64(10, 'ms')
        mset.tendencies.advection.disable()
        mset.setup()
        z = nh.initial_conditions.KelvinWave(mset, 'N', kh=1, kz=2)
        model = nh.Model(mset)
        model.z = z
        model.run(runlen=np.timedelta64(5, 's'))


    A vertical mode may be constructed with

    .. code-block:: python

        z  = nh.initial_conditions.kelvin_wave(mset, 'N', kh=1, kz=2)
        z += nh.initial_conditions.kelvin_wave(
            mset, 'N', kh=1, kz=-2, phase=np.pi)
    """

    def __init__(self,  # noqa: C901
                 mset: nh.ModelSettings,
                 side: str,
                 kz: int,
                 k_parallel: int,
                 phase: float = 0) -> None:
        super().__init__(mset, is_spectral=False)

        if side not in {"N", "S", "E", "W"}:
            msg = (f"Invalid side '{side}'. "
                   "Must be one of 'N', 'S', 'E', or 'W'.")
            raise ValueError(msg)

        # convert the wavenumbers
        lx, ly, lz = mset.grid.domain_size
        kz = 2 * jnp.pi * kz / lz
        l_parallel = lx if side in {"N", "S"} else ly
        k_parallel = 2 * jnp.pi * k_parallel / l_parallel

        # calculate the frequency
        om = jnp.sqrt((k_parallel**2 * mset.stratification_n2) /
                      (kz**2 + mset.dsqr * k_parallel**2))

        # calculate the polarizations
        pol_u_normal = - kz
        pol_w = k_parallel
        pol_b = - 1j * mset.stratification_n2 * k_parallel / om

        # define function to get the exponential decay and the wave pattern
        def wave(x_parallel: ndarray,
                 x_normal: ndarray,
                 z: ndarray) -> ndarray:
            wave = jnp.exp(1j * (k_parallel * x_parallel + kz * z + phase))
            wave *= jnp.exp(- mset.f0 * k_parallel / om * x_normal)
            return wave

        if side == "N":
            def get_wave(f: nh.ScalarField) -> ndarray:
                x, y, z = f.get_mesh()
                x_normal = ly - y
                x_parallel = lx - x
                return wave(x_parallel, x_normal, z)
            self.u.arr = (- pol_u_normal * get_wave(self.u)).imag
        elif side == "S":
            def get_wave(f: nh.ScalarField) -> ndarray:
                x, y, z = f.get_mesh()
                x_normal = y
                x_parallel = x
                return wave(x_parallel, x_normal, z)
            self.u.arr = (pol_u_normal * get_wave(self.u)).imag
        elif side == "E":
            def get_wave(f: nh.ScalarField) -> ndarray:
                x, y, z = f.get_mesh()
                x_normal = lx - x
                x_parallel = y
                return wave(x_parallel, x_normal, z)
            self.v.arr = (pol_u_normal * get_wave(self.v)).imag
        elif side == "W":
            def get_wave(f: nh.ScalarField) -> ndarray:
                x, y, z = f.get_mesh()
                x_normal = x
                x_parallel = ly - y
                return wave(x_parallel, x_normal, z)
            self.v.arr = (- pol_u_normal * get_wave(self.v)).imag

        self.w.arr = (pol_w * get_wave(self.w)).imag
        self.b.arr = (pol_b * get_wave(self.b)).imag

        # save some parameters
        self.k_parallel = k_parallel
        self.kz = kz
        self.om = om
