"""Biharmonic friction + mixing for the nonhydrostatic model."""
from __future__ import annotations

from functools import partial

import fridom.framework as fr
import fridom.nonhydro as nh


@partial(fr.utils.jaxify, dynamic=("kh", "kv"))
class BiharmonicClosure(fr.modules.Module):

    r"""
    Biharmonic friction + mixing for the nonhydrostatic model.

    Parameters
    ----------
    kh : float
        horizontal diffusion coefficient
    kv : float
        vertical diffusion coefficient

    """

    name = "Biharmonic Closure"
    def __init__(self, kh: float, kv: float, velocity_scale: float = 1.0) -> None:
        super().__init__()
        self.kh = kh
        self.kv = kv
        self.velocity_scale = velocity_scale
        self.required_halo = 2

    def _on_setup(self) -> None:
        ncp = fr.config.ncp
        rossby_number = self.mset.Ro
        velocity_scale = self.velocity_scale

        dx, dy, dz = self.grid.dx
        lx, ly, lz = self.mset.grid.L

        aspect_ratio = lz / lx * (self.mset.dsqr ** 0.5)

        kh_max = ncp.pi / dx
        hor_diff_coeff = velocity_scale * rossby_number / kh_max**3

        kv_max = ncp.pi / dz
        ver_diff_coeff = aspect_ratio * velocity_scale * rossby_number / kv_max**3

        self._kh = self.kh * hor_diff_coeff
        self._kv = self.kv * ver_diff_coeff

    @fr.utils.jaxjit
    def _compute_tendency(self, z: nh.State, dz: nh.State) -> nh.State:
        diff = self.diff_module.diff
        for f in z:
            # first two derivatives
            f_hor = (diff(f, axis=0, order=2) +
                          diff(f, axis=1, order=2) )
            f_ver = diff(f, axis=2, order=2)

            # multiply diffusion coefficients
            f_hor *= self._kh
            f_ver *= self._kv

            # second two derivatives
            dz[f.name] -= (diff(f_hor, axis=0, order=2) +
                           diff(f_hor, axis=1, order=2) +
                           diff(f_ver, axis=2, order=2) )
        return dz

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        mz.dz = self._compute_tendency(mz.z, mz.dz)
        return mz
