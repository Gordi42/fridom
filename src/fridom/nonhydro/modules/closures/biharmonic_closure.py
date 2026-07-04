"""Biharmonic friction + mixing for the nonhydrostatic model."""
from __future__ import annotations

from functools import partial
from typing import Literal

import fridom.framework as fr


@partial(fr.utils.jaxify,
         dynamic=("_kh", "_kv", "_hor_diff_coeff", "_ver_diff_coeff",
                  "_water_mask"))
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
    def __init__(self,
                 kh: float,
                 kv: float,
                 velocity_scale: float = 1.0,
                 # free-slip and no-slip modes
                 mode: Literal["free-slip", "no-slip"] = "free-slip",
                 ) -> None:
        super().__init__()
        self._kh = kh
        self._kv = kv
        self.velocity_scale = velocity_scale
        self.required_halo = 2
        self._hor_diff_coeff = None
        self._ver_diff_coeff = None
        self._hor_grid_coeff = None
        self._ver_grid_coeff = None

        match mode:
            case "free-slip":
                self._second_derivative = self._second_derivative_free_slip
            case "no-slip":
                self._second_derivative = self._second_derivative_no_slip
            case _:
                msg = (f"Invalid mode '{mode}'."
                       " Must be 'free-slip' or 'no-slip'.")
                raise ValueError(msg)

    def _on_setup(self) -> None:
        ncp = fr.config.ncp
        rossby_number = self.mset.rossby_number
        velocity_scale = self.velocity_scale

        dx, _dy, dz = self.grid.dx
        lx, _ly, lz = self.mset.grid.domain_size

        aspect_ratio = lz / lx * (self.mset.dsqr ** 0.5)

        kh_max = ncp.pi / dx
        hor_diff_coeff = velocity_scale * rossby_number / kh_max**3

        kv_max = ncp.pi / dz
        ver_diff_coeff = (aspect_ratio * velocity_scale * rossby_number
                          / kv_max**3)

        self._hor_grid_coeff = hor_diff_coeff
        self._ver_grid_coeff = ver_diff_coeff

        self._hor_diff_coeff = self._kh * hor_diff_coeff
        self._ver_diff_coeff = self._kv * ver_diff_coeff

        self._water_mask = self.mset.grid.water_mask

    def _second_derivative_no_slip(self,
                                   f: fr.ScalarField,
                                   axis: int) -> fr.ScalarField:
        # when not applying the water mask, results look like a no-slip
        # condition. I am not entirely sure why, maybe because gradients
        # at the boundary can be non-zero when not applying the water
        # mask, which leads to stronger diffusion at the boundary leading
        # to a no-slip like condition
        first_derivative = self.diff_module.diff(f, axis)
        return self.diff_module.diff(first_derivative, axis)

    def _second_derivative_free_slip(self,
                                     f: fr.ScalarField,
                                     axis: int) -> fr.ScalarField:
        first_derivative = self.diff_module.diff(f, axis)
        first_derivative = self._water_mask.apply_mask(first_derivative)
        second_derivative = self.diff_module.diff(first_derivative, axis)
        return self._water_mask.apply_mask(second_derivative)

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        for f in mz.z:
            # first two derivatives
            f_hor = (self._second_derivative(f, 0)
                     + self._second_derivative(f, 1))
            f_ver = self._second_derivative(f, 2)

            # multiply diffusion coefficients
            f_hor *= self._hor_diff_coeff
            f_ver *= self._ver_diff_coeff

            # second two derivatives
            mz.dz[f.name] -= (self._second_derivative(f_hor, 0) +
                           self._second_derivative(f_hor, 1) +
                           self._second_derivative(f_ver, 2))

        return mz

    @property
    def kh(self) -> float:
        """The horizontal diffusion coefficient."""
        return self._kh

    @kh.setter
    def kh(self, value: float) -> None:
        self._hor_diff_coeff = value * self._hor_grid_coeff
        self._kh = value

    @property
    def kv(self) -> float:
        """The vertical diffusion coefficient."""
        return self._kv

    @kv.setter
    def kv(self, value: float) -> None:
        self._ver_diff_coeff = value * self._ver_grid_coeff
        self._kv = value
