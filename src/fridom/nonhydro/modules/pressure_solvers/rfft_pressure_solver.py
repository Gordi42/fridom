"""Pressure solver using rfft."""
from __future__ import annotations

from functools import partial

import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

import fridom.framework as fr
import fridom.nonhydro as nh


@partial(fr.utils.jaxify, dynamic=("k_squared_inv",))
class RFFTPressureSolver(fr.modules.Module):

    """Solve for the pressure field with a spectral solver."""

    name = "Pressure Solver (RFFTN)"

    def _on_setup(self) -> None:
        self._check_grid_type()
        self._set_dct_axes()
        self._check_multiple_gpus()
        self._determine_rfft_axis()
        self._setup_k_squared_inv()
        self._setup_transform_functions()

    def _check_grid_type(self) -> None:
        if not isinstance(self.mset.grid, nh.grid.cartesian.Grid):
            msg = "The spectral solver only supports cartesian grids."
            raise TypeError(msg)

    def _set_dct_axes(self) -> None:
        periodic_axes = self.grid.periodic_bounds
        self.fft_axes = {i for i, periodic in enumerate(periodic_axes)
                         if periodic}
        self.dct_axes = {i for i, periodic in enumerate(periodic_axes)
                         if not periodic}

        # we don't need to perform any transform in axis with only one
        # grid point
        single_point_axes = {i for i, N in enumerate(self.grid.shape)
                             if N == 1}
        self.fft_axes -= single_point_axes
        self.dct_axes -= single_point_axes

    def _check_multiple_gpus(self) -> None:
        dd = self.mset.grid.domain_decomp
        if (isinstance(dd, fr.domain_decomposition.JaxDecomposition)
                and dd.n_devices > 1):
            self.multiple_gpus = True
            return
        self.multiple_gpus = False

    def _determine_rfft_axis(self) -> None:
        if self.multiple_gpus:
            self._determine_rfft_axis_multiple_gpus()
        else:
            self._determine_rfft_axis_single_gpu()

    def _determine_rfft_axis_single_gpu(self) -> None:
        if len(self.fft_axes) == 0:
            self.rfft_axis = None
        else:
            self.rfft_axis = max(self.fft_axes)

    def _setup_k_squared_inv(self) -> None:
        # use jaxjit here to avoid allocating memory for intermediate arrays
        # when computing the wave numbers
        @fr.utils.jaxjit
        def _setup_k_squared_inv() -> jnp.ndarray:
            grid = self.mset.grid
            k = list(grid.k_global)
            if self.rfft_axis is not None:
                i = self.rfft_axis
                k[self.rfft_axis] = jnp.fft.rfftfreq(
                    grid.shape[i], d=grid.dx[i] / (2 * np.pi),
                )
            dso = fr.grid.cartesian.discrete_spectral_operators
            k = [dso.k_hat_squared(kx, dx, use_discrete=True)
                    for (kx,dx) in zip(k, grid.dx, strict=False)]
            k = grid.domain_decomp.create_meshgrid(
                *k, pad=False, spectral=True)
            k_squared = k[0] + k[1] + k[2] / self.mset.dsqr
            with np.errstate(divide="ignore", invalid="ignore"):
                k_squared_inv = 1 / k_squared
            return - jnp.where(k_squared == 0, 0, k_squared_inv)

        self.k_squared_inv = _setup_k_squared_inv()

    def _setup_transform_functions(self) -> None:
        if self.rfft_axis is None:
            self._setup_default_transform_functions()
            return
        if self.multiple_gpus:
            self._setup_transform_functions_multiple_gpus()
        else:
            self._setup_transform_functions_single_gpu()

    def _setup_default_transform_functions(self) -> None:
        def forward_transform(
                x: jnp.ndarray, axes: set[int] | None = None) -> jnp.ndarray:
            axes = axes or set(range(self.mset.grid.n_dims))
            axes = set(axes)
            dct_axes = axes & self.dct_axes
            fft_axes = axes & self.fft_axes
            if len(dct_axes) > 0:
                x = jsp.fft.dctn(x, type=2, axes=list(dct_axes))
            if len(fft_axes) > 0:
                x = jnp.fft.fftn(x, axes=list(fft_axes))
            return x

        def backward_transform(
                x: jnp.ndarray, axes: set[int] | None = None) -> jnp.ndarray:
            axes = axes or set(range(self.mset.grid.n_dims))
            axes = set(axes)
            dct_axes = axes & self.dct_axes
            fft_axes = axes & self.fft_axes
            if len(fft_axes) > 0:
                x = jnp.fft.ifftn(x, axes=list(fft_axes))
            if len(dct_axes) > 0:
                x = jsp.fft.idctn(x, type=2, axes=list(dct_axes))
            return x

        dd = self.mset.grid.domain_decomp
        self.forward_transform = dd.parallel_forward_transform(
            forward_transform)
        self.backward_transform = dd.parallel_backward_transform(
            backward_transform)

    def _setup_transform_functions_single_gpu(self) -> None:
        dd = self.mset.grid.domain_decomp
        def forward_transform(x: jnp.ndarray) -> jnp.ndarray:
            x = dd.unpad(x)
            if len(self.dct_axes) > 0:
                x = jsp.fft.dctn(x, type=2, axes=list(self.dct_axes))
            if len(self.fft_axes) > 0:
                x = jnp.fft.rfftn(x, axes=list(self.fft_axes))
            return x

        def backward_transform(x: jnp.ndarray) -> jnp.ndarray:
            if len(self.fft_axes) > 0:
                x = jnp.fft.irfftn(x, axes=list(self.fft_axes))
            if len(self.dct_axes) > 0:
                x = jsp.fft.idctn(x, type=2, axes=list(self.dct_axes))
            return dd.sync(dd.pad(x))
        self.forward_transform = forward_transform
        self.backward_transform = backward_transform

    def _setup_transform_functions_multiple_gpus(self) -> None:
        def forward_transform(
                x: jnp.ndarray, axes: set[int] | None = None) -> jnp.ndarray:
            axes = axes or set(range(self.mset.grid.n_dims))
            axes = set(axes)
            dct_axes = axes & self.dct_axes
            rfft_axis = self.rfft_axis
            fft_axes = axes & self.fft_axes - {rfft_axis}
            if len(dct_axes) > 0:
                x = jsp.fft.dctn(x, type=2, axes=list(dct_axes))
            if rfft_axis in axes:
                x = jnp.fft.rfft(x, axis=rfft_axis)
            if len(fft_axes) > 0:
                x = jnp.fft.fftn(x, axes=list(fft_axes))
            return x

        def backward_transform(
                x: jnp.ndarray, axes: set[int] | None = None) -> jnp.ndarray:
            axes = axes or set(range(self.mset.grid.n_dims))
            axes = set(axes)
            dct_axes = axes & self.dct_axes
            rfft_axis = self.rfft_axis
            fft_axes = axes & self.fft_axes - {rfft_axis}
            if len(fft_axes) > 0:
                x = jnp.fft.ifftn(x, axes=list(fft_axes))
            if rfft_axis in axes:
                x = jnp.fft.irfft(x, axis=rfft_axis)
            if len(dct_axes) > 0:
                x = jsp.fft.idctn(x, type=2, axes=list(dct_axes))
            return x

        dd = self.mset.grid.domain_decomp
        self.forward_transform = dd.parallel_forward_transform(
            forward_transform)
        self.backward_transform = dd.parallel_backward_transform(
            backward_transform)

    @fr.modules.module_method
    def update(self, mz: fr.ModelState) -> fr.ModelState:  # noqa: D102
        div_hat = self.forward_transform(mz.z_diag.div.arr)
        mz.z_diag.p.arr = self.backward_transform(
            div_hat * self.k_squared_inv).real
        return mz

    @property
    def info(self) -> dict:  # noqa: D102
        res = super().info
        res["Solver"] = "Spectral (RFFTN)"
        return res
